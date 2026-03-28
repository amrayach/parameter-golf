# A100 Development Evidence Summary

Date: 2026-03-28
Hardware: NVIDIA A100-SXM4-80GB (serv-3333, Pegasus A100-80GB partition)
Purpose: Development evidence for compute grant application

---

## Runs Completed

### Run 1: Smoke Test (200 iterations, reduced batch)

| Metric | Value |
|--------|-------|
| RUN_ID | a100_baseline_smoke |
| Config | 9L 512d, batch 65536, 200 iter, 1 shard |
| Steps | 200/200 |
| Final post-quant val_bpb | 2.1737 |
| Step avg | 154.57 ms |
| Peak memory | 1548 MiB |
| Artifact int8+zlib | 7,066,088 bytes |

### Run 2: 600s Baseline (full batch, wallclock-capped)

| Metric | Value |
|--------|-------|
| RUN_ID | a100_baseline_600s |
| Config | 9L 512d, batch 524288, 10 shards, 600s cap |
| Steps | 907/20000 |
| Final post-quant val_bpb | **1.3714** |
| Step avg | 661.65 ms |
| Peak memory | 10253 MiB |
| Artifact int8+zlib | 12,046,627 bytes |
| Val BPB progression | 4.11 → 1.66 → 1.50 → 1.43 → 1.38 → 1.37 |

### Run 3: 600s LowerLR Comparison (controlled variant)

| Metric | Value |
|--------|-------|
| RUN_ID | a100_lowerlr_600s |
| Config | Same as baseline, MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03 |
| Steps | 908/20000 |
| Final post-quant val_bpb | **1.3776** |
| Step avg | 661.00 ms |
| Peak memory | 10253 MiB |
| Artifact int8+zlib | 10,723,611 bytes |
| Val BPB progression | 4.11 → 1.64 → 1.49 → 1.42 → 1.38 → 1.37 |

### Run 4: 600s Baseline Reproducibility Check (seed 42)

| Metric | Value |
|--------|-------|
| RUN_ID | a100_baseline_seed42_600s |
| Config | Same as baseline, `SEED=42` |
| Steps | 900/20000 |
| Final post-quant val_bpb | **1.3746** |
| Step avg | 667.26 ms |
| Peak memory | 10253 MiB |
| Artifact int8+zlib | 12,018,778 bytes |
| Val BPB progression | 4.11 → 1.65 → 1.50 → 1.43 → 1.38 → 1.37 |

### Run 5: 600s Warmdown-Only Variant

| Metric | Value |
|--------|-------|
| RUN_ID | a100_warmdown3600_600s |
| Config | Same as baseline, `WARMDOWN_ITERS=3600` |
| Steps | 903/20000 |
| Final post-quant val_bpb | **1.4106** |
| Step avg | 665.18 ms |
| Peak memory | 10253 MiB |
| Artifact int8+zlib | 9,951,155 bytes |
| Val BPB progression | 4.11 → 1.72 → 1.52 → 1.45 → 1.41 → 1.40 |

---

## Controlled Comparison

| Metric | Baseline (default LR) | LowerLR | Delta |
|--------|----------------------|---------|-------|
| MATRIX_LR | 0.04 | 0.02 | -50% |
| SCALAR_LR | 0.04 | 0.02 | -50% |
| TIED_EMBED_LR | 0.05 | 0.03 | -40% |
| val_bpb | **1.3714** | 1.3776 | +0.0062 |
| Artifact size | 12.0 MB | 10.7 MB | -1.3 MB |

Interpretation: Default LR produces marginally better val_bpb. Lower LR produces a smaller artifact (lower weights = better compressibility). Both runs structurally identical in step count, memory, and stability.

## Reproducibility Check

| Metric | Baseline seed 1337 | Baseline seed 42 | Delta |
|--------|--------------------|------------------|-------|
| val_bpb | **1.3714** | 1.3746 | +0.0032 |
| Steps | 907 | 900 | -7 |
| Step avg | 661.65 ms | 667.26 ms | +5.61 ms |
| Artifact size | 12.05 MB | 12.02 MB | -0.03 MB |

Interpretation: The baseline remains materially stable across two seeds on 1xA100. The seed-42 run is slightly worse, but the spread is small enough to support a grant claim of reproducible operator behavior rather than one-off luck.

## Schedule Negative Control

| Metric | Baseline | Warmdown3600 | Delta |
|--------|----------|--------------|-------|
| WARMDOWN_ITERS | 1200 | 3600 | +2400 |
| val_bpb | **1.3714** | 1.4106 | +0.0392 |
| Steps | 907 | 903 | -4 |
| Step avg | 661.65 ms | 665.18 ms | +3.53 ms |
| Artifact size | 12.05 MB | 9.95 MB | -2.10 MB |

Interpretation: Extending warmdown alone is clearly harmful on this 1xA100 600s setup. It improves compressibility but hurts the actual objective. This is a useful negative control because it shows the pipeline is discriminating between plausible schedule changes rather than producing noisy ties.

---

## What This Demonstrates

1. **End-to-end pipeline works:** Training, evaluation, int8 quantization, zlib compression, and post-quantization round-trip validation all execute correctly on Pegasus.
2. **AMP dtype auto-detection works:** bf16 selected automatically on A100.
3. **Controlled experimentation capability:** Baseline vs variant comparison with only LR changed, a baseline seed-repeat for reproducibility, and a clearly negative warmdown-only schedule test.
4. **Artifact fits challenge budget:** Both runs produce artifacts well under the 16 MB cap.
5. **Operator readiness:** Dataset download, environment setup, and training execution completed without external assistance on shared HPC infrastructure.

## What This Does Not Demonstrate

- H100 SXM parity (A100 step time ~661ms vs expected ~85ms on 8xH100)
- Competitive val_bpb (baseline 1.37 on 1xA100 vs leaderboard 1.22 on 8xH100)
- Multi-GPU distributed training
- Advanced techniques (int6, XSA, EMA, TTT, etc.)

## Hardware Note

These runs used `1xA100-SXM4-80GB`. The challenge leaderboard target is `8xH100-SXM5`. This summary therefore demonstrates development readiness and reproducibility on available Pegasus hardware, not challenge-parity performance. H100-class time is still needed for meaningful parity claims and for measuring the pre-TTT anchor under closer-to-target conditions.

---

## Grant Application Support

This evidence package supports a Development-level compute grant request:
- Demonstrates working infrastructure and operator competence
- Shows controlled experimental methodology
- Shows small seed sensitivity on the current baseline
- Shows that plausible schedule changes can be rejected cleanly with current evidence
- Provides a clear path from current state to competitive submission
- H100 SXM time is needed for challenge-parity validation and the pre-TTT anchor port
