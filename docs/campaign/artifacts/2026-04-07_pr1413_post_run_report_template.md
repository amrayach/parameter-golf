# PR #1413 A/B/C/D/E Batch Results — RunPod 8xH100 SXM

Date: 2026-04-__
Branch: `07c1-base-runpod-strict-submission`

## 1. Run Environment

| Field | Value |
|-------|-------|
| Pod ID | ___ |
| GPU Type | 8x NVIDIA H100 80GB HBM3 SXM |
| Template ID | ___ |
| Cost/hr | $21.52 |
| Repo commit | ___ |
| PR_REF_HEAD (pr1413) | ___ |
| Branch | `07c1-base-runpod-strict-submission` |
| RUN_STAMP | ___ |
| PyTorch version | ___ |
| CUDA version | ___ |
| NCCL_NET | Socket |

## 2. Archive Location

Archive root:
```
/workspace/pr1413_archive___/seed0/
```

Per-run subdirectories:

| Run | Archive path |
|-----|-------------|
| A | `/workspace/pr1413_archive___/seed0/pr1413_ctrl_s0/` |
| B | `/workspace/pr1413_archive___/seed0/pr1413_par7_s0/` |
| C | `/workspace/pr1413_archive___/seed0/pr1413_loop35_s0/` |
| D | `/workspace/pr1413_archive___/seed0/pr1413_combo_s0/` |
| E | `/workspace/pr1413_archive___/seed0/pr1413_ngram_eval_s0/` |

Each directory contains:
- `console.txt` (full stdout+stderr)
- `logs/<RUN_ID>.txt` (structured log)
- `final_model.int6.ptz` (quantized checkpoint)
- `final_model.pt` (full checkpoint, absent for Run E)
- `run_meta.env` (env snapshot)
- `source_sha256.txt`
- `artifact_sha256.txt`
- `train_gpt.py`, `README.md`, `submission.json`
- `variant_manifest.json` (B/D/E only)
- `ngram_tilt.py`, `fused_expert_kernel.cpp` (B/D/E only)
- `libfused_ngram.so` (E only, if compiled)

## 3. Per-Run Metrics Table

| Run | RUN_ID | Status | Steps | Train ms | Eval ms (roundtrip) | Eval ms (sliding) | TTT ms | Sliding BPB | TTT BPB | Roundtrip BPB | Artifact bytes | Notes |
|-----|--------|--------|-------|----------|---------------------|-------------------|--------|-------------|---------|---------------|----------------|-------|
| A | `pr1413_ctrl_s0` | ___ | ___/20000 | ___ | ___ | ___ | ___ | ___ | ___ | ___ | ___ | faithful #1413 control |
| B | `pr1413_par7_s0` | ___ | ___/20000 | ___ | ___ | ___ | ___ | ___ | ___ | ___ | ___ | PARALLEL_RESIDUAL_START=7 |
| C | `pr1413_loop35_s0` | ___ | ___/20000 | ___ | ___ | ___ | ___ | ___ | ___ | ___ | ___ | LOOP_START=3 LOOP_END=5 |
| D | `pr1413_combo_s0` | ___ | ___/20000 | ___ | ___ | ___ | ___ | ___ | ___ | ___ | ___ | par7 + loop3-5 |
| E | `pr1413_ngram_eval_s0` | ___ | N/A | N/A | ___ | ___ | ___ | ___ | ___ | ___ | ___ | eval-only tilt on D ckpt |

## 4. Comparison Matrix

### vs Run A (faithful #1413 control)

| Run | Sliding BPB | delta vs A | TTT BPB | delta vs A | Direction |
|-----|-------------|-----------|---------|-----------|-----------|
| A | ___ | 0.00000 | ___ | 0.00000 | baseline |
| B | ___ | ___ | ___ | ___ | ___ |
| C | ___ | ___ | ___ | ___ | ___ |
| D | ___ | ___ | ___ | ___ | ___ |
| E | ___ | ___ | ___ | ___ | ___ |

### vs upstream references

| Run | TTT BPB | delta vs #1413 ref (1.08279) | delta vs #1394 3-seed mean (1.08521) | delta vs #1394 sliding-only (1.08521) |
|-----|---------|------------------------------|--------------------------------------|---------------------------------------|
| A | ___ | ___ | ___ | ___ |
| B | ___ | ___ | ___ | ___ |
| C | ___ | ___ | ___ | ___ |
| D | ___ | ___ | ___ | ___ |
| E | ___ | ___ | ___ | ___ |

Sign convention: negative = better than reference.

## 5. Timing Budget

| Item | Value |
|------|-------|
| Pod start time (UTC) | ___ |
| Pod stop time (UTC) | ___ |
| Total pod wall time | ___ min |
| Total cost | $ ___ |
| Setup/prep overhead | ___ min |

Per-run breakdown:

| Run | Start (UTC) | End (UTC) | Wall min | Train ms | Eval ms total | Status |
|-----|-------------|-----------|----------|----------|---------------|--------|
| A | ___ | ___ | ___ | ___ | ___ | ___ |
| B | ___ | ___ | ___ | ___ | ___ | ___ |
| C | ___ | ___ | ___ | ___ | ___ | ___ |
| D | ___ | ___ | ___ | ___ | ___ | ___ |
| E | ___ | ___ | ___ | ___ | ___ | ___ |

Estimated total run time (5 runs x ~17 min each): ~85 min
Estimated total cost at $21.52/hr: ~$30.48

## 6. Interpretation

### Winner Identification

Best sliding BPB (pre-TTT): Run ___ at ___
Best TTT BPB (post-TTT): Run ___ at ___
Best overall: Run ___ (reason: ___)

### Component Attribution

| Component | Isolated effect (sliding) | Isolated effect (TTT) | Source |
|-----------|--------------------------|----------------------|--------|
| PARALLEL_RESIDUAL_START=7 | B minus A = ___ | B minus A = ___ | B vs A |
| LOOP_START=3 LOOP_END=5 | C minus A = ___ | C minus A = ___ | C vs A |
| Interaction term | D minus (B+C-A) = ___ | D minus (B+C-A) = ___ | additivity check |
| N-gram tilt (eval-only) | N/A | E minus D = ___ | E vs D |

Additivity check: if interaction term is near zero, components are orthogonal.

### Reproduction Fidelity Assessment

| Metric | Upstream #1413 ref | Run A (our control) | Delta | Acceptable? |
|--------|-------------------|---------------------|-------|-------------|
| TTT BPB | 1.08279 | ___ | ___ | ___ |
| Steps at stop | ~5088 | ___ | ___ | ___ |
| Train time | ~588,000 ms | ___ | ___ | ___ |
| Artifact bytes | ~15,991,000 | ___ | ___ | ___ |

If Run A delta from upstream > 0.001 BPB, investigate before trusting B/C/D/E deltas.

## 7. Artifact Status

| Run | Code bytes | Model bytes (.int6.ptz) | Total bytes | Under 16MB cap? | Margin |
|-----|-----------|------------------------|-------------|-----------------|--------|
| A | ___ | ___ | ___ | ___ | ___ |
| B | ___ | ___ | ___ | ___ | ___ |
| C | ___ | ___ | ___ | ___ | ___ |
| D | ___ | ___ | ___ | ___ | ___ |
| E | ___ | ___ | ___ | ___ | ___ |

Cap: 16,000,000 bytes decimal.
Expected code bytes: ~16,719 (LocalBase) or ~17,390 (ParallelResid7 stack).

## 8. Next Decision

Winner: Run ___
Action: ___

| Option | Condition | Seeds needed | Est cost |
|--------|-----------|-------------|----------|
| Promote Run ___ to 3-seed validation | TTT BPB < 1.08279 by > 0.0005 | 3 (seeds 0, 42, 1337) | ~$15 (3 x ~17 min at $21.52/hr) |
| Promote Run ___ + tilt (E) to 3-seed | E improves on D by > 0.0003 | 3 | ~$15 |
| Abandon batch, return to #1394 base | No run beats A by > 0.0003 | 0 | $0 |
| Try different PARALLEL_RESIDUAL_START | B shows promise but not enough | 1 per value | ~$6/run |

Kill criterion: if best variant minus A < -0.0003 BPB, promote. If all variants are within noise of A (+/-0.0003), the modifications are neutral and not worth pursuing.

## 9. Appendix: Log Extraction Commands

Run these from the archive root (`/workspace/pr1413_archive___/seed0/`).

### Training phase

```bash
# Final training step and wallclock cap
grep 'stopping_early' */console.txt

# Last training step logged (step count, loss, time, tok/s)
grep -E '^[0-9]+/20000 train_loss' */console.txt | sort -t/ -k1 -n | tail -5

# Layer loop activation
grep 'layer_loop:enabled' */console.txt

# Peak GPU memory
grep 'peak memory' */console.txt
```

### EMA and serialization

```bash
# EMA application
grep 'ema:applying' */console.txt

# Pre-quantization post-EMA eval
grep 'pre-quantization post-ema' */console.txt

# Serialized model sizes (uncompressed and compressed)
grep 'Serialized model' */console.txt

# Code size
grep 'Code size' */console.txt

# Total submission size
grep 'Total submission' */console.txt
```

### GPTQ quantization

```bash
# GPTQ Hessian collection
grep 'GPTQ:collected' */console.txt
```

### Eval phase (the key results)

```bash
# Roundtrip exact BPB (quantized model, no sliding window)
grep 'quantized val_loss' */console.txt

# Sliding window BPB (pre-TTT)
grep 'quantized_sliding_window' */console.txt

# TTT final result (the primary metric for #1413 runs)
grep 'ttt_sliding:done' */console.txt

# Eval times (in the val_loss/val_bpb lines)
grep -E 'eval_time=[0-9]+ms' */console.txt
```

### N-gram tilt (Run E only)

```bash
# Tilt precompute stats
grep 'ngram_tilt:precompute' pr1413_ngram_eval_s0/console.txt

# Tilt final result (same ttt_sliding:done line with tilt applied)
grep 'ttt_sliding:done' pr1413_ngram_eval_s0/console.txt
```

### Quick one-liner summary extraction

```bash
# Pull the three critical BPB values per run
for d in pr1413_*/; do
  echo "=== ${d%/} ==="
  grep 'quantized val_loss' "$d/console.txt" 2>/dev/null | tail -1
  grep 'quantized_sliding_window' "$d/console.txt" 2>/dev/null | tail -1
  grep 'ttt_sliding:done' "$d/console.txt" 2>/dev/null | tail -1
  echo
done
```

### Artifact size verification

```bash
# Actual compressed model file size
for d in pr1413_*/; do
  echo -n "${d%/}: "
  ls -l "$d/final_model.int6.ptz" 2>/dev/null | awk '{print $5}'
done

# Cross-check with Total submission size from log
grep 'Total submission' */console.txt
```

### Run identity verification

```bash
# Confirm env overrides per run
grep -E 'PARALLEL_RESIDUAL_START|LOOP_START|LOOP_END|SKIP_TRAINING|NGRAM_TILT_ENABLED' */run_meta.env
```

### Timing extraction

```bash
# Wall time from stopping_early line (train_time field)
grep -oP 'train_time: \K[0-9]+ms' */console.txt

# Eval elapsed from ttt_sliding:done line
grep -oP 'elapsed=\K[0-9.]+s' */console.txt
```
