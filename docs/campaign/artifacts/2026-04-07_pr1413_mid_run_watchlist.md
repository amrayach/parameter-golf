# PR #1413 Batch Run Operator Watchlist

Pod: RunPod 8xH100 SXM | Batch script: `scripts/runpod_1413_batch.sh`
Archive root: `/workspace/pr1413_archive_<STAMP>/seed0/`

---

## Per-Run Severity Reference

### RED -- stop the batch immediately

```
grep -iE 'CUDA error|OutOfMemoryError|NCCL timeout|NCCL error|nan|ChildFailedError|FAILED' console.txt
```

| Pattern | Meaning |
|---------|---------|
| `CUDA error` or `OutOfMemoryError` | GPU OOM or driver fault; all later runs will fail too |
| `NCCL timeout` or `NCCL error` | Inter-GPU comms broken; pod may need restart |
| `train_loss: nan` or `val_loss: nan` | Divergence; config error or numerical instability |
| `train_gpt.py FAILED` | torchrun reports child process crash |
| `ChildFailedError` | Same as above, in torchrun wrapper |
| `[rank*]: Traceback` within first 60s | Startup crash (missing data, import error) |
| `No such file or directory` (tokenizer/data) | PREPARE_SP8192 failed silently or data symlinks broken |

### YELLOW -- note it, continue, investigate after batch

| Pattern | Meaning |
|---------|---------|
| `Total submission size quantized+brotli:` > 16000000 | Artifact exceeds 16MB limit; run is not submittable |
| `stopping_early` at step < 3000 | Training terminated abnormally early |
| `tok/s:` < 5000000 | Throughput 35%+ below expected ~7.7M tok/s |
| `quantized val_bpb:` > 1.15 (A) or > 0.05 delta from A (B/C/D) | Quality regression worth investigating |
| `peak memory allocated:` > 70000 MiB | Near OOM territory |
| `GPTQ:collected 0 Hessians` | Quantization will be garbage |

### GREEN -- expected, proceed

| Pattern | Meaning |
|---------|---------|
| `OMP_NUM_THREADS environment variable` warning | Standard torchrun boilerplate |
| `stopping_early: wallclock_cap` | Normal; training hits 588s budget |
| `step: 5000-5200/20000` | Expected step count for ~588s at ~7.7M tok/s |
| `peak memory allocated: 34000-36000 MiB` | Normal H100 usage |
| `tok/s: 6800000-7800000` | Healthy throughput range |

---

## Per-Run File Checklist (A/B/C/D)

After each training run completes, verify these exist in `<ARCHIVE_DIR>/<RUN_ID>/`:

```
ls -la console.txt run_meta.env source_sha256.txt artifact_sha256.txt \
      train_gpt.py submission.json \
      logs/<RUN_ID>.txt \
      final_model.pt final_model.int6.ptz
```

| File | Must exist | Sanity check |
|------|-----------|--------------|
| `console.txt` | YES | Non-empty, last line is not a traceback |
| `logs/<RUN_ID>.txt` | YES | Contains `quantized_sliding_window val_loss:` |
| `final_model.pt` | YES (A-D) | ~135MB (raw float checkpoint) |
| `final_model.int6.ptz` | YES | < 16MB minus code size (~15.97MB) |
| `run_meta.env` | YES | Verify SEED, LOOP_START, PARALLEL_RESIDUAL_START match intent |
| `source_sha256.txt` | YES | Matches across runs using same RECORD_REL |
| `artifact_sha256.txt` | YES | Exists after copy step |

### Success grep (must all match in logs/<RUN_ID>.txt)

```bash
# Training completed
grep 'stopping_early: wallclock_cap' logs/<RUN_ID>.txt

# EMA applied
grep 'ema:applying EMA weights' logs/<RUN_ID>.txt

# Pre-quant eval ran
grep 'pre-quantization post-ema val_loss:' logs/<RUN_ID>.txt

# GPTQ ran
grep 'GPTQ:collected.*Hessians' logs/<RUN_ID>.txt

# Artifact size printed
grep 'Total submission size quantized+brotli:' logs/<RUN_ID>.txt

# Final quantized eval ran
grep 'quantized val_loss:' logs/<RUN_ID>.txt

# Sliding window eval ran (this is the headline number)
grep 'quantized_sliding_window val_loss:' logs/<RUN_ID>.txt
```

If TTT is active (PR #1413 has legal score-first TTT by default), also expect:

```bash
# TTT started
grep 'ttt_sliding:start' logs/<RUN_ID>.txt

# TTT chunk progress (should reach final chunk)
grep 'ttt_chunk \[.*/.*\]' logs/<RUN_ID>.txt

# TTT completed
grep 'ttt_sliding:done' logs/<RUN_ID>.txt

# Legal TTT final number (headline BPB)
grep 'legal_ttt_exact val_loss:' logs/<RUN_ID>.txt
```

### Quick size check

```bash
# Artifact must be under 16,000,000 bytes total
stat -c%s final_model.int6.ptz
# Expected: ~15,968,000-15,990,000 bytes for the model
# Code size adds ~17,000-18,000 bytes
# Total must be < 16,000,000
```

---

## Run-Specific Notes

| Run | RUN_ID | Record dir | Key env overrides | What to watch |
|-----|--------|-----------|-------------------|---------------|
| A | pr1413_ctrl_s0 | LocalBase | (none - faithful control) | Baseline BPB; PREPARE_SP8192=1 runs here |
| B | pr1413_par7_s0 | ParallelResid7 | PARALLEL_RESIDUAL_START=7 | BPB delta vs A |
| C | pr1413_loop35_s0 | LocalBase | LOOP_START=3 LOOP_END=5 | BPB delta vs A |
| D | pr1413_combo_s0 | ParallelResid7 | PAR_RESID=7 + LOOP=3-5 | BPB; its checkpoint feeds E |
| E | pr1413_ngram_eval_s0 | ParallelResid7 | SKIP_TRAINING=1 NGRAM_TILT_ENABLED=1 | Uses D's .int6.ptz; eval-only |

### Expected A baseline (from PR #1413 upstream reference)

- `quantized_sliding_window val_bpb`: ~1.082-1.086
- `legal_ttt val_bpb`: ~1.080-1.084
- `Total submission size`: < 16,000,000
- `train_time`: ~588,000ms (wallclock cap)
- `step`: ~5000-5100

---

## Pre-E Gate Checklist

Verify ALL before allowing run E to proceed. E depends on D's checkpoint.

### 1. D checkpoint exists and is valid

```bash
# Must exist in the RECORD dir (not just archive), because E reads from it
ls -la records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/final_model.int6.ptz

# Should be ~15.97MB
stat -c%s records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/final_model.int6.ptz
```

If the file does not exist, D either crashed or the copy step failed. **Do not run E.**

### 2. D's BPB is reasonable

```bash
grep 'quantized_sliding_window val_loss:.*val_bpb:' <D_ARCHIVE>/logs/pr1413_combo_s0.txt
```

- If D's sliding BPB is > 0.010 worse than A's, something is wrong with the combo. Note it as YELLOW but let E run anyway (the tilt might still be informative).
- If D's sliding BPB is > 0.030 worse than A's, or D's log shows nan/crash, **do not run E**.

### 3. g++ is available (for ngram kernel compilation)

```bash
command -v g++ && echo OK || echo "MISSING -- apt-get install -y g++"
```

### 4. Time budget

E is eval-only with TTT + ngram tilt. Expected wall time: ~5-8 minutes.
No training budget concern, but if the pod billing window is tight, verify enough time remains.

---

## E-Specific Checks

Run E does NOT produce `final_model.pt` (no training). Expected outputs:

| File | Must exist | Notes |
|------|-----------|-------|
| `console.txt` | YES | |
| `logs/pr1413_ngram_eval_s0.txt` | YES | |
| `final_model.int6.ptz` | YES | Copied from D's record dir, not newly created |
| `final_model.pt` | NO | Script prints "Note: final_model.pt not found" -- this is expected |
| `libfused_ngram.so` | YES | Compiled from fused_expert_kernel.cpp |

### E success patterns

```bash
# Ngram kernel compiled
grep 'ngram_tilt:precompute' logs/pr1413_ngram_eval_s0.txt

# TTT with tilt completed
grep 'ttt_sliding:done' logs/pr1413_ngram_eval_s0.txt
grep 'legal_ttt_exact val_loss:' logs/pr1413_ngram_eval_s0.txt
```

### E failure patterns (RED)

```bash
# Kernel compilation failed
grep -i 'error.*fused_expert_kernel\|cannot find.*g++\|compilation failed' console.txt

# Checkpoint load failed
grep -i 'No such file.*int6.ptz\|Error loading' console.txt
```

---

## Quick Comparison Table (fill in after each run)

```
Run | sliding_bpb | ttt_bpb | submission_bytes | steps | train_ms | status
----|-------------|---------|------------------|-------|----------|-------
A   |             |         |                  |       |          |
B   |             |         |                  |       |          |
C   |             |         |                  |       |          |
D   |             |         |                  |       |          |
E   |     n/a     |         |       n/a        |  n/a  |   n/a    |
```

Extract with:
```bash
for f in */logs/*.txt; do
  echo "=== $f ==="
  grep -E 'quantized_sliding_window|legal_ttt_exact|Total submission size|stopping_early' "$f"
done
```
