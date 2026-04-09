# 2026-04-09 Pre-Launch Audit

Scope:

- `scripts/run_pod_batch.sh`
- regenerated wrapper at `records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/train_gpt.py`

Wrapper verification:

- Decoded the LZMA/base85 wrapper and verified its SHA256 matches `decoded_source_sha256` in `records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/variant_manifest.json:4`.
- Decoded SHA256: `897388636a2fbf42848ce3993f6d361e9947f1482e5a818050daaf90332f52e9`

## File 1: `scripts/run_pod_batch.sh`

### 1. Env var correctness per run

- PASS: All runs inherit the required common overrides `SEED=0 PARALLEL_RESIDUAL_START=7 LOOP_START=3 LOOP_END=5` from `scripts/run_pod_batch.sh:96-107`.
- PASS: R1-R4 match the documented eval-only configs, with only `TTT_ENABLED=1` added explicitly. That extra flag is already the default in `scripts/runpod_1413.sh:124-125`, so it does not change the intended config. Evidence: `scripts/run_pod_batch.sh:155-186`, `scripts/runpod_1413.sh:163-178`.
- FAIL: `tier2` mode can silently run the wrong eval config. `best_eval_env()` claims to “fall back to R1 baseline” but actually returns an empty string when tier 1 has not run or all tier-1 BPB parses fail. The caller uses `eval_env_str=$(best_eval_env 2>/dev/null || echo "NGRAM_TILT_ENABLED=1")`, but an empty echo is still exit code 0, so the fallback never fires. That leaves R6-R9 eval passes without `NGRAM_TILT_ENABLED=1` and without any best-TTT settings. Evidence: `scripts/run_pod_batch.sh:239-257`, `scripts/run_pod_batch.sh:270-273`, `scripts/run_pod_batch.sh:313-319`, `scripts/run_pod_batch.sh:337-343`, `scripts/run_pod_batch.sh:406-412`, `scripts/run_pod_batch.sh:435-444`.
- WARN: The tier-2 eval passes redundantly pass training-side flags (`CAUTIOUS_MUON`, `OWC_*`, `CDQUANT_*`) even though `SKIP_TRAINING=1` only deserializes the existing quantized artifact and never re-trains or re-quantizes. This does not change BPB, but it makes the eval-pass env strings look more active than they really are. Evidence: `scripts/run_pod_batch.sh:313-319`, `scripts/run_pod_batch.sh:337-343`, `scripts/run_pod_batch.sh:406-412`, `scripts/run_pod_batch.sh:435-444`; decoded wrapper behavior in `train_gpt.py` is `skip_training -> deserialize -> eval` only.

### 2. BPB extraction

- FAIL: The primary extraction regex does not match the actual final eval log format. `scripts/run_pod_batch.sh:59` expects `legal_ttt_exact ... val_bpb=...`, but the wrapper prints `legal_ttt_exact val_loss:... val_bpb:... eval_time:...`. The same colon-vs-equals mismatch exists for the `quantized_sliding_window` and `quantized` fallbacks in `scripts/run_pod_batch.sh:64-67`. The only pattern that currently matches is `ttt_sliding:done ... val_bpb=...` in `scripts/run_pod_batch.sh:61`, which is an internal log inside the TTT function, not the final timed eval line. Evidence: `scripts/run_pod_batch.sh:53-69`; decoded wrapper lines `423-424`, `495-504`.
- FAIL: This mismatch fails silently. `extract_bpb()` returns the literal string `FAILED` on an otherwise successful run, and `run_experiment()` records that in `summary.tsv` without erroring. Evidence: `scripts/run_pod_batch.sh:69`, `scripts/run_pod_batch.sh:121-124`.
- PASS: For the current batch as written, every run sets `TTT_ENABLED=1`, so the `ttt_sliding:done` fallback will usually recover the TTT BPB. The issue is brittleness and silent misclassification, not immediate complete breakage.

### 3. Checkpoint backup / restore logic

- PASS: The original D checkpoint `final_model.int6.ptz` is backed up before any tier-2 run starts. Evidence: `scripts/run_pod_batch.sh:275-281`.
- PASS: The backup is restored before each new training run. Evidence: `scripts/run_pod_batch.sh:283-288`, `scripts/run_pod_batch.sh:303`, `scripts/run_pod_batch.sh:327`, `scripts/run_pod_batch.sh:394`, `scripts/run_pod_batch.sh:428`.
- PASS: The original checkpoint is restored again after all tier-2 runs complete. Evidence: `scripts/run_pod_batch.sh:449-450`.
- PASS: Each eval-only pass installs the just-trained run’s quantized checkpoint before launching `SKIP_TRAINING=1`, so R7/R8/R9 evals do not accidentally reuse the previous run’s checkpoint. Evidence: `scripts/run_pod_batch.sh:290-297`, `scripts/run_pod_batch.sh:311-319`, `scripts/run_pod_batch.sh:335-343`, `scripts/run_pod_batch.sh:404-412`, `scripts/run_pod_batch.sh:433-444`.
- FAIL: A mid-run failure can leave the record folder dirty. `serialize()` overwrites `final_model.int6.ptz` in place, and `run_pod_batch.sh` has no `trap` to restore the backup on failure. Combined with `set -e`, a failure aborts the batch before the final restore runs. That means a crash during or after serialization can strand a partially overwritten checkpoint in the shared record dir for the next manual retry. Evidence: `scripts/run_pod_batch.sh:2`, `scripts/run_pod_batch.sh:95-109`, `scripts/run_pod_batch.sh:275-288`, `scripts/run_pod_batch.sh:449-450`; decoded wrapper overwrite path at `serialize()` writes `h.quantized_model_path` directly.

### 4. Tier 2 -> E eval pass

- PASS: Each tier-2 variant is structured as two separate runs: one `SKIP_TRAINING=0` training run, then one `SKIP_TRAINING=1` eval-only run on the installed quantized checkpoint. Evidence: `scripts/run_pod_batch.sh:300-446`.
- PASS: The BPB for each `*_eval` row comes from the second pass, because `run_experiment()` captures one run per `output.log`. Evidence: `scripts/run_pod_batch.sh:72-124` together with the separate R6/R7/R8/R9 eval invocations in `scripts/run_pod_batch.sh:317-319`, `341-343`, `410-412`, `442-444`.
- FAIL: The second pass is not guaranteed to be the intended `E`-style eval. In `tier2` mode, or after tier-1 BPB parse failure, the silent-empty `best_eval_env()` path removes `NGRAM_TILT_ENABLED=1` entirely, so the script can look like it completed both passes while evaluating the wrong config. Evidence: `scripts/run_pod_batch.sh:239-257`, `scripts/run_pod_batch.sh:270-273`, `scripts/run_pod_batch.sh:313-319`, `scripts/run_pod_batch.sh:337-343`, `scripts/run_pod_batch.sh:406-412`, `scripts/run_pod_batch.sh:435-444`.
- WARN: The final ranking mixes `_train` and `_eval` rows together instead of filtering to the second-pass eval rows. The run names make the distinction visible, but the ranking itself does not enforce it. Evidence: `scripts/run_pod_batch.sh:478-484`.

### 5. CDQuant timing probe

- FAIL: The “timing probe” does not run CDQuant on a real tensor, does not import the wrapper, and does not call `gptq_quantize_weight()` or any quantization path. It only imports `spec_from_loader` and prints a PASS message. Evidence: `scripts/run_pod_batch.sh:353-380`.
- FAIL: The documented 3-second threshold is not implemented. The script measures `probe_wall` but never compares it against any threshold. Evidence: `scripts/run_pod_batch.sh:355-383`, `scripts/run_pod_batch.sh:385-391`.
- FAIL: R8/R9 are skipped only when the probe exits nonzero, not when it is slow. Evidence: `scripts/run_pod_batch.sh:385-416`, `scripts/run_pod_batch.sh:423-438`.

### 6. `RESULTS_DIR`

- PASS: The batch creates a per-invocation results root at `/workspace/optimization_batch_${RUN_STAMP}` and a unique per-run subdirectory beneath it. Evidence: `scripts/run_pod_batch.sh:39-42`, `scripts/run_pod_batch.sh:79-92`.
- PASS: Each run passes a unique `RUN_ID` and `ARCHIVE_DIR`, so per-run logs and artifacts are archived separately. Evidence: `scripts/run_pod_batch.sh:96-107`, `scripts/runpod_1413.sh:69-71`, `scripts/runpod_1413.sh:101-144`.

### 7. Failure handling

- FAIL: `set -e` will abort the whole batch on the first failed run before `local rc=$?` can execute. The intended “record FAILED and continue” logic in `scripts/run_pod_batch.sh:115-119` is dead code for command failures in the subshell at `scripts/run_pod_batch.sh:95-108`. Evidence: `scripts/run_pod_batch.sh:2`, `scripts/run_pod_batch.sh:95-119`.

### 8. File paths

- PASS: `REPO_ROOT` and `STACK_RECORD_DIR` are derived from the script location, so the record payload path itself is stable. Evidence: `scripts/run_pod_batch.sh:30-37`.
- WARN: The batch wrapper does not `cd "${REPO_ROOT}"` before invoking `bash scripts/runpod_prepare_sp8192.sh` and `bash scripts/runpod_1413.sh`, so it assumes it is launched from the repo root. If run from elsewhere, those helper-script paths fail noisily. Evidence: `scripts/run_pod_batch.sh:107`, `scripts/run_pod_batch.sh:152-153`. `scripts/runpod_1413.sh` does `cd "${REPO_ROOT}"` once it starts, but the wrapper has to find that script first. Evidence: `scripts/runpod_1413.sh:57-73`.

## File 2: Regenerated wrapper

Implementation was checked by decoding `records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/train_gpt.py`; the decoded hash matched `variant_manifest.json:4`.

### 1. CDQuant fix is applied

- PASS: The inner GPTQ loop is single-pass, uses `d = Hinv_block[j,j]`, compares floor vs ceil with Hessian-weighted squared error, and does not use an iteration loop, a separate `h_j`, or plain `abs(error)`. Evidence: decoded wrapper lines `259-271`, especially `267-270`.

### 2. OWC defaults

- PASS: `OWC_GAMMA_STEPS` defaults to `10`. Evidence: decoded wrapper hyperparameter parse at line `7`.
- PASS: `log_gamma` is initialized from the log of the initial sigma value, using a per-row tensor fill with `math.log(max(initial_sigmas,1e-6))`. This is functionally equivalent to `torch.log(initial_sigmas)` for the current scalar `initial_sigmas` input. Evidence: decoded wrapper lines `251-258`.

### 3. Cautious Muon ordering

- PASS: The cautious mask is applied to the momentum-adjusted gradient before the Newton-Schulz whitening call. Evidence: decoded wrapper lines `188-192`.

### 4. No regressions in existing features

- PASS: `PARALLEL_RESIDUAL_START` is still parsed and still enables parallel residuals on the tail blocks. Evidence: decoded wrapper line `7`, decoded wrapper lines `107-130`.
- PASS: `SKIP_TRAINING` is still parsed and still switches `train_and_eval()` into eval-only reuse of `final_model.int6.ptz`. Evidence: decoded wrapper line `7`, decoded wrapper lines `485-505`.
- PASS: `NGRAM_TILT_ENABLED` is still parsed and still gates `NgramTiltState` creation before `legal_ttt_exact`. Evidence: decoded wrapper line `7`, decoded wrapper lines `500-504`.
- PASS: `TTT_OPTIMIZER` and `TTT_FREEZE_BLOCKS` are still parsed and still control the TTT optimizer selection and frozen-block set in `eval_val_sliding_ttt()`. Evidence: decoded wrapper line `7`, decoded wrapper lines `373-385`.

### 5. Env var parse for the 5 new vars

- PASS: All five new env vars are parsed in `Hyperparameters` with the expected types and defaults:
  - `CAUTIOUS_MUON`: `bool(int(...))`, default `0`
  - `OWC_ENABLED`: `bool(int(...))`, default `0`
  - `OWC_GAMMA_STEPS`: `int(...)`, default `10`
  - `CDQUANT_ENABLED`: `bool(int(...))`, default `0`
  - `CDQUANT_ITERS`: `int(...)`, default `3`
  Evidence: decoded wrapper line `7`.

## Recommendation

NO-GO.

Blocking issues before spending GPU money:

1. `scripts/run_pod_batch.sh` does not actually continue after a failed run because `set -e` kills the batch first.
2. `tier2` mode can silently omit `NGRAM_TILT_ENABLED=1` and the best eval-time settings because `best_eval_env()` returns an empty success instead of a real fallback.
3. The BPB extractor is keyed to the wrong log format and silently records `FAILED` if the internal `ttt_sliding:done` line is missing.
4. The CDQuant “timing probe” is a stub: it does not test real quantization, does not enforce the 3-second threshold, and therefore does not actually gate R8/R9.
5. A tier-2 crash can leave the shared checkpoint dirty because there is no failure-path restore.

The regenerated wrapper itself looks correct for the requested CDQuant / OWC / Cautious-Muon changes. The no-go call is entirely about the launcher.

## Re-audit

Date: 2026-04-09

This section re-audits `scripts/run_pod_batch.sh` after the launcher fixes landed. It supersedes the earlier launcher `NO-GO` call above.

### 1. `set -e` / batch continuation

- PASS: The outer script no longer runs with `set -e`; it is now `set -uo pipefail`, so one failed run does not abort the batch. Evidence: `scripts/run_pod_batch.sh:1-2`.
- PASS: `run_experiment()` now wraps the run in a subshell with local `set -e`, captures failures with `|| rc=$?`, and returns `0` to the caller after recording `FAILED`. Evidence: `scripts/run_pod_batch.sh:100-125`.
- PASS: If R3 exits with code `1`, `run_experiment("R3...")` records the failure and returns normally, so the next top-level call to `run_experiment("R4...")` still executes. Evidence: tier-1 calls are linear at `scripts/run_pod_batch.sh:162-236`, and there is no outer `set -e` left to short-circuit them.

### 2. `best_eval_env()`

- PASS: `best_eval_env()` now falls back to `SKIP_TRAINING=1 NGRAM_TILT_ENABLED=1 TTT_OPTIMIZER=rmsdecay TTT_DECAY=0.001` when tier-1 parsing yields no winner. Evidence: `scripts/run_pod_batch.sh:246-271`.
- PASS: `best_eval_env()` now force-injects `NGRAM_TILT_ENABLED=1` if it is missing from the chosen env string. Evidence: `scripts/run_pod_batch.sh:273-276`.
- PASS: Every tier-2 eval pass still appends `eval_env_str`, so after this fix all R6-R9 eval launches inherit `NGRAM_TILT_ENABLED=1` even if tier 1 never ran. Evidence: `scripts/run_pod_batch.sh:342-349`, `scripts/run_pod_batch.sh:366-373`, `scripts/run_pod_batch.sh:394-401`, `scripts/run_pod_batch.sh:420-428`.
- WARN: `NGRAM_TILT_ENABLED=1` is enforced centrally by `best_eval_env()`, not duplicated literally at each eval call site. That is safe against the original silent-failure mode, but it is not “hardcoded at every call site” in the narrow syntactic sense.

### 3. BPB extractor

- PASS: The extractor now matches the real archived output format with `:` separators for timed eval lines and `=` for the internal `ttt_sliding:done` line. Evidence: `scripts/run_pod_batch.sh:53-75`.
- PASS: The archived seed-0 eval log contains exactly these strings:
  - `legal_ttt_exact val_loss:2.79177797 val_bpb:1.08078425 eval_time:324216ms`
  - `ttt_sliding:done val_loss=2.791778 val_bpb=1.080784 elapsed=324.0s`
  - `quantized_sliding_window val_loss:2.79649863 val_bpb:1.08261177 eval_time:91365ms`
  - `quantized val_loss:2.83976191 val_bpb:1.09936033 eval_time:9661ms`
  Evidence: `artifacts/runpod_pull/pr1413_archive_20260407_213205/seed0/pr1413_ngram_eval_s0/console.txt:104-105`, `artifacts/runpod_pull/pr1413_archive_20260407_213205/seed0/pr1413_ngram_eval_s0/console.txt:234-235`.
- PASS: Running the exact regexes against that archive returns the expected values:
  - `legal_ttt_exact.*?val_bpb:\K...` -> `1.08078425`
  - `ttt_sliding:done.*?val_bpb=\K...` -> `1.080784`
  - `quantized_sliding_window.*?val_bpb:\K...` -> `1.08261177`
  - `quantized val_loss:.*?val_bpb:\K...` -> `1.09936033`

### 4. CDQuant probe removal

- PASS: The standalone CDQuant timing probe is gone. R8 now runs directly with `CDQUANT_ENABLED=1 CDQUANT_ITERS=3 OWC_ENABLED=1 OWC_GAMMA_STEPS=10`. Evidence: `scripts/run_pod_batch.sh:378-404`.
- PASS: R9 also runs directly with CDQuant enabled, without any stub gate in front of it. Evidence: `scripts/run_pod_batch.sh:406-431`.
- PASS: The training path still carries its own GPTQ reserve guard. Archived real logs show `gptq_reserve_seconds: 12.0` and `gptq:reserving 12s, effective=588000ms`, which is the expected budget clamp. Evidence: `artifacts/runpod_pull/pr1413_archive_20260407_213205/seed0/pr1413_combo_s0/console.txt:24`, `artifacts/runpod_pull/pr1413_archive_20260407_213205/seed0/pr1413_combo_s0/console.txt:104`.

### 5. Checkpoint trap

- PASS: A cleanup trap now exists and is armed with `trap _ckpt_cleanup EXIT`. On normal script exit or whole-script crash/signal-driven exit, it restores the backed-up checkpoint. Evidence: `scripts/run_pod_batch.sh:304-311`.
- PASS: Between tier-2 runs, the real protection is still the explicit `restore_checkpoint()` call before every training run. Evidence: `scripts/run_pod_batch.sh:313-318`, `scripts/run_pod_batch.sh:333`, `scripts/run_pod_batch.sh:357`, `scripts/run_pod_batch.sh:383`, `scripts/run_pod_batch.sh:409`.
- PASS: Exact trace, R6 crashes mid-training:
  - `run_experiment("R6...")` returns with `rc != 0` and records `FAILED` instead of exiting the batch. Evidence: `scripts/run_pod_batch.sh:100-125`.
  - `install_checkpoint(...)` for the R6 eval pass fails, so the eval pass is skipped. Evidence: `scripts/run_pod_batch.sh:340-352`.
  - The script then reaches R7 and runs `restore_checkpoint` before launching R7 training. Evidence: `scripts/run_pod_batch.sh:354-358`.
  - The `EXIT` trap does not fire before R7 starts; it is exit-scoped, not per-run scoped. That is acceptable here because `restore_checkpoint()` repairs the shared checkpoint before the next training run.
- WARN: The trap message says “restore checkpoint on any exit,” but between-run restoration still comes from `restore_checkpoint()`, not the trap itself. The implementation is safe; the nuance is only about where the recovery actually happens.

### WARN re-check

- PASS: The previously flagged WARN-class issues in the original audit did not escalate to new FAILs after these launcher changes.
- WARN: Tier-2 eval passes still carry training-side flags (`CAUTIOUS_MUON`, `OWC_*`, `CDQUANT_*`) even though `SKIP_TRAINING=1` only reuses the quantized artifact. This remains a readability risk, not a correctness bug.
- WARN: The final ranking still mixes `_train` and `_eval` rows instead of filtering to second-pass eval rows only. This remains a presentation risk, not a silent wrong-result path.
- WARN: The wrapper still assumes it is launched from the repo root when calling `bash scripts/...`. This still fails noisily rather than silently if invoked from the wrong directory.

## Re-audit Recommendation

GO.

The 5 original blocking launcher FAILs are fixed. Residual issues are still WARN-level only and do not create a silent wrong-result path under the current launcher logic.
