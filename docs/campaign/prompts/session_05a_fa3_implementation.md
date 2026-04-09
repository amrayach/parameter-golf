# Session 05a: FA3 Implementation Prompt

Paste everything below the line into a fresh Claude Code session.

---

Session 05a: Implement Flash Attention 3 (FW-1) as an isolated delta on the Session 03 anchor.

## Read order

1. `docs/campaign/AGENT_SYNC.md`
2. `CLAUDE.md`
3. `docs/campaign/artifacts/05_ttt_correctness_audit.md` — the Session 05 audit plan (read the throughput audit section and FW-1)
4. `records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/train_gpt.py` — our anchor (read the CausalSelfAttention class, lines 567-638, and SDPA backend config, lines 984-988)
5. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py` — porting reference (read the CausalSelfAttention class and FA3 usage)

## Current fixed facts

- Session 03 anchor sliding s64: `1.12904446`, step_avg: `91.37 ms`, steps: `6564`
- Session 03 anchor uses SDPA with `enable_math_sdp(True)` at time of measured run
- The 2026-03-22 record uses FA3 in the same CastedLinear architecture, achieving `1.1233` at `~84ms/step`
- FA3 import path: `from flash_attn_interface import flash_attn_func as flash_attn_3_func`
- FA3 tensor layout: `(B, T, H, D)` — no transposes needed after reshape
- FA3 call: `flash_attn_3_func(q, k, v, causal=True)`
- FA3 benchmark is already done:
  - `26.03` SDPA flash: `1.967 ms/iter`
  - `25.02` SDPA flash: `1.889 ms/iter`
  - `25.02` direct FA3: `0.165 ms/iter`
  - this is attention-kernel-only evidence, not end-to-end training speed

## Pre-implementation stance

- The explicit FA3 experiment path is now known:
  - use the saved Pegasus FA3 container at `/netscratch/$USER/containers/pytorch_25.02_fa3.sqsh`
- Do not treat the microbenchmark as proof of full training speedup.
- The implementation must still earn promotion through a short real-training smoke and then a full `600s` run.
- Do not use `--no-deps` against the stock `25.02` container. FA3 imports fail against the bundled PyTorch ABI.
- Do not hide output with `| tail -1`. Use unbuffered Python instead.

## Task

1. Work in `records/track_non_record_16mb/2026-03-29_fa3_port/`
2. Keep the delta isolated against the anchor — no unrelated code changes
3. Apply these changes to CausalSelfAttention.forward():
   - Add import: `from flash_attn_interface import flash_attn_func as flash_attn_3_func`
   - Remove `.transpose(1, 2)` after q/k reshape (anchor lines 616-617)
   - Remove `v_sdpa = v.transpose(1, 2)` (anchor line 619)
   - Replace `F.scaled_dot_product_attention(q, k, v_sdpa, ...)` with `flash_attn_3_func(q, k, v, causal=True)`
   - Remove `.transpose(1, 2).contiguous()` after attention output (anchor line 633)
   - Update RoPE cache shape from `(1, 1, T, rd/2)` to `(1, T, 1, rd/2)` if needed
   - Verify XSA still works post-swap (it operates on `(B, T, H, D)` which is FA3's native layout)
4. Restore `enable_math_sdp(True)` to match the measured anchor state (same isolation as Delta 2)
5. Update docstring to describe FW-1
6. Keep README.md and submission.json in sync with the saved-container Pegasus path
7. Verify with `python3 -c "import ast; ast.parse(open('train_gpt.py').read())"`

## Isolation constraint

- The measured anchor result (`1.12904446`) was from the pre-563700f state where `enable_math_sdp(True)`
- FW-1 must be a pure FA3 delta against that measured state
- No other changes (no activation change, no VE, no SWA, no warmdown change)

## Success criteria

- sliding s64 val_bpb < `1.12904446` (improvement from more training steps)
- step_avg < `88 ms` (meaningful throughput improvement)
- steps > `6800` (at least ~250 more steps than anchor)
- bytes_total < `16,000,000`

## After implementation

- Commit with: `research(protocol): Session 05 FW-1 — FA3 isolated delta, pre-run`
- Push to remote
- Tell user to run on Pegasus:
  ```bash
  cd /netscratch/$USER/parameter-golf && git pull
  srun -K -p H100 --nodes=1 --ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6 --time=02:00:00 \
    --container-image=/netscratch/$USER/containers/pytorch_25.02_fa3.sqsh \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/fscratch/$USER:/fscratch/$USER \
    --container-workdir=/netscratch/$USER/parameter-golf \
    bash -c '
      export LOCAL_RANK=$SLURM_LOCALID
      export RANK=$SLURM_PROCID
      export WORLD_SIZE=$SLURM_NTASKS
      export PYTHONUNBUFFERED=1
      export MKL_NUM_THREADS=1
      export NUMEXPR_NUM_THREADS=1
      export OMP_NUM_THREADS=1
      export USE_OPENMP=1
      export NCCL_IB_DISABLE=1
      export NCCL_SOCKET_IFNAME=bond,eth
      export NCCL_P2P_LEVEL=NVL
      cd /netscratch/$USER/parameter-golf
      RUN_ID=fa3_port_8xh100 \
      DATA_PATH=/fscratch/$USER/parameter-golf-data/datasets/fineweb10B_sp1024 \
      TOKENIZER_PATH=/fscratch/$USER/parameter-golf-data/tokenizers/fineweb_1024_bpe.model \
      VOCAB_SIZE=1024 AMP_DTYPE=auto MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=200 TRAIN_LOG_EVERY=50 \
      python3 -u records/track_non_record_16mb/2026-03-29_fa3_port/train_gpt.py
    ' 2>&1 | tee /netscratch/$USER/fa3_port_8xh100.log
  ```

## Git conventions

- `research(protocol):` — before running
- `research(results):` — after run with measured results
- Stage specific files only — do NOT stage:
  - `records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/README.md` (unrelated)
  - `.serena/`, `docs/*.pdf`, `docs/*.txt` (untracked, unrelated)

## Pegasus conventions

- NEVER use torchrun
- salloc: `salloc -p H100 --nodes=1 --ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6 --mem=200G --time=02:00:00`
- Standard container: NGC `26.03`
- Explicit FA3 experiment container: saved Pegasus `25.02` FA3 container
- Current launcher note: `scripts/pegasus_optimized_launcher.sh` still auto-detects the latest NGC image, so it is not yet the right default launcher for FW-1 unless updated to pin `25.02`
- Data: `/fscratch` preferred
- Logs: `/netscratch/$USER/<run_id>.log`

## Documentation conventions (after results come back)

Update in this order:
1. `records/track_non_record_16mb/2026-03-29_fa3_port/submission.json` — fill metrics
2. `records/track_non_record_16mb/2026-03-29_fa3_port/README.md` — add results
3. `docs/campaign/AGENT_SYNC.md` — add FW-1 measured results
4. `docs/codex-memory/project-state.md` — update what's been demonstrated
5. `docs/codex-memory/next-session.md` — update next action (FW-2 or stack)
6. `docs/codex-memory/decisions.md` — record FA3 decision
7. Claude memory: `~/.claude/projects/-home-amay-Work-parameter-golf/memory/project_parameter_golf.md`
