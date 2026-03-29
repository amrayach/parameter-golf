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

## Pre-implementation check

Before writing any code, the user needs to verify FA3 availability on Pegasus. Ask the user to run:
```bash
srun -p H100 --ntasks=1 --gpus-per-task=1 --cpus-per-task=6 --mem=64G --time=00:10:00 \
  --container-image=$(ls -1 /enroot/nvcr.io_nvidia_pytorch_*.sqsh | sort -V | tail -1) \
  bash -c 'python -c "from flash_attn_interface import flash_attn_func; print(\"FA3 OK\")"'
```
If FA3 is not present, try: `pip install flash-attn` inside the container.
If neither works, stop and reassess — do not proceed without FA3 confirmed.

## Task

1. Create `records/track_non_record_16mb/2026-03-29_fw1_fa3/`
2. Copy anchor files (train_gpt.py, requirements.txt) — no __pycache__
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
6. Create README.md and submission.json (flat schema matching anchor format)
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
  cd /netscratch/ayach/parameter-golf && git pull
  bash scripts/pegasus_optimized_launcher.sh fw1_fa3_8xh100 \
    records/track_non_record_16mb/2026-03-29_fw1_fa3/train_gpt.py
  ```

## Git conventions

- `research(protocol):` — before running
- `research(results):` — after run with measured results
- Stage specific files only — do NOT stage:
  - `records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/README.md` (unrelated)
  - `.serena/`, `docs/*.pdf`, `docs/*.txt` (untracked, unrelated)

## Pegasus conventions

- NEVER use torchrun. Use `scripts/pegasus_optimized_launcher.sh`
- salloc: `salloc -p H100 --nodes=1 --ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6 --mem=200G --time=1-00:00:00`
- Container: NGC 26.03, auto-detected by launcher
- Data: `/fscratch` preferred
- Logs: `/netscratch/ayach/<run_id>.log`

## Documentation conventions (after results come back)

Update in this order:
1. `records/track_non_record_16mb/2026-03-29_fw1_fa3/submission.json` — fill metrics
2. `records/track_non_record_16mb/2026-03-29_fw1_fa3/README.md` — add results
3. `docs/campaign/AGENT_SYNC.md` — add FW-1 measured results
4. `docs/codex-memory/project-state.md` — update what's been demonstrated
5. `docs/codex-memory/next-session.md` — update next action (FW-2 or stack)
6. `docs/codex-memory/decisions.md` — record FA3 decision
7. Claude memory: `~/.claude/projects/-home-amay-Work-parameter-golf/memory/project_parameter_golf.md`
