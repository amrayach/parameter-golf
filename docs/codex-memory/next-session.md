# Next Session

## Phase

**Session 03 anchor is COMPLETE. Session 04 targeted deltas start now.**

## Immediate next action

**Session 04: Targeted Delta Sweep** — choose at most three cheap, attributable deltas on top of the Session 03 anchor.

## Prerequisites (all satisfied)

- Session 03 anchor verified: sliding s64 val_bpb `1.12904446`, 6564 steps, 91.37ms/step
- Remaining donor gap is small (`+0.00419944` on final sliding), so broad redesign is unnecessary
- NGC container + fscratch path confirmed on Pegasus
- Launcher lesson locked: use `srun --ntasks=8 --gpus-per-task=1 --gpu-bind=none`, NOT torchrun
- int6+zstd roundtrip artifact: `15751324` bytes, headroom `248676` bytes

## Session 04 implementation order

1. Read:
   - `docs/campaign/artifacts/03_pre_ttt_anchor_summary.md`
   - `docs/campaign/sessions/04_targeted_delta_sweep.md`
   - `records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/README.md`

2. Choose at most three deltas from the Session 04 pool:
   - GPTQ-lite percentile clip search
   - LeakyReLU^2
   - one warmdown/EMA threshold change
   - one small bigram or smear change

3. Recommended order from the current anchor result:
   - GPTQ-lite clip search first
   - LeakyReLU^2 second
   - one small schedule or token-path tweak third

4. Keep backend/perf parity as a separate control if throughput becomes the dominant bottleneck.
   - Do not bundle backend work with export or model deltas in the same run.

## Measurement discipline

- Each delta is a separate run with one change
- Compare against Session 03 anchor as the fixed reference
- Record: GPU, steps, step_avg, sliding s64 val_bpb, pre-quant EMA val_bpb, int6 roundtrip val_bpb, artifact size
- Only combine deltas after each is measured in isolation

## Target

Session 04 goal: beat `1.12904446` on final sliding s64 with an attributable single-delta improvement.

## Launcher template for 8xH100 on Pegasus (NGC container)

```bash
salloc -p H100 --nodes=1 --ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6 --time=02:00:00

srun --gpu-bind=none bash -c '
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
export NCCL_IB_DISABLE=1
cd /netscratch/ayach/parameter-golf
RUN_ID=<run_id> \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
AMP_DTYPE=auto \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
python3 -u records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/train_gpt.py
' 2>&1 | tee /netscratch/ayach/<run_id>.log
```
