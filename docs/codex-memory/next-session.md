# Next Session

## Phase

**Session 05f: BigramHash 3072x112 + Warmdown 4000 — smoke, then 8xH100 run.**

GPTQ is permanently parked. Focus on naive int6 export only.

## Immediate next action

1. Smoke 05f on 1xGPU (Pegasus A100-80GB or H100)
2. If smoke passes, launch 05f on `8xH100`
3. Compare against both anchor and 05c-plus:
   - Quality target: sliding s64 val_bpb < 1.1256 (05c-plus was 1.1256, anchor was 1.1290)
   - Throughput: step_avg within +5ms of anchor (91.37ms) — 05c-plus regressed to 100.39ms
   - Artifact: <= 16,000,000 bytes

## What happened in Session 05c-plus (MEASURED)

8xH100 result:
- sliding s64 val_bpb: `1.12557920` (anchor delta: **-0.00347**, positive)
- pre_quant EMA: `1.14186715`
- int6 roundtrip: `1.14933197`
- step_avg: `100.39 ms` (+9.02ms vs anchor, **regressed**)
- steps: `5977` (587 fewer than anchor due to throughput)
- artifact: `15,589,271` bytes

Quality-positive but throughput regressed materially. Not a seed-validation branch.

## 05f changes (on 05c-plus base)

| Change | Type | Risk |
|--------|------|------|
| BigramHash vocab 2048→3072 | constant | low (hash collision reduction) |
| BigramHash dim 128→112 | constant | low (offsets param increase) |
| Warmdown 3500→4000 | constant | none |

Base: 05c-plus (`records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py`)

## Files to read first

1. `docs/campaign/AGENT_SYNC.md`
2. `CLAUDE.md`
3. `records/track_non_record_16mb/2026-03-31_05f_refine_bigram3072_warmdown4000/train_gpt.py`
4. `records/track_non_record_16mb/2026-03-31_05f_refine_bigram3072_warmdown4000/README.md`

## 1xGPU smoke command

```bash
cd /netscratch/$USER/parameter-golf && git pull

srun -K -p A100-80GB --nodes=1 --ntasks=1 --gpus-per-task=1 \
  --cpus-per-task=6 --mem=80G --time=00:10:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
  --container-mounts="$(pwd)":"$(pwd)" --container-workdir="$(pwd)" \
  bash -c '
    export PYTHONUNBUFFERED=1
    export ITERATIONS=100 MAX_WALLCLOCK_SECONDS=120
    pip install --no-cache-dir sentencepiece zstandard 2>/dev/null
    python -u records/track_non_record_16mb/2026-03-31_05f_refine_bigram3072_warmdown4000/train_gpt.py
  '
```

## 8xH100 launch command

```bash
cd /netscratch/$USER/parameter-golf && git pull

srun -K -p H100 --nodes=1 --ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6 \
  --mem=200G --time=00:20:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
  --container-mounts="$(pwd)":"$(pwd)" --container-workdir="$(pwd)" \
  bash -c '
    export LOCAL_RANK=$SLURM_LOCALID RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS
    export PYTHONUNBUFFERED=1
    export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
    export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=bond,eth NCCL_P2P_LEVEL=NVL
    pip install --no-cache-dir sentencepiece zstandard 2>/dev/null
    python -u records/track_non_record_16mb/2026-03-31_05f_refine_bigram3072_warmdown4000/train_gpt.py
  '
```
