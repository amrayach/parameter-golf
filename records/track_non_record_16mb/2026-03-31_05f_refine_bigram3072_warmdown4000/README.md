# Session 05f: BigramHash 3072x112 + Warmdown 4000

Prepared next candidate on the 05c-plus base. Not mainline unless 05c-plus results justify promotion.

## Changes from 05c-plus

| # | Change | Type | Lines changed |
|---|--------|------|---------------|
| 1 | BigramHash vocab 2048 -> 3072 | constant | 1 |
| 2 | BigramHash dim 128 -> 112 | constant | 1 |
| 3 | warmdown 3500 -> 4000 | constant | 1 |

## Inherited from 05c-plus (on Session 03 anchor)

| # | Change | Type |
|---|--------|------|
| 1 | warmdown 3000 -> 3500 | constant |
| 2 | XSA 4 -> 11 (all layers) | constant |
| 3 | LeakyReLU(0.5)^2 (replaces ReLU^2) | activation |
| 4 | VE128 on layers 9-10 | new module |

## Base

`records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py`

## 05c-plus reference result (8xH100)

- Sliding s64 val_bpb: 1.12557920 (anchor delta: -0.00347)
- step_avg: 100.39 ms (+9.02ms vs anchor)
- artifact: 15,589,271 bytes

## NOT included

- SWA (dead code in PR #1019 and #634)
- GPTQ (permanently parked)
- FA3 (container ABI issue unresolved)
- TTT, Parallel Muon, Parameter Banking, Late QAT, LZMA

## 1xGPU smoke

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

## 8xH100 launch

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
