# Session 05g: XSA-8 Throughput Recovery

## Change
- Base: 05c-plus (`records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py`)
- Single change: `xsa_last_n` 11 → 8 (XSA on layers 3-10, removed from layers 0-2)

## Hypothesis
XSA on all 11 layers is the leading hypothesis for the +9ms throughput regression in 05c-plus. Removing XSA from the 3 shallowest layers should partially recover throughput while retaining most of the quality gain.

## Unchanged from 05c-plus
- VE128 on layers 9-10
- LeakyReLU(0.5)²
- warmdown_iters = 3500
- bigram 2048x128
- mlp_mult = 3.0
- All other hyperparameters

## Reference results

| Metric | 05c-plus | Session 03 anchor |
|--------|----------|-------------------|
| sliding s64 val_bpb | 1.12557920 | 1.12904446 |
| step_avg_ms | 100.39 | 91.37 |
| steps | 5977 | 6564 |

## Commands

### 1xGPU smoke
```bash
srun --immediate=30 -K -p H100 --nodes=1 --ntasks=1 --gpus-per-task=1 \
  --cpus-per-task=6 --mem=80G --time=00:10:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
  --container-mounts=/netscratch/$USER/parameter-golf:/netscratch/$USER/parameter-golf \
  --container-workdir=/netscratch/$USER/parameter-golf \
  bash -lc 'export PYTHONUNBUFFERED=1 && pip install --no-cache-dir sentencepiece zstandard 2>/dev/null && \
  WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 ITERATIONS=50 MAX_WALLCLOCK_SECONDS=120 \
  VAL_LOSS_EVERY=25 TRAIN_LOG_EVERY=10 SEED=1337 \
  python -u records/track_non_record_16mb/2026-03-31_05g_xsa8_throughput/train_gpt.py'
```

### 8xH100 full run
```bash
srun --immediate=30 -K -p H100 --nodes=1 --ntasks=8 --gpus-per-task=1 --gpu-bind=none \
  --cpus-per-task=6 --mem=200G --time=00:20:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
  --container-mounts=/netscratch/$USER/parameter-golf:/netscratch/$USER/parameter-golf,/fscratch/$USER:/fscratch/$USER \
  --container-workdir=/netscratch/$USER/parameter-golf \
  bash -lc 'export PYTHONUNBUFFERED=1 && \
  export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 && \
  export NCCL_IB_DISABLE=1 && export NCCL_SOCKET_IFNAME=bond,eth && export NCCL_P2P_LEVEL=NVL && \
  pip install --no-cache-dir sentencepiece zstandard 2>/dev/null && \
  LOCAL_RANK=$SLURM_LOCALID RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS \
  python -u records/track_non_record_16mb/2026-03-31_05g_xsa8_throughput/train_gpt.py'
```

## Status
- [ ] 1xGPU smoke passed
- [ ] 8xH100 measured
