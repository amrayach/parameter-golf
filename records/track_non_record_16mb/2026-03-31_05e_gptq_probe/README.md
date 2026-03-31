# Session 05e: GPTQ Probe on 05c-plus Architecture

Falsification test: does LeakyReLU(0.5)^2 + VE128 unblock GPTQ?

For the 1xGPU compatibility probe, `probe.sh` defaults `ENABLE_TORCH_COMPILE=0`.
This avoids RTXA6000 Triton shared-memory compile failures and does not change the
GPTQ question being tested.

## Commands

```bash
# Run from repo root. 1xGPU probe (Pegasus A100):
srun -K -p A100-80GB --nodes=1 --ntasks=1 --gpus-per-task=1 \
  --cpus-per-task=6 --mem=80G --time=00:15:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
  --container-mounts="$(pwd)":"$(pwd)" --container-workdir="$(pwd)" \
  bash records/track_non_record_16mb/2026-03-31_05e_gptq_probe/probe.sh
```

## Decision Rule

GPTQ must beat same-script naive replay on the same checkpoint:

| Condition | Verdict |
|---|---|
| GPTQ BPB < naive BPB AND worse_than_naive_rowmax < 10 | GPTQ unblocked |
| worse_than_naive_rowmax > 50% of layers | Park GPTQ permanently |
| Intermediate | Try on full 8xH100 checkpoint |
