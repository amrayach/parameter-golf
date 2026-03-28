# Pegasus H100 Runbook

This is a lightweight operating note for Parameter Golf work on Pegasus.

## Why this partition

From `@docs/Pegasus_Server_documentation.txt`:
- `H100` is `H100-SXM5`
- 8 GPUs per node
- NVSwitch connectivity
- 80 GB per GPU

This is the right Pegasus partition for challenge-like iteration.

Do not use:
- `batch` for multi-GPU challenge work
- `H100-PCI` when the goal is closest parity with the challenge hardware

## Minimum scheduling stance

Prefer:
- one node
- 8 GPUs on that node
- one process per GPU

Important scheduler notes from the Pegasus docs:
- keep all GPUs on one node for this workload
- add `--gpu-bind=none` for peer-to-peer visibility in multi-GPU jobs
- avoid mixed-model allocations from the `batch` partition

## Suggested job metadata to record

For every real run, capture:
- partition
- node name
- GPU type
- GPU count
- seed
- training script path
- dataset path
- tokenizer path
- train batch tokens
- train seq len
- iterations
- wallclock cap
- step average
- pre-quant `val_bpb`
- post-quant `val_bpb`
- sliding `val_bpb`
- artifact bytes
- code bytes
- total bytes

Use `templates/RUN_MANIFEST_TEMPLATE.md` and `templates/EXPERIMENT_SUMMARY_TEMPLATE.md`.

## Practical command shape

This is not a final cluster script, just the scheduling shape to preserve:

```bash
srun \
  --partition=H100 \
  --nodes=1 \
  --ntasks=8 \
  --gpus=8 \
  --cpus-per-gpu=<set deliberately> \
  --gpu-bind=none \
  --time=01:00:00 \
  bash -lc '
    cd <repo-path> &&
    torchrun --standalone --nproc_per_node=8 <script>
  '
```

Tune `--cpus-per-gpu` to your actual data pipeline needs, but keep the partition and single-node assumptions fixed unless a session explicitly changes them.

## Storage stance

From the Pegasus docs:
- keep source in `$HOME`
- keep datasets, logs, checkpoints, and scratch outputs in `/netscratch/$USER`
- use `/fscratch/$USER` only if low-latency access is clearly needed

For this campaign:
- repo clone can stay in home or project space
- large run logs and temporary artifacts should live on scratch and be copied back only when needed
