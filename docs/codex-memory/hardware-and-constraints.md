# Hardware And Constraints

## Challenge constraints

- train in under 10 minutes on `8xH100 SXM`
- artifact under `16,000,000` bytes total
- final metric is `val_bpb`

## Pegasus

Docs currently claim:
- `H100` = `H100-SXM5`
- 8 GPUs per node
- NVSwitch connectivity

But this is not yet trusted for your user until live Slurm verification passes.

Things that remain unknown until verified:
- your account access
- single-node 8-GPU schedulability
- QoS and fairshare limits
- actual node-level GPU model string today

## RunPod

- budget available: about `$25`
- practical interpretation: reserve for final validation only
- do not plan broad hyperparameter search there

## Time budget

- target effort: about `10-15` hours per week
- increase only if early evidence is strong
- stop or downshift if the anchor never becomes competitive
