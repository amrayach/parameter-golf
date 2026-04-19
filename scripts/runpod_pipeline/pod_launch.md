# Pod Launch Instructions

## Image

```
amay01/parameter-golf@sha256:6206b37e0f363c3886323b391e64bb0e46b1623e203b6b9e55f165fd774ea2cf
```

Contains: torch 2.9.1+cu128, flash_attn_interface (FA3 3.0.0), all Python deps.
Venv: `/opt/pg-venv` — **do not reinstall packages**.

Until the image is rebuilt with the local Dockerfile fix, you must still override the container start command explicitly at launch. The pinned digest above predates the `sleep infinity` default.

## RunPod Web UI

1. Go to **Deploy** → **GPU Pods**
2. Click **Deploy** on an **8× H100 SXM 80GB** configuration
3. **Select region**: CA-MTL-1 or EU-RO-1 preferred — **avoid AP-IN-1** (high latency)
4. Under **Container Image**, paste:
   ```
   amay01/parameter-golf@sha256:6206b37e0f363c3886323b391e64bb0e46b1623e203b6b9e55f165fd774ea2cf
   ```
5. **Container disk**: set to **120 GB** (model + dataset + checkpoints)
6. **Volume disk**: leave unset if possible. Do **not** accept a small default volume mounted at `/workspace` without checking the quota implications below.
7. **SSH key**: add your public key under **SSH Public Key** so you can connect
8. **Container Start Command**: set it to:
   ```
   sleep infinity
   ```
9. Click **Deploy**

## RunPod API (alternative)

```bash
# Requires: pip install runpod  +  RUNPOD_API_KEY set
runpodctl create pod \
  --name "pr1610-corrector" \
  --imageName "amay01/parameter-golf@sha256:6206b37e0f363c3886323b391e64bb0e46b1623e203b6b9e55f165fd774ea2cf" \
  --gpuType "NVIDIA H100 80GB HBM3" \
  --gpuCount 8 \
  --containerDiskInGb 120 \
  --dataCenter "CA-MTL-1"
```

If you use the REST API directly, set the keepalive command in exec-array form:

```json
{
  "dockerStartCmd": ["sleep", "infinity"]
}
```

RunPod rejects shell strings here with HTTP 400 (`got string, want array`).

## Container Start Command

The image's historical default was `CMD ["/bin/bash"]`. On a non-interactive RunPod start, that exits immediately and leaves the pod thrashing, which breaks SSH. Always launch with `dockerStartCmd: ["sleep", "infinity"]` until the rebuilt image digest is published and repinned here.

## Storage Layout

Treat `/workspace` as a quota-constrained MooseFS mount unless you have explicitly verified otherwise. `df -h` may show TBs free while the pod still enforces a much smaller per-pod quota (the Session 3 failure mode was effectively `~20 GB`).

Default write paths for Session 4:
- HF cache: `/root/.hf/`
- XDG cache: `/root/.cache/`
- dataset: `/root/data/`
- checkpoints: `/root/checkpoints/`

Do not put dataset cache, dataset shards, or checkpoints under `/workspace` on a pod with a small mounted volume. Use symlinks back into the repo tree only as a salvage path, not the default plan.

## Cold-start Polling

After a `stop` / `start` cycle, poll for `runtime != null` for at least `300 s`. The image can take 4+ minutes to restage on the assigned machine. Do not declare failure after a `180 s` poll window if `dockerStartCmd` is already correct.

## Cost Reference

| Phase           | Est. time | Cost at $21.52/hr |
|-----------------|-----------|-------------------|
| Stage 0–1       | ~25 min   | ~$9               |
| Stage 2 Gate A  | ~20 min   | ~$7               |
| Stage 3 (×3 evals) | ~15 min | ~$5              |
| Stage 4a Gate B (×2 seeds) | ~40 min | ~$14  |
| Stage 5 preserve| ~5 min    | ~$2               |
| **Total (primary path)** | **~105 min** | **~$38** |
| Fallback (4b instead of 4a) | ~10 min | ~$4  |

**Budget: $138 remaining** → $38 primary, $98 reserve for one retry.

## SSH Connection

After pod shows **Running** status and an SSH port appears:

```bash
ssh -p <PORT> root@<POD_IP>
# or use the "Connect" button in RunPod web UI → "Start Web Terminal"
```

Confirm the pod is ready before continuing:
```bash
nvidia-smi   # should show 8× H100 80GB
df -h /workspace   # inspect the raw mount, do not trust the headline number blindly
/opt/pg-venv/bin/python -c "import torch; print(torch.__version__)"  # 2.9.1+cu128
```
