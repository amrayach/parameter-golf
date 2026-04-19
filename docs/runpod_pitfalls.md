# RunPod Pitfalls

Date: 2026-04-19

Canonical merge of the Session 3 deployment and execution failures from the two postmortem reports. This is required reading before Session 4.

## 1. Dead container blocks SSH

Symptom:
- SSH prints the NGC banner, then fails with `container ... is not running`

Root cause:
- The image historically ended with `CMD ["/bin/bash"]`
- RunPod proxy SSH is `docker exec` into a running container, not host SSH
- Non-interactive bash sees EOF and exits immediately, so the container never stays up

Fix applied:
- Override start command to `dockerStartCmd: ["sleep", "infinity"]`

Prevention:
- Rebuild the image with `CMD ["sleep", "infinity"]`
- Document the keepalive requirement in `pod_launch.md`
- Verify SSH within the first minute after launch

## 2. RunPod REST API requires exec-array form for `dockerStartCmd`

Symptom:
- PATCH with `"dockerStartCmd": "sleep infinity"` returns HTTP 400 with `got string, want array`

Root cause:
- RunPod enforces Docker exec-form command arrays for this field

Fix applied:
- Use `"dockerStartCmd": ["sleep", "infinity"]`

Prevention:
- Treat command-like RunPod API fields as JSON arrays unless confirmed otherwise
- Read the 4xx body; RunPod returns precise schema errors

## 3. Cold-start polling window was too short

Symptom:
- `runtime` stayed null for 180 seconds after `stop` / `start`, which looked like another dead pod

Root cause:
- The image had to restage after restart; 180 seconds was a warm-start assumption applied to a cold-start path

Fix applied:
- Waited longer and confirmed the pod eventually came up

Prevention:
- Poll for at least `300–360 s` after `stop` / `start`
- Distinguish `dockerStartCmd` misconfiguration from slow image staging

## 4. `/workspace` quota was much smaller than `df -h` implied

Symptom:
- Dataset download died mid-run with `OSError: [Errno 122] Disk quota exceeded`
- `df -h /workspace` still showed enormous backing capacity

Root cause:
- `/workspace` was a MooseFS mount with a small per-pod quota
- The pod also had a 120 GB container disk, but caches/data/checkpoints were being written to `/workspace` instead

Fix applied:
- Move HF cache, dataset, and checkpoints to container-disk-backed paths under `/root`
- Use symlinks from repo-expected locations only as a salvage path

Prevention:
- Default HF cache to `/root/.hf/`, XDG cache to `/root/.cache/`, data to `/root/data/`, checkpoints to `/root/checkpoints/`
- Compare `df -h /workspace` against the pod's `volumeInGb` before Stage 1

## 5. Stage 1 fresh-pod glob bug

Symptom:
- `01_download_data.sh` printed only its header line, then the pipeline aborted

Root cause:
- `ls ... | wc -l` under `set -euo pipefail` on a fresh pod caused `ls` to exit non-zero before any download started

Fix applied:
- Replace the count logic with `nullglob` + bash arrays
- Add `mkdir -p "${DATA_DIR}"` before the first count

Prevention:
- Do not use `ls | wc -l` for file counts in bash under `pipefail`
- Fresh-pod test Stage 1 explicitly before launch day

## 6. Compile warmup looked like a hang

Symptom:
- First run spent several minutes with low GPU utilization and growing memory

Root cause:
- Inductor/Triton compile warmup takes minutes on the first pass
- Session 3 also added a corrector bias path that triggered more compilation

Fix applied:
- Confirmed liveness via process state and memory movement rather than interrupting

Prevention:
- Expect 3–5 minute first-run compile warmup
- Add a periodic heartbeat log during compile-heavy sections

## 7. Eval-only quantized path bug A: pre-quant diagnostic ran with `compiled_model=None`

Symptom:
- Eval-only ablation crashed with `AttributeError: 'NoneType' object has no attribute 'forward_logits'`

Root cause:
- `quantized_eval_only` nulled the pre-quant model handles but the code still ran the pre-quant diagnostic eval

Fix applied:
- Guard the pre-quant diagnostic with `if not quantized_eval_only`

Prevention:
- Keep explicit coverage for the multi-GPU quantized-eval-only path

## 8. Eval-only quantized path bug B: cleanup deleted `eval_model` before it existed

Symptom:
- Follow-up ablation crashed with `UnboundLocalError: local variable 'eval_model' referenced before assignment`

Root cause:
- Cleanup block assumed the non-quantized branch had created `eval_model`

Fix applied:
- Change the guard to `if not ttt_only_eval and not quantized_eval_only`

Prevention:
- Test cleanup paths separately for eval-only and train+eval modes

## 9. Gate A false failure from stale internal headroom math

Symptom:
- Gate A summary flagged `FAIL` even though the artifact was under the competition cap

Root cause:
- The internal safety threshold (`15,997,520 B`) was computed earlier against pristine `#1610` HEAD and drifted out of date as local code size changed

Fix applied:
- Manual interpretation: Gate A counted as a scientific pass because the artifact was `15,999,394 B`, which is `606 B` under the official cap

Prevention:
- Recompute internal thresholds from the current working tree, not from historical estimates
- Reserve a fixed extra safety buffer below `16,000,000 B`

## 10. Artifact preservation failed late when `HF_TOKEN` was missing

Symptom:
- Tarball creation succeeded, then upload failed with `401 Unauthorized`

Root cause:
- HF auth was not validated before the expensive preservation step

Fix applied:
- Set `HF_TOKEN`, then upload the existing tarball directly

Prevention:
- Validate `HF_TOKEN` with `whoami` before tarball creation
- Fail fast if the upload target is `hf:...` and the token is unset

## 11. Long jobs ran outside `tmux`

Symptom:
- Training, downloads, and ablations were launched in the foreground SSH shell

Root cause:
- No enforced `tmux` gate in the stage scripts

Fix applied:
- Session 4 scripts now refuse to run without `tmux` unless `ALLOW_NO_TMUX=1`

Prevention:
- Start `tmux` before any long-running stage

## 12. `scp` failed because RunPod's SSH proxy lacked SFTP support

Symptom:
- `scp` failed with `subsystem request failed on channel 0`

Root cause:
- Modern `scp` defaults to SFTP; RunPod's proxy did not support it

Fix applied:
- Use HuggingFace Hub as the canonical preservation channel; if needed, use `scp -O`

Prevention:
- Prefer HF upload or legacy-mode `scp -O`

## 13. Huge diagnostic pastes slowed the session down

Symptom:
- Full process trees and large command outputs consumed conversation bandwidth and slowed debugging

Root cause:
- Diagnostic commands were not scoped to the minimum useful output

Fix applied:
- None during Session 3 beyond manual tightening of commands

Prevention:
- Use filtered one-liners, counts, and `tail -20`, not full trees or full logs

## 14. Stale or unsynchronized AI guidance caused patch confusion

Symptom:
- One follow-up patch recommendation ignored the already-applied state and would have made a redundant change

Root cause:
- The recommender was operating on a stale read of the file

Fix applied:
- Re-read the current file and diff before applying any follow-up patch

Prevention:
- On shared-file debugging, verify current state with `sed -n` or `git diff` before accepting a second patch recommendation

## 15. Composio RunPod connector was stale

Symptom:
- RunPod actions through the connector returned `401 Unauthorized`

Root cause:
- Stale Composio credentials

Fix applied:
- Switched to direct RunPod REST/GraphQL

Prevention:
- Verify connector auth before relying on it for paid pod operations

## 16. RunPod MCP/connector surface was incomplete

Symptom:
- The available higher-level tool surfaces did not expose `dockerStartCmd`

Root cause:
- The wrapper lagged the REST API surface

Fix applied:
- Use REST directly for pod patching

Prevention:
- Treat pod creation/patching as REST-first whenever start-command or storage-layout control matters
