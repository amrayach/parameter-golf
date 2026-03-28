# Pegasus Live Verification

Date: 2026-03-27
Status: **PARTIAL — hardware access confirmed, GPU type unverified, allocation not yet tested under load**

---

## Results

### 1. Partition visibility — PASS

```
sinfo -N -p H100,H100-RP,H100-SEE,H100-PCI -o "%P %N %G %t %c %m"

PARTITION NODELIST GRES STATE CPUS MEMORY
H100-PCI serv-3310 gpu:6(S:0,2-3) mix 192 1060783
H100 serv-3340 gpu:8(S:0-1) mix 224 1960580
H100 serv-3341 gpu:8(S:0-1) mix- 224 1960580
H100-RP serv-3341 gpu:8(S:0-1) mix- 224 1960580
H100 serv-3342 gpu:8(S:0-1) mix- 224 1960580
H100-RP serv-3342 gpu:8(S:0-1) mix- 224 1960580
H100 serv-3343 gpu:8(S:0-1) mix 224 1960580
H100-SEE serv-3343 gpu:8(S:0-1) mix 224 1960580
```

User can see all H100-class partitions. Nodes confirm 8 GPUs each on H100/H100-RP/H100-SEE.

### 2. Account and limits — NO IMMEDIATE REJECTION

```
sacctmgr show assoc where user="$USER" format=Account,User,QOS,GrpTRES,MaxTRES,MaxJobs,MaxWall
(empty output — no explicit per-partition associations returned)

sshare -u "$USER"
Account: compute-account
RawShares: 1, NormShares: 0.500000
RawUsage: 0, EffectvUsage: 0.000000
```

Account is in `compute-account`. No explicit partition restrictions returned by sacctmgr. Job submissions were accepted (queued), not rejected. However, empty sacctmgr output does not prove absence of all limits — it may mean limits are set at a level not visible to this query. Actual allocation still needed to confirm no policy blocks exist.

### 3. Allocation test — NOT YET VERIFIED

All attempts queued due to cluster saturation:

- `salloc -p H100 --nodes=1 --gpus=8 --time=00:15:00` → queued, cancelled after wait
- `salloc -p H100-RP --nodes=1 --gpus=8 --time=00:15:00` → queued ("not available now")
- `salloc -p H100 --nodes=1 --gpus=1 --time=00:10:00` → queued
- `salloc -p H200 --nodes=1 --gpus=1 --time=00:10:00` → queued (Priority)

Grafana dashboard at time of test showed:
- H100: 32/32 GPUs allocated (100%), 52 jobs queued
- H100-RP: 16/16 allocated (100%), 51 queued
- H100-SEE: 8/8 allocated (100%)
- H200: 24 GPUs, some free but 3 queued with higher priority

**Conclusion:** Account can submit jobs to H100 partitions (not immediately rejected). Pending reason is `Priority` (cluster load), not `PartitionNotAvail` or `AccountNotAllowed`. But actual allocation has not succeeded, so hidden policy limits cannot be ruled out.

### 4. GPU type verification — PARTIAL (A100 confirmed, H100 still pending)

V100 allocation succeeded on glendale but PyTorch 2.11 does not support CC 7.0 (V100). Abandoned.

A100-80GB allocation succeeded on serv-3333:
```
GPU 0: NVIDIA A100 80GB PCIe (UUID: ...)
```

H100 SXM verification still pending (cluster saturated during test window).

### 5. Baseline smoke test — PASS (A100-80GB, 1 GPU)

```
GPU: A100-80GB (serv-3333, A100-IML partition)
AMP dtype: bf16 (auto-detected)
Model params: 17,059,912
Config: 9L, 512d, 1024 vocab, seq 1024, batch 65536, 200 iterations
Steps: 200/200 completed
Train loss: 6.937 → 3.684
Val loss: 3.6186
Val BPB: 2.1432 (expected high — only 200 steps, reduced batch, 1 shard)
Step avg: 154.57 ms
Peak memory: 1548 MiB
Serialized model int8+zlib: 7,017,338 bytes (7.0 MB)
Total submission size: 7,066,088 bytes (7.1 MB)
```

Evidence established:
- Training loop runs end-to-end without errors
- AMP_DTYPE auto-detection works (bf16 on A100)
- Muon optimizer + compiled training loop stable
- Int8 + zlib export/quantization pipeline works
- Model fits well within 16MB budget

### 6. 8-GPU single-node feasibility — PENDING

Only 1-GPU allocation tested so far. 8-GPU H100 allocation blocked by cluster load during test window.

---

## Assessment

| Check | Result |
|-------|--------|
| Account can see H100 partitions | **PASS** |
| No immediate account/partition rejection | **PASS** (jobs queued, not denied) |
| Actual policy limits fully cleared | **UNVERIFIED** (empty sacctmgr ≠ no limits) |
| Scheduler accepts 8-GPU requests | **PASS** (queued, not rejected) |
| Actual 1-GPU allocation (A100-80GB) | **PASS** (serv-3333) |
| Actual 8-GPU allocation on H100 | **PENDING** (cluster saturated) |
| Baseline smoke test (A100, 200 steps) | **PASS** (training + export + eval end-to-end) |
| H100 SXM GPU type via nvidia-smi | **PENDING** |
| NVSwitch topology confirmed | **PENDING** |

## Next Steps

1. Retry allocation during off-peak hours (evening/night/weekend)
2. Preferred: `salloc -p H100 --nodes=1 --gpus=1 --time=00:10:00` (just 1 GPU to verify hardware)
3. Once allocated: run `nvidia-smi -L` and `nvidia-smi topo -m`
4. After GPU type confirmed, attempt full 8-GPU allocation
5. Only after full verification: proceed to baseline training

## Fallback

If H100 remains unavailable for extended periods:
- H200 (SXM5, NVSwitch) is suitable for development but not leaderboard parity
- A100-80GB is a lower-priority fallback
- RunPod for final validation only (1-2 runs)
