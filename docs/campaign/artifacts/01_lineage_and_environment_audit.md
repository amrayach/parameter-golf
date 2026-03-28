# Session 01: Lineage and Environment Audit

Date: 2026-03-27
Status: Complete

---

## 1. Stack Progression: Baseline → Non-TTT SOTA → TTT SOTA

### 1.1 Baseline (2026-03-18, val_bpb = 1.2244)

Config: 9 layers, 512-dim, 8 heads (4 KV), MLP 2x, vocab 1024, seq_len 1024, tied embeddings.
Optimizer: Muon (lr=0.04, matrices) + Adam (lr=0.05, embeddings). Batch: 524,288 tokens/step.
Quantization: int8 + zlib level 9. Artifact: 15.86 MB.
Training: 13,780 steps in 600s on 8xH100 (43.5 ms/step). Pre-quant val_bpb: 1.2172. Post-quant: 1.2244.

### 1.2 Mid-tier stack (2026-03-19 to 2026-03-20, val_bpb 1.19–1.15 range)

Additive gains from baseline:
1. **Seq len 2048** (-0.015): Longer context improves perplexity, doubles token-level context at fixed batch cost.
2. **11 layers** (-0.02): Trade wider for deeper. Under int6+zstd, 11×512 compresses into 16MB budget.
3. **MLP 3x** (-0.01): 1536 hidden instead of 1024. More capacity per layer.
4. **Int6 quantization** (-0.005): 64 levels vs 256. Requires per-row scaling with fp16 scale factors.
5. **zstd-22** (vs zlib-9): Better compression ratio, fits more params in 16MB.
6. **Muon WD 0.04** (-0.002): Weight decay on Muon-optimized parameters. Regularization.
7. **SmearGate + BigramHash** (-0.005): Causal cumulative-mean blending + bigram embeddings (2048 buckets, dim=128). Cheap token-level features.
8. **OrthoInit + muP scaling** (-0.001): Orthogonal weight initialization + maximal update parameterization.
9. **Sliding window eval** (-0.005 apparent, not real training improvement): stride=64 evaluation gives more context per scored token. Does not improve the model, only the measurement.

### 1.3 Strong non-TTT (2026-03-20 to 2026-03-22, val_bpb 1.127–1.123)

Built on mid-tier stack, adding:
1. **U-Net skip connections** (-0.003): 5 encoder + 6 decoder layers with learned skip weights. Allows gradient flow from early to late layers.
2. **XSA on last 4 layers** (-0.0047): Exclusive Self Attention subtracts self-aligned value component. Encourages learning of orthogonal-to-self information. Zero new params, ~2ms/step overhead.
3. **EMA weight averaging** (-0.0006 vs SWA): Exponential moving average (decay=0.997, every step) replaces periodic SWA checkpoints. Smoother optimization landscape.
4. **Partial RoPE (16/64 dims)** (-0.0023): Apply rotary position embeddings to only 25% of head dimensions. Remaining 75% learn position-invariant patterns. Zero new params.
5. **LN Scale (1/sqrt(layer+1))** (-0.0005): RMSNorm output scaled by inverse sqrt of layer index. Dampens deep-layer contribution magnitudes. Zero new params.
6. **Shared Value Embeddings** (param efficiency): dim=128, layers 9-10 with per-layer learned scales.
7. **GPTQ-lite** (-0.0006): Per-row optimal clip percentile search (5 candidates) instead of fixed row max for int6 quantization. Zero training cost.
8. **Late QAT** (partially effective): STE int6 fake-quantization when LR scale < 0.15. Note: in some implementations this was dead code due to torch.compile constant-folding.
9. **Warmdown 3500 iter** (-0.001): Longer warmdown phase than default.
10. **FlashAttention 3** (throughput): Hopper-optimized attention kernel. Enables faster steps → more training steps in 600s.

Best non-TTT: **1.1228** (signalrush, PR #374 stack + GPTQ-lite).

### 1.4 TTT stack (2026-03-23, val_bpb = 1.1194)

All of the above, plus:
1. **LeakyReLU²(0.5)** (-0.0025): Replaces relu² activation. `F.leaky_relu(x, 0.5).square()`. Allows negative gradient flow, eliminates dead neurons.
2. **Legal Score-First TTT** (-0.0025): Validation split into 1,893 × 32K-token chunks. Score chunk first under `torch.inference_mode()`, then train on already-scored tokens via SGD(lr=0.002, momentum=0.9, 3 epochs/chunk, cosine LR, grad clip=1.0). Last chunk scored but never trained on. ~410s eval time budget.
3. **Parallel Muon with Parameter Banking** (throughput): 4 contiguous 3D `nn.Parameter` banks replace 66 separate `nn.Linear` weights. Batched Newton-Schulz via `torch.bmm`. Step time: 83.3 ms → more training steps.
4. **Muon momentum 0.99** (warmup 0.92→0.99 over 1500 steps): Higher final momentum.
5. **Gradient clip 0.3**: Tighter gradient clipping.
6. **Batch: 786,432 tokens/step**: Larger batch with seq_len=2048.

---

## 2. Ranked Lever Map

### Tier 1: High-yield, proven, low-risk (implement first)

| Lever | BPB Impact | Effort | Notes |
|-------|-----------|--------|-------|
| 11 layers + MLP 3x + seq 2048 | ~-0.045 combined | Low | Config-level changes. Core architecture. |
| Int6 + zstd-22 | ~-0.005 | Medium | Enables fitting 11L model in 16MB. |
| XSA (last 4 layers) | -0.0047 | Medium | Zero params. Small step-time cost. |
| EMA (decay=0.997) | -0.0006 | Low | Drop-in replacement for SWA. |
| Partial RoPE (16/64) | -0.0023 | Low | Zero params. Straightforward. |
| Sliding window eval (stride=64) | -0.005 apparent | Low | Not a model improvement — just better measurement. Must be in eval pipeline. |
| SmearGate + BigramHash | ~-0.005 | Medium | Token-level features. Requires careful implementation. |
| U-Net skip connections | ~-0.003 | Medium | Gradient flow improvement. |
| GPTQ-lite clip search | -0.0006 | Medium | Post-training quantization improvement. Zero training cost. |

### Tier 2: High-yield but higher complexity or risk

| Lever | BPB Impact | Effort | Risk |
|-------|-----------|--------|------|
| LeakyReLU²(0.5) | -0.0025 | Low | Well-validated. Low risk. |
| Legal TTT (score-first) | -0.0025 | High | Complex implementation. Legality must be audited. 410s eval budget. |
| Warmdown tuning (3500 iter) | -0.001 | Low | Sensitive to other hyperparameters. |
| Parameter Banking + Parallel Muon | throughput | High | Complex distributed code. Benefits depend on step-time bottleneck. |
| Late QAT (STE int6) | unclear | Medium | Dead code in some implementations. Needs careful integration with torch.compile. |

### Tier 3: Speculative / lower-yield

| Lever | Expected Impact | Notes |
|-------|----------------|-------|
| 12 layers (under int5) | unclear | Would need int5 to fit. Untested trade-off. |
| Different MLP ratios per layer | unclear | No public data. |
| Ternary/binary quantization | proven at 1.157 | Different architecture entirely. Not composable with current stack. |
| State-space models | unknown | On OpenAI's wishlist. No submissions yet. |
| Attribution-graph sidecar | research | Would inform quantization decisions. Not a direct BPB lever. |

### Tier 4: Not recommended for this campaign

| Lever | Reason |
|-------|--------|
| Full RFN pipeline | No implementation exists. Multi-week research project. |
| JEPA / Text diffusion | No existing submissions or evidence of viability within constraints. |
| Custom CUDA megakernels | Engineering effort disproportionate to expected gain. |

---

## 3. Pegasus Environment: Confirmed Assumptions

### H100 Partition — CONFIRMED SXM5

From Pegasus documentation (line 1103 of `docs/Pegasus_Server_documentation.txt`):

```
| H100 | H100-SXM5 | Hopper | 80 | 8 | 28 | 224 | 1 - 1 days |
```

- **GPU model:** H100-SXM5, 80GB HBM3
- **GPUs per node:** 8
- **Interconnect:** NVSwitch (full mesh GPU-to-GPU within node)
- **CPUs per node:** 28 (→ 3-4 CPUs per GPU for data loading)
- **Max time limit:** 1 day
- **Container requirement:** Enroot image version 22.09 or newer
- **No InfiniBand** (single-node only, which is what we need)

This is the correct hardware class. The challenge specifies 8xH100 SXM. Pegasus `H100` partition provides exactly that.

**Do NOT use:**
- `H100-PCI` (H100 NVL, 6 GPUs, different topology)
- `batch` (mixed GPUs, PCIe only)

### MANDATORY Live Verification (Session 02, before any training)

Documentation says SXM5 but this is NOT sufficient. The following must be verified with live Slurm commands before trusting Pegasus as the campaign base:

1. **Account access:** Can your user access H100-class partitions at all?
2. **Exact GPU type:** What does `nvidia-smi -L` report on allocated H100/H100-RP/H100-SEE nodes?
3. **8-GPU single-node allocation:** Can you request `--nodes=1 --gpus=8` and get all 8 GPUs on one node?
4. **QoS/fairshare limits:** What are your user's time limits, max TRES, max concurrent jobs?
5. **Schedulability:** Does a short 8-GPU job actually get scheduled in reasonable time (<30 min queue)?

**Verification commands (run on Pegasus login node):**

```bash
# 1. Partition and node info
sinfo -N -p H100,H100-RP,H100-SEE,H100-PCI -o "%P %N %G %t %c %m"

# 2. Partition details (limits, max time, allowed accounts)
scontrol show partition H100
scontrol show partition H100-RP
scontrol show partition H100-SEE

# 3. Node hardware details
scontrol show node serv-3340
scontrol show node serv-3341
scontrol show node serv-3342
scontrol show node serv-3343

# 4. Your fairshare and account standing
sshare -u "$USER"

# 5. Your account/QoS associations and limits
sacctmgr show assoc where user="$USER" format=Account,User,Partition,QOS,GrpTRES,MaxTRES,MaxJobs

# 6. Actual allocation test (the real proof)
salloc -p H100 --nodes=1 --gpus=8 --time=00:15:00 --gpu-bind=none
# If that queues >15 min, Ctrl-C and try:
salloc -p H100-RP --nodes=1 --gpus=8 --time=00:15:00 --gpu-bind=none
salloc -p H100-SEE --nodes=1 --gpus=8 --time=00:15:00 --gpu-bind=none

# 7. Once allocated, verify GPU model
nvidia-smi -L
nvidia-smi topo -m
```

**If any of these fail** (account denied, can't get 8 GPUs, QoS blocks it), the fallback plan is:
- Try H100-RP or H100-SEE partitions
- Try 4 GPUs instead of 8 (development only, not submission-grade)
- Fall back to A100-80GB partition (different GPU class, development only)
- Use RunPod for all 8xH100 work (extremely limited budget)

**Session 02 cannot proceed to training until this verification artifact is produced.**

### Scheduling Template

```bash
srun \
  --partition=H100 \
  --nodes=1 \
  --ntasks=8 \
  --gpus=8 \
  --cpus-per-gpu=3 \
  --gpu-bind=none \
  --time=01:00:00 \
  bash -lc '
    cd /home/amay/Work/parameter-golf &&
    torchrun --standalone --nproc_per_node=8 train_gpt.py
  '
```

Notes:
- `--gpu-bind=none` required for peer-to-peer visibility
- `--cpus-per-gpu=3` sufficient for data pipeline (DistributedTokenLoader is I/O-light)
- Store datasets on `/netscratch/$USER` for faster I/O
- Source code stays in `$HOME`

### Container Environment

Pegasus uses Enroot containers. The challenge requirements.txt lists: numpy, tqdm, torch, huggingface-hub, kernels, setuptools, typing-extensions, datasets, tiktoken, sentencepiece.

Options:
1. Use an NGC PyTorch container (≥22.09) with pip install of missing packages
2. Use conda environment via pegasus-bridle's activate_and_execute.sh

The pegasus-bridle wrapper handles conda activation, pip requirements, and environment variables. Set `PIP_REQUIREMENTS_FILE=requirements.txt` in `.env` or `.pegasus-bridle.env`.

---

## 4. RFN / Attribution-Graph Scope Assessment

### Current BachExpGraph Implementation

- **Architecture:** 75→50→30→2 MLP on synthetic 5×5 RGB toy images
- **Explainers:** Captum LayerLRP (primary), also supports Saliency, IntegratedGradients, DeepLift (unused)
- **RFN extraction:** Eq 3.2 (`E = R^{L+1} ⊙ W^T`) for gradient, Eq 3.3-3.4 for LRP — both derived for dense layers only
- **Graph infrastructure:** Adjacency matrix construction from RFN edge-relevance matrix → NetworkX graph
- **Validation:** Sanity checks only. No compression decisions, no pruning, no quantization guidance.
- **Dependencies:** PyTorch 1.10 (2021), Captum 0.4 (outdated)
- **Code maturity:** Early prototype. Hardcoded layer indices, no model abstraction, no tests.

### Gap: What a transformer-compatible probe requires

1. **New node types:** Attention heads (QKV projections), MLP sublayers, residual branches, norm layers
2. **New edge computation:** Local Jacobian or gradient attribution under frozen attention (per Anthropic's attribution-graph methodology). The thesis formulas (Eq 3.1-3.4) do not apply to multi-head attention or residual connections.
3. **Aggregation across sequences:** Need to average influence scores over representative validation batches, not single samples
4. **Validation task:** Must compare module rankings against ablation/quantization sensitivity, not just visualize heatmaps

### Recommendation for this campaign

**Do not make RFN/attribution-graphs the main campaign bet.**

Instead, pursue a **cheap sidecar probe in Session 06** with these constraints:
- Nodes = existing model components (attention heads, MLP channels, blocks)
- Edges = local Jacobian under frozen attention on ~100 validation sequences
- Score = indirect node-to-logit influence (ablation impact prediction)
- Validation = does this ranking predict int5-vs-int8 sensitivity better than weight magnitude?
- Budget: ≤2 hours on 1xH100, <200 lines of instrumentation code
- Gate: if attribution ranking does not beat magnitude by ≥0.0002 BPB on controlled experiment → stop

If the probe succeeds, use the ranking to inform mixed-precision quantization decisions (which layers get int5 vs int6 vs int8). This is a realistic, falsifiable contribution.

If the probe fails, stop and redirect effort to engineering levers from Tier 1-2.

---

## 5. Reproduction Targets for Sessions 02-03

### Session 02: Baseline Ladder

| Run | Expected val_bpb | Purpose |
|-----|-----------------|---------|
| Naive baseline (9L, 512d, seq1024) | ~1.2244 | Environment validation |
| Cheap variant: seq2048 + 10L | ~1.19-1.20 | Confirm cheap wins work on Pegasus |

Success: both runs complete under 10 min, val_bpb within 0.005 of expected.

### Session 03: Pre-TTT Anchor

Target: **val_bpb 1.123-1.128** (non-TTT). This is the band occupied by submissions #2-#4 on the leaderboard.

Build from the strongest non-TTT lineage (PR #374 stack):
- 11 layers, 512-dim, 8 heads (4 KV)
- MLP 3x (1536), LeakyReLU²(0.5) or relu²
- U-Net skips, XSA last 4 layers, Partial RoPE (16/64), LN Scale
- SmearGate + BigramHash, shared value embeddings
- EMA (0.997), warmdown 3500, gradient clip 0.3
- Int6 + GPTQ-lite + zstd-22
- Sliding window eval (stride=64)

If the anchor lands in this band, proceed to delta sweep. If it lands > 1.130, diagnose bottleneck before broadening.

---

## 6. RunPod Budget Constraint

The $25 RunPod credit allows approximately 3-4 usable 8xH100 runs after accounting for setup overhead, failed starts, and evaluation time.

**Policy:** All seed sweeps, experimentation, and debugging on Pegasus. RunPod reserved for exactly 1-2 final confirmation runs on official hardware. Do not budget 3 seeds on RunPod.

If Pegasus H100 SXM5 timing closely matches RunPod H100 SXM timing (which it should — same hardware class, same NVSwitch topology), a single RunPod confirmation run is sufficient to validate that our Pegasus numbers are representative.

---

## 7. Summary

| Item | Status |
|------|--------|
| H100 SXM5 availability on Pegasus | **Docs say yes** (line 1103). Live verification REQUIRED in Session 02. |
| NVSwitch topology | **Docs say yes** (line 1132). Unverified for user account access. |
| Correct partition | `H100` preferred, fall back to `H100-RP` or `H100-SEE` |
| Account can allocate H100 | **UNKNOWN — must verify with salloc** |
| 8 GPUs on single node feasible | **UNKNOWN — must verify with salloc** |
| QoS/fairshare allows short jobs | **UNKNOWN — must verify with sacctmgr/sshare** |
| Baseline reproduction target | val_bpb ~1.2244 |
| Pre-TTT anchor target | val_bpb 1.123-1.128 |
| RFN scope | Sidecar probe only (Session 06), gated on beating magnitude baseline |
| RunPod usage | 1-2 final confirmation runs only |
| Next action | **Session 02: Pegasus Baseline Ladder** |
