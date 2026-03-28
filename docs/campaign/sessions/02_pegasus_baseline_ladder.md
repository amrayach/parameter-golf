# Session 02: Pegasus Baseline Ladder

Preferred mode: Execution

Use `/research-engineer` if it exists locally and is helpful for experiment planning, but execute the work directly in this session.

Goal:
- Validate the Pegasus environment and produce a trustworthy baseline ladder from the public baseline toward a cheap mid-tier checkpoint.

Read these first:
- @README.md
- @train_gpt.py
- @docs/campaign/PEGASUS_H100_RUNBOOK.md
- @docs/campaign/templates/RUN_MANIFEST_TEMPLATE.md
- @docs/campaign/templates/EXPERIMENT_SUMMARY_TEMPLATE.md
- @records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md
- @records/track_10min_16mb/2026-03-18_LowerLR/README.md
- @records/track_10min_16mb/2026-03-19_SlidingWindowEval/README.md
- @records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md

Constraints:
- Keep this session focused on environment trust and cheap reproduction, not SOTA chasing.
- Work in a new self-contained folder under `records/track_non_record_16mb/<today>_pegasus_baseline_ladder/`, where `<today>` is the current date in `YYYY-MM-DD` format.
- Preserve reproducibility details and the exact commands used.

Required workflow:

**Step 0 — MANDATORY Pegasus Live Verification (before any training)**

Run the following on the Pegasus login node and record all output:

```bash
# Partition and node availability
sinfo -N -p H100,H100-RP,H100-SEE,H100-PCI -o "%P %N %G %t %c %m"

# Partition config and limits
scontrol show partition H100
scontrol show partition H100-RP
scontrol show partition H100-SEE

# Node hardware details (expected: H100-SXM5, 8 GPUs)
scontrol show node serv-3340
scontrol show node serv-3341
scontrol show node serv-3342
scontrol show node serv-3343

# Your account fairshare
sshare -u "$USER"

# Your QoS/account limits
sacctmgr show assoc where user="$USER" format=Account,User,Partition,QOS,GrpTRES,MaxTRES,MaxJobs

# ACTUAL ALLOCATION TEST — this is the real proof
salloc -p H100 --nodes=1 --gpus=8 --time=00:15:00 --gpu-bind=none
# If queues >15 min, Ctrl-C and try H100-RP, then H100-SEE

# Once allocated:
nvidia-smi -L
nvidia-smi topo -m
```

Write results to `docs/campaign/artifacts/02a_pegasus_verification.md`. If any of the following are true, STOP and reassess before proceeding:
- Account cannot access H100-class partitions
- Cannot get 8 GPUs on a single node
- QoS blocks short 8-GPU jobs
- GPU type is not H100 SXM (80GB HBM3)

Only after verification passes, proceed to step 1.

1. Inspect the baseline and selected ladder records to decide the smallest credible set of reproduction runs.
2. Create the non-record experiment folder with a README and any run helper files you need.
3. Run or stage the baseline and one or two cheap follow-up variants that test:
   - baseline parity,
   - one simple LR or warmdown improvement,
   - one eval-side control such as sliding window if feasible.
4. Record all settings in a run manifest and summarize the measured outputs.
5. End with a clear judgment: trusted environment or not yet trusted.

Deliverables:
- `records/track_non_record_16mb/<today>_pegasus_baseline_ladder/README.md`
- `records/track_non_record_16mb/<today>_pegasus_baseline_ladder/train_gpt.py` or a clearly justified wrapper path
- `records/track_non_record_16mb/<today>_pegasus_baseline_ladder/` run logs and manifests
- `docs/campaign/artifacts/02_pegasus_baseline_ladder_summary.md`

Definition of done:
- `docs/campaign/artifacts/02a_pegasus_verification.md` exists with live Slurm output confirming access, GPU type, and 8-GPU allocation feasibility.
- There is at least one measured Pegasus baseline run.
- There is at least one measured comparison run.
- The summary states whether session 3 should proceed unchanged or first fix infrastructure.

Commit message:
- `feat(campaign): add pegasus baseline ladder experiment`
