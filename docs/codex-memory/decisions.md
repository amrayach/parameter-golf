# Locked Decisions

## Strategy

- Do not lead with the RFN thesis idea.
- Build a strong non-TTT anchor first.
- Treat TTT as a later integration, only after audit and only if the anchor is already competitive.
- Treat RFN or attribution-graph work as a sidecar probe, not the main competition bet.
- Treat open public PR claims as horizon signals, not as accepted leaderboard facts, until they are merged or independently validated.
- Do not skip the clean anchor to chase open-claim branches like GDN or two-pass n-gram rescoring before the current stack is running.

## Hardware

- Pegasus `H100` is the primary intended execution base.
- RunPod is reserved for final challenge-style validation only.
- Because the RunPod budget is only about `$25`, assume at most `1-2` meaningful final validations there, not broad sweeps.
- Non-H100 Pegasus or RunPod runs are allowed only as development or grant-support evidence when clearly labeled as non-parity evidence.

## Workflow

- Keep competitive experiments in self-contained folders under `records/track_non_record_16mb/YYYY-MM-DD_<tag>/`
- Do not modify existing public record folders
- Document every run with manifests and experiment summaries
- Prefer additive, well-understood public techniques over speculative novelty

## Hard gates

- No baseline training before live Pegasus verification
- No TTT implementation before a correctness and legality audit
- No RFN continuation unless it beats or complements magnitude-based heuristics on a controlled test
- No claim of top-competitor status or `Advanced` compute-grant posture before there is an owned competitive result

## Compute-grant stance

- Preferred compute-grant tier, if reapplying now: `Development grant`
- Strongest supporting evidence package:
  - `1` root baseline run
  - `1` narrow clean-anchor smoke run
  - logs for GPU type, steps, wallclock, `val_bpb`, artifact size, eval mode, and compile/export warnings

## Memory design

- shared memory in repo: `docs/codex-memory/`
- private Codex mirror: `~/.codex/memories/parameter-golf/`
- keep this separate from Claude's own built-in memory or `CLAUDE.md` workflows
