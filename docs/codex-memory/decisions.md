# Locked Decisions

## Mainline

- `07c` and `07c1` are now background evidence lines, not the foreground branch hunt.
- The completed strict RunPod `#1394` SP8192 baseline is the new stable base.
- The next foreground branch is faithful `#1413`, not more `#1394` reruns and not more `07c1` polish.
- `#1437` stays the later stacked frontier target.
- `#1420` is a component / legality ablation, not the main endpoint.
- `#1416` is a later, higher-review-risk lane only after `#1413` / `#1437` are understood.

## A/B/C/D/E offline prep strategy (2026-04-07)

Decision: instead of launching a single faithful `#1413` seed-0 as the sole next paid pod action, materialize the full A/B/C/D/E variant suite offline first and launch them as a batch on the next pod session.

Rationale:

- reduces on-pod debugging time by proving the wrapper format and patch logic locally before spending GPU-hours
- enables a clean sequential ablation (A=faithful control, B=parallel residual, C=loop adjustment, D=combined, E=ngram eval-only) in a single reserved session
- the offline prep is cheap (no GPU cost); the only cost is one more pre-pod review pass

What was fixed as part of this prep (2026-04-07):

- `_wrap_source` in `scripts/prepare_pr1413_variants.py` now uses `FORMAT_RAW + FILTER_LZMA2` and ASCII string literal — exactly matching upstream `#1413` wrapper format
- a decode roundtrip assertion was added to `main()` to prove compress → decompress → compare succeeds before writing the folder
- `scripts/runpod_1413.sh` now copies `final_model.pt` conditionally — Run E with `SKIP_TRAINING=1` does not produce a fresh `.pt`; the copy is skipped gracefully when the file is absent

Remaining decisions deferred until first pod result is in hand:

- which of B/D/E wins and whether to run additional seeds on the winner
- whether `#1437` should be climbed directly after `#1413` or requires intermediate ablation
- artifact-cap margin for B/D/E is unmeasured until a real stack run completes

## `#1394` archive handling

- The preserved `#1394` archive has now been fetched locally:
  - `artifacts/runpod_pull/pr1394_archive_2026-04-07/pr1394_archive_2026-04-07.tgz`
  - SHA256 `95e3a38cc89a160469d8417d0d4dbd40ef6a5106803d25ccdab5c0f86e2c0b07`
- The local strict proof folder now exists:
  - `records/track_10min_16mb/2026-04-07_pr1394_sp8192_runpod_strict`
- The old pod/volume is no longer the only durable copy of the `#1394` evidence.
- The old CPU-only recovery pod may be terminated/deleted; it no longer carries unique `#1394` evidence.
- Do not spend future paid RunPod time on archive recovery; the next paid RunPod session should start directly on `#1413` on a fresh `8xH100 SXM` pod.

## Record arithmetic

- Official threshold calculations stay in **nats**, not BPB shortcuts.
- Current merged official `#1019` remains `1.88217853` nats.
- The current record bar remains `1.87717853` nats.
- `07c1` evidence remains useful for official-record arithmetic, but it is no longer the foreground route to recognition or the open frontier.

## `07c1` interpretation

- Keep Pegasus `07c1` TTT reruns alive as background evidence only:
  - `2740306` `07c1_ttt_s1337_fix`
  - `2740307` `07c1_ttt_s2025_fix`
- Latest checked state for both reruns: `FAILED`
- Do not let unresolved `07c1` TTT block the SP8192 line.
- Falsified `07c1` levers remain falsified:
  - `QK_GAIN=5.0`
  - `MLP_MULT=3.08`
- `MLP_MULT=3.5` remains over cap under the current export path.

## SP8192 branch priority

- The live clean / reviewer-facing branch order remains:
  - `#1394`
  - `#1413`
  - `#1437`
- Treat SLOT as out of mainline scope until the legality picture is explicit.

## RunPod workflow

- RunPod is the foreground execution lane for SP8192 reproductions when credits are available.
- Pegasus remains the canonical validator and background TTT lane.
- For the current migration pod shape, `/workspace` is the persistence boundary; `/root` is not reliable across `Stop`.
- Do not assume the old direct-TCP `rsync` / `scp` path works on this pod.
- The successful `#1394` recovery path was a PTY-driven gateway shell with local base64 capture and checksum verification.

## Documentation discipline

- Keep mutable current state in `docs/campaign/AGENT_SYNC.md`.
- Append measured runs to `docs/campaign/results_log.jsonl`.
- Update `docs/codex-memory/project-state.md` and `docs/codex-memory/next-session.md` when the branch priority or preserved artifact state changes materially.
