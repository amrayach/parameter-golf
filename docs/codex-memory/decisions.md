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
  - preserved proof base: `#1394`
  - measured clean foreground base: local canonical `D`
  - next attack: corrected token-only `E` / `#1437`-style eval-time tilt
- Treat SLOT and pre-quant validation fine-tuning as out of mainline scope.

## Frontier reinterpretation (2026-04-08)

Decision: reinterpret the `pr1413` RunPod batch results around the measured `D` bundle, not around a future faithful `#1413` restart.

Rationale:

- local `D` now has a canonical 5-seed mean of `1.08128837`, which is already ahead of the currently titled `#1420` number (`1.08309`) and ahead of `#1413` / `#1460`
- `#1416` is no longer a clean anchor after the author admitted the pre-quant validation TTT issue
- `#1423` is no longer a clean anchor after the comment thread pointed out direct validation fine-tuning before quantization
- the clean public target is now the corrected `#1437` number (`1.08091`), only `0.00037837` BPB ahead of local `D`
- the old `E` seed-0 gain is interesting, but that run predates the causal correction discussion and should not be treated as clean evidence

Operational consequence:

- do not spend the next paid H100 session on a faithful `#1413` replay
- spend it on a causally corrected, token-only, clearly auditable `E`-style eval-time n-gram layer on top of the existing `D` stack

## RFN direction triage (2026-04-08, post-Feynman)

Decision: kill full RFN / attribution-graph work as a competition-critical path. Treat Fisher / gradient-sensitivity-guided export as an optional sidecar, not the mainline.

Rationale:

- Feynman `compare` preferred Fisher / gradient sensitivity over RFN-lite and full attribution graphs, but only inside the RFN-derived option set
- Feynman `deepresearch` concluded the highest-EV competition path is still the corrected standard `E` path, with Fisher / Hessian sensitivity as an optional 2-day side experiment
- the literature review on sensitivity-guided compression found real prior art (GPTQ, AWQ, HAWQ, ECQx, Taylor/Fisher pruning), so this is not a thesis-exclusive novelty claim
- the literature review on tiny-transformer BPB improvements found only small quantization deltas at int6 scale; these are stackable refinements, not a replacement for the `E` attack
- the thesis-to-transformer RFN bridge remains a research problem, not an engineering task suitable for the April 30 deadline

Operational consequence:

- primary path: corrected token-only `E` on top of `D`
- secondary path: implement training-split-only Fisher / `(g*w)^2` sensitivity extraction and test one guided export policy
- defer block-level RFN-lite unless the sidecar wins and there is slack left before the deadline
- reject any “target-data calibration” idea that would consume validation/eval tokens before scoring; keep all calibration on the training side only

## Post-literature reprioritization (2026-04-08, code-grounded)

Decision: move export-only and eval-only `D`-checkpoint sidecars ahead of both Fisher and training-side optimizer work.

Rationale:

- the preserved `pr1413_archive_20260407_213205` bundle contains `final_model.pt` for the `D` runs, so export-only and eval-only experiments can be tested on the real measured checkpoints without paying for retraining first
- the current codebase already uses `torch.compile` in the active training / eval path and uses SGD+momentum for TTT, so some of Claude's proposed "missing" interventions are either already partly present or aimed at the wrong baseline
- CDQuant / OWC attack the current GPTQ export path directly
- RMS+decay / middle-layer-only TTT attack the current legal TTT path directly
- Cautious Muon and AdaMuon-style training changes remain interesting, but they require retraining and therefore sit behind the cheaper checkpoint-reuse levers

Operational consequence:

- first: corrected token-only `E` on top of `D`
- second: export-only GPTQ refinements (CDQuant / OWC) using preserved `final_model.pt`
- third: eval-only TTT refinements (optimizer / freeze policy) using preserved `final_model.pt`
- fourth: only then consider training-side optimizer variants such as Cautious Muon
- Fisher is now a later diagnostic / sidecar, not an immediate next action

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
