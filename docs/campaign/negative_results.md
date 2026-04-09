# Negative Results

Structured from `docs/codex-memory/decisions.md`, `docs/campaign/results_log.jsonl`, and `docs/codex-memory/project-state.md` for Section VII of PR `#1017`. Rows below are limited to what those three sources directly support.

Delta conventions used below:

- Session 03 pre-TTT anchor (`1.12904446` sliding-window s64 BPB) for the early Session 04/05 deltas
- `05c-plus` (`1.12557920` sliding-window s64 BPB) for the `05f` / `05g` follow-ons
- `07c1` base seed `1337` (`1.11416225` sliding-window s64 BPB) for the `07c1` ablations
- `no improvement` means no clean positive sliding-window result, or no measured run at all

| Intervention name | Hypothesis | Measured BPB delta | Number of seeds tested | Kill criterion | Status |
| --- | --- | --- | --- | --- | --- |
| GPTQ family (documented probes: GPTQ-lite clip search, full-Hessian replay, export-only probe) | Smarter post-training quantization would recover roundtrip loss without retraining. | no improvement; best logged delta was `+0.00036910` s64 vs Session 03 anchor, and later GPTQ replay/export probes degraded to `2.15605819` and `3.96902897` roundtrip BPB | 3 logged single-seed probes | No positive measured delta survived logging; replay/export correctness was bad enough that the whole GPTQ lane was treated as archival rather than foreground. | killed |
| FA3 saved-container port | FlashAttention-3 would improve throughput or quality "for free" on the same model family. | `+0.00054538` s64 vs Session 03 anchor | 1 | Worse BPB than the anchor on the same metric, so extra FA3 integration work was not justified. | killed |
| LeakyReLU^2 / SQ delta | Activation-shape changes could buy a small free gain. | `-0.00000323` s64 vs Session 03 anchor | 1 | Logged as neutral; the effect was too small to justify keeping the branch alive. | killed |
| Mixed-bit int5/int6 lane (`05c-plus`) | Mixed int5/int6 compression plus the `05c-plus` training bundle could open a better mainline than the clean SP8192 branch. | `-0.00346526` s64 vs Session 03 anchor | 1 | Quality moved the right way, but throughput regressed and the lane was later explicitly marked superseded by the cleaner SP8192 frontier. | demoted |
| `05f` bigram3072 + warmdown4000 refinement | Extra bigram/warmdown tuning would improve on `05c-plus`. | `+0.00102744` s64 vs `05c-plus` | 1 | Worse than the parent `05c-plus` bundle. | killed |
| `05g` XSA8 throughput variant | XSA8 throughput tuning would preserve `05c-plus` quality while improving runtime characteristics. | `+0.00026314` s64 vs `05c-plus` | 1 | Worse BPB than `05c-plus`, and the artifact grew to `16,475,467` bytes. | killed |
| `QK_GAIN=5.0` | Stronger Q/K gain would improve the `07c1` line under the same export budget. | `+0.00098947` s64 vs `07c1` base `s1337` | 1 | Explicitly listed as falsified in both `decisions.md` and `project-state.md`. | killed |
| MLP width expansion thesis (`MLP_MULT=3.08`, `MLP_MULT=3.5`) | Wider MLPs would deliver enough BPB gain to justify the added bytes. | `MLP_MULT=3.08`: `+0.00111782` s64 vs `07c1` base; `MLP_MULT=3.5`: `-0.00520350` s64 but over cap | 2 | The in-cap width lost quality, while the quality-positive width exceeded the `16,000,000`-byte limit. | killed |
| `07c1` legal TTT reruns | Score-first TTT on `07c1` would produce a clean measured gain worth carrying forward. | no improvement | 5 logged attempts across seeds `1337`, `42`, and `2025`; repaired Pegasus reruns later failed operationally | The logged variants timed out before sliding-window eval, and the later repaired Pegasus reruns still failed, so `07c1` TTT was kept background-only. | demoted |
| RFN / attribution-graph pipeline | Thesis-derived relevance / attribution graphs would produce a novel compression/export win on the competition timeline. | no improvement | 0 | Literature plus strategy review concluded the bridge from thesis RFN ideas to this tiny-transformer setting was too slow for the deadline, with only small int6-scale deltas expected and substantial prior art already covering the space. | killed |
| Fisher / Hessian sensitivity-guided export sidecar | Lightweight sensitivity-guided export might still help after the mainline stack was stabilized. | no improvement | 0 | Kept only as a later sidecar behind corrected `E` and cheaper checkpoint-reuse export/TTT experiments. | deferred |
