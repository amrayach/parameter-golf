# Diagnostics Index

This directory keeps local copies of analysis outputs pulled from Pegasus so
diagnostic state survives future training runs.

## Current contents

- `2026-03-31_05c_plus/`
  - float and float-vs-int6 reports for the best measured branch
- `2026-03-31_05f/`
  - cross-run comparison reports against `05c-plus`

## Canonical utilities

- `scripts/diagnostics/diagnose_weights.py`
  - single-checkpoint weight statistics
  - float-vs-int6 comparison on the same checkpoint
- `scripts/diagnostics/compress_probe.py`
  - export-path feasibility probe for saved `.int6.ptz` artifacts

## Typical commands

From the repo root:

```bash
python scripts/diagnostics/diagnose_weights.py final_model.pt
python scripts/diagnostics/diagnose_weights.py final_model.pt final_model.int6.ptz
python scripts/diagnostics/compress_probe.py diagnostics/2026-03-31_05c_plus/final_model.int6.ptz
```

## Notes

- The authoritative preserved artifacts live on Pegasus under `/netscratch/$USER/parameter-golf/diagnostics/`.
- This directory is for pulled reports and local interpretation, not for live training outputs.
