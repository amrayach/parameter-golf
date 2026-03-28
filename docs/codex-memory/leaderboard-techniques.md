# Leaderboard Techniques

## Current public shape

Best public stack on 2026-03-27:
- `1.1194`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

Best public non-TTT stack:
- `1.1228`
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

## What matters most

Large practical gains:
- 11 layers
- MLP 3x
- seq len 2048
- int6-aware compression and export
- strong eval protocol

Mid-size gains:
- XSA on late layers
- SmearGate plus BigramHash
- U-Net skip connections
- partial RoPE

Small but real gains:
- EMA
- GPTQ-lite clip search
- warmdown tuning
- LeakyReLU squared

Later complexity:
- legal score-first TTT
- parameter banking plus parallel Muon

## Current recommendation

Build upward:
1. trusted baseline on Pegasus
2. strong non-TTT anchor in the `1.123-1.128` band
3. narrow delta sweep
4. only then TTT

Do not start with the full top stack.
