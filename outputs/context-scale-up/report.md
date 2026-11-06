# Training through 64 records

Five fresh seeds share a checkpoint trained through 32 records at 832,000 input tokens. Each branch spends 280,864 additional tokens. Continuing at 32 takes 131 updates. Moving to 64 takes 67 updates. Batch size is 32 and AdamW uses a constant 0.002 learning rate. Each final model has seen 1,112,864 input tokens.

The primary extrapolation test uses 128 records. Each length and position has two held-out populations of 128 examples. At 128 records, each context contains all 128 key identities exactly once. Positions refer to the first queried record. The second is half a context away with wraparound.

| Records | Metric | Cap 32 | Cap 64 | Paired change, pp | Descriptive 95% seed interval, pp |
|---:|---|---:|---:|---:|---:|
| 32 | single | 99.87% | 99.55% | -0.33 | [-0.51, -0.14] |
| 32 | two | 99.75% | 99.25% | -0.50 | [-1.06, +0.06] |
| 32 | both_reads | 99.80% | 99.27% | -0.53 | [-0.85, -0.21] |
| 64 | single | 83.28% | 98.44% | +15.16 | [+12.59, +17.73] |
| 64 | two | 68.27% | 97.00% | +28.73 | [+24.40, +33.07] |
| 64 | both_reads | 68.69% | 97.67% | +28.98 | [+25.19, +32.78] |
| 128 | single | 50.48% | 76.72% | +26.23 | [+20.41, +32.06] |
| 128 | two | 26.33% | 56.06% | +29.73 | [+27.65, +31.81] |
| 128 | both_reads | 22.39% | 57.03% | +34.64 | [+32.21, +37.07] |

## Evidence position

| Records | First evidence position | Retrieval cap 32 → cap 64 | Composition cap 32 → cap 64 |
|---:|---:|---:|---:|
| 32 | 0% | 99.69% → 99.77% | 99.69% → 99.38% |
| 32 | 25% | 100.00% → 100.00% | 99.84% → 99.38% |
| 32 | 50% | 99.84% → 99.38% | 99.45% → 99.22% |
| 32 | 75% | 99.92% → 99.45% | 99.84% → 99.38% |
| 32 | 100% | 99.92% → 99.14% | 99.92% → 98.91% |
| 64 | 0% | 52.50% → 96.72% | 51.80% → 96.17% |
| 64 | 25% | 81.64% → 99.22% | 80.23% → 98.12% |
| 64 | 50% | 99.77% → 100.00% | 52.27% → 97.34% |
| 64 | 75% | 99.53% → 99.53% | 74.92% → 97.19% |
| 64 | 100% | 82.97% → 96.72% | 82.11% → 96.17% |
| 128 | 0% | 33.91% → 44.69% | 20.47% → 42.19% |
| 128 | 25% | 42.19% → 72.03% | 32.50% → 63.36% |
| 128 | 50% | 42.34% → 94.61% | 18.12% → 39.61% |
| 128 | 75% | 67.11% → 90.94% | 31.17% → 61.33% |
| 128 | 100% | 66.88% → 81.33% | 29.38% → 73.83% |

Single measures the first readout. Two measures learned composition. Both reads measures exact accuracy on both underlying fact readouts. The paired intervals describe seed variation on shared test populations. Fixed batch size makes update count differ across the two token-matched schedules.

Run `python test_context_scale_up.py` alongside the imported experiment scripts. Checkpoints store weights, optimizer state and random generator states.