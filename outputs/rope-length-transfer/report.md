# Frozen RoPE rule across context lengths

Five saved seeds per training cap are tested on fresh examples at 32, 64, 96 and 128 records. The compression rule preserves distances through 16 tokens and halves the excess in each attention layer. Both inference modes see identical records. Each length has a balanced 5 × 5 grid of ordered evidence regions and two independently seeded batches of 32 examples per cell. Early retrieval means the queried fact is in the first fifth of that context.

The gate dispatches to ordinary RoPE at or below the training cap and compression above it. Its scores are computed exactly from the two measured inference modes, since the gate chooses a single mode per input. The policy assumes total record count is known at inference.

| Cap | Records | Metric | Ordinary | Compressed | Gated | Compression delta, pp | Descriptive 95% seed interval, pp |
|---:|---:|---|---:|---:|---:|---:|---:|
| 32 | 32 | composition | 99.91% | 99.84% | 99.91% | -0.08 | [-0.31, +0.16] |
| 32 | 32 | early_first | 99.94% | 100.00% | 99.94% | +0.06 | [-0.11, +0.24] |
| 32 | 32 | early_second | 99.94% | 99.94% | 99.94% | +0.00 | [-0.27, +0.27] |
| 32 | 64 | composition | 77.12% | 98.89% | 98.89% | +21.76 | [+20.93, +22.59] |
| 32 | 64 | early_first | 62.19% | 99.44% | 99.44% | +37.25 | [+35.92, +38.58] |
| 32 | 64 | early_second | 61.88% | 99.31% | 99.31% | +37.44 | [+34.61, +40.27] |
| 32 | 96 | composition | 42.76% | 86.06% | 86.06% | +43.30 | [+38.46, +48.14] |
| 32 | 96 | early_first | 38.19% | 74.12% | 74.12% | +35.94 | [+29.25, +42.62] |
| 32 | 96 | early_second | 37.06% | 73.75% | 73.75% | +36.69 | [+27.26, +46.11] |
| 32 | 128 | composition | 28.75% | 60.09% | 60.09% | +31.34 | [+29.06, +33.62] |
| 32 | 128 | early_first | 40.88% | 49.50% | 49.50% | +8.62 | [-2.02, +19.27] |
| 32 | 128 | early_second | 37.94% | 46.75% | 46.75% | +8.81 | [+2.09, +15.53] |
| 64 | 32 | composition | 99.21% | 99.24% | 99.21% | +0.02 | [-0.27, +0.32] |
| 64 | 32 | early_first | 99.94% | 99.62% | 99.94% | -0.31 | [-0.70, +0.08] |
| 64 | 32 | early_second | 99.81% | 99.81% | 99.81% | +0.00 | [-0.27, +0.27] |
| 64 | 64 | composition | 98.02% | 98.50% | 98.02% | +0.48 | [-0.14, +1.09] |
| 64 | 64 | early_first | 98.12% | 99.94% | 98.12% | +1.81 | [+0.93, +2.70] |
| 64 | 64 | early_second | 98.94% | 99.75% | 98.94% | +0.81 | [+0.16, +1.46] |
| 64 | 96 | composition | 85.40% | 97.64% | 97.64% | +12.24 | [+10.05, +14.42] |
| 64 | 96 | early_first | 79.06% | 98.62% | 98.62% | +19.56 | [+13.94, +25.19] |
| 64 | 96 | early_second | 78.06% | 99.25% | 99.25% | +21.19 | [+14.34, +28.03] |
| 64 | 128 | composition | 62.78% | 94.51% | 94.51% | +31.74 | [+29.54, +33.93] |
| 64 | 128 | early_first | 51.25% | 96.50% | 96.50% | +45.25 | [+33.61, +56.89] |
| 64 | 128 | early_second | 51.31% | 96.00% | 96.00% | +44.69 | [+37.86, +51.51] |

The intervals describe variation across five paired checkpoint seeds on the same fresh test sets. Run `python test_rope_length_transfer.py` with the saved context-scale-up checkpoints in place.