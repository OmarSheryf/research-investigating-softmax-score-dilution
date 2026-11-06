# Preserve local RoPE distances and halve the excess

All 128 records remain visible. At each layer and head, causal query–key separations through 16 tokens use ordinary RoPE. A larger separation d uses 16 + 0.5 × (d − 16). The fixed rule is evaluated on the same checkpoints and crossed position grid as ordinary and uniform half-scale RoPE.

The 16-token threshold was chosen in advance. The primary metrics are composition and retrieval when the queried fact lies in the earliest region, records 0–25. Five seeds per cap share all held-out examples.

| Cap | Condition | First fact | Second fact | Both | Composition | Composition given both correct |
|---:|---|---:|---:|---:|---:|---:|
| 32 | full | 50.62% | 50.81% | 25.91% | 29.29% | 90.04% |
| 32 | full_scaled | 27.34% | 26.86% | 7.74% | 15.79% | 89.67% |
| 32 | local_preserved | 78.79% | 78.26% | 61.49% | 61.43% | 95.98% |
| 64 | full | 79.53% | 80.33% | 63.88% | 62.75% | 93.22% |
| 64 | full_scaled | 35.43% | 34.83% | 12.74% | 18.68% | 88.47% |
| 64 | local_preserved | 97.67% | 97.78% | 95.49% | 94.64% | 98.44% |

## Retrieval by evidence region

| Cap | Query | Region | Ordinary | Uniform 0.5 | Local preserved |
|---:|---|---|---:|---:|---:|
| 32 | first | 0–25 | 40.66% | 22.72% | 50.81% |
| 32 | first | 26–51 | 37.72% | 26.12% | 71.22% |
| 32 | first | 52–76 | 42.75% | 30.72% | 94.22% |
| 32 | first | 77–102 | 63.16% | 28.84% | 97.59% |
| 32 | first | 103–127 | 68.84% | 28.28% | 80.12% |
| 32 | second | 0–25 | 42.16% | 21.31% | 50.03% |
| 32 | second | 26–51 | 38.78% | 25.12% | 69.47% |
| 32 | second | 52–76 | 42.78% | 29.22% | 94.31% |
| 32 | second | 77–102 | 62.34% | 30.06% | 97.75% |
| 32 | second | 103–127 | 67.97% | 28.56% | 79.72% |
| 64 | first | 0–25 | 50.06% | 37.00% | 96.28% |
| 64 | first | 26–51 | 74.25% | 35.47% | 98.38% |
| 64 | first | 52–76 | 93.25% | 35.84% | 98.78% |
| 64 | first | 77–102 | 93.78% | 35.44% | 99.22% |
| 64 | first | 103–127 | 86.31% | 33.38% | 95.72% |
| 64 | second | 0–25 | 52.16% | 37.47% | 96.50% |
| 64 | second | 26–51 | 74.91% | 33.47% | 98.16% |
| 64 | second | 52–76 | 93.69% | 35.25% | 99.34% |
| 64 | second | 77–102 | 93.75% | 33.66% | 98.53% |
| 64 | second | 103–127 | 87.12% | 34.28% | 96.38% |

## Paired changes from each baseline

| Cap | Baseline | Metric | Change, pp | Descriptive 95% seed interval, pp |
|---:|---|---|---:|---:|
| 32 | full | first | +28.17 | [+23.74, +32.60] |
| 32 | full | second | +27.45 | [+23.29, +31.61] |
| 32 | full | both | +35.59 | [+32.30, +38.87] |
| 32 | full | composition | +32.14 | [+29.80, +34.47] |
| 32 | full | composition_given_both | +6.05 | [+3.99, +8.10] |
| 32 | full | early_first | +10.16 | [+3.53, +16.79] |
| 32 | full | early_second | +7.88 | [+1.10, +14.65] |
| 32 | full_scaled | first | +51.46 | [+44.01, +58.90] |
| 32 | full_scaled | second | +51.40 | [+43.63, +59.17] |
| 32 | full_scaled | both | +53.75 | [+49.73, +57.77] |
| 32 | full_scaled | composition | +45.64 | [+42.89, +48.38] |
| 32 | full_scaled | composition_given_both | +5.86 | [+4.68, +7.05] |
| 32 | full_scaled | early_first | +28.09 | [+24.18, +32.00] |
| 32 | full_scaled | early_second | +28.72 | [+25.03, +32.40] |
| 64 | full | first | +18.14 | [+16.43, +19.86] |
| 64 | full | second | +17.46 | [+15.25, +19.66] |
| 64 | full | both | +31.61 | [+28.50, +34.72] |
| 64 | full | composition | +31.89 | [+29.34, +34.45] |
| 64 | full | composition_given_both | +5.19 | [+3.24, +7.15] |
| 64 | full | early_first | +46.22 | [+36.78, +55.66] |
| 64 | full | early_second | +44.34 | [+33.59, +55.10] |
| 64 | full_scaled | first | +62.25 | [+56.96, +67.54] |
| 64 | full_scaled | second | +62.96 | [+58.09, +67.82] |
| 64 | full_scaled | both | +82.75 | [+79.17, +86.33] |
| 64 | full_scaled | composition | +75.96 | [+73.09, +78.84] |
| 64 | full_scaled | composition_given_both | +9.71 | [+5.58, +13.83] |
| 64 | full_scaled | early_first | +59.28 | [+55.17, +63.40] |
| 64 | full_scaled | early_second | +59.03 | [+54.85, +63.21] |

The position-dependent score path matched uniform scaling at window zero and ordinary RoPE when the window covered the full input. Both saved reference conditions were replayed on a sample from each checkpoint. The reported intervals describe variation across five seeds on the shared held-out examples.

Run `python test_local_rope_compression.py` with the preceding experiment results and checkpoints in place.