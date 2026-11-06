# Uniform RoPE compression at 128 records

Original and half-scale position IDs are crossed with full visibility and the previous 64-record retention mask. Fractional IDs are applied in every attention layer. Scaling preserves the same inputs, ordering and causal visibility within each comparison. Compaction is the earlier reference with contiguous IDs on the retained records.

The five seeds per cap share all held-out inputs. Full-scale, masked and compacted reference predictions are reused from the previous experiment and spot-checked on each checkpoint. Each cap and condition has 16,000 model-example observations.

| Cap | Condition | Fact 1 | Fact 2 | Both | Composition | Composition given both correct |
|---:|---|---:|---:|---:|---:|---:|
| 32 | full | 50.62% | 50.81% | 25.91% | 29.29% | 90.04% |
| 32 | full_scaled | 27.34% | 26.86% | 7.74% | 15.79% | 89.67% |
| 32 | masked | 63.44% | 63.31% | 40.29% | 41.64% | 92.43% |
| 32 | masked_scaled | 58.29% | 58.33% | 34.25% | 36.73% | 95.57% |
| 32 | compact | 88.92% | 88.65% | 78.75% | 78.80% | 98.18% |
| 64 | full | 79.53% | 80.33% | 63.88% | 62.75% | 93.22% |
| 64 | full_scaled | 35.43% | 34.83% | 12.74% | 18.68% | 88.47% |
| 64 | masked | 86.91% | 86.81% | 75.43% | 74.45% | 95.80% |
| 64 | masked_scaled | 68.48% | 69.03% | 47.43% | 48.68% | 96.57% |
| 64 | compact | 99.19% | 99.28% | 98.48% | 98.00% | 99.25% |

## Paired contrasts

| Cap | Contrast | Metric | Change, pp | Descriptive 95% seed interval, pp |
|---:|---|---|---:|---:|
| 32 | full_scaling | first | -23.29 | [-29.02, -17.55] |
| 32 | full_scaling | second | -23.95 | [-31.18, -16.72] |
| 32 | full_scaling | both | -18.16 | [-23.05, -13.27] |
| 32 | full_scaling | composition | -13.50 | [-17.47, -9.53] |
| 32 | full_scaling | composition_given_both | +0.18 | [-2.02, +2.38] |
| 32 | masked_scaling | first | -5.15 | [-8.32, -1.98] |
| 32 | masked_scaling | second | -4.98 | [-9.23, -0.72] |
| 32 | masked_scaling | both | -6.04 | [-10.88, -1.21] |
| 32 | masked_scaling | composition | -4.91 | [-8.76, -1.06] |
| 32 | masked_scaling | composition_given_both | +3.08 | [+1.27, +4.89] |
| 32 | interaction | first | +18.14 | [+15.07, +21.20] |
| 32 | interaction | second | +18.97 | [+15.56, +22.39] |
| 32 | interaction | both | +12.12 | [+11.33, +12.91] |
| 32 | interaction | composition | +8.59 | [+6.86, +10.32] |
| 32 | interaction | composition_given_both | +2.90 | [-0.39, +6.18] |
| 32 | compact_minus_masked_scaled | first | +30.64 | [+24.03, +37.25] |
| 32 | compact_minus_masked_scaled | second | +30.32 | [+24.13, +36.51] |
| 32 | compact_minus_masked_scaled | both | +44.50 | [+37.38, +51.62] |
| 32 | compact_minus_masked_scaled | composition | +42.07 | [+35.39, +48.75] |
| 32 | compact_minus_masked_scaled | composition_given_both | +2.61 | [+1.24, +3.99] |
| 64 | full_scaling | first | -44.11 | [-48.56, -39.65] |
| 64 | full_scaling | second | -45.50 | [-49.97, -41.03] |
| 64 | full_scaling | both | -51.14 | [-54.28, -48.00] |
| 64 | full_scaling | composition | -44.07 | [-46.60, -41.54] |
| 64 | full_scaling | composition_given_both | -4.52 | [-7.19, -1.84] |
| 64 | masked_scaling | first | -18.43 | [-21.38, -15.47] |
| 64 | masked_scaling | second | -17.78 | [-20.36, -15.20] |
| 64 | masked_scaling | both | -28.00 | [-31.73, -24.27] |
| 64 | masked_scaling | composition | -25.77 | [-30.10, -21.44] |
| 64 | masked_scaling | composition_given_both | +0.74 | [-0.58, +2.07] |
| 64 | interaction | first | +25.68 | [+23.83, +27.54] |
| 64 | interaction | second | +27.72 | [+25.62, +29.82] |
| 64 | interaction | both | +23.14 | [+21.44, +24.83] |
| 64 | interaction | composition | +18.30 | [+14.71, +21.89] |
| 64 | interaction | composition_given_both | +5.26 | [+1.74, +8.78] |
| 64 | compact_minus_masked_scaled | first | +30.71 | [+26.86, +34.57] |
| 64 | compact_minus_masked_scaled | second | +30.26 | [+27.10, +33.41] |
| 64 | compact_minus_masked_scaled | both | +51.05 | [+45.96, +56.14] |
| 64 | compact_minus_masked_scaled | composition | +49.32 | [+43.99, +54.65] |
| 64 | compact_minus_masked_scaled | composition_given_both | +2.70 | [+1.59, +3.80] |

## Retrieval by evidence position

| Cap | Query | Record region | Full | Full scaled | Masked | Masked scaled | Compact |
|---:|---|---|---:|---:|---:|---:|---:|
| 32 | first | 0–25 | 40.66% | 22.72% | 49.38% | 43.44% | 64.44% |
| 32 | first | 26–51 | 37.72% | 26.12% | 47.44% | 55.34% | 86.25% |
| 32 | first | 52–76 | 42.75% | 30.72% | 56.09% | 65.91% | 99.06% |
| 32 | first | 77–102 | 63.16% | 28.84% | 80.62% | 65.03% | 99.81% |
| 32 | first | 103–127 | 68.84% | 28.28% | 83.66% | 61.72% | 95.06% |
| 32 | second | 0–25 | 42.16% | 21.31% | 47.53% | 44.00% | 64.28% |
| 32 | second | 26–51 | 38.78% | 25.12% | 47.41% | 55.97% | 84.94% |
| 32 | second | 52–76 | 42.78% | 29.22% | 57.72% | 64.72% | 98.88% |
| 32 | second | 77–102 | 62.34% | 30.06% | 79.50% | 64.44% | 99.72% |
| 32 | second | 103–127 | 67.97% | 28.56% | 84.38% | 62.53% | 95.44% |
| 64 | first | 0–25 | 50.06% | 37.00% | 61.81% | 68.03% | 98.59% |
| 64 | first | 26–51 | 74.25% | 35.47% | 84.31% | 66.41% | 99.38% |
| 64 | first | 52–76 | 93.25% | 35.84% | 96.50% | 70.75% | 99.81% |
| 64 | first | 77–102 | 93.78% | 35.44% | 97.69% | 69.47% | 99.69% |
| 64 | first | 103–127 | 86.31% | 33.38% | 94.22% | 67.75% | 98.50% |
| 64 | second | 0–25 | 52.16% | 37.47% | 61.66% | 69.97% | 98.75% |
| 64 | second | 26–51 | 74.91% | 33.47% | 83.03% | 68.19% | 99.50% |
| 64 | second | 52–76 | 93.69% | 35.25% | 96.88% | 69.69% | 99.84% |
| 64 | second | 77–102 | 93.75% | 33.66% | 97.53% | 68.75% | 99.53% |
| 64 | second | 103–127 | 87.12% | 34.28% | 94.94% | 68.53% | 98.78% |

The interaction subtracts the full-context scaling effect from the masked scaling effect. Conditional composition comparisons use the correctly retrieved subset for each condition. This is a fixed-factor inference test. Uniform scaling changes both local and long-range relative RoPE distances. The masking arms retain facts using known locations.

Run `python test_rope_compression.py` with the prior experiment results and checkpoints in place.