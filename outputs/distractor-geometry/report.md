# Distractor masking and positional compaction

All ten checkpoints use the same 128-record contexts and crossed position grid as the previous diagnostic. Full shows all records. Masked retains 64 records, including both facts and 62 uniformly sampled distractors, with their original RoPE position IDs. Compact uses the same retained records and contiguous IDs. All conditions preserve the separator and query order.

Masking changes the accessible evidence and contextual representations throughout the network. The contrast measures distractor removal at fixed positions. The compact-minus-masked contrast measures the effect of changing positional geometry with identical visible input.

| Training cap | Condition | Fact 1 | Fact 2 | Both facts | Composition | Composition given both correct |
|---:|---|---:|---:|---:|---:|---:|
| 32 | full | 50.62% | 50.81% | 25.91% | 29.29% | 90.04% |
| 32 | masked | 63.44% | 63.31% | 40.29% | 41.64% | 92.43% |
| 32 | compact | 88.92% | 88.65% | 78.75% | 78.80% | 98.18% |
| 64 | full | 79.53% | 80.33% | 63.88% | 62.75% | 93.22% |
| 64 | masked | 86.91% | 86.81% | 75.43% | 74.45% | 95.80% |
| 64 | compact | 99.19% | 99.28% | 98.48% | 98.00% | 99.25% |

## Position breakdown

Accuracy is averaged over the other fact’s five regions.

| Training cap | Query | Record region | Full | Masked | Compact |
|---:|---|---|---:|---:|---:|
| 32 | first | 0–25 | 40.66% | 49.38% | 64.44% |
| 32 | first | 26–51 | 37.72% | 47.44% | 86.25% |
| 32 | first | 52–76 | 42.75% | 56.09% | 99.06% |
| 32 | first | 77–102 | 63.16% | 80.62% | 99.81% |
| 32 | first | 103–127 | 68.84% | 83.66% | 95.06% |
| 32 | second | 0–25 | 42.16% | 47.53% | 64.28% |
| 32 | second | 26–51 | 38.78% | 47.41% | 84.94% |
| 32 | second | 52–76 | 42.78% | 57.72% | 98.88% |
| 32 | second | 77–102 | 62.34% | 79.50% | 99.72% |
| 32 | second | 103–127 | 67.97% | 84.38% | 95.44% |
| 64 | first | 0–25 | 50.06% | 61.81% | 98.59% |
| 64 | first | 26–51 | 74.25% | 84.31% | 99.38% |
| 64 | first | 52–76 | 93.25% | 96.50% | 99.81% |
| 64 | first | 77–102 | 93.78% | 97.69% | 99.69% |
| 64 | first | 103–127 | 86.31% | 94.22% | 98.50% |
| 64 | second | 0–25 | 52.16% | 61.66% | 98.75% |
| 64 | second | 26–51 | 74.91% | 83.03% | 99.50% |
| 64 | second | 52–76 | 93.69% | 96.88% | 99.84% |
| 64 | second | 77–102 | 93.75% | 97.53% | 99.53% |
| 64 | second | 103–127 | 87.12% | 94.94% | 98.78% |

## Paired changes

| Training cap | Contrast | Metric | Mean change, pp | Descriptive 95% seed interval, pp |
|---:|---|---|---:|---:|
| 32 | masked_minus_full | first | +12.81 | [+11.69, +13.94] |
| 32 | masked_minus_full | second | +12.50 | [+11.08, +13.92] |
| 32 | masked_minus_full | both | +14.39 | [+11.52, +17.25] |
| 32 | masked_minus_full | composition | +12.35 | [+9.74, +14.96] |
| 32 | masked_minus_full | composition_given_both | +2.55 | [+0.39, +4.70] |
| 32 | compact_minus_masked | first | +25.49 | [+19.41, +31.56] |
| 32 | compact_minus_masked | second | +25.34 | [+19.43, +31.26] |
| 32 | compact_minus_masked | both | +38.46 | [+30.92, +45.99] |
| 32 | compact_minus_masked | composition | +37.16 | [+30.61, +43.70] |
| 32 | compact_minus_masked | composition_given_both | +5.69 | [+3.98, +7.41] |
| 64 | masked_minus_full | first | +7.38 | [+6.41, +8.34] |
| 64 | masked_minus_full | second | +6.48 | [+5.48, +7.49] |
| 64 | masked_minus_full | both | +11.56 | [+10.20, +12.92] |
| 64 | masked_minus_full | composition | +11.70 | [+10.14, +13.26] |
| 64 | masked_minus_full | composition_given_both | +2.56 | [+1.60, +3.53] |
| 64 | compact_minus_masked | first | +12.29 | [+10.09, +14.48] |
| 64 | compact_minus_masked | second | +12.48 | [+10.45, +14.50] |
| 64 | compact_minus_masked | both | +23.05 | [+19.24, +26.86] |
| 64 | compact_minus_masked | composition | +23.55 | [+20.17, +26.93] |
| 64 | compact_minus_masked | composition_given_both | +3.44 | [+2.33, +4.55] |

Each conditional composition score is calculated on that condition’s correctly retrieved subset. The subsets can differ. This inference-only experiment retains facts using their known locations and samples one distractor subset per example. It estimates effects for this retention policy and the shared test populations.

For each checkpoint, the sparse masked computation was checked against a dense layerwise key mask and saved full-context predictions were spot-checked by replay. Run `python test_distractor_geometry.py` with the prior results and checkpoints in place.