# Training-span ratio compression

The 16-token local window is fixed. Beyond it, the ratio rule scales distances by min(1, (training span − 16)/(input span − 16)). A context of N records has 2N+3 tokens and maximum causal separation 2N+2. At 128 records the resulting factors are 0.206612 for cap 32 and 0.471074 for cap 64. All records remain visible.

All three modes use the same fresh 5 × 5 position grid, two batches of 32 examples per cell, and five saved checkpoint seeds per cap. The primary outcomes are early retrieval at cap 32 and composition. Early means the first fifth of the context.

| Cap | Mode | Early fact 1 | Early fact 2 | Both facts | Composition |
|---:|---|---:|---:|---:|---:|
| 32 | ordinary | 41.56% | 37.50% | 25.85% | 28.76% |
| 32 | fixed | 49.44% | 48.44% | 60.89% | 61.25% |
| 32 | ratio | 99.69% | 99.19% | 97.95% | 97.94% |
| 64 | ordinary | 53.81% | 50.06% | 63.66% | 62.35% |
| 64 | fixed | 96.44% | 96.19% | 95.26% | 94.58% |
| 64 | ratio | 97.25% | 97.00% | 95.88% | 95.25% |

| Cap | Baseline | Metric | Ratio change, pp | Descriptive 95% seed interval, pp |
|---:|---|---|---:|---:|
| 32 | ordinary | composition | +69.17 | [+65.06, +73.29] |
| 32 | ordinary | early_first | +58.13 | [+47.24, +69.01] |
| 32 | ordinary | early_second | +61.69 | [+50.63, +72.74] |
| 32 | ordinary | both | +72.10 | [+67.75, +76.45] |
| 32 | fixed | composition | +36.69 | [+33.35, +40.02] |
| 32 | fixed | early_first | +50.25 | [+44.55, +55.95] |
| 32 | fixed | early_second | +50.75 | [+45.69, +55.81] |
| 32 | fixed | both | +37.06 | [+33.81, +40.31] |
| 64 | ordinary | composition | +32.90 | [+30.42, +35.38] |
| 64 | ordinary | early_first | +43.44 | [+33.51, +53.37] |
| 64 | ordinary | early_second | +46.94 | [+36.38, +57.50] |
| 64 | ordinary | both | +32.21 | [+29.26, +35.17] |
| 64 | fixed | composition | +0.67 | [+0.03, +1.32] |
| 64 | fixed | early_first | +0.81 | [-0.20, +1.82] |
| 64 | fixed | early_second | +0.81 | [-0.04, +1.66] |
| 64 | fixed | both | +0.61 | [-0.10, +1.32] |

The factor uses the checkpoint’s training cap and the total input length. It was frozen before this test. The mapping aligns the farthest pair with the training span and changes distances throughout the far region. The intervals describe five-seed variability on shared examples. Scale-one equivalence against ordinary RoPE is checked for every checkpoint.

Run `python outputs/test_rope_length_ratio.py` from the repository root. Raw results retain per-example correctness codes.