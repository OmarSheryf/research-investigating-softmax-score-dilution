# Composition and evidence position at 128 records

Five paired saved checkpoints per training cap. All 25 ordered combinations of five position regions receive equal weight. Positions are independently sampled within their assigned regions, subject to the two records being distinct. Each cell uses two independently seeded sets of 64 contexts, shared across checkpoints. Regions use zero-based record indices.

| Training cap | Fact 1 | Fact 2 | Both facts | Composition | Composition given both | Composition given a missed fact |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 50.62% | 50.81% | 25.91% | 29.29% | 90.04% | 8.06% |
| 64 | 79.53% | 80.33% | 63.88% | 62.75% | 93.22% | 8.88% |

| Training cap | Composition errors | Both readouts correct | At least one readout wrong |
|---:|---:|---:|---:|
| 32 | 11,313 | 413 | 10,900 |
| 64 | 5,960 | 693 | 5,267 |

## Cap 32, Both readouts correct

Rows show the first query region. Columns show the second.

| First query | 0–25 | 26–51 | 52–76 | 77–102 | 103–127 |
|---|---:|---:|---:|---:|---:|
| 0–25 | 16.56% | 16.25% | 16.88% | 26.25% | 26.25% |
| 26–51 | 16.88% | 14.69% | 15.47% | 21.88% | 28.28% |
| 52–76 | 17.81% | 19.06% | 18.91% | 24.06% | 30.00% |
| 77–102 | 27.03% | 25.47% | 27.50% | 40.47% | 43.59% |
| 103–127 | 28.12% | 28.28% | 27.81% | 42.81% | 47.34% |

## Cap 32, Composition accuracy

Rows show the first query region. Columns show the second.

| First query | 0–25 | 26–51 | 52–76 | 77–102 | 103–127 |
|---|---:|---:|---:|---:|---:|
| 0–25 | 22.50% | 22.19% | 21.56% | 30.47% | 29.22% |
| 26–51 | 22.19% | 20.47% | 19.06% | 23.91% | 29.38% |
| 52–76 | 23.44% | 23.91% | 22.97% | 27.50% | 32.97% |
| 77–102 | 29.38% | 28.28% | 29.84% | 43.12% | 46.88% |
| 103–127 | 28.91% | 30.78% | 31.87% | 42.34% | 49.22% |

## Cap 32, Composition given both readouts correct

Rows show the first query region. Columns show the second.

| First query | 0–25 | 26–51 | 52–76 | 77–102 | 103–127 |
|---|---:|---:|---:|---:|---:|
| 0–25 | 88.68% | 86.54% | 75.93% | 91.67% | 91.07% |
| 26–51 | 86.11% | 87.23% | 85.86% | 87.86% | 88.95% |
| 52–76 | 88.60% | 85.25% | 80.99% | 88.96% | 92.19% |
| 77–102 | 86.71% | 87.12% | 88.64% | 93.44% | 98.21% |
| 103–127 | 89.44% | 91.16% | 88.20% | 93.43% | 97.36% |

## Cap 64, Both readouts correct

Rows show the first query region. Columns show the second.

| First query | 0–25 | 26–51 | 52–76 | 77–102 | 103–127 |
|---|---:|---:|---:|---:|---:|
| 0–25 | 27.81% | 41.41% | 47.66% | 46.41% | 42.19% |
| 26–51 | 40.47% | 54.53% | 68.28% | 70.00% | 65.78% |
| 52–76 | 47.81% | 72.81% | 86.09% | 86.25% | 78.12% |
| 77–102 | 48.91% | 70.31% | 88.44% | 87.97% | 80.31% |
| 103–127 | 44.84% | 62.81% | 81.09% | 81.56% | 75.00% |

## Cap 64, Composition accuracy

Rows show the first query region. Columns show the second.

| First query | 0–25 | 26–51 | 52–76 | 77–102 | 103–127 |
|---|---:|---:|---:|---:|---:|
| 0–25 | 29.84% | 38.28% | 46.09% | 46.09% | 40.62% |
| 26–51 | 40.94% | 52.66% | 65.62% | 69.38% | 65.47% |
| 52–76 | 47.66% | 70.16% | 82.97% | 85.94% | 77.97% |
| 77–102 | 48.91% | 67.34% | 86.41% | 86.41% | 80.94% |
| 103–127 | 43.91% | 61.56% | 79.53% | 79.69% | 74.38% |

## Cap 64, Composition given both readouts correct

Rows show the first query region. Columns show the second.

| First query | 0–25 | 26–51 | 52–76 | 77–102 | 103–127 |
|---|---:|---:|---:|---:|---:|
| 0–25 | 83.15% | 80.38% | 88.52% | 91.92% | 87.41% |
| 26–51 | 85.33% | 86.25% | 90.62% | 95.09% | 93.82% |
| 52–76 | 90.20% | 92.27% | 94.56% | 97.64% | 97.20% |
| 77–102 | 90.42% | 92.89% | 97.00% | 97.34% | 98.05% |
| 103–127 | 89.90% | 92.04% | 95.76% | 96.74% | 96.67% |

Conditioning uses correctness of the auxiliary readout heads. The composition head uses the underlying query states directly, so these counts measure association. The balanced position grid differs from the earlier half-context separation test.

Correctness codes use bit 0 for fact 1, bit 1 for fact 2 and bit 2 for composition. Raw results include exact sampled positions and per-example codes. Run `python test_composition_positions.py` with the context-scale-up checkpoints in place.