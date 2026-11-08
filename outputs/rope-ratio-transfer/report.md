# Training-span ratio transfer across lengths

The evaluation reused 39 completed Windows runs and finished the remaining 21 on a Linux runner with CPU PyTorch 2.13.0. All 39 cached runs matched the validation branch files after artifact download. [GitHub Actions run](https://github.com/OmarSheryf/research-investigating-softmax-score-dilution/actions/runs/42534235638)

The frozen rule preserves the first 16 tokens of query–key separation. It scales the excess by min(1, (2 × training cap + 2 − 16)/(2 × input records + 2 − 16)). The rule equals ordinary RoPE at or below the training cap. Ordinary, fixed-half, and ratio modes share every test example.

Five saved checkpoint seeds per training cap are evaluated at 32, 48, 64, 80, 96, and 128 records. The 48- and 80-record lengths extend the preceding tests. Every length has a balanced 5 × 5 position grid with two independently seeded batches of 16 examples per cell. This sample count was chosen to bound local CPU and memory use. Each cap/length/mode has 4,000 model-example observations.

Primary checks are early first-fact retrieval and composition for cap-32 checkpoints at 48 and 80 records. Other lengths and cap-64 checkpoints test consistency and preservation of prior gains.

| Cap | Records | Ratio scale | Metric | Ordinary | Fixed half | Ratio |
|---:|---:|---:|---|---:|---:|---:|
| 32 | 32 | 1.0000 | composition | 99.78% | 99.83% | 99.78% |
| 32 | 32 | 1.0000 | early_first | 100.00% | 99.88% | 100.00% |
| 32 | 32 | 1.0000 | early_second | 99.50% | 99.88% | 99.50% |
| 32 | 32 | 1.0000 | both | 99.80% | 99.85% | 99.80% |
| 32 | 48 | 0.6098 | composition | 94.83% | 99.62% | 99.45% |
| 32 | 48 | 0.6098 | early_first | 88.88% | 100.00% | 100.00% |
| 32 | 48 | 0.6098 | early_second | 87.88% | 99.75% | 99.50% |
| 32 | 48 | 0.6098 | both | 95.03% | 99.67% | 99.52% |
| 32 | 64 | 0.4386 | composition | 77.72% | 99.15% | 99.50% |
| 32 | 64 | 0.4386 | early_first | 64.75% | 99.75% | 99.88% |
| 32 | 64 | 0.4386 | early_second | 60.88% | 99.88% | 100.00% |
| 32 | 64 | 0.4386 | both | 77.95% | 99.17% | 99.60% |
| 32 | 80 | 0.3425 | composition | 57.65% | 95.43% | 98.78% |
| 32 | 80 | 0.3425 | early_first | 42.25% | 94.00% | 99.38% |
| 32 | 80 | 0.3425 | early_second | 42.25% | 94.00% | 99.75% |
| 32 | 80 | 0.3425 | both | 57.23% | 95.88% | 98.72% |
| 32 | 96 | 0.2809 | composition | 42.95% | 86.95% | 99.00% |
| 32 | 96 | 0.2809 | early_first | 33.00% | 75.50% | 99.75% |
| 32 | 96 | 0.2809 | early_second | 40.25% | 76.38% | 99.38% |
| 32 | 96 | 0.2809 | both | 41.77% | 87.25% | 98.83% |
| 32 | 128 | 0.2066 | composition | 29.25% | 61.68% | 97.92% |
| 32 | 128 | 0.2066 | early_first | 40.38% | 50.88% | 99.62% |
| 32 | 128 | 0.2066 | early_second | 39.50% | 51.12% | 99.38% |
| 32 | 128 | 0.2066 | both | 26.02% | 62.10% | 97.85% |
| 64 | 32 | 1.0000 | composition | 99.12% | 98.98% | 99.12% |
| 64 | 32 | 1.0000 | early_first | 99.88% | 99.50% | 99.88% |
| 64 | 32 | 1.0000 | early_second | 99.75% | 99.50% | 99.75% |
| 64 | 32 | 1.0000 | both | 99.33% | 99.15% | 99.33% |
| 64 | 48 | 1.0000 | composition | 98.75% | 98.90% | 98.75% |
| 64 | 48 | 1.0000 | early_first | 99.88% | 100.00% | 99.88% |
| 64 | 48 | 1.0000 | early_second | 98.88% | 99.75% | 98.88% |
| 64 | 48 | 1.0000 | both | 99.15% | 99.15% | 99.15% |
| 64 | 64 | 1.0000 | composition | 97.97% | 98.32% | 97.97% |
| 64 | 64 | 1.0000 | early_first | 98.25% | 99.62% | 98.25% |
| 64 | 64 | 1.0000 | early_second | 98.88% | 99.88% | 98.88% |
| 64 | 64 | 1.0000 | both | 98.25% | 98.35% | 98.25% |
| 64 | 80 | 0.7808 | composition | 93.58% | 97.88% | 96.88% |
| 64 | 80 | 0.7808 | early_first | 93.00% | 99.88% | 98.62% |
| 64 | 80 | 0.7808 | early_second | 89.88% | 99.62% | 97.88% |
| 64 | 80 | 0.7808 | both | 94.77% | 98.32% | 97.78% |
| 64 | 96 | 0.6404 | composition | 86.33% | 97.60% | 96.70% |
| 64 | 96 | 0.6404 | early_first | 79.25% | 99.00% | 98.12% |
| 64 | 96 | 0.6404 | early_second | 82.75% | 99.12% | 97.88% |
| 64 | 96 | 0.6404 | both | 87.35% | 98.08% | 97.08% |
| 64 | 128 | 0.4711 | composition | 60.88% | 93.58% | 93.65% |
| 64 | 128 | 0.4711 | early_first | 51.00% | 96.25% | 96.62% |
| 64 | 128 | 0.4711 | early_second | 50.25% | 96.00% | 96.38% |
| 64 | 128 | 0.4711 | both | 61.75% | 94.65% | 95.03% |

## Paired changes

| Cap | Records | Metric | Baseline | Ratio change, pp | Descriptive 95% seed interval, pp |
|---:|---:|---|---|---:|---:|
| 32 | 32 | composition | ordinary | +0.00 | [+0.00, +0.00] |
| 32 | 32 | composition | fixed | -0.05 | [-0.33, +0.23] |
| 32 | 32 | early_first | ordinary | +0.00 | [+0.00, +0.00] |
| 32 | 32 | early_first | fixed | +0.12 | [-0.22, +0.47] |
| 32 | 48 | composition | ordinary | +4.62 | [+3.91, +5.34] |
| 32 | 48 | composition | fixed | -0.18 | [-0.38, +0.03] |
| 32 | 48 | early_first | ordinary | +11.12 | [+8.05, +14.20] |
| 32 | 48 | early_first | fixed | +0.00 | [+0.00, +0.00] |
| 32 | 64 | composition | ordinary | +21.78 | [+21.03, +22.52] |
| 32 | 64 | composition | fixed | +0.35 | [-0.00, +0.70] |
| 32 | 64 | early_first | ordinary | +35.12 | [+31.04, +39.21] |
| 32 | 64 | early_first | fixed | +0.12 | [-0.22, +0.47] |
| 32 | 80 | composition | ordinary | +41.12 | [+35.16, +47.09] |
| 32 | 80 | composition | fixed | +3.35 | [+2.46, +4.24] |
| 32 | 80 | early_first | ordinary | +57.12 | [+51.59, +62.66] |
| 32 | 80 | early_first | fixed | +5.38 | [+1.79, +8.96] |
| 32 | 96 | composition | ordinary | +56.05 | [+50.12, +61.98] |
| 32 | 96 | composition | fixed | +12.05 | [+10.25, +13.85] |
| 32 | 96 | early_first | ordinary | +66.75 | [+55.30, +78.20] |
| 32 | 96 | early_first | fixed | +24.25 | [+18.78, +29.72] |
| 32 | 128 | composition | ordinary | +68.68 | [+65.66, +71.69] |
| 32 | 128 | composition | fixed | +36.25 | [+33.96, +38.54] |
| 32 | 128 | early_first | ordinary | +59.25 | [+50.90, +67.60] |
| 32 | 128 | early_first | fixed | +48.75 | [+41.66, +55.84] |
| 64 | 32 | composition | ordinary | +0.00 | [+0.00, +0.00] |
| 64 | 32 | composition | fixed | +0.15 | [-0.02, +0.32] |
| 64 | 32 | early_first | ordinary | +0.00 | [+0.00, +0.00] |
| 64 | 32 | early_first | fixed | +0.37 | [-0.05, +0.80] |
| 64 | 48 | composition | ordinary | +0.00 | [+0.00, +0.00] |
| 64 | 48 | composition | fixed | -0.15 | [-0.43, +0.13] |
| 64 | 48 | early_first | ordinary | +0.00 | [+0.00, +0.00] |
| 64 | 48 | early_first | fixed | -0.12 | [-0.47, +0.22] |
| 64 | 64 | composition | ordinary | +0.00 | [+0.00, +0.00] |
| 64 | 64 | composition | fixed | -0.35 | [-0.92, +0.22] |
| 64 | 64 | early_first | ordinary | +0.00 | [+0.00, +0.00] |
| 64 | 64 | early_first | fixed | -1.37 | [-2.39, -0.36] |
| 64 | 80 | composition | ordinary | +3.30 | [+1.81, +4.79] |
| 64 | 80 | composition | fixed | -1.00 | [-2.24, +0.24] |
| 64 | 80 | early_first | ordinary | +5.62 | [+2.72, +8.53] |
| 64 | 80 | early_first | fixed | -1.25 | [-2.20, -0.30] |
| 64 | 96 | composition | ordinary | +10.37 | [+7.90, +12.85] |
| 64 | 96 | composition | fixed | -0.90 | [-2.12, +0.32] |
| 64 | 96 | early_first | ordinary | +18.87 | [+16.46, +21.29] |
| 64 | 96 | early_first | fixed | -0.88 | [-2.48, +0.73] |
| 64 | 128 | composition | ordinary | +32.78 | [+30.27, +35.28] |
| 64 | 128 | composition | fixed | +0.08 | [-0.77, +0.92] |
| 64 | 128 | early_first | ordinary | +45.62 | [+34.08, +57.17] |
| 64 | 128 | early_first | fixed | +0.38 | [-0.67, +1.42] |

Early retrieval averages the other query’s five regions and restricts the queried fact to the first fifth of the context. Both measures exact correctness of both auxiliary fact readouts. Composition is the trained neural composition readout. The paired intervals describe variability across five existing checkpoint seeds on fresh shared examples. These checkpoints have participated in earlier experiments. The formula was frozen before these data were generated.

Within the training cap, ratio results reuse the ordinary predictions because scale one is the same policy. No hyperparameters were selected on this test. Raw results include correctness vectors for all conditions. Run `python outputs/test_rope_ratio_transfer.py` from the repository root. Completed per-checkpoint/length files can be reused on restart.
