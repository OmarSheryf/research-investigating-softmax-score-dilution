# Causal transformer experiment

## Saved run

126,608 parameters; 3 causal layers; 4 heads per layer; width 64; RoPE base 10000. No imposed position penalty. Five seeds (11, 22, 33, 44, 55), each trained for 1,200 updates with batch size 32. Training contexts cycle through 4, 8, and 16 key/value records.

Each record uses two tokens. A separator and two query tokens follow the records, giving 2N+3 input tokens. A read classifier predicts each queried value; a neural MLP predicts the sum modulo 8 from the two contextual query states. Labels supervise composition, but retrieved values or a fixed addition rule do not enter its input. This is a synthetic supervised transformer, not a pretrained autoregressive language model.

Loss is read cross-entropy plus composition cross-entropy. AdamW learning rate 0.002; gradient clipping 1.0. Keys are unique within each context; values have eight possible classes. Chance answer accuracy is 12.5%.

## Results across all five seeds

| Records | Input tokens | Single fact | Two-fact composition | Scaled single fact | Scaled composition |
|---:|---:|---:|---:|---:|---:|
| 8 | 19 | 73.94% | 67.42% | 73.94% | 67.42% |
| 16 | 35 | 69.88% | 63.81% | 69.88% | 63.81% |
| 32 | 67 | 47.55% | 32.33% | 47.17% | 32.28% |
| 64 | 131 | 23.88% | 14.36% | 23.11% | 14.33% |

## Training stability and length generalization

Seeds 11, 22, and 55 learned both tasks well at training lengths. Seeds 33 and 44 did not learn reliably within the fixed budget. Aggregate scores mix optimization failures and length generalization failures.

The following subset is selected after observing task acquisition and is descriptive, not a prespecified confirmatory result.

| Records | Single fact, three learned seeds | Composition, three learned seeds |
|---:|---:|---:|
| 8 | 99.43% | 98.70% |
| 16 | 98.05% | 96.38% |
| 32 | 64.66% | 45.29% |
| 64 | 28.46% | 14.58% |

## Paired scaling effects

Scaling multiplies causal attention logits by 1 + max(0, log(candidate_count)/log(35) - 1). It operates in every head and layer, with each row using only its available candidate count. It is inactive within training lengths. Each intervention uses identical checkpoint weights and identical examples. Unlike rescaling a fixed score vector, scaling in multiple layers can change later representations and rankings.

| Records | Task | Mean change, percentage points | Descriptive 95% seed interval |
|---:|---|---:|---|
| 32 | single | -0.375 | [-0.794, +0.044] |
| 32 | two | -0.047 | [-0.590, +0.496] |
| 64 | single | -0.766 | [-1.421, -0.110] |
| 64 | two | -0.031 | [-1.157, +1.095] |

Independent validation chooses among coefficients 0, 0.5, and 1, scoring the mean of retrieval and composition accuracy at 16/32 records. Four seeds selected 0, and seed 22 selected 0.5. This provides no consistent evidence of a useful scaling benefit.

## Position sensitivity

| Evidence position | Single fact, 16 records | Single fact, 32 records | Single fact, 64 records |
|---|---:|---:|---:|
| Beginning | 69.45% | 37.73% | 21.09% |
| Quarter | 70.00% | 55.23% | 25.39% |
| Middle | 70.00% | 66.95% | 21.64% |
| Three quarters | 70.16% | 46.41% | 27.19% |
| End | 69.77% | 31.41% | 24.06% |

No consistent U-shaped pattern appeared. At 32 records the middle was best on the average single-fact score. Long-context degradation can therefore appear here without the classic lost-in-the-middle curve. This experiment does not establish the internal cause of that degradation.

## Measurement and limitations

Evaluation uses five positions, two independently generated test populations, and 128 examples per cell. Near/far evidence separation is tested at 16/64 records. Intervals are descriptive Student-t intervals over five seed means after averaging test populations and positions. They are not multiplicity-corrected and do not capture all new-test-set uncertainty. Raw normal-approximation intervals may exceed [0,1]; accuracy intervals in the explorer are clipped for display only.

Last-layer, head-averaged attention mass and entropy are diagnostics, not causal explanations. The composition head is a task-specific neural readout of two query states. This small experiment cannot establish mechanisms or remedies in Claude or another production model.

## Conclusions

1. Length generalization failure appeared without an imposed positional penalty, including in seeds that learned the short-context tasks.
2. This length-aware attention scaling rule did not recover performance.
3. A consistent U-shaped positional failure was not required for degradation.
4. Training instability is a separate bottleneck and must be resolved before strong architecture comparisons.

## Prioritized next steps

1. Stabilize task acquisition across seeds using validation-selected optimization settings or a training curriculum; retain unsuccessful seeds in reporting.
2. Compare short-context and longer-context training under matched training-token budgets. Include tests both within and beyond each training range.
3. Separate positional-range extrapolation from distractor competition with controlled position/length interventions and genuine causal ablations.
4. Only then compare positional encodings, selective retrieval/routing, and attention normalizers on unseen test sets.

## Reproduction and checkpoints

Run from the parent outputs directory: `python causal_transformer_experiment.py`. This retrains all seeds and writes into `causal-run/`.
Resume a checkpoint: `python causal_transformer_experiment.py --resume causal-run/checkpoints/seed-11-step-1200.pt --steps 1400`. This continues that seed; it does not rebuild the full evaluation report.
Each checkpoint includes weights, AdamW optimizer state, data generator state, torch RNG state, seed, and step. Checkpoints are saved every 200 updates. Existing files for matching seed/step are replaced when rerunning; copy a run directory before preserving an independent variant.

Validation: 1,060 unique evaluation records; 30 checkpoint files; maximum difference from the original retained aggregate results: 0.00000000. All causal-masking and diagnostic-equivalence preflight checks passed.