# Controlled training follow-ups

Five fresh seeds share initialization and exact training batches across recipes. Each recipe uses 1,200 updates, 832,000 tokens, and AdamW at a constant learning rate of 0.002. Interleaving cycles through 4, 8, and 16 records. The ordering curriculum groups those same batches by increasing length. Retrieval-first keeps the interleaved order and omits the composition loss for the first 600 updates.

Validation is measured every 300 updates. A pass requires at least 95% on both tasks at both 8 and 16 records. Final test populations are separate from validation. First measured validation pass is interval-censored by checkpoint spacing, and absent passes are right-censored at the training budget.

| Recipe | Held-out passes | Mean accuracy | Worst seed/task/length |
|---|---:|---:|---:|
| joint_interleaved | 0/5 | 21.86% | 12.73% |
| joint_curriculum | 3/5 | 68.00% | 12.50% |
| retrieval_first | 3/5 | 82.47% | 14.77% |

| Seed | Joint interleaved minimum | Ordered joint minimum | Retrieval-first minimum |
|---:|---:|---:|---:|
| 116 | 13.28% | 96.09% | 14.77% |
| 127 | 12.73% | 99.61% | 99.30% |
| 138 | 16.02% | 15.39% | 96.64% |
| 149 | 12.73% | 99.45% | 87.81% |
| 160 | 14.37% | 12.50% | 97.66% |

Paired changes in mean accuracy and descriptive 95% intervals across five seeds.
- joint_curriculum +46.14 pp [-7.26, +99.55].
- retrieval_first +60.62 pp [+19.17, +102.06].

| Seed | Joint interleaved | Ordered joint | Retrieval-first |
|---:|---:|---:|---:|
| 116 | No measured pass | 832,000 | No measured pass |
| 127 | No measured pass | 496,000 | 624,000 |
| 138 | No measured pass | No measured pass | 832,000 |
| 149 | No measured pass | 496,000 | No measured pass |
| 160 | No measured pass | No measured pass | 624,000 |

The table shows tokens consumed at the first measured validation pass.

| Seed | Retrieval-first single fact at 8 records | At 16 records |
|---:|---:|---:|
| 116 | 27.58% | 23.36% |
| 127 | 99.06% | 96.41% |
| 138 | 27.97% | 23.44% |
| 149 | 27.34% | 22.27% |
| 160 | 88.20% | 70.00% |

Retrieval scores above are measured at the switch to joint training.

Retrieval-first changes the number and timing of composition-supervised updates together. The comparison tests the complete schedule. Validation traces and per-example held-out results are in `results.json`. Checkpoints at 600 and 1200 updates include optimizer state, batch-bank identity, and consumed batch counts. Run `python test_training_approaches.py` alongside the two imported experiment scripts to reproduce the study.