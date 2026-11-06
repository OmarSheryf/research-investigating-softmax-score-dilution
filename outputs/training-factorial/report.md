# Batch order and loss timing factorial

Five new paired seeds, 1,200 updates per run, and 832,000 input tokens. All four recipes share initial weights and the same 400 batches per length within a seed. AdamW uses a constant 0.002 learning rate. Both retrieval-first arms activate composition loss at update 601.

Pass requires at least 95% on both tasks at both 8 and 16 records. Validation and held-out populations use separate generator seeds.

| Recipe | Passes | Mean held-out accuracy |
|---|---:|---:|
| joint_interleaved | 0/5 | 51.27% |
| joint_curriculum | 4/5 | 84.45% |
| retrieval_first | 5/5 | 99.25% |
| retrieval_first_curriculum | 5/5 | 99.93% |

| Seed | Interleaved joint | Ordered joint | Interleaved retrieval-first | Ordered retrieval-first |
|---:|---:|---:|---:|---:|
| 171 | 94.45% | 99.06% | 97.97% | 99.92% |
| 182 | 87.42% | 99.77% | 99.84% | 99.92% |
| 193 | 12.81% | 99.77% | 99.30% | 99.84% |
| 204 | 13.67% | 14.30% | 96.88% | 100.00% |
| 215 | 14.84% | 99.30% | 98.05% | 99.61% |

The seed table shows each run’s minimum required held-out score.

| Contrast | Change in mean accuracy, pp | Descriptive 95% seed interval, pp |
|---|---:|---:|
| order_under_joint | +33.18 | [-17.28, +83.64] |
| order_under_retrieval_first | +0.68 | [+0.01, +1.36] |
| retrieval_first_under_interleaving | +47.97 | [-1.52, +97.46] |
| retrieval_first_under_curriculum | +15.48 | [-27.00, +57.95] |
| interaction | -32.50 | [-83.21, +18.21] |

The interaction is the change in the curriculum effect when switching from joint training to retrieval-first. The loss schedule changes both composition timing and the number of composition-supervised updates. At the common switch, the ordered arm has seen 400 length-4 and 200 length-8 batches, while the interleaved arm has seen 200 at each length. Consequently, the interaction includes differences in which lengths receive early composition supervision.

All 40 checkpoints loaded and matched their run’s batch-bank identifier. The raw results include validation traces and per-example test correctness. Run `python test_training_factorial.py` alongside the imported experiment scripts.