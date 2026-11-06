# Training stabilization experiment

## Design

The architecture, loss, positional encoding and attention mechanism are unchanged. The goal is reliable acquisition of both synthetic tasks, before interpreting context-length failures. All runs receive 1,200 updates, 38,400 examples, and exactly 832,000 input tokens. No imposed positional penalty or attention scaling is used.

Three development seeds (11, 33, 44) include previously observed successes/failures. Three recipes are screened:

- Baseline: constant learning rate 0.002; lengths 4/8/16 alternate each update.
- Warmup/cosine: same length schedule, 100-update linear warmup to 0.002, then cosine decay to 0.0002.
- Curriculum: the warmup/cosine schedule, with 400 updates at length 4, then 400 at length 8, then 400 at length 16. Length counts and total tokens match the baseline.

Selection maximizes development success count, then the worst seed/task/length score, then mean accuracy. The chosen recipe is frozen before five fresh training seeds (61, 72, 83, 94, 105) are trained and evaluated. Confirmation compares baseline and selected recipe on the same seeds and new test populations.

A seed passes only if BOTH single-fact and learned composition accuracy reach 95% at BOTH 8 and 16 records. Each score averages five evidence positions, two populations, and 128 examples per cell. This criterion does not require every individual position to pass 95%.

## Development selection

| Recipe | Seeds passing / 3 | Worst score | Mean score |
|---|---:|---:|---:|
| baseline | 1 | 12.50% | 48.80% |
| warmup_cosine | 0 | 13.67% | 26.31% |
| curriculum | 3 | 99.92% | 99.99% |

Selected recipe: **curriculum**.

## Fresh-seed confirmation

| Recipe | Seeds passing / 5 | 95% Wilson interval for pass rate | Mean in-domain accuracy | Worst seed/task/length score |
|---|---:|---|---:|---:|
| baseline | 0 | [0.0%, 43.4%] | 41.53% | 13.52% |
| curriculum | 1 | [3.6%, 62.4%] | 53.32% | 15.94% |

Paired change in mean in-domain accuracy: +11.79 percentage points; descriptive 95% Student-t interval across five seeds [-63.65, +87.23].
Paired pass/fail changes: 1 improvements, 0 regressions. Exact two-sided McNemar p-value: 1.0000. Five seeds provide limited evidence about the population failure rate.

| Seed | Baseline pass | Selected pass | Baseline minimum | Selected minimum |
|---:|---|---|---:|---:|
| 61 | False | False | 13.52% | 15.94% |
| 72 | False | False | 13.67% | 82.11% |
| 83 | False | False | 20.94% | 16.48% |
| 94 | False | False | 93.91% | 17.58% |
| 105 | False | True | 14.45% | 95.70% |

The selected recipe improved the observed acquisition count, but failures persist. Training stabilization remains incomplete.

## Secondary length-generalization results

Only 8/16-record results determine acquisition success. Longer lengths are secondary held-out diagnostics.

| Recipe | Records | Single fact | Learned composition |
|---|---:|---:|---:|
| baseline | 8 | 52.38% | 37.45% |
| baseline | 16 | 45.00% | 31.30% |
| baseline | 32 | 33.17% | 24.05% |
| baseline | 64 | 24.22% | 15.06% |
| curriculum | 8 | 61.41% | 50.25% |
| curriculum | 16 | 56.08% | 45.56% |
| curriculum | 32 | 38.48% | 24.59% |
| curriculum | 64 | 24.38% | 14.23% |

## Interpretation and limits

This tests a selected training recipe, not a production-model fix. The curriculum recipe changes both learning-rate scheduling and the temporal ordering of lengths relative to baseline; the development screen helps distinguish these, but fresh-seed confirmation does not independently identify each component.

Development seeds were intentionally chosen from prior observations. Confirmation seeds and test populations were unused for selection. Test examples are shared across training seeds to support pairing. Intervals over seeds do not represent all test-population uncertainty, and secondary comparisons are exploratory.

Each training seed controls both parameter initialization and the training-data generator. Recipe pairs share initialization, but the curriculum changes generator consumption order, so the exact training example sets are not identical. Context-length counts, example counts, and token budgets are identical. A future curriculum-isolation test should replay the same pregenerated per-length examples in different orders.

## Next decision

If the recipe improves acquisition, confirm on more fresh seeds before treating it as reliable, then compare short- and longer-context training under matched token budgets. If failures persist, prioritize optimization/task acquisition rather than assigning failures to attention dilution.

Concrete follow-ups: complete the missing constant-learning-rate plus length-curriculum condition using controlled example replay; then compare retrieval-only pretraining against joint retrieval/composition training, tracking tokens-to-acquisition on validation data. These are hypotheses to test, not established causes. Defer strong long-context mechanism claims until the short-context control learns reliably.

## Files and reproduction

`protocol.json` records the prespecified rule and source hashes. `selection-frozen.json` records development-only selection. `results.json` contains all evaluation correctness vectors. Per-run directories contain checkpoints with model, optimizer, generator and torch RNG states.

Run `python stabilize_causal_training.py`, then `python analyze_stability.py`. Completed per-run results are reused; use a separate copy/run directory for a new protocol. Development baseline reuse requires the original seed-11/33/44 update-1200 checkpoints under `causal-run/checkpoints/`.

To resume an intermediate recipe checkpoint, run `python resume_stability.py CHECKPOINT --until 1200 --output resumed.pt`. The original length and learning-rate schedules are preserved. This does not rebuild evaluation summaries.

Validated 580 evaluation cells and 51 checkpoints; all training budgets match exactly.