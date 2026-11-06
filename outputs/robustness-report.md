# Controlled attention robustness experiment

## Protocol

Five training seeds; two training regimes (imposed middle penalty 2 or 0); 600 optimizer updates; training length 16; 32 value classes. Shared-weight reader answers two independently supervised queries. The two-fact answer is their fixed sum modulo 32, not learned multi-hop reasoning. Training uses random distractors; confusable distractors are an evaluation distribution shift. Query is a noisy copy of its target key, not semantic paraphrase.

Four evaluations use identical checkpoint weights and held-out examples: penalty 2 / scale 1 (baseline), penalty 2 / length scaling, penalty 0 / scale 1 (position), penalty 0 / length scaling (combined). Scaling is max(1, log(n)/log(16)). Thus baseline always includes the imposed penalty, even for the model trained without it.

128 examples per length × position × difficulty cell, shared across all seeds and interventions. Three positions; four lengths; three difficulty levels. Similar and near-duplicate conditions insert four distractors per target at uniformly sampled non-target positions, with Gaussian noise SD 0.6 and 0.15 before normalization. Targets receive query noise SD 0.05.

Intervals are descriptive 95% Student-t intervals across five training-seed means, averaged over three positions. They capture training variability on this fixed test set, not new-test-set uncertainty; comparisons are exploratory and not corrected for multiplicity.

## Final accuracy: model trained with penalty

| Records | Distractors | Task | Baseline | Scaling | Position | Combined |
|---:|---|---|---:|---:|---:|---:|
| 16 | random | one | 100.0% | 100.0% | 100.0% | 100.0% |
| 16 | random | two | 100.0% | 100.0% | 100.0% | 100.0% |
| 16 | similar | one | 71.9% | 71.9% | 86.8% | 86.8% |
| 16 | similar | two | 65.3% | 65.3% | 73.6% | 73.6% |
| 16 | near_duplicate | one | 44.4% | 44.4% | 16.6% | 16.6% |
| 16 | near_duplicate | two | 29.2% | 29.2% | 5.3% | 5.3% |
| 64 | random | one | 99.9% | 99.9% | 100.0% | 100.0% |
| 64 | random | two | 99.8% | 99.8% | 100.0% | 100.0% |
| 64 | similar | one | 76.4% | 77.1% | 86.2% | 95.9% |
| 64 | similar | two | 69.7% | 70.3% | 73.0% | 91.7% |
| 64 | near_duplicate | one | 56.5% | 61.7% | 15.8% | 17.2% |
| 64 | near_duplicate | two | 46.5% | 57.3% | 5.3% | 6.0% |
| 256 | random | one | 100.0% | 100.0% | 100.0% | 100.0% |
| 256 | random | two | 99.7% | 99.8% | 100.0% | 100.0% |
| 256 | similar | one | 75.4% | 77.8% | 83.9% | 98.0% |
| 256 | similar | two | 69.1% | 71.0% | 70.9% | 95.4% |
| 256 | near_duplicate | one | 56.5% | 63.3% | 17.2% | 21.5% |
| 256 | near_duplicate | two | 46.1% | 59.3% | 4.4% | 5.4% |
| 1024 | random | one | 98.4% | 99.5% | 100.0% | 100.0% |
| 1024 | random | two | 97.2% | 99.1% | 100.0% | 100.0% |
| 1024 | similar | one | 75.2% | 78.4% | 82.0% | 98.9% |
| 1024 | similar | two | 69.6% | 72.2% | 70.5% | 97.5% |
| 1024 | near_duplicate | one | 55.6% | 64.9% | 16.1% | 21.1% |
| 1024 | near_duplicate | two | 47.0% | 64.6% | 5.5% | 7.0% |

## Paired effects at 1024 records, two-fact task

| Training regime | Distractors | Intervention | Accuracy difference (percentage points) | 95% interval |
|---|---|---|---:|---|
| penalty | random | scale | +1.88 | [+1.34, +2.41] |
| penalty | random | position | +2.76 | [+1.98, +3.54] |
| penalty | random | combined | +2.76 | [+1.98, +3.54] |
| penalty | similar | scale | +2.60 | [+1.72, +3.49] |
| penalty | similar | position | +0.89 | [-2.95, +4.72] |
| penalty | similar | combined | +27.92 | [+26.68, +29.16] |
| penalty | near_duplicate | scale | +17.60 | [+15.77, +19.44] |
| penalty | near_duplicate | position | -41.56 | [-44.10, -39.02] |
| penalty | near_duplicate | combined | -40.00 | [-42.53, -37.47] |
| no_penalty | random | scale | +9.64 | [+7.92, +11.35] |
| no_penalty | random | position | +10.52 | [+8.44, +12.60] |
| no_penalty | random | combined | +15.05 | [+12.54, +17.56] |
| no_penalty | similar | scale | +2.50 | [+0.75, +4.25] |
| no_penalty | similar | position | -20.99 | [-24.83, -17.15] |
| no_penalty | similar | combined | +23.85 | [+22.15, +25.56] |
| no_penalty | near_duplicate | scale | +17.66 | [+14.65, +20.67] |
| no_penalty | near_duplicate | position | -41.04 | [-45.43, -36.65] |
| no_penalty | near_duplicate | combined | -40.05 | [-44.31, -35.79] |

## Exact two-value accuracy at 1024 records

Model trained with penalty; requires both decoded values to be correct, excluding accidental modular-sum successes.

| Distractors | Baseline | Scaling | Position | Combined |
|---|---:|---:|---:|---:|
| random | 97.2% | 99.1% | 100.0% | 100.0% |
| similar | 68.8% | 71.6% | 70.4% | 97.5% |
| near_duplicate | 46.2% | 63.0% | 2.9% | 4.5% |

## Position breakdown: near duplicates at 1024 records

Final checkpoint, penalty-trained model, two-fact modular answer accuracy.

| Evidence position | Baseline | Scaling | Position | Combined |
|---|---:|---:|---:|---:|
| beginning | 70.8% | 95.6% | 5.9% | 6.4% |
| middle | 2.5% | 5.0% | 5.5% | 6.9% |
| end | 67.8% | 93.3% | 5.0% | 7.8% |

The imposed penalty becomes a useful location prior for endpoint targets when confusable distractors are scattered throughout the context. Removing it can hurt endpoint accuracy dramatically. The three-position average must not be read as location-independent retrieval ability. Scaling preserves evidence rankings and can amplify this useful or harmful prior. These are properties of this synthetic setup, not evidence that positional bias generally helps.

## Sampled reliability boundary

Lengths whose lower descriptive 95% seed interval reaches 90% answer accuracy. These are sampled passing lengths, not interpolated limits or a guarantee of monotonic performance.

| Training regime | Distractors | Task | Baseline | Scaling | Position | Combined |
|---|---|---|---|---|---|---|
| penalty | random | one | 16, 64, 256, 1024 | 16, 64, 256, 1024 | 16, 64, 256, 1024 | 16, 64, 256, 1024 |
| penalty | random | two | 16, 64, 256, 1024 | 16, 64, 256, 1024 | 16, 64, 256, 1024 | 16, 64, 256, 1024 |
| penalty | similar | one | None | None | None | 64, 256, 1024 |
| penalty | similar | two | None | None | None | 64, 256, 1024 |
| penalty | near_duplicate | one | None | None | None | None |
| penalty | near_duplicate | two | None | None | None | None |
| no_penalty | random | one | 16, 64, 256 | 16, 64, 256, 1024 | 16, 64, 256, 1024 | 16, 64, 256, 1024 |
| no_penalty | random | two | 16, 64, 256 | 16, 64, 256, 1024 | 16, 64, 256, 1024 | 16, 64, 256, 1024 |
| no_penalty | similar | one | None | None | None | 256, 1024 |
| no_penalty | similar | two | None | None | None | None |
| no_penalty | near_duplicate | one | None | None | None | None |
| no_penalty | near_duplicate | two | None | None | None | None |

## Validation and limits

Validated all 5,760 evaluation records: complete factorial coverage, finite bounded metrics, answer counts consistent with accuracy, training loss reduction in all ten runs, unchanged evidence rankings under positive scaling. Runtime: 60.1 seconds.

This is a single attention readout with feed-forward decoder, not a causal language model. The position penalty is imposed rather than learned. Two-read queries are supplied explicitly. Modular sums can be correct through cancelling read errors; evidence retrieval is therefore reported separately in raw results. Three tested positions can miss other positional failures. No production-model claims follow.

## Next experiment

Use a small causal transformer with genuine positional encodings and no imposed middle penalty. Preserve paired evaluation, add independently resampled held-out sets, multiple-fact exact-match metrics, and validation-selected attention scales. Test whether the gains survive semantic matching and learned composition before claiming a general fix.