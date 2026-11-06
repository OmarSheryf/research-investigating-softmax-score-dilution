# (research) Investigating softmax score dilution

A reproducible study of long-context retrieval and composition in synthetic attention models and a small causal RoPE transformer. The experiments separate attention concentration, evidence ranking, positional effects, training stability, and answer accuracy. All training and evaluation inputs are generated synthetic records.

## Results in order

| Study | Finding |
|---|---|
| [Controlled attention robustness](outputs/robustness-report.md) | Scaling and positional correction helped under some distractor regimes. Near duplicates exposed limitations. |
| [Causal transformer](outputs/causal-run/report.md) | Accuracy fell outside the training range without an imposed positional penalty. Uniform logit scaling did not recover it. |
| [Training stabilization](outputs/stability-run/report.md), [paired follow-ups](outputs/training-approaches/report.md), [factorial](outputs/training-factorial/report.md) | The combined short-to-long curriculum and retrieval-first schedule passed on all five fresh seeds in the factorial. |
| [Longer training contexts](outputs/context-exposure/report.md), [scale-up](outputs/context-scale-up/report.md) | At matched token budgets, training through longer contexts improved performance at the next unseen length in both comparisons. |
| [Position and composition](outputs/composition-positions/report.md) | For the 64-record-trained models, 88.4% of composition errors coincided with at least one wrong fact readout. |
| [Masking and compaction](outputs/distractor-geometry/report.md), [uniform RoPE scaling](outputs/rope-compression/report.md) | Removing distractors helped. Compacting the same retained records helped more. Uniform position halving hurt. |
| [Preserving local RoPE distances](outputs/local-rope-compression/report.md), [fresh length transfer](outputs/rope-length-transfer/report.md) | Preserving the first 16 tokens of separation and halving the excess raised composition from 62.8% to 94.5% at 128 records on fresh examples for the 64-record-trained models. |

The results describe these synthetic tasks and checkpoints. The experiments report paired seed variation, training and test budgets, and where interventions change multiple aspects of the computation.

## Reproduce and explore

The recorded training environment used Python 3.11.9 and PyTorch 2.13.0+cpu. Run scripts from the repository root. The later evaluations read the preceding experiment results and saved checkpoints under `outputs/`. A fresh training run writes to the corresponding output directory, so use a separate checkout to preserve the published results.

```sh
python outputs/test_rope_length_transfer.py
```

See each linked report for its protocol, commands, and interpretation. Raw evaluations, generator seeds, model and optimizer states, and the earlier portable archives are included in `outputs/`. The [standalone checkpoint explorer](outputs/causal-run/explorer.html) can be downloaded and opened in a browser. The `visualizations/` directory contains the original in-chat visualization fragments, which use the visualization host’s styles. The `work/` directory contains early training code and verification utilities.
