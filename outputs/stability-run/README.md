# Training stabilization study

Start with `report.md`. The selected curriculum passed the acquisition criterion on 3/3 development seeds but only 1/5 fresh confirmation seeds; baseline passed on 0/5 fresh seeds. This study did not establish reliable training stabilization.

Every run received 1,200 updates and exactly 832,000 input tokens. A pass requires at least 95% accuracy for single-fact retrieval and learned composition at both 8- and 16-record lengths, averaging five positions and two test populations.

Files:

- `protocol.json`: prespecified selection rule, datasets, budgets and source hashes.
- `selection-frozen.json`: recipe selection made before confirmation training.
- `results.json`: full development and confirmation results, including per-example correctness.
- `summary.json`: paired comparisons, success counts and uncertainty.
- `checkpoint-validation.json`: exact replay validation.
- `development/` and `confirmation/`: per-run results and 51 model checkpoints, including optimizer and random-generator states.
- `manifest.json`: SHA-256 checksums of packaged files.

Scripts are one directory above this README. From that directory:

```powershell
python stabilize_causal_training.py
python analyze_stability.py
python verify_stability_checkpoints.py
```

Completed per-run results are reused. Preserve this run and use a separate copy for any changed protocol. The bundle includes the three original baseline checkpoint dependencies under `causal-run/checkpoints/`.

Resume an intermediate checkpoint using its original recipe:

```powershell
python resume_stability.py stability-run/confirmation/curriculum-seed-61/step-800.pt --until 1200 --output resumed.pt
```

The resume command preserves the original 1,200-update schedule; it does not generate evaluation reports. The recorded environment is Python 3.11.9 and PyTorch 2.13.0+cpu with two CPU threads.

Next experiments should complete the constant-learning-rate/curriculum comparison with controlled example replay, then test retrieval-only pretraining against joint training. These are proposed tests, not established explanations for the failures.
