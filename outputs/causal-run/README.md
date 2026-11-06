# Causal transformer experiment

Open `explorer.html` to inspect measured checkpoints, individual seeds, positions, and attention interventions. It replays saved measurements; it does not train a model in the browser. Model states are saved every 200 updates; evaluation measurements are available at updates 400 and 1200.

Read `report.md` for conclusions, limitations, and next experiments. `results.json` holds aggregate results and protocol metadata. `raw-evaluations.json` holds 1,060 evaluation records including per-example correctness. `validation.json` compares the saved rerun to the original in-memory results. `checkpoint-validation.json` verifies exact continuation from update 1000 to update 1200.

The `checkpoints/` directory contains 30 PyTorch checkpoint files: five seeds times six updates. Each contains model weights, optimizer state, training-data RNG state, torch RNG state, and the seed/step. `manifest.json` records file sizes and SHA-256 checksums.

The experiment source and verification scripts sit one directory above this README. From that directory:

```powershell
python causal_transformer_experiment.py
python build_causal_report.py
python verify_causal_checkpoints.py
```

To continue a specific seed:

```powershell
python causal_transformer_experiment.py --resume causal-run/checkpoints/seed-11-step-1200.pt --steps 1400
```

This writes a new step-1400 checkpoint. It does not rebuild evaluation summaries. Rerunning the full experiment replaces files with matching names, so preserve a copy before an independent experiment.

Recorded environment: Python 3.11.9, PyTorch 2.13.0+cpu, two CPU threads. Exact numerical reproduction across other platforms or library versions is not guaranteed.

The original in-memory run's retained summaries are included as `causal-original-results.json` beside the scripts. The saved rerun reproduced their reported means exactly. No imposed position penalty is used in this transformer.
