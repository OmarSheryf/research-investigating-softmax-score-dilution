# Training context exposure

Five fresh seeds use the selected short-to-long, retrieval-first recipe. Composition loss starts at update 601. Each seed branches from one shared checkpoint at update 999. The control continues for 201 batches at 16 records. The extended arm spends that same token budget on 105 batches at 32 records. Both train with batch size 32, AdamW at 0.002, and exactly 832,000 input tokens.

The extended recipe uses 1,104 updates and the control uses 1,200. This compares allocation of a fixed token budget, with context length and update count changing together after the fork. Test inputs are paired across arms. Each length uses five positions and two independently seeded populations, with 128 examples per cell.

At 16 records, both tasks reach 95% in 5/5 control runs and 5/5 extended runs.

| Records | Metric | Cap 16 | Cap 32 | Paired change, pp | Descriptive 95% seed interval, pp |
|---:|---|---:|---:|---:|---:|
| 16 | single | 99.83% | 99.81% | -0.02 | [-0.20, +0.17] |
| 16 | two | 99.69% | 99.69% | -0.00 | [-0.27, +0.27] |
| 16 | both_reads | 99.69% | 99.66% | -0.03 | [-0.34, +0.28] |
| 32 | single | 85.20% | 99.61% | +14.41 | [+11.60, +17.21] |
| 32 | two | 77.36% | 99.17% | +21.81 | [+16.94, +26.68] |
| 32 | both_reads | 77.95% | 99.31% | +21.36 | [+16.95, +25.77] |
| 64 | single | 48.41% | 81.59% | +33.19 | [+27.42, +38.95] |
| 64 | two | 28.41% | 65.73% | +37.33 | [+33.65, +41.00] |
| 64 | both_reads | 24.53% | 66.08% | +41.55 | [+36.81, +46.28] |

Single is the first fact readout. Two is learned composition accuracy. Both reads is exact accuracy on both underlying fact readouts. Only the extended arm trains at 32 records. Both arms extrapolate at 64. Intervals summarize paired differences across five training seeds and the shared fixed test populations.

Run `python test_context_exposure.py` alongside the two imported experiment scripts. Shared and final checkpoints save weights, optimizer and generator states.