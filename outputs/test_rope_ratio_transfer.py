"""Evaluate the frozen training-span ratio rule across fresh context lengths."""
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from causal_transformer_experiment import Model
from stabilize_causal_training import atomic_json
from test_composition_positions import CAPS, CHECKPOINTS, SEEDS, aggregate, pct
from test_context_exposure import paired_interval
from test_distractor_geometry import codes
from test_local_rope_compression import forward
from test_rope_length_ratio import compression
import test_rope_length_transfer as data

OUT = Path(__file__).resolve().parent / 'rope-ratio-transfer'
LENGTHS = [32,48,64,80,96,128]
MODES = ['ordinary','fixed','ratio']
POPULATION = 121000000
EXAMPLES = 16
METRICS = ['composition','early_first','early_second','both']


@torch.inference_mode()
def evaluate(seed, cap, length):
    torch.set_num_threads(1)
    destination = OUT/f'seed-{seed}-cap{cap}-length{length}.json'
    if destination.exists():
        return json.loads(destination.read_text())
    data.POPULATION = POPULATION
    data.EXAMPLES = EXAMPLES
    state = torch.load(CHECKPOINTS/f'seed-{seed}'/f'cap{cap}.pt',weights_only=True,map_location='cpu')
    assert (state['seed'],state['length'],state['tokens']) == (seed,cap,1112864)
    model = Model().eval()
    model.load_state_dict(state['model'])
    scale = compression(cap,length)
    assert 0 < scale <= 1
    cells = []
    for first in range(5):
        for second in range(5):
            for replicate in range(2):
                tokens, labels = data.examples(length,first,second,replicate)
                predictions = {mode:[] for mode in MODES}
                for chunk, truth in zip(tokens.split(8),labels.split(8)):
                    reads, answer, _ = model(chunk)
                    ordinary = codes(reads,answer,truth)
                    predictions['ordinary'].extend(ordinary)
                    reads, answer = forward(model,chunk,scale=.5)
                    predictions['fixed'].extend(codes(reads,answer,truth))
                    if scale == 1:
                        predictions['ratio'].extend(ordinary)
                    else:
                        reads, answer = forward(model,chunk,scale=scale)
                        predictions['ratio'].extend(codes(reads,answer,truth))
                for mode, values in predictions.items():
                    assert len(values) == EXAMPLES
                    cells.append(dict(first_region=first,second_region=second,replicate=replicate,mode=mode,
                                      codes=values,counts=[values.count(k) for k in range(8)]))
    result = dict(seed=seed,cap=cap,length=length,scale=scale,population=POPULATION,cells=cells)
    atomic_json(destination,result)
    print(f'Completed seed {seed}, cap {cap}, length {length}',flush=True)
    return result


def metric_values(runs, mode):
    cells = [c for r in runs for c in r['cells'] if c['mode']==mode]
    assert len(cells)==50*len(runs)
    result = aggregate(cells)
    return dict(composition=result['composition'],both=result['both'],
                early_first=aggregate([c for c in cells if c['first_region']==0])['first'],
                early_second=aggregate([c for c in cells if c['second_region']==0])['second'])


def summarize(runs):
    expected_runs = {(s,c,n) for s in SEEDS for c in CAPS for n in LENGTHS}
    assert len(runs)==len(expected_runs)
    assert {(r['seed'],r['cap'],r['length']) for r in runs}==expected_runs
    expected_cells = {(a,b,p,m) for a in range(5) for b in range(5) for p in range(2) for m in MODES}
    for run in runs:
        assert run['scale']==compression(run['cap'],run['length']) and run['population']==POPULATION
        assert len(run['cells'])==150
        indexed = {(c['first_region'],c['second_region'],c['replicate'],c['mode']):c for c in run['cells']}
        assert set(indexed)==expected_cells
        for key,cell in indexed.items():
            assert len(cell['codes'])==EXAMPLES and cell['counts']==[cell['codes'].count(k) for k in range(8)]
            if run['length']<=run['cap'] and cell['mode']=='ratio':
                assert cell['codes']==indexed[(*key[:3],'ordinary')]['codes']
    summary = []
    for cap in CAPS:
        for length in LENGTHS:
            selected = [r for r in runs if r['cap']==cap and r['length']==length]
            pooled = {mode:metric_values(selected,mode) for mode in MODES}
            for metric in METRICS:
                changes = {baseline:paired_interval([metric_values([r],'ratio')[metric]-metric_values([r],baseline)[metric]
                                                     for r in selected]) for baseline in MODES[:2]}
                summary.append(dict(cap=cap,length=length,metric=metric,scale=compression(cap,length),
                                    scores={mode:pooled[mode][metric] for mode in MODES},contrasts=changes))
    return summary


def report(summary):
    lines = ['# Training-span ratio transfer across lengths','',
             'The frozen rule preserves the first 16 tokens of query–key separation. It scales the excess by min(1, (2 × training cap + 2 − 16)/(2 × input records + 2 − 16)). The rule equals ordinary RoPE at or below the training cap. Ordinary, fixed-half, and ratio modes share every test example.','',
             'Five saved checkpoint seeds per training cap are evaluated at 32, 48, 64, 80, 96, and 128 records. The 48- and 80-record lengths extend the preceding tests. Every length has a balanced 5 × 5 position grid with two independently seeded batches of 16 examples per cell. This sample count was chosen to bound local CPU and memory use. Each cap/length/mode has 4,000 model-example observations.','',
             'Primary checks are early first-fact retrieval and composition for cap-32 checkpoints at 48 and 80 records. Other lengths and cap-64 checkpoints test consistency and preservation of prior gains.','',
             '| Cap | Records | Ratio scale | Metric | Ordinary | Fixed half | Ratio |',
             '|---:|---:|---:|---|---:|---:|---:|']
    for row in summary:
        lines.append(f"| {row['cap']} | {row['length']} | {row['scale']:.4f} | {row['metric']} | "
                     + ' | '.join(pct(row['scores'][m]) for m in MODES) + ' |')
    lines += ['','## Paired changes','',
              '| Cap | Records | Metric | Baseline | Ratio change, pp | Descriptive 95% seed interval, pp |',
              '|---:|---:|---|---|---:|---:|']
    for row in summary:
        if row['metric'] in ['composition','early_first']:
            for baseline,(mean,lo,hi) in row['contrasts'].items():
                lines.append(f"| {row['cap']} | {row['length']} | {row['metric']} | {baseline} | {100*mean:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines += ['',
              'Early retrieval averages the other query’s five regions and restricts the queried fact to the first fifth of the context. Both measures exact correctness of both auxiliary fact readouts. Composition is the trained neural composition readout. The paired intervals describe variability across five existing checkpoint seeds on fresh shared examples. These checkpoints have participated in earlier experiments. The formula was frozen before these data were generated.','',
              'Within the training cap, ratio results reuse the ordinary predictions because scale one is the same policy. No hyperparameters were selected on this test. Raw results include correctness vectors for all conditions. Run `python outputs/test_rope_ratio_transfer.py` from the repository root. Completed per-checkpoint/length files can be reused on restart.']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    atomic_json(OUT/'protocol.json',dict(seeds=SEEDS,caps=CAPS,lengths=LENGTHS,modes=MODES,population=POPULATION,
                examples_per_cell=EXAMPLES,replicates=2,window=16,workers=1,inference_batch=8,
                primary=dict(cap=32,lengths=[48,80],metrics=['early_first','composition'])))
    runs = []
    with ProcessPoolExecutor(max_workers=1) as pool:
        jobs = [pool.submit(evaluate,s,c,n) for n in LENGTHS for s in SEEDS for c in CAPS]
        for job in as_completed(jobs):
            runs.append(job.result())
            atomic_json(OUT/'progress.json',dict(completed=len(runs),total=len(jobs)))
    runs.sort(key=lambda r:(r['cap'],r['length'],r['seed']))
    summary = summarize(runs)
    report(summary)
    atomic_json(OUT/'results.json',dict(summary=summary,runs=runs))
    print(json.dumps([r for r in summary if r['metric']=='composition']),flush=True)


if __name__=='__main__':
    main()
