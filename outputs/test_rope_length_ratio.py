"""Freeze a training-span ratio rule and compare three modes on fresh 128-record inputs."""
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from causal_transformer_experiment import Model
from stabilize_causal_training import atomic_json
from test_composition_positions import CAPS, CHECKPOINTS, SEEDS, aggregate, pct
from test_context_exposure import paired_interval
from test_distractor_geometry import codes
from test_local_rope_compression import WINDOW, forward
import test_rope_length_transfer as data

OUT = Path(__file__).resolve().parent / 'rope-length-ratio'
POPULATION = 111000000
RECORDS = 128
MODES = ['ordinary', 'fixed', 'ratio']
METRICS = ['composition', 'early_first', 'early_second', 'both']


def compression(cap, records):
    # The farthest causal token pair spans 2*N+2 positions.
    train_span = 2*cap + 2
    test_span = 2*records + 2
    return min(1., (train_span-WINDOW)/(test_span-WINDOW))


@torch.inference_mode()
def evaluate(seed, cap):
    torch.set_num_threads(1)
    data.POPULATION = POPULATION
    checkpoint = torch.load(CHECKPOINTS/f'seed-{seed}'/f'cap{cap}.pt', weights_only=True, map_location='cpu')
    assert (checkpoint['seed'], checkpoint['length'], checkpoint['tokens']) == (seed,cap,1112864)
    model = Model().eval()
    model.load_state_dict(checkpoint['model'])
    factor = compression(cap, RECORDS)
    assert abs(WINDOW+factor*(2*RECORDS+2-WINDOW)-(2*cap+2)) < 1e-9
    cells = []
    for first in range(5):
        for second in range(5):
            for replicate in range(2):
                tokens, labels = data.examples(RECORDS, first, second, replicate)
                predictions = {mode: [] for mode in MODES}
                for chunk, truth in zip(tokens.split(8), labels.split(8)):
                    reads, composition, _ = model(chunk)
                    predictions['ordinary'].extend(codes(reads, composition, truth))
                    for mode, scale in [('fixed',.5), ('ratio',factor)]:
                        reads, composition = forward(model, chunk, scale=scale)
                        predictions[mode].extend(codes(reads, composition, truth))
                    if (first,second,replicate)==(0,0,0):
                        actual = forward(model,chunk,scale=1.)
                        expected = model(chunk)[:2]
                        for got,want in zip(actual,expected):
                            torch.testing.assert_close(got,want,atol=3e-4,rtol=3e-4)
                for mode, values in predictions.items():
                    assert len(values)==data.EXAMPLES
                    cells.append(dict(first_region=first, second_region=second, replicate=replicate,
                                      mode=mode, codes=values, counts=[values.count(k) for k in range(8)]))
    result = dict(seed=seed,cap=cap,scale=factor,cells=cells)
    atomic_json(OUT/f'seed-{seed}-cap{cap}.json',result)
    print(f'Completed seed {seed}, cap {cap}',flush=True)
    return result


def metrics(runs, mode):
    cells = [c for r in runs for c in r['cells'] if c['mode']==mode]
    assert len(cells)==50*len(runs)
    result = aggregate(cells)
    return dict(composition=result['composition'],both=result['both'],
                early_first=aggregate([c for c in cells if c['first_region']==0])['first'],
                early_second=aggregate([c for c in cells if c['second_region']==0])['second'])


def report(runs):
    assert len(runs)==10 and {(r['seed'],r['cap']) for r in runs}=={(s,c) for s in SEEDS for c in CAPS}
    expected = {(a,b,p,m) for a in range(5) for b in range(5) for p in range(2) for m in MODES}
    for run in runs:
        assert len(run['cells'])==150
        assert {(c['first_region'],c['second_region'],c['replicate'],c['mode']) for c in run['cells']}==expected
        for cell in run['cells']:
            assert len(cell['codes'])==32 and cell['counts']==[cell['codes'].count(k) for k in range(8)]
    lines = ['# Training-span ratio compression', '',
             'The 16-token local window is fixed. Beyond it, the ratio rule scales distances by min(1, (training span − 16)/(input span − 16)). A context of N records has 2N+3 tokens and maximum causal separation 2N+2. At 128 records the resulting factors are 0.206612 for cap 32 and 0.471074 for cap 64. All records remain visible.', '',
             'All three modes use the same fresh 5 × 5 position grid, two batches of 32 examples per cell, and five saved checkpoint seeds per cap. The primary outcomes are early retrieval at cap 32 and composition. Early means the first fifth of the context.', '',
             '| Cap | Mode | Early fact 1 | Early fact 2 | Both facts | Composition |',
             '|---:|---|---:|---:|---:|---:|']
    summary = {}
    for cap in CAPS:
        group = [r for r in runs if r['cap']==cap]
        pooled = {mode:metrics(group,mode) for mode in MODES}
        contrasts = {baseline: {metric:paired_interval([metrics([r],'ratio')[metric]-metrics([r],baseline)[metric]
                                                       for r in group]) for metric in METRICS}
                     for baseline in ['ordinary','fixed']}
        summary[cap] = dict(pooled=pooled,contrasts=contrasts)
        for mode,result in pooled.items():
            lines.append(f'| {cap} | {mode} | ' + ' | '.join(pct(result[k]) for k in ['early_first','early_second','both','composition']) + ' |')
    lines += ['', '| Cap | Baseline | Metric | Ratio change, pp | Descriptive 95% seed interval, pp |',
              '|---:|---|---|---:|---:|']
    for cap,result in summary.items():
        for baseline,values in result['contrasts'].items():
            for metric,(mean,lo,hi) in values.items():
                lines.append(f'| {cap} | {baseline} | {metric} | {100*mean:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |')
    lines += ['', 'The factor uses the checkpoint’s training cap and the total input length. It was frozen before this test. The mapping aligns the farthest pair with the training span and changes distances throughout the far region. The intervals describe five-seed variability on shared examples. Scale-one equivalence against ordinary RoPE is checked for every checkpoint.', '',
              'Run `python outputs/test_rope_length_ratio.py` from the repository root. Raw results retain per-example correctness codes.']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return summary


def main():
    atomic_json(OUT/'protocol.json',dict(seeds=SEEDS,caps=CAPS,records=RECORDS,window=WINDOW,
                population=POPULATION,examples_per_cell=32,replicates=2,modes=MODES,
                factors={cap:compression(cap,RECORDS) for cap in CAPS}))
    runs = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(evaluate,s,c) for s in SEEDS for c in CAPS]
        for job in as_completed(jobs):
            runs.append(job.result())
            atomic_json(OUT/'progress.json',dict(completed=len(runs),total=len(jobs)))
    runs.sort(key=lambda r:(r['seed'],r['cap']))
    summary = report(runs)
    atomic_json(OUT/'results.json',dict(summary=summary,runs=runs))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
