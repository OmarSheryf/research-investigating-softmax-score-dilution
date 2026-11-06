"""Test a frozen RoPE compression rule and a training-cap gate on fresh contexts."""
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from causal_transformer_experiment import KEYS, Model, batch
from stabilize_causal_training import atomic_json
from test_composition_positions import CAPS, CHECKPOINTS, SEEDS, aggregate, pct
from test_context_exposure import paired_interval
from test_distractor_geometry import codes
from test_local_rope_compression import SCALE, WINDOW, forward

OUT = Path(__file__).resolve().parent / 'rope-length-transfer'
LENGTHS = [32, 64, 96, 128]
POPULATION = 101000000
EXAMPLES = 32
CONDITIONS = ['ordinary', 'compressed', 'gated']


def examples(length, first, second, replicate):
    generator = torch.Generator().manual_seed(POPULATION + length*1000 + replicate*100 + first*5 + second)
    tokens, _, _ = batch(length, EXAMPLES, generator)
    lo1, hi1 = first*length//5, (first+1)*length//5
    lo2, hi2 = second*length//5, (second+1)*length//5
    p1 = torch.randint(lo1, hi1, (EXAMPLES,), generator=generator)
    if first == second:
        p2 = torch.randint(lo2, hi2-1, (EXAMPLES,), generator=generator)
        p2 += (p2 >= p1).long()
    else:
        p2 = torch.randint(lo2, hi2, (EXAMPLES,), generator=generator)
    assert ((p1 >= lo1) & (p1 < hi1) & (p2 >= lo2) & (p2 < hi2) & (p1 != p2)).all()
    positions = torch.stack((p1,p2),dim=1)
    tokens[:, -2:] = tokens.gather(1, 2*positions)
    return tokens, tokens.gather(1, 2*positions+1)-KEYS


@torch.inference_mode()
def evaluate(seed, cap):
    torch.set_num_threads(1)
    state = torch.load(CHECKPOINTS/f'seed-{seed}'/f'cap{cap}.pt',weights_only=True,map_location='cpu')
    assert (state['seed'],state['length'],state['tokens']) == (seed,cap,1112864)
    model = Model().eval()
    model.load_state_dict(state['model'])
    cells = []
    for n in LENGTHS:
        for a in range(5):
            for b in range(5):
                for rep in range(2):
                    tokens, truth = examples(n,a,b,rep)
                    reads, answer, _ = model(tokens)
                    ordinary = codes(reads,answer,truth)
                    compressed = []
                    for chunk, labels in zip(tokens.split(16), truth.split(16)):
                        reads, answer = forward(model,chunk)
                        compressed.extend(codes(reads,answer,labels))
                    for condition, predictions in [('ordinary',ordinary),('compressed',compressed)]:
                        assert len(predictions)==EXAMPLES
                        cells.append(dict(length=n,first_region=a,second_region=b,replicate=rep,
                                          condition=condition,codes=predictions,
                                          counts=[predictions.count(i) for i in range(8)]))
        print(f'Seed {seed}, cap {cap}, length {n} complete',flush=True)
    result = dict(seed=seed,cap=cap,cells=cells)
    atomic_json(OUT/f'seed-{seed}-cap{cap}.json',result)
    return result


def metrics(runs,n,condition):
    cap = runs[0]['cap']
    assert all(r['cap']==cap for r in runs)
    if condition=='gated':
        condition = 'compressed' if n>cap else 'ordinary'
    cells = [c for r in runs for c in r['cells'] if c['length']==n and c['condition']==condition]
    assert len(cells)==50*len(runs)
    return dict(composition=aggregate(cells)['composition'],
                early_first=aggregate([c for c in cells if c['first_region']==0])['first'],
                early_second=aggregate([c for c in cells if c['second_region']==0])['second'])


def report(runs):
    assert len(runs)==10 and {(r['seed'],r['cap']) for r in runs}=={(s,c) for s in SEEDS for c in CAPS}
    expected = {(n,a,b,p,c) for n in LENGTHS for a in range(5) for b in range(5)
                for p in range(2) for c in CONDITIONS[:2]}
    for run in runs:
        cells = run['cells']
        assert len(cells)==len(expected)
        assert {(c['length'],c['first_region'],c['second_region'],c['replicate'],c['condition']) for c in cells}==expected
        assert all(len(c['codes'])==EXAMPLES and c['counts']==[c['codes'].count(i) for i in range(8)] for c in cells)
    lines = ['# Frozen RoPE rule across context lengths', '',
             'Five saved seeds per training cap are tested on fresh examples at 32, 64, 96 and 128 records. The compression rule preserves distances through 16 tokens and halves the excess in each attention layer. Both inference modes see identical records. Each length has a balanced 5 × 5 grid of ordered evidence regions and two independently seeded batches of 32 examples per cell. Early retrieval means the queried fact is in the first fifth of that context.', '',
             'The gate dispatches to ordinary RoPE at or below the training cap and compression above it. Its scores are computed exactly from the two measured inference modes, since the gate chooses a single mode per input. The policy assumes total record count is known at inference.', '',
             '| Cap | Records | Metric | Ordinary | Compressed | Gated | Compression delta, pp | Descriptive 95% seed interval, pp |',
             '|---:|---:|---|---:|---:|---:|---:|---:|']
    summary = []
    for cap in CAPS:
        selected = [r for r in runs if r['cap']==cap]
        for n in LENGTHS:
            pooled = {c:metrics(selected,n,c) for c in CONDITIONS}
            for metric in pooled['ordinary']:
                interval = paired_interval([metrics([r],n,'compressed')[metric]-metrics([r],n,'ordinary')[metric]
                                            for r in selected])
                mean,lo,hi = interval
                values = {c:pooled[c][metric] for c in CONDITIONS}
                summary.append(dict(cap=cap,length=n,metric=metric,interval=interval,**values))
                lines.append(f'| {cap} | {n} | {metric} | ' + ' | '.join(pct(values[c]) for c in CONDITIONS)
                             + f' | {100*mean:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |')
    lines += ['', 'The intervals describe variation across five paired checkpoint seeds on the same fresh test sets. Run `python test_rope_length_transfer.py` with the saved context-scale-up checkpoints in place.']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return summary


def main():
    assert WINDOW==16 and SCALE==.5
    atomic_json(OUT/'protocol.json',dict(seeds=SEEDS,caps=CAPS,lengths=LENGTHS,window=WINDOW,scale=SCALE,
                population=POPULATION,examples_per_cell=EXAMPLES,replicates=2,gate='compress iff records > training cap'))
    runs = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(evaluate,s,c) for s in SEEDS for c in CAPS]
        for job in as_completed(jobs):
            runs.append(job.result())
            atomic_json(OUT/'progress.json',dict(completed=len(runs),total=len(jobs)))
    runs.sort(key=lambda r:(r['seed'],r['cap']))
    summary = report(runs)
    atomic_json(OUT/'results.json',dict(summary=summary,runs=runs))
    print(json.dumps([r for r in summary if r['metric']=='composition']),flush=True)


if __name__=='__main__':
    main()
