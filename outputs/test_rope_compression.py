"""Cross distractor masking with uniform 0.5 RoPE position scaling."""
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from test_composition_positions import CAPS, CHECKPOINTS, REGIONS, SEEDS, aggregate, make_examples, pct
from test_context_exposure import paired_interval
from test_distractor_geometry import OUT as PREVIOUS, Model, atomic_json, codes, positional_forward, retained_indices

OUT = Path(__file__).resolve().parent / 'rope-compression'
CONDITIONS = ['full', 'full_scaled', 'masked', 'masked_scaled', 'compact']
METRICS = ['first', 'second', 'both', 'composition', 'composition_given_both']


@torch.inference_mode()
def evaluate(previous):
    torch.set_num_threads(1)
    seed, cap = previous['seed'], previous['cap']
    state = torch.load(CHECKPOINTS / f'seed-{seed}' / f'cap{cap}.pt', weights_only=True, map_location='cpu')
    assert (state['seed'], state['length'], state['tokens']) == (seed, cap, 1112864)
    model = Model().eval()
    model.load_state_dict(state['model'])
    reference = {(c['first_region'], c['second_region'], c['replicate'], c['condition']): c for c in previous['cells']}
    assert len(reference) == 150
    cells = list(previous['cells'])
    for first in range(5):
        for second in range(5):
            for replicate in range(2):
                tokens, labels, locations = make_examples(first, second, replicate)
                positions = torch.arange(tokens.shape[1]).expand(len(tokens), -1)
                indices = retained_indices(locations, 92000000 + replicate*100 + first*5 + second)
                kept = tokens.gather(1, indices)
                if (first, second, replicate) == (0, 0, 0):
                    for condition, inputs, ids in [('full',tokens,positions), ('masked',kept,indices),
                                                    ('compact',kept,torch.arange(kept.shape[1]).expand(len(kept),-1))]:
                        reads, answer = positional_forward(model, inputs[:32], ids[:32])
                        assert codes(reads, answer, labels[:32]) == reference[0,0,0,condition]['codes'][:32]
                for condition, inputs, ids in [('full_scaled',tokens,positions), ('masked_scaled',kept,indices)]:
                    predictions = []
                    for chunk, original, truth in zip(inputs.split(32), ids.split(32), labels.split(32)):
                        reads, answer = positional_forward(model, chunk, original * .5)
                        predictions.extend(codes(reads, answer, truth))
                    assert len(predictions) == len(labels)
                    cells.append(dict(first_region=first, second_region=second, replicate=replicate,
                                      condition=condition, codes=predictions,
                                      counts=[predictions.count(i) for i in range(8)]))
    result = dict(seed=seed, cap=cap, cells=cells)
    atomic_json(OUT / f'seed-{seed}-cap{cap}.json', result)
    print(f'Completed seed {seed}, cap {cap}', flush=True)
    return result


def scores(runs, condition, role=None, region=None):
    return aggregate([c for r in runs for c in r['cells'] if c['condition'] == condition
                      and (role is None or c[f'{role}_region'] == region)])


def report(runs):
    assert {(r['seed'], r['cap']) for r in runs} == {(s,c) for s in SEEDS for c in CAPS}
    expected = {(a,b,p,c) for a in range(5) for b in range(5) for p in range(2) for c in CONDITIONS}
    for run in runs:
        assert len(run['cells']) == len(expected)
        assert {(c['first_region'],c['second_region'],c['replicate'],c['condition']) for c in run['cells']} == expected
        for cell in run['cells']:
            assert len(cell['codes']) == 64 and cell['counts'] == [cell['codes'].count(i) for i in range(8)]
    lines = ['# Uniform RoPE compression at 128 records', '',
             'Original and half-scale position IDs are crossed with full visibility and the previous 64-record retention mask. Fractional IDs are applied in every attention layer. Scaling preserves the same inputs, ordering and causal visibility within each comparison. Compaction is the earlier reference with contiguous IDs on the retained records.', '',
             'The five seeds per cap share all held-out inputs. Full-scale, masked and compacted reference predictions are reused from the previous experiment and spot-checked on each checkpoint. Each cap and condition has 16,000 model-example observations.', '',
             '| Cap | Condition | Fact 1 | Fact 2 | Both | Composition | Composition given both correct |',
             '|---:|---|---:|---:|---:|---:|---:|']
    summary = {}
    for cap in CAPS:
        selected = [r for r in runs if r['cap'] == cap]
        pooled = {c: scores(selected,c) for c in CONDITIONS}
        for condition, row in pooled.items():
            lines.append(f'| {cap} | {condition} | ' + ' | '.join(pct(row[m]) for m in METRICS) + ' |')
        seed_scores = [{c: scores([r],c) for c in CONDITIONS} for r in selected]
        contrasts = {name: {m: paired_interval([sum(weight*row[c][m] for c,weight in terms)
                                              for row in seed_scores]) for m in METRICS}
                     for name, terms in [('full_scaling',[('full_scaled',1),('full',-1)]),
                                         ('masked_scaling',[('masked_scaled',1),('masked',-1)]),
                                         ('interaction',[('masked_scaled',1),('masked',-1),('full_scaled',-1),('full',1)]),
                                         ('compact_minus_masked_scaled',[('compact',1),('masked_scaled',-1)])]}
        regions = {role: [{c: scores(selected,c,role,region) for c in CONDITIONS} for region in range(5)]
                   for role in ['first','second']}
        summary[cap] = dict(pooled=pooled, contrasts=contrasts, regions=regions, seed_scores=seed_scores)
    lines += ['', '## Paired contrasts', '', '| Cap | Contrast | Metric | Change, pp | Descriptive 95% seed interval, pp |',
              '|---:|---|---|---:|---:|']
    for cap, result in summary.items():
        for contrast, metrics in result['contrasts'].items():
            for metric, (mean,lo,hi) in metrics.items():
                lines.append(f'| {cap} | {contrast} | {metric} | {100*mean:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |')
    lines += ['', '## Retrieval by evidence position', '', '| Cap | Query | Record region | Full | Full scaled | Masked | Masked scaled | Compact |',
              '|---:|---|---|---:|---:|---:|---:|---:|']
    for cap, result in summary.items():
        for role, rows in result['regions'].items():
            for (lo,hi), row in zip(REGIONS, rows):
                lines.append(f'| {cap} | {role} | {lo}–{hi-1} | ' + ' | '.join(pct(row[c][role]) for c in CONDITIONS) + ' |')
    lines += ['', 'The interaction subtracts the full-context scaling effect from the masked scaling effect. Conditional composition comparisons use the correctly retrieved subset for each condition. This is a fixed-factor inference test. Uniform scaling changes both local and long-range relative RoPE distances. The masking arms retain facts using known locations.', '',
              'Run `python test_rope_compression.py` with the prior experiment results and checkpoints in place.']
    (OUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def main():
    previous = json.loads((PREVIOUS / 'results.json').read_text())
    atomic_json(OUT / 'protocol.json', dict(seeds=SEEDS, caps=CAPS, conditions=CONDITIONS,
                position_scale=.5, records=128, masked_records=64, reference=str(PREVIOUS/'results.json')))
    runs = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(evaluate, run) for run in previous['runs']]
        for job in as_completed(jobs):
            runs.append(job.result())
            atomic_json(OUT / 'progress.json', dict(completed=len(runs), total=len(jobs)))
    runs.sort(key=lambda r: (r['seed'],r['cap']))
    summary = report(runs)
    atomic_json(OUT / 'results.json', dict(summary=summary, runs=runs))
    print(json.dumps({cap: {c: row['composition'] for c,row in result['pooled'].items()}
                      for cap,result in summary.items()}), flush=True)


if __name__ == '__main__':
    main()
