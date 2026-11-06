"""Evaluate fact readouts and composition on a crossed position grid."""
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from causal_transformer_experiment import KEYS, Model, VALUES, batch
from stabilize_causal_training import atomic_json
from test_context_scale_up import OUT as CHECKPOINTS, SEEDS

OUT = Path(__file__).resolve().parent / 'composition-positions'
REGIONS = [(0, 26), (26, 52), (52, 77), (77, 103), (103, 128)]
CAPS = [32, 64]
POPULATION = 91000000
EXAMPLES = 64


def make_examples(first, second, replicate):
    generator = torch.Generator().manual_seed(POPULATION + replicate * 100 + first * 5 + second)
    tokens, _, _ = batch(128, EXAMPLES, generator)
    lo1, hi1 = REGIONS[first]
    lo2, hi2 = REGIONS[second]
    p1 = torch.randint(lo1, hi1, (EXAMPLES,), generator=generator)
    if first == second:
        p2 = torch.randint(lo2, hi2 - 1, (EXAMPLES,), generator=generator)
        p2 += (p2 >= p1).long()
    else:
        p2 = torch.randint(lo2, hi2, (EXAMPLES,), generator=generator)
    assert ((p1 >= lo1) & (p1 < hi1) & (p2 >= lo2) & (p2 < hi2) & (p1 != p2)).all()
    positions = torch.stack((p1, p2), dim=1)
    tokens[:, -2:] = tokens.gather(1, 2 * positions)
    labels = tokens.gather(1, 2 * positions + 1) - KEYS
    assert ((labels >= 0) & (labels < VALUES)).all()
    return tokens, labels, positions


@torch.inference_mode()
def evaluate(seed, cap):
    torch.set_num_threads(1)
    state = torch.load(CHECKPOINTS / f'seed-{seed}' / f'cap{cap}.pt', weights_only=True, map_location='cpu')
    assert (state['seed'], state['length'], state['tokens']) == (seed, cap, 1112864)
    model = Model().eval()
    model.load_state_dict(state['model'])
    cells = []
    for first in range(5):
        for second in range(5):
            for replicate in range(2):
                tokens, labels, positions = make_examples(first, second, replicate)
                codes = []
                for inputs, truth in zip(tokens.split(32), labels.split(32)):
                    reads, composition, _ = model(inputs)
                    correct = reads.argmax(-1) == truth
                    answer = composition.argmax(-1) == truth.sum(-1) % VALUES
                    codes.extend((correct[:, 0].int() + 2*correct[:, 1].int() + 4*answer.int()).tolist())
                counts = [codes.count(code) for code in range(8)]
                assert sum(counts) == EXAMPLES
                cells.append(dict(first_region=first, second_region=second, replicate=replicate,
                                  counts=counts, positions=positions.tolist(), codes=codes))
    result = dict(seed=seed, cap=cap, cells=cells)
    atomic_json(OUT / f'seed-{seed}-cap{cap}.json', result)
    print(f'Finished seed {seed}, cap {cap}', flush=True)
    return result


def aggregate(cells):
    counts = [sum(c['counts'][code] for c in cells) for code in range(8)]
    total = sum(counts)
    both = counts[3] + counts[7]
    errors = sum(counts[:4])
    return dict(total=total, first=(counts[1]+counts[3]+counts[5]+counts[7])/total,
                second=(counts[2]+counts[3]+counts[6]+counts[7])/total, both=both/total,
                composition=sum(counts[4:])/total, both_count=both,
                composition_given_both=counts[7]/both if both else None,
                composition_given_missed=sum(counts[4:7])/(total-both) if total>both else None,
                composition_errors=errors, errors_both_correct=counts[3],
                errors_with_missed_fact=sum(counts[:3]), counts=counts)


def pct(value):
    return '—' if value is None else f'{value:.2%}'


def report(runs):
    assert {(r['seed'], r['cap']) for r in runs} == {(s,c) for s in SEEDS for c in CAPS}
    expected = {(a,b,p) for a in range(5) for b in range(5) for p in range(2)}
    for run in runs:
        assert len(run['cells']) == 50
        assert {(c['first_region'],c['second_region'],c['replicate']) for c in run['cells']} == expected
        for cell in run['cells']:
            assert len(cell['positions']) == len(cell['codes']) == EXAMPLES
            assert cell['counts'] == [cell['codes'].count(code) for code in range(8)]
    summary = {}
    lines = ['# Composition and evidence position at 128 records', '',
             'Five paired saved checkpoints per training cap. All 25 ordered combinations of five position regions receive equal weight. Positions are independently sampled within their assigned regions, subject to the two records being distinct. Each cell uses two independently seeded sets of 64 contexts, shared across checkpoints. Regions use zero-based record indices.', '',
             '| Training cap | Fact 1 | Fact 2 | Both facts | Composition | Composition given both | Composition given a missed fact |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for cap in CAPS:
        cells = [c for r in runs if r['cap'] == cap for c in r['cells']]
        pooled = aggregate(cells)
        grid = [[aggregate([c for c in cells if c['first_region']==a and c['second_region']==b])
                 for b in range(5)] for a in range(5)]
        summary[cap] = dict(pooled=pooled, grid=grid,
                            per_seed={r['seed']: aggregate(r['cells']) for r in runs if r['cap']==cap})
        metrics = ['first','second','both','composition','composition_given_both','composition_given_missed']
        lines.append(f'| {cap} | ' + ' | '.join(pct(pooled[m]) for m in metrics) + ' |')
    lines += ['', '| Training cap | Composition errors | Both readouts correct | At least one readout wrong |',
              '|---:|---:|---:|---:|']
    for cap in CAPS:
        p = summary[cap]['pooled']
        lines.append(f"| {cap} | {p['composition_errors']:,} | {p['errors_both_correct']:,} | {p['errors_with_missed_fact']:,} |")
    labels = [f'{lo}–{hi-1}' for lo,hi in REGIONS]
    for cap in CAPS:
        for metric, title in [('both', 'Both readouts correct'), ('composition', 'Composition accuracy'),
                              ('composition_given_both', 'Composition given both readouts correct')]:
            lines += ['', f'## Cap {cap}, {title}', '', 'Rows show the first query region. Columns show the second.', '',
                      '| First query | ' + ' | '.join(labels) + ' |', '|---|---:|---:|---:|---:|---:|']
            for index, row in enumerate(summary[cap]['grid']):
                lines.append('| ' + labels[index] + ' | ' + ' | '.join(pct(r[metric]) for r in row) + ' |')
    lines += ['', 'Conditioning uses correctness of the auxiliary readout heads. The composition head uses the underlying query states directly, so these counts measure association. The balanced position grid differs from the earlier half-context separation test.', '',
              'Correctness codes use bit 0 for fact 1, bit 1 for fact 2 and bit 2 for composition. Raw results include exact sampled positions and per-example codes. Run `python test_composition_positions.py` with the context-scale-up checkpoints in place.']
    (OUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def main():
    atomic_json(OUT / 'protocol.json', dict(seeds=SEEDS, caps=CAPS, records=128, regions=REGIONS,
                population=POPULATION, replicates=2, examples_per_cell_per_replicate=EXAMPLES))
    runs = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(evaluate, seed, cap) for seed in SEEDS for cap in CAPS]
        for job in as_completed(jobs):
            runs.append(job.result())
            atomic_json(OUT / 'progress.json', dict(completed=len(runs), total=len(jobs)))
    runs.sort(key=lambda r: (r['seed'],r['cap']))
    summary = report(runs)
    atomic_json(OUT / 'results.json', dict(summary=summary, runs=runs))
    print(json.dumps({cap: s['pooled'] for cap,s in summary.items()}), flush=True)


if __name__ == '__main__':
    main()
