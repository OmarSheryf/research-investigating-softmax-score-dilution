"""Token-matched 32 versus 64 record exposure with a primary 128-record test."""
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from causal_transformer_experiment import KEYS, Model, batch, verify
from stabilize_causal_training import POSITIONS, atomic_json, evaluate
from test_context_exposure import checkpoint, paired_interval, train_segment

OUT = Path(__file__).resolve().parent / 'context-scale-up'
SEEDS = [281, 292, 303, 314, 325]
PREFIX = [(4, 400), (8, 400), (16, 199), (32, 105)]
SUFFIXES = {32: 131, 64: 67}
LENGTHS = [32, 64, 128]
METRICS = ['single', 'two', 'both_reads']
POPULATION = 81000000
BUDGET = 1112864


def run(seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    generator = torch.Generator().manual_seed(seed + 1000)
    directory = OUT / f'seed-{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    step = tokens = 0
    for length, count in PREFIX:
        step, tokens = train_segment(model, optimizer, generator, length, count, step, tokens)
        print(f'{seed} prefix length {length} complete at update {step}', flush=True)
    assert (step, tokens) == (1104, 832000)
    shared = directory / 'shared.pt'
    checkpoint(shared, model, optimizer, generator, seed, step, tokens, 32)
    runs = []
    for length, count in SUFFIXES.items():
        state = torch.load(shared, weights_only=True, map_location='cpu')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['batch_rng'])
        torch.set_rng_state(state['torch_rng'])
        end_step, end_tokens = train_segment(model, optimizer, generator, length, count, step, tokens)
        assert end_tokens == BUDGET
        checkpoint(directory / f'cap{length}.pt', model, optimizer, generator, seed, end_step, end_tokens, length)
        cells = evaluate(model, POPULATION, LENGTHS)
        result = dict(seed=seed, cap=length, tokens=end_tokens, updates=end_step, test=cells)
        atomic_json(directory / f'cap{length}.json', result)
        runs.append(result)
        scores = {m: mean_score(result, 128, m) for m in METRICS}
        print(f'{seed} cap {length} at 128 records {json.dumps(scores)}', flush=True)
    return runs


def mean_score(run, length, metric, position=None):
    cells = [r for r in run['test'] if r['n'] == length and (position is None or r['position'] == position)]
    assert len(cells) == (10 if position is None else 2)
    return sum(r[metric] for r in cells) / len(cells)


def summarize(runs):
    indexed = {(r['seed'], r['cap']): r for r in runs}
    assert len(runs) == len(indexed) == 10
    expected = {(n, p, rep) for n in LENGTHS for p in POSITIONS for rep in [0, 1]}
    for run in runs:
        assert run['tokens'] == BUDGET and len(run['test']) == 30
        assert {(r['n'], r['position'], r['replicate']) for r in run['test']} == expected
        for row in run['test']:
            for metric in ['single', 'two']:
                correct = row['correct_' + metric]
                assert len(correct) == 128 and sum(correct) / 128 == row[metric]
    summary = []
    for n in LENGTHS:
        for position in [None, *POSITIONS]:
            for metric in METRICS:
                control = [mean_score(indexed[s, 32], n, metric, position) for s in SEEDS]
                extended = [mean_score(indexed[s, 64], n, metric, position) for s in SEEDS]
                deltas = [b-a for a, b in zip(control, extended)]
                summary.append(dict(length=n, position=position, metric=metric, cap32=sum(control)/5,
                                    cap64=sum(extended)/5, deltas=deltas, interval=paired_interval(deltas)))
    return summary


def report(summary):
    lines = ['# Training through 64 records', '',
             'Five fresh seeds share a checkpoint trained through 32 records at 832,000 input tokens. Each branch spends 280,864 additional tokens. Continuing at 32 takes 131 updates. Moving to 64 takes 67 updates. Batch size is 32 and AdamW uses a constant 0.002 learning rate. Each final model has seen 1,112,864 input tokens.', '',
             'The primary extrapolation test uses 128 records. Each length and position has two held-out populations of 128 examples. At 128 records, each context contains all 128 key identities exactly once. Positions refer to the first queried record. The second is half a context away with wraparound.', '',
             '| Records | Metric | Cap 32 | Cap 64 | Paired change, pp | Descriptive 95% seed interval, pp |',
             '|---:|---|---:|---:|---:|---:|']
    for row in summary:
        if row['position'] is None:
            change, lo, hi = row['interval']
            lines.append(f"| {row['length']} | {row['metric']} | {row['cap32']:.2%} | {row['cap64']:.2%} | {100*change:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines += ['', '## Evidence position', '',
              '| Records | First evidence position | Retrieval cap 32 → cap 64 | Composition cap 32 → cap 64 |',
              '|---:|---:|---:|---:|']
    for n in LENGTHS:
        for position in POSITIONS:
            rows = [r for r in summary if r['length'] == n and r['position'] == position and r['metric'] in ['single','two']]
            lines.append(f'| {n} | {position:.0%} | ' + ' | '.join(f"{r['cap32']:.2%} → {r['cap64']:.2%}" for r in rows) + ' |')
    lines += ['', 'Single measures the first readout. Two measures learned composition. Both reads measures exact accuracy on both underlying fact readouts. The paired intervals describe seed variation on shared test populations. Fixed batch size makes update count differ across the two token-matched schedules.', '',
              'Run `python test_context_scale_up.py` alongside the imported experiment scripts. Checkpoints store weights, optimizer state and random generator states.']
    (OUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    torch.set_num_threads(1)
    verify()
    tokens, labels, _ = batch(128, 4, torch.Generator().manual_seed(404))
    assert KEYS == 128 and tokens.shape == (4, 259) and labels.shape == (4, 2)
    assert (tokens[:, :256:2].sort(dim=1).values == torch.arange(128)).all()
    atomic_json(OUT / 'protocol.json', dict(seeds=SEEDS, prefix=PREFIX, suffixes=SUFFIXES, budget=BUDGET,
                primary_length=128, test_lengths=LENGTHS, test_population=POPULATION, positions=POSITIONS,
                composition_starts=601, learning_rate=.002, batch_size=32))
    runs = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(run, seed) for seed in SEEDS]
        for job in as_completed(jobs):
            runs.extend(job.result())
            atomic_json(OUT / 'progress.json', dict(completed_pairs=len(runs)//2, total_pairs=len(SEEDS)))
    summary = summarize(runs)
    report(summary)
    atomic_json(OUT / 'results.json', dict(summary=summary, runs=runs))
    print(json.dumps([r for r in summary if r['position'] is None]), flush=True)


if __name__ == '__main__':
    main()
