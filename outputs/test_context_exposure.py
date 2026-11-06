"""Compare 16- and 32-record training caps at an exact input-token budget."""
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from torch.nn import functional as F

from causal_transformer_experiment import Model, batch, verify
from stabilize_causal_training import atomic_json, evaluate

OUT = Path(__file__).resolve().parent / 'context-exposure'
SEEDS = [226, 237, 248, 259, 270]
ARMS = {'cap16': (16, 201), 'cap32': (32, 105)}
TEST_LENGTHS = [16, 32, 64]
METRICS = ['single', 'two', 'both_reads']
TEST_POPULATION = 71000000
BUDGET = 832000


def train_segment(model, optimizer, generator, length, count, step, tokens_seen):
    model.train()
    for _ in range(count):
        step += 1
        tokens, labels, _ = batch(length, 32, generator)
        reads, composition, _ = model(tokens)
        loss = F.cross_entropy(reads.flatten(0, 1), labels.flatten())
        if step > 600:
            loss = loss + F.cross_entropy(composition, labels.sum(-1) % 8)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        tokens_seen += tokens.numel()
    return step, tokens_seen


def checkpoint(path, model, optimizer, generator, seed, step, tokens_seen, length):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    batch_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
                    seed=seed, step=step, tokens=tokens_seen, length=length), temporary)
    temporary.replace(path)


def run(seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    generator = torch.Generator().manual_seed(seed + 1000)
    directory = OUT / f'seed-{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    step = tokens_seen = 0
    for length, count in [(4, 400), (8, 400), (16, 199)]:
        step, tokens_seen = train_segment(model, optimizer, generator, length, count, step, tokens_seen)
        print(f'Seed {seed} shared prefix: {step} updates, {tokens_seen} tokens', flush=True)
    assert (step, tokens_seen) == (999, 606880)
    prefix = directory / 'shared-step-999.pt'
    checkpoint(prefix, model, optimizer, generator, seed, step, tokens_seen, 16)
    results = []
    for arm, (length, count) in ARMS.items():
        state = torch.load(prefix, weights_only=True, map_location='cpu')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['batch_rng'])
        torch.set_rng_state(state['torch_rng'])
        end_step, end_tokens = train_segment(model, optimizer, generator, length, count, step, tokens_seen)
        assert end_tokens == BUDGET
        checkpoint(directory / f'{arm}.pt', model, optimizer, generator, seed, end_step, end_tokens, length)
        cells = evaluate(model, TEST_POPULATION, TEST_LENGTHS)
        scores = {str(n): {metric: sum(row[metric] for row in cells if row['n'] == n) / 10
                           for metric in METRICS} for n in TEST_LENGTHS}
        result = dict(seed=seed, arm=arm, updates=end_step, tokens=end_tokens, scores=scores, test=cells)
        atomic_json(directory / f'{arm}.json', result)
        results.append(result)
        print(f'Seed {seed} {arm}: {json.dumps(scores)}', flush=True)
    return results


def paired_interval(values):
    mean = sum(values) / len(values)
    half = 2.776 * math.sqrt(sum((value - mean)**2 for value in values) / 20)
    return [mean, mean-half, mean+half]


def summarize(runs):
    indexed = {(run['seed'], run['arm']): run for run in runs}
    assert set(indexed) == {(seed, arm) for seed in SEEDS for arm in ARMS}
    expected_cells = {(n, p, r) for n in TEST_LENGTHS for p in [0, .25, .5, .75, 1] for r in [0, 1]}
    for run in runs:
        assert run['tokens'] == BUDGET and len(run['test']) == 30
        assert {(row['n'], row['position'], row['replicate']) for row in run['test']} == expected_cells
        for row in run['test']:
            for metric in ['single', 'two']:
                answers = row['correct_' + metric]
                assert len(answers) == 128 and sum(answers) / 128 == row[metric]
    summary = []
    for length in TEST_LENGTHS:
        for metric in METRICS:
            control = [indexed[s, 'cap16']['scores'][str(length)][metric] for s in SEEDS]
            extended = [indexed[s, 'cap32']['scores'][str(length)][metric] for s in SEEDS]
            deltas = [new - old for old, new in zip(control, extended)]
            summary.append(dict(length=length, metric=metric, cap16=sum(control)/5, cap32=sum(extended)/5,
                                seed_deltas=deltas, paired_interval=paired_interval(deltas)))
    passes = {arm: sum(min(indexed[s, arm]['scores']['16'][m] for m in ['single', 'two']) >= .95
                       for s in SEEDS) for arm in ARMS}
    lines = ['# Training context exposure', '',
             'Five fresh seeds use the selected short-to-long, retrieval-first recipe. Composition loss starts at update 601. Each seed branches from one shared checkpoint at update 999. The control continues for 201 batches at 16 records. The extended arm spends that same token budget on 105 batches at 32 records. Both train with batch size 32, AdamW at 0.002, and exactly 832,000 input tokens.', '',
             'The extended recipe uses 1,104 updates and the control uses 1,200. This compares allocation of a fixed token budget, with context length and update count changing together after the fork. Test inputs are paired across arms. Each length uses five positions and two independently seeded populations, with 128 examples per cell.', '',
             f"At 16 records, both tasks reach 95% in {passes['cap16']}/5 control runs and {passes['cap32']}/5 extended runs.", '',
             '| Records | Metric | Cap 16 | Cap 32 | Paired change, pp | Descriptive 95% seed interval, pp |',
             '|---:|---|---:|---:|---:|---:|']
    for row in summary:
        mean, lo, hi = row['paired_interval']
        lines.append(f"| {row['length']} | {row['metric']} | {row['cap16']:.2%} | {row['cap32']:.2%} | {100*mean:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines += ['', 'Single is the first fact readout. Two is learned composition accuracy. Both reads is exact accuracy on both underlying fact readouts. Only the extended arm trains at 32 records. Both arms extrapolate at 64. Intervals summarize paired differences across five training seeds and the shared fixed test populations.', '',
              'Run `python test_context_exposure.py` alongside the two imported experiment scripts. Shared and final checkpoints save weights, optimizer and generator states.']
    (OUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return dict(metrics=summary, passes_at_16=passes)


def main():
    torch.set_num_threads(1)
    verify()
    atomic_json(OUT / 'protocol.json', dict(seeds=SEEDS, shared_schedule=[[4,400],[8,400],[16,199]],
                suffixes=ARMS, budget=BUDGET, composition_starts=601, test_lengths=TEST_LENGTHS,
                test_population=TEST_POPULATION, learning_rate=.002))
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(run, seed) for seed in SEEDS]
        for job in as_completed(jobs):
            results.extend(job.result())
            atomic_json(OUT / 'progress.json', dict(completed_pairs=len(results)//2, total_pairs=len(SEEDS)))
    summary = summarize(results)
    atomic_json(OUT / 'results.json', dict(summary=summary, runs=results))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
