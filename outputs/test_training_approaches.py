"""Paired training-order and loss-schedule experiments with exact batch replay."""
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from causal_transformer_experiment import Model, batch, verify
from stabilize_causal_training import atomic_json, evaluate, task_scores

OUT = Path(__file__).resolve().parent / 'training-approaches'
SEEDS = [116, 127, 138, 149, 160]
LENGTHS = [4, 8, 16]
ARMS = ['joint_interleaved', 'joint_curriculum', 'retrieval_first']
STEPS = 1200
VALIDATION = 31000000
TEST = 41000000


def bank_for(seed):
    banks = {}
    digest = hashlib.sha256()
    for length in LENGTHS:
        generator = torch.Generator().manual_seed(seed * 1000 + length)
        banks[length] = []
        for _ in range(400):
            tokens, labels, _ = batch(length, 32, generator)
            banks[length].append((tokens, labels))
            digest.update(tokens.numpy().tobytes())
            digest.update(labels.numpy().tobytes())
    return banks, digest.hexdigest()


def train(seed, arm, banks, digest):
    torch.manual_seed(seed)
    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    directory = OUT / f'{seed}-{arm}'
    directory.mkdir(parents=True, exist_ok=True)
    used = dict.fromkeys(LENGTHS, 0)
    trace = []
    tokens_seen = 0
    for step in range(1, STEPS + 1):
        ordered = arm in ('joint_curriculum', 'retrieval_first_curriculum')
        index = (step - 1) // 400 if ordered else (step - 1) % 3
        length = LENGTHS[index]
        tokens, labels = banks[length][used[length]]
        used[length] += 1
        tokens_seen += tokens.numel()
        reads, composition, _ = model(tokens)
        read_loss = F.cross_entropy(reads.flatten(0, 1), labels.flatten())
        sum_loss = F.cross_entropy(composition, labels.sum(-1) % 8)
        retrieval_only = arm in ('retrieval_first', 'retrieval_first_curriculum') and step <= 600
        loss = read_loss if retrieval_only else read_loss + sum_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % 300 == 0:
            scores = task_scores(evaluate(model, VALIDATION, [8, 16]))
            trace.append(dict(step=step, tokens=tokens_seen, scores=scores))
            model.train()
            print(f'{seed} {arm} step {step}: validation minimum {min(scores.values()):.3f}', flush=True)
        if step in [600, 1200]:
            path = directory / f'step-{step}.pt'
            temp = path.with_suffix('.tmp')
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), seed=seed,
                            arm=arm, step=step, bank_hash=digest, used=used, tokens=tokens_seen,
                            rng=torch.get_rng_state()), temp)
            temp.replace(path)
    assert used == dict.fromkeys(LENGTHS, 400) and tokens_seen == 832000
    rows = evaluate(model, TEST, [8, 16])
    scores = task_scores(rows)
    first = next((entry['tokens'] for entry in trace if min(entry['scores'].values()) >= .95), None)
    result = dict(seed=seed, arm=arm, bank_hash=digest, tokens=tokens_seen, scores=scores,
                  passed=min(scores.values()) >= .95, first_validation_pass_tokens=first,
                  validation_trace=trace, test=rows)
    atomic_json(directory / 'result.json', result)
    print(f'FINISHED {seed} {arm}: held-out minimum {min(scores.values()):.3f}', flush=True)
    return result


def report(results):
    indexed = {(r['seed'], r['arm']): r for r in results}
    assert len(indexed) == len(SEEDS) * len(ARMS)
    for seed in SEEDS:
        assert len({indexed[seed, arm]['bank_hash'] for arm in ARMS}) == 1
    for result in results:
        assert len(result['test']) == 20
        for cell in result['test']:
            for metric in ['single', 'two']:
                assert len(cell['correct_' + metric]) == 128
                assert sum(cell['correct_' + metric]) / 128 == cell[metric]
    lines = ['# Controlled training follow-ups', '',
             'Five fresh seeds share initialization and exact training batches across recipes. Each recipe uses 1,200 updates, 832,000 tokens, and AdamW at a constant learning rate of 0.002. Interleaving cycles through 4, 8, and 16 records. The ordering curriculum groups those same batches by increasing length. Retrieval-first keeps the interleaved order and omits the composition loss for the first 600 updates.', '',
             'Validation is measured every 300 updates. A pass requires at least 95% on both tasks at both 8 and 16 records. Final test populations are separate from validation. First measured validation pass is interval-censored by checkpoint spacing, and absent passes are right-censored at the training budget.', '',
             '| Recipe | Held-out passes | Mean accuracy | Worst seed/task/length |',
             '|---|---:|---:|---:|']
    summaries = {}
    for arm in ARMS:
        rows = [indexed[seed, arm] for seed in SEEDS]
        means = [sum(r['scores'].values()) / 4 for r in rows]
        deltas = [value - sum(indexed[seed, ARMS[0]]['scores'].values()) / 4 for value, seed in zip(means, SEEDS)]
        delta = sum(deltas) / 5
        half = 2.776 * math.sqrt(sum((x - delta) ** 2 for x in deltas) / 20)
        summary = dict(passes=sum(r['passed'] for r in rows), mean=sum(means) / 5,
                       worst=min(min(r['scores'].values()) for r in rows), paired_delta=delta,
                       paired_interval=[delta - half, delta + half])
        summaries[arm] = summary
        lines.append(f"| {arm} | {summary['passes']}/5 | {summary['mean']:.2%} | {summary['worst']:.2%} |")
    lines += ['', '| Seed | Joint interleaved minimum | Ordered joint minimum | Retrieval-first minimum |', '|---:|---:|---:|---:|']
    for seed in SEEDS:
        lines.append(f'| {seed} | ' + ' | '.join(f"{min(indexed[seed, arm]['scores'].values()):.2%}" for arm in ARMS) + ' |')
    lines += ['', 'Paired changes in mean accuracy and descriptive 95% intervals across five seeds.']
    for arm in ARMS[1:]:
        s = summaries[arm]
        lo, hi = s['paired_interval']
        lines.append(f"- {arm} {100*s['paired_delta']:+.2f} pp [{100*lo:+.2f}, {100*hi:+.2f}].")
    lines += ['', '| Seed | Joint interleaved | Ordered joint | Retrieval-first |',
              '|---:|---:|---:|---:|']
    for seed in SEEDS:
        passes = [indexed[seed, arm]['first_validation_pass_tokens'] for arm in ARMS]
        lines.append(f'| {seed} | ' + ' | '.join(f'{tokens:,}' if tokens is not None else 'No measured pass' for tokens in passes) + ' |')
    lines += ['', 'The table shows tokens consumed at the first measured validation pass.']
    lines += ['', '| Seed | Retrieval-first single fact at 8 records | At 16 records |',
              '|---:|---:|---:|']
    for seed in SEEDS:
        switch = next(entry for entry in indexed[seed, 'retrieval_first']['validation_trace'] if entry['step'] == 600)
        lines.append(f"| {seed} | {switch['scores']['8-single']:.2%} | {switch['scores']['16-single']:.2%} |")
    lines += ['', 'Retrieval scores above are measured at the switch to joint training.']
    lines += ['', 'Retrieval-first changes the number and timing of composition-supervised updates together. The comparison tests the complete schedule. Validation traces and per-example held-out results are in `results.json`. Checkpoints at 600 and 1200 updates include optimizer state, batch-bank identity, and consumed batch counts. Run `python test_training_approaches.py` alongside the two imported experiment scripts to reproduce the study.']
    (OUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return summaries


def main():
    start = time.time()
    verify()
    atomic_json(OUT / 'protocol.json', dict(seeds=SEEDS, arms=ARMS, steps=STEPS, learning_rate=.002,
                retrieval_only_updates=600, validation_population=VALIDATION, test_population=TEST,
                validation_steps=[300, 600, 900, 1200], success_threshold=.95))
    results = []
    for seed in SEEDS:
        banks, digest = bank_for(seed)
        for arm in ARMS:
            results.append(train(seed, arm, banks, digest))
            atomic_json(OUT / 'progress.json', dict(completed=len(results), total=len(SEEDS)*len(ARMS)))
    summaries = report(results)
    atomic_json(OUT / 'results.json', dict(summaries=summaries, runs=results, seconds=time.time()-start))
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
