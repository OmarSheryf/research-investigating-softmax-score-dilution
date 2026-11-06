"""Cross batch order with a fixed retrieval-first schedule on five fresh seeds."""
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
import test_training_approaches as experiment

OUT = Path(__file__).resolve().parent / 'training-factorial'
SEEDS = [171, 182, 193, 204, 215]
ARMS = ['joint_interleaved', 'joint_curriculum', 'retrieval_first', 'retrieval_first_curriculum']
VALIDATION, TEST = 51000000, 61000000


def run(seed, arm):
    torch.set_num_threads(1)
    experiment.OUT = OUT
    experiment.VALIDATION = VALIDATION
    experiment.TEST = TEST
    banks, digest = experiment.bank_for(seed)
    return experiment.train(seed, arm, banks, digest)


def interval(values):
    mean = sum(values) / 5
    half = 2.776 * math.sqrt(sum((x - mean) ** 2 for x in values) / 20)
    return [mean, mean - half, mean + half]


def report(runs):
    rows = {(r['seed'], r['arm']): r for r in runs}
    assert len(rows) == 20
    for seed in SEEDS:
        assert len({rows[seed, arm]['bank_hash'] for arm in ARMS}) == 1
    for r in runs:
        assert r['tokens'] == 832000 and len(r['test']) == 20
        assert r['passed'] == (min(r['scores'].values()) >= .95)
        for cell in r['test']:
            for metric in ('single', 'two'):
                answers = cell['correct_' + metric]
                assert len(answers) == 128 and sum(answers) / 128 == cell[metric]
        for step in (600, 1200):
            checkpoint = OUT / f"{r['seed']}-{r['arm']}" / f'step-{step}.pt'
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert state['bank_hash'] == r['bank_hash'] and state['step'] == step
            assert sum(state['used'].values()) == step
    averages = {arm: [sum(rows[seed, arm]['scores'].values()) / 4 for seed in SEEDS] for arm in ARMS}
    a, b, c, d = (averages[arm] for arm in ARMS)
    contrasts = dict(order_under_joint=interval([y-x for x,y in zip(a,b)]),
                     order_under_retrieval_first=interval([y-x for x,y in zip(c,d)]),
                     retrieval_first_under_interleaving=interval([y-x for x,y in zip(a,c)]),
                     retrieval_first_under_curriculum=interval([y-x for x,y in zip(b,d)]),
                     interaction=interval([w-z-y+x for x,y,z,w in zip(a,b,c,d)]))
    summaries = {arm: dict(passes=sum(rows[seed, arm]['passed'] for seed in SEEDS),
                           mean=sum(averages[arm])/5) for arm in ARMS}
    lines = ['# Batch order and loss timing factorial', '',
             'Five new paired seeds, 1,200 updates per run, and 832,000 input tokens. All four recipes share initial weights and the same 400 batches per length within a seed. AdamW uses a constant 0.002 learning rate. Both retrieval-first arms activate composition loss at update 601.', '',
             'Pass requires at least 95% on both tasks at both 8 and 16 records. Validation and held-out populations use separate generator seeds.', '',
             '| Recipe | Passes | Mean held-out accuracy |', '|---|---:|---:|']
    for arm, summary in summaries.items():
        lines.append(f"| {arm} | {summary['passes']}/5 | {summary['mean']:.2%} |")
    lines += ['', '| Seed | Interleaved joint | Ordered joint | Interleaved retrieval-first | Ordered retrieval-first |', '|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        lines.append(f'| {seed} | ' + ' | '.join(f"{min(rows[seed, arm]['scores'].values()):.2%}" for arm in ARMS) + ' |')
    lines += ['', 'The seed table shows each run’s minimum required held-out score.', '',
              '| Contrast | Change in mean accuracy, pp | Descriptive 95% seed interval, pp |', '|---|---:|---:|']
    for name, (mean, lo, hi) in contrasts.items():
        lines.append(f'| {name} | {100*mean:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |')
    lines += ['', 'The interaction is the change in the curriculum effect when switching from joint training to retrieval-first. The loss schedule changes both composition timing and the number of composition-supervised updates. At the common switch, the ordered arm has seen 400 length-4 and 200 length-8 batches, while the interleaved arm has seen 200 at each length. Consequently, the interaction includes differences in which lengths receive early composition supervision.', '',
              'All 40 checkpoints loaded and matched their run’s batch-bank identifier. The raw results include validation traces and per-example test correctness. Run `python test_training_factorial.py` alongside the imported experiment scripts.']
    (OUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return dict(arms=summaries, contrasts=contrasts)


def main():
    torch.set_num_threads(1)
    experiment.verify()
    experiment.atomic_json(OUT / 'protocol.json', dict(seeds=SEEDS, arms=ARMS, updates=1200,
                           input_tokens=832000, composition_switch=601, validation_population=VALIDATION,
                           test_population=TEST, pass_threshold=.95))
    runs = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(run, seed, arm) for seed in SEEDS for arm in ARMS]
        for job in as_completed(jobs):
            runs.append(job.result())
            experiment.atomic_json(OUT / 'progress.json', dict(completed=len(runs), total=len(jobs)))
    summaries = report(runs)
    experiment.atomic_json(OUT / 'results.json', dict(summary=summaries, runs=runs))
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
