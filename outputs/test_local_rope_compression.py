"""Test distance-dependent RoPE compression with full causal visibility."""
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from causal_transformer_experiment import D, H, Model
from test_composition_positions import CAPS, CHECKPOINTS, REGIONS, SEEDS, make_examples, pct
from test_context_exposure import paired_interval
from test_distractor_geometry import atomic_json, codes, positional_forward, rotate
from test_rope_compression import OUT as PREVIOUS, scores

OUT = Path(__file__).resolve().parent / 'local-rope-compression'
WINDOW = 16
SCALE = .5
CONDITIONS = ['full', 'full_scaled', 'local_preserved']
METRICS = ['first', 'second', 'both', 'composition', 'composition_given_both']


def forward(model, tokens, window=WINDOW, scale=SCALE):
    hidden = model.embed(tokens)
    batch_size, length = tokens.shape
    ids = torch.arange(length, device=tokens.device)
    distance = ids[:, None] - ids[None, :]
    original = ids.expand(batch_size, -1)
    scaled = original * scale
    for block in model.blocks:
        qkv = block.attn.qkv(block.n1(hidden)).view(batch_size, length, 3, H, D//H)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        freq = block.attn.freq
        near = rotate(q, original, freq) @ rotate(k, original, freq).transpose(-1,-2)
        # The query offset makes the far-region distance continuous at the window edge.
        far = rotate(q, scaled + (1-scale)*window, freq) @ rotate(k, scaled, freq).transpose(-1,-2)
        logits = torch.where(distance > window, far, near) / ((D//H)**.5)
        attention = logits.masked_fill(distance < 0, float('-inf')).softmax(-1) @ v
        hidden = hidden + block.attn.out(attention.transpose(1,2).reshape(batch_size,length,D))
        hidden = hidden + block.ff(block.n2(hidden))
    queries = model.norm(hidden)[:, -2:]
    return model.read(queries), model.compose(queries.flatten(1))


@torch.inference_mode()
def evaluate(previous):
    torch.set_num_threads(1)
    seed, cap = previous['seed'], previous['cap']
    checkpoint = torch.load(CHECKPOINTS / f'seed-{seed}' / f'cap{cap}.pt', weights_only=True, map_location='cpu')
    assert (checkpoint['seed'], checkpoint['length'], checkpoint['tokens']) == (seed, cap, 1112864)
    model = Model().eval()
    model.load_state_dict(checkpoint['model'])
    cells = [c for c in previous['cells'] if c['condition'] in CONDITIONS[:2]]
    reference = {(c['first_region'], c['second_region'], c['replicate'], c['condition']): c for c in cells}
    assert len(reference) == len(cells) == 100
    for a in range(5):
        for b in range(5):
            for rep in range(2):
                tokens, truth, _ = make_examples(a,b,rep)
                if (a,b,rep) == (0,0,0):
                    sample = tokens[:4]
                    ids = torch.arange(sample.shape[1]).expand(len(sample), -1)
                    for window, factor in [(0,SCALE), (sample.shape[1],1.)]:
                        actual = forward(model, sample, window)
                        expected = positional_forward(model, sample, ids*factor)
                        for got, want in zip(actual,expected):
                            torch.testing.assert_close(got,want,atol=3e-4,rtol=3e-4)
                    for condition, factor in [('full',1.), ('full_scaled',SCALE)]:
                        reads, answer = positional_forward(model,tokens[:32],ids[:1].expand(32,-1)*factor)
                        assert codes(reads,answer,truth[:32]) == reference[a,b,rep,condition]['codes'][:32]
                predictions = []
                for chunk, labels in zip(tokens.split(16), truth.split(16)):
                    reads, answer = forward(model,chunk)
                    predictions.extend(codes(reads,answer,labels))
                cells.append(dict(first_region=a, second_region=b, replicate=rep, condition='local_preserved',
                                  codes=predictions, counts=[predictions.count(i) for i in range(8)]))
    result = dict(seed=seed,cap=cap,cells=cells)
    atomic_json(OUT / f'seed-{seed}-cap{cap}.json',result)
    print(f'Completed seed {seed}, cap {cap}',flush=True)
    return result


def report(runs):
    assert len(runs) == 10 and {(r['seed'],r['cap']) for r in runs} == {(s,c) for s in SEEDS for c in CAPS}
    expected = {(a,b,p,c) for a in range(5) for b in range(5) for p in range(2) for c in CONDITIONS}
    for run in runs:
        assert len(run['cells']) == len(expected)
        assert {(c['first_region'],c['second_region'],c['replicate'],c['condition']) for c in run['cells']} == expected
        for cell in run['cells']:
            assert len(cell['codes']) == 64 and cell['counts'] == [cell['codes'].count(i) for i in range(8)]
    lines = ['# Preserve local RoPE distances and halve the excess', '',
             'All 128 records remain visible. At each layer and head, causal query–key separations through 16 tokens use ordinary RoPE. A larger separation d uses 16 + 0.5 × (d − 16). The fixed rule is evaluated on the same checkpoints and crossed position grid as ordinary and uniform half-scale RoPE.', '',
             'The 16-token threshold was chosen in advance. The primary metrics are composition and retrieval when the queried fact lies in the earliest region, records 0–25. Five seeds per cap share all held-out examples.', '',
             '| Cap | Condition | First fact | Second fact | Both | Composition | Composition given both correct |',
             '|---:|---|---:|---:|---:|---:|---:|']
    summary = {}
    for cap in CAPS:
        group = [r for r in runs if r['cap'] == cap]
        pooled = {c: scores(group,c) for c in CONDITIONS}
        positions = {role: [{c: scores(group,c,role,p)[role] for c in CONDITIONS} for p in range(5)]
                     for role in ['first','second']}
        contrasts = {}
        for baseline in CONDITIONS[:2]:
            values = {metric: paired_interval([scores([r],'local_preserved')[metric]-scores([r],baseline)[metric]
                                              for r in group]) for metric in METRICS}
            for role in ['first','second']:
                values[f'early_{role}'] = paired_interval([scores([r],'local_preserved',role,0)[role]
                                                           -scores([r],baseline,role,0)[role] for r in group])
            contrasts[baseline] = values
        summary[cap] = dict(pooled=pooled,positions=positions,contrasts=contrasts)
        for c,result in pooled.items():
            lines.append(f'| {cap} | {c} | ' + ' | '.join(pct(result[k]) for k in METRICS) + ' |')
    lines += ['', '## Retrieval by evidence region', '',
              '| Cap | Query | Region | Ordinary | Uniform 0.5 | Local preserved |', '|---:|---|---|---:|---:|---:|']
    for cap,result in summary.items():
        for role,rows in result['positions'].items():
            for (lo,hi),row in zip(REGIONS,rows):
                lines.append(f'| {cap} | {role} | {lo}–{hi-1} | ' + ' | '.join(pct(row[c]) for c in CONDITIONS) + ' |')
    lines += ['', '## Paired changes from each baseline', '',
              '| Cap | Baseline | Metric | Change, pp | Descriptive 95% seed interval, pp |', '|---:|---|---|---:|---:|']
    for cap,result in summary.items():
        for baseline,metrics in result['contrasts'].items():
            for metric,(mean,lo,hi) in metrics.items():
                lines.append(f'| {cap} | {baseline} | {metric} | {100*mean:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |')
    lines += ['', 'The position-dependent score path matched uniform scaling at window zero and ordinary RoPE when the window covered the full input. Both saved reference conditions were replayed on a sample from each checkpoint. The reported intervals describe variation across five seeds on the shared held-out examples.', '',
              'Run `python test_local_rope_compression.py` with the preceding experiment results and checkpoints in place.']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return summary


def main():
    prior = json.loads((PREVIOUS/'results.json').read_text())
    atomic_json(OUT/'protocol.json', dict(seeds=SEEDS,caps=CAPS,window=WINDOW,scale=SCALE,records=128,
                conditions=CONDITIONS,primary=['composition','early_first','early_second']))
    runs = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(evaluate,run) for run in prior['runs']]
        for job in as_completed(jobs):
            runs.append(job.result())
            atomic_json(OUT/'progress.json',dict(completed=len(runs),total=len(jobs)))
    runs.sort(key=lambda r:(r['seed'],r['cap']))
    summary = report(runs)
    atomic_json(OUT/'results.json',dict(summary=summary,runs=runs))
    print(json.dumps({cap:{c:r['composition'] for c,r in result['pooled'].items()} for cap,result in summary.items()}),flush=True)


if __name__ == '__main__':
    main()
