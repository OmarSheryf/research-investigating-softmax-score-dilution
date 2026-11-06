"""Compare full, distractor-masked, and compacted contexts on saved checkpoints."""
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from torch.nn import functional as F

from causal_transformer_experiment import D, H, Model, VALUES
from stabilize_causal_training import atomic_json
from test_composition_positions import CAPS, CHECKPOINTS, OUT as PREVIOUS, REGIONS, SEEDS, aggregate, make_examples, pct
from test_context_exposure import paired_interval

OUT = Path(__file__).resolve().parent / 'distractor-geometry'
CONDITIONS = ['full', 'masked', 'compact']
RETAINED = 64


def retained_indices(positions, seed):
    generator = torch.Generator().manual_seed(seed)
    priorities = torch.rand(len(positions), 128, generator=generator)
    priorities.scatter_(1, positions, -1.)
    records = priorities.argsort(dim=1)[:, :RETAINED].sort(dim=1).values
    assert (records[:, :, None] == positions[:, None, :]).any(dim=1).all()
    tail = torch.arange(256, 259).expand(len(positions), -1)
    return torch.cat((torch.stack((2*records, 2*records+1), dim=-1).flatten(1), tail), dim=1)


def rotate(x, positions, freq):
    phase = positions[:, None, :, None] * freq
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even*phase.cos()-odd*phase.sin(), even*phase.sin()+odd*phase.cos()), dim=-1).flatten(-2)


def positional_forward(model, tokens, positions, key_mask=None):
    hidden = model.embed(tokens)
    batch_size, length = tokens.shape
    for block in model.blocks:
        qkv = block.attn.qkv(block.n1(hidden)).view(batch_size, length, 3, H, D//H)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = rotate(q, positions, block.attn.freq)
        k = rotate(k, positions, block.attn.freq)
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask, is_causal=key_mask is None)
        hidden = hidden + block.attn.out(attention.transpose(1, 2).reshape(batch_size, length, D))
        hidden = hidden + block.ff(block.n2(hidden))
    queries = model.norm(hidden)[:, -2:]
    return model.read(queries), model.compose(queries.flatten(1))


def codes(reads, answer, truth):
    correct = reads.argmax(-1) == truth
    composed = answer.argmax(-1) == truth.sum(-1) % VALUES
    return (correct[:, 0].int() + 2*correct[:, 1].int() + 4*composed.int()).tolist()


@torch.inference_mode()
def evaluate(run):
    torch.set_num_threads(1)
    seed, cap = run['seed'], run['cap']
    state = torch.load(CHECKPOINTS / f'seed-{seed}' / f'cap{cap}.pt', weights_only=True, map_location='cpu')
    assert (state['seed'], state['length'], state['tokens']) == (seed, cap, 1112864)
    model = Model().eval()
    model.load_state_dict(state['model'])
    cells = []
    for index, cell in enumerate(run['cells']):
        a, b, rep = (cell[key] for key in ['first_region', 'second_region', 'replicate'])
        tokens, truth, positions = make_examples(a, b, rep)
        assert positions.tolist() == cell['positions']
        indices = retained_indices(positions, 92000000 + rep*100 + a*5 + b)
        kept = tokens.gather(1, indices)
        # Dropping masked rows is equivalent here because all layers mask their keys,
        # kept rows retain causal order, and RoPE uses the original position IDs.
        if index == 0:
            original = torch.arange(tokens.shape[1]).expand(2, -1)
            visible = torch.zeros_like(original, dtype=torch.bool).scatter_(1, indices[:2], True)
            mask = visible[:, None, None, :] & torch.ones(259, 259, dtype=torch.bool).tril()[None, None]
            dense = positional_forward(model, tokens[:2], original, mask)
            sparse = positional_forward(model, kept[:2], indices[:2])
            for got, expected in zip(sparse, dense):
                torch.testing.assert_close(got, expected, atol=2e-4, rtol=2e-4)
            reads, answer, _ = model(tokens[:32])
            assert codes(reads, answer, truth[:32]) == cell['codes'][:32]
        values = dict(full=cell['codes'], masked=[], compact=[])
        for inputs, original, labels in zip(kept.split(32), indices.split(32), truth.split(32)):
            reads, answer = positional_forward(model, inputs, original)
            values['masked'].extend(codes(reads, answer, labels))
            reads, answer, _ = model(inputs)
            values['compact'].extend(codes(reads, answer, labels))
        for condition, predictions in values.items():
            assert len(predictions) == len(truth)
            cells.append(dict(first_region=a, second_region=b, replicate=rep, condition=condition,
                              counts=[predictions.count(i) for i in range(8)], codes=predictions))
    result = dict(seed=seed, cap=cap, cells=cells)
    atomic_json(OUT / f'seed-{seed}-cap{cap}.json', result)
    print(f'Completed seed {seed}, cap {cap}', flush=True)
    return result


def select_cells(runs, condition, role=None, region=None):
    return [cell for run in runs for cell in run['cells'] if cell['condition'] == condition
            and (role is None or cell[f'{role}_region'] == region)]


def report(runs):
    assert {(r['seed'], r['cap']) for r in runs} == {(seed, cap) for seed in SEEDS for cap in CAPS}
    assert all(len(r['cells']) == 150 for r in runs)
    lines = ['# Distractor masking and positional compaction', '',
             'All ten checkpoints use the same 128-record contexts and crossed position grid as the previous diagnostic. Full shows all records. Masked retains 64 records, including both facts and 62 uniformly sampled distractors, with their original RoPE position IDs. Compact uses the same retained records and contiguous IDs. All conditions preserve the separator and query order.', '',
             'Masking changes the accessible evidence and contextual representations throughout the network. The contrast measures distractor removal at fixed positions. The compact-minus-masked contrast measures the effect of changing positional geometry with identical visible input.', '',
             '| Training cap | Condition | Fact 1 | Fact 2 | Both facts | Composition | Composition given both correct |',
             '|---:|---|---:|---:|---:|---:|---:|']
    summary = {}
    for cap in CAPS:
        pair = [r for r in runs if r['cap'] == cap]
        pooled = {c: aggregate(select_cells(pair, c)) for c in CONDITIONS}
        regions = {role: [{c: aggregate(select_cells(pair, c, role, p)) for c in CONDITIONS}
                          for p in range(5)] for role in ['first','second']}
        contrasts = {}
        for before, after in [('full','masked'),('masked','compact')]:
            differences = {}
            for metric in ['first','second','both','composition','composition_given_both']:
                values = [aggregate(select_cells([r], after))[metric] - aggregate(select_cells([r], before))[metric]
                          for r in pair]
                differences[metric] = dict(seed_deltas=values, interval=paired_interval(values))
            contrasts[f'{after}_minus_{before}'] = differences
        summary[cap] = dict(pooled=pooled, regions=regions, contrasts=contrasts)
        for condition, result in pooled.items():
            lines.append(f'| {cap} | {condition} | ' + ' | '.join(pct(result[k]) for k in ['first','second','both','composition','composition_given_both']) + ' |')
    lines += ['', '## Position breakdown', '',
              'Accuracy is averaged over the other fact’s five regions.', '',
              '| Training cap | Query | Record region | Full | Masked | Compact |', '|---:|---|---|---:|---:|---:|']
    for cap, result in summary.items():
        for role, regions in result['regions'].items():
            for (lo, hi), row in zip(REGIONS, regions):
                lines.append(f'| {cap} | {role} | {lo}–{hi-1} | ' + ' | '.join(pct(row[c][role]) for c in CONDITIONS) + ' |')
    lines += ['', '## Paired changes', '',
              '| Training cap | Contrast | Metric | Mean change, pp | Descriptive 95% seed interval, pp |',
              '|---:|---|---|---:|---:|']
    for cap, result in summary.items():
        for contrast, metrics in result['contrasts'].items():
            for metric, values in metrics.items():
                mean, lo, hi = values['interval']
                lines.append(f'| {cap} | {contrast} | {metric} | {100*mean:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |')
    lines += ['', 'Each conditional composition score is calculated on that condition’s correctly retrieved subset. The subsets can differ. This inference-only experiment retains facts using their known locations and samples one distractor subset per example. It estimates effects for this retention policy and the shared test populations.', '',
              'For each checkpoint, the sparse masked computation was checked against a dense layerwise key mask and saved full-context predictions were spot-checked by replay. Run `python test_distractor_geometry.py` with the prior results and checkpoints in place.']
    (OUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def main():
    previous = json.loads((PREVIOUS / 'results.json').read_text())
    atomic_json(OUT / 'protocol.json', dict(seeds=SEEDS, caps=CAPS, conditions=CONDITIONS, retained_records=RETAINED,
                regions=REGIONS, retention_seed_base=92000000))
    runs = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(evaluate, run) for run in previous['runs']]
        for job in as_completed(jobs):
            runs.append(job.result())
            atomic_json(OUT / 'progress.json', dict(completed=len(runs), total=len(jobs)))
    runs.sort(key=lambda r: (r['seed'],r['cap']))
    summary = report(runs)
    atomic_json(OUT / 'results.json', dict(summary=summary, runs=runs))
    print(json.dumps({cap: s['pooled'] for cap,s in summary.items()}), flush=True)


if __name__ == '__main__':
    main()
