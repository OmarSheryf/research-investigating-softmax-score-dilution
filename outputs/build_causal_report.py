"""Validate saved evaluations, compare the original run, and build reports."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'causal-run'

def main():
    d=json.loads((RUN/'results.json').read_text())
    original=json.loads((ROOT/'causal-original-results.json').read_text())
    raw=json.loads((RUN/'raw-evaluations.json').read_text())
    assert len(raw)==d['validated_records']
    index={tuple(r[k] for k in ['seed','step','n','position','replicate','dispersion','coefficient']):r for r in raw}
    assert len(index)==len(raw)
    for seed in d['protocol']['seeds']:
        for step in range(200,1201,200):
            assert (RUN/'checkpoints'/f'seed-{seed}-step-{step}.pt').is_file()
    for r in raw:
        for metric in ['single','two']:
            assert abs(sum(r['correct_'+metric])/128-r[metric])<1e-6
        if r['coefficient']==1 and r['n']<=16:
            base=index[tuple(r[k] if k!='coefficient' else 0. for k in ['seed','step','n','position','replicate','dispersion','coefficient'])]
            assert r['correct_single']==base['correct_single']
            assert r['correct_two']==base['correct_two']
    assert len(d['summary'])==len(original['summary'])
    max_diff=max(abs(a['interval'][0]-b['interval'][0]) for a,b in zip(d['summary'],original['summary']))
    validation=dict(raw_records=len(raw),checkpoint_count=30,max_original_summary_difference=max_diff,exact_summary_reproduction=max_diff==0)
    (RUN/'validation.json').write_text(json.dumps(validation,indent=2))
    def summary(n,metric,variant='baseline'):
        return next(r for r in d['summary'] if r['step']==1200 and r['n']==n and r['metric']==metric and r['variant']==variant)
    lines=['# Causal transformer experiment','',
    '## Saved run','',
    f"{d['protocol']['parameters']:,} parameters; 3 causal layers; 4 heads per layer; width 64; RoPE base 10000. No imposed position penalty. Five seeds (11, 22, 33, 44, 55), each trained for 1,200 updates with batch size 32. Training contexts cycle through 4, 8, and 16 key/value records.",'',
    'Each record uses two tokens. A separator and two query tokens follow the records, giving 2N+3 input tokens. A read classifier predicts each queried value; a neural MLP predicts the sum modulo 8 from the two contextual query states. Labels supervise composition, but retrieved values or a fixed addition rule do not enter its input. This is a synthetic supervised transformer, not a pretrained autoregressive language model.','',
    'Loss is read cross-entropy plus composition cross-entropy. AdamW learning rate 0.002; gradient clipping 1.0. Keys are unique within each context; values have eight possible classes. Chance answer accuracy is 12.5%.','',
    '## Results across all five seeds','',
    '| Records | Input tokens | Single fact | Two-fact composition | Scaled single fact | Scaled composition |',
    '|---:|---:|---:|---:|---:|---:|']
    for n in [8,16,32,64]:
        vals=[summary(n,m,v)['interval'][0] for v in ['baseline','scale'] for m in ['single','two']]
        lines.append(f'| {n} | {2*n+3} | '+' | '.join(f'{100*x:.2f}%' for x in vals)+' |')
    lines+=['','## Training stability and length generalization','',
    'Seeds 11, 22, and 55 learned both tasks well at training lengths. Seeds 33 and 44 did not learn reliably within the fixed budget. Aggregate scores mix optimization failures and length generalization failures.','',
    'The following subset is selected after observing task acquisition and is descriptive, not a prespecified confirmatory result.','',
    '| Records | Single fact, three learned seeds | Composition, three learned seeds |',
    '|---:|---:|---:|']
    for n in [8,16,32,64]:
        vals=[sum(summary(n,m)['seed_values'][i] for i in [0,1,4])/3 for m in ['single','two']]
        lines.append(f'| {n} | '+' | '.join(f'{100*x:.2f}%' for x in vals)+' |')
    lines+=['','## Paired scaling effects','',
    'Scaling multiplies causal attention logits by 1 + max(0, log(candidate_count)/log(35) - 1). It operates in every head and layer, with each row using only its available candidate count. It is inactive within training lengths. Each intervention uses identical checkpoint weights and identical examples. Unlike rescaling a fixed score vector, scaling in multiple layers can change later representations and rankings.','',
    '| Records | Task | Mean change, percentage points | Descriptive 95% seed interval |',
    '|---:|---|---:|---|']
    for r in d['paired']:
        if r['step']==1200 and r['variant']=='scale' and r['metric'] in ['single','two'] and r['n']>=32:
            mean,lo,hi=r['interval']
            lines.append(f"| {r['n']} | {r['metric']} | {100*mean:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] |")
    lines+=['',
    'Independent validation chooses among coefficients 0, 0.5, and 1, scoring the mean of retrieval and composition accuracy at 16/32 records. Four seeds selected 0, and seed 22 selected 0.5. This provides no consistent evidence of a useful scaling benefit.','',
    '## Position sensitivity','',
    '| Evidence position | Single fact, 16 records | Single fact, 32 records | Single fact, 64 records |',
    '|---|---:|---:|---:|']
    for p,label in zip([0,.25,.5,.75,1],['Beginning','Quarter','Middle','Three quarters','End']):
        vals=[next(r['interval'][0] for r in d['position_summary'] if r['step']==1200 and r['n']==n and r['position']==p and r['metric']=='single' and r['coefficient']==0) for n in [16,32,64]]
        lines.append('| '+label+' | '+' | '.join(f'{100*x:.2f}%' for x in vals)+' |')
    lines+=['',
    'No consistent U-shaped pattern appeared. At 32 records the middle was best on the average single-fact score. Long-context degradation can therefore appear here without the classic lost-in-the-middle curve. This experiment does not establish the internal cause of that degradation.','',
    '## Measurement and limitations','',
    'Evaluation uses five positions, two independently generated test populations, and 128 examples per cell. Near/far evidence separation is tested at 16/64 records. Intervals are descriptive Student-t intervals over five seed means after averaging test populations and positions. They are not multiplicity-corrected and do not capture all new-test-set uncertainty. Raw normal-approximation intervals may exceed [0,1]; accuracy intervals in the explorer are clipped for display only.','',
    'Last-layer, head-averaged attention mass and entropy are diagnostics, not causal explanations. The composition head is a task-specific neural readout of two query states. This small experiment cannot establish mechanisms or remedies in Claude or another production model.','',
    '## Conclusions','',
    '1. Length generalization failure appeared without an imposed positional penalty, including in seeds that learned the short-context tasks.',
    '2. This length-aware attention scaling rule did not recover performance.',
    '3. A consistent U-shaped positional failure was not required for degradation.',
    '4. Training instability is a separate bottleneck and must be resolved before strong architecture comparisons.','',
    '## Prioritized next steps','',
    '1. Stabilize task acquisition across seeds using validation-selected optimization settings or a training curriculum; retain unsuccessful seeds in reporting.',
    '2. Compare short-context and longer-context training under matched training-token budgets. Include tests both within and beyond each training range.',
    '3. Separate positional-range extrapolation from distractor competition with controlled position/length interventions and genuine causal ablations.',
    '4. Only then compare positional encodings, selective retrieval/routing, and attention normalizers on unseen test sets.','',
    '## Reproduction and checkpoints','',
    'Run from the parent outputs directory: `python causal_transformer_experiment.py`. This retrains all seeds and writes into `causal-run/`.',
    'Resume a checkpoint: `python causal_transformer_experiment.py --resume causal-run/checkpoints/seed-11-step-1200.pt --steps 1400`. This continues that seed; it does not rebuild the full evaluation report.',
    'Each checkpoint includes weights, AdamW optimizer state, data generator state, torch RNG state, seed, and step. Checkpoints are saved every 200 updates. Existing files for matching seed/step are replaced when rerunning; copy a run directory before preserving an independent variant.','',
    f"Validation: {len(raw):,} unique evaluation records; 30 checkpoint files; maximum difference from the original retained aggregate results: {max_diff:.8f}. All causal-masking and diagnostic-equivalence preflight checks passed."]
    (RUN/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    visual=dict(protocol=d['protocol'],training=d['training'],selection=d['selection'],rows=[{k:r[k] for k in ['seed','step','n','position','replicate','dispersion','coefficient','single','two','both_reads','mass','entropy']} for r in raw])
    (RUN/'explorer-data.json').write_text(json.dumps(visual,separators=(',',':')),encoding='utf-8')
    print(json.dumps(validation))

if __name__=='__main__':main()
