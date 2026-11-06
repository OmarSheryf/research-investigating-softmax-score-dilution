"""Validate stabilization results and report paired fresh-seed outcomes."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'stability-run'

def wilson(success,total):
    z=1.95996398454
    p=success/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [center-half,center+half]

def t_interval(xs):
    mean=sum(xs)/len(xs)
    half=2.776*math.sqrt(sum((x-mean)**2 for x in xs)/4/5)
    return [mean,mean-half,mean+half]

def main():
    d=json.loads((OUT/'results.json').read_text())
    p=d['protocol'];winner=d['selection']['winner']
    assert not set(p['development_seeds'])&set(p['confirmation_seeds'])
    assert hashlib.sha256((ROOT/'stabilize_causal_training.py').read_bytes()).hexdigest()==p['source_sha256']
    assert hashlib.sha256((ROOT/'causal_transformer_experiment.py').read_bytes()).hexdigest()==p['model_source_sha256']
    assert json.loads((OUT/'selection-frozen.json').read_text())==d['selection']
    settings=list(dict.fromkeys(['baseline',winner]))
    assert len(d['development'])==9
    assert len(d['confirmation'])==5*len(settings)
    checkpoint_count=0;cells=0
    for r in d['development']+d['confirmation']:
        assert r['tokens_seen']==832000 and r['steps']==1200
        assert r['acquired']==all(v>=.95 for v in r['scores'].values())
        keys=set()
        for row in r['evaluation']:
            key=tuple(row[k] for k in ['n','position','replicate'])
            assert key not in keys;keys.add(key)
            for metric in ['single','two']:
                assert len(row['correct_'+metric])==128
                assert abs(sum(row['correct_'+metric])/128-row[metric])<1e-9
                assert math.isfinite(row[metric]) and 0<=row[metric]<=1
            cells+=1
        expected=20 if r['stage']=='development' else 40
        assert len(keys)==expected
        directory=OUT/r['stage']/f"{r['setting']}-seed-{r['seed']}"
        steps=[1200] if r['reused_baseline_checkpoint'] else [400,800,1200]
        for step in steps:
            assert (directory/f'step-{step}.pt').is_file();checkpoint_count+=1
    result={}
    for setting in settings:
        rs=[r for r in d['confirmation'] if r['setting']==setting]
        acquired=sum(r['acquired'] for r in rs)
        result[setting]=dict(success_count=acquired,total=5,wilson95=wilson(acquired,5),mean_score=sum(r['mean_score'] for r in rs)/5,worst_score=min(r['minimum_score'] for r in rs))
    indexed={(r['setting'],r['seed']):r for r in d['confirmation']}
    deltas=[indexed[(winner,s)]['mean_score']-indexed[('baseline',s)]['mean_score'] for s in p['confirmation_seeds']]
    improved=sum(indexed[(winner,s)]['acquired'] and not indexed[('baseline',s)]['acquired'] for s in p['confirmation_seeds'])
    regressed=sum(indexed[('baseline',s)]['acquired'] and not indexed[(winner,s)]['acquired'] for s in p['confirmation_seeds'])
    discordant=improved+regressed
    exact_p=min(1,2*sum(math.comb(discordant,k) for k in range(min(improved,regressed)+1))/2**discordant) if discordant else 1.
    summary=dict(winner=winner,confirmation=result,paired_mean_accuracy_change=t_interval(deltas),paired_success_improvements=improved,paired_success_regressions=regressed,exact_mcnemar_p=exact_p,evaluation_cells=cells,checkpoint_count=checkpoint_count)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Training stabilization experiment','',
    '## Design','',
    'The architecture, loss, positional encoding and attention mechanism are unchanged. The goal is reliable acquisition of both synthetic tasks, before interpreting context-length failures. All runs receive 1,200 updates, 38,400 examples, and exactly 832,000 input tokens. No imposed positional penalty or attention scaling is used.','',
    'Three development seeds (11, 33, 44) include previously observed successes/failures. Three recipes are screened:','',
    '- Baseline: constant learning rate 0.002; lengths 4/8/16 alternate each update.',
    '- Warmup/cosine: same length schedule, 100-update linear warmup to 0.002, then cosine decay to 0.0002.',
    '- Curriculum: the warmup/cosine schedule, with 400 updates at length 4, then 400 at length 8, then 400 at length 16. Length counts and total tokens match the baseline.','',
    'Selection maximizes development success count, then the worst seed/task/length score, then mean accuracy. The chosen recipe is frozen before five fresh training seeds (61, 72, 83, 94, 105) are trained and evaluated. Confirmation compares baseline and selected recipe on the same seeds and new test populations.','',
    'A seed passes only if BOTH single-fact and learned composition accuracy reach 95% at BOTH 8 and 16 records. Each score averages five evidence positions, two populations, and 128 examples per cell. This criterion does not require every individual position to pass 95%.','',
    '## Development selection','',
    '| Recipe | Seeds passing / 3 | Worst score | Mean score |','|---|---:|---:|---:|']
    for r in d['selection']['rankings']:
        lines.append(f"| {r['setting']} | {r['success_count']} | {100*r['worst_score']:.2f}% | {100*r['mean_score']:.2f}% |")
    lines+=['',f'Selected recipe: **{winner}**.','',
    '## Fresh-seed confirmation','',
    '| Recipe | Seeds passing / 5 | 95% Wilson interval for pass rate | Mean in-domain accuracy | Worst seed/task/length score |',
    '|---|---:|---|---:|---:|']
    for setting,r in result.items():
        lo,hi=r['wilson95']
        lines.append(f"| {setting} | {r['success_count']} | [{100*lo:.1f}%, {100*hi:.1f}%] | {100*r['mean_score']:.2f}% | {100*r['worst_score']:.2f}% |")
    mean,lo,hi=summary['paired_mean_accuracy_change']
    lines+=['',f'Paired change in mean in-domain accuracy: {100*mean:+.2f} percentage points; descriptive 95% Student-t interval across five seeds [{100*lo:+.2f}, {100*hi:+.2f}].',
    f'Paired pass/fail changes: {improved} improvements, {regressed} regressions. Exact two-sided McNemar p-value: {exact_p:.4f}. Five seeds provide limited evidence about the population failure rate.','',
    '| Seed | Baseline pass | Selected pass | Baseline minimum | Selected minimum |',
    '|---:|---|---|---:|---:|']
    for seed in p['confirmation_seeds']:
        a=indexed[('baseline',seed)];b=indexed[(winner,seed)]
        lines.append(f"| {seed} | {a['acquired']} | {b['acquired']} | {100*a['minimum_score']:.2f}% | {100*b['minimum_score']:.2f}% |")
    selected_passes=result[winner]['success_count']
    baseline_passes=result['baseline']['success_count']
    if selected_passes==5:
        conclusion='The selected recipe passed on every sampled fresh seed. This is promising evidence within this small experiment, not a guarantee of reliability across seeds.'
    elif selected_passes>baseline_passes:
        conclusion='The selected recipe improved the observed acquisition count, but failures persist. Training stabilization remains incomplete.'
    elif selected_passes==baseline_passes:
        conclusion='The selected recipe did not improve the observed acquisition count on fresh seeds. The development result did not establish a reliable stabilization fix.'
    else:
        conclusion='The selected recipe reduced the observed acquisition count on fresh seeds. It should not replace the baseline as a stabilization fix.'
    lines+=['',conclusion]
    lines+=['','## Secondary length-generalization results','',
    'Only 8/16-record results determine acquisition success. Longer lengths are secondary held-out diagnostics.','',
    '| Recipe | Records | Single fact | Learned composition |','|---|---:|---:|---:|']
    for setting in settings:
        for n in [8,16,32,64]:
            rs=[row for r in d['confirmation'] if r['setting']==setting for row in r['evaluation'] if row['n']==n]
            vals=[sum(r[k] for r in rs)/len(rs) for k in ['single','two']]
            lines.append(f'| {setting} | {n} | {100*vals[0]:.2f}% | {100*vals[1]:.2f}% |')
    lines+=['','## Interpretation and limits','',
    'This tests a selected training recipe, not a production-model fix. The curriculum recipe changes both learning-rate scheduling and the temporal ordering of lengths relative to baseline; the development screen helps distinguish these, but fresh-seed confirmation does not independently identify each component.','',
    'Development seeds were intentionally chosen from prior observations. Confirmation seeds and test populations were unused for selection. Test examples are shared across training seeds to support pairing. Intervals over seeds do not represent all test-population uncertainty, and secondary comparisons are exploratory.','',
    'Each training seed controls both parameter initialization and the training-data generator. Recipe pairs share initialization, but the curriculum changes generator consumption order, so the exact training example sets are not identical. Context-length counts, example counts, and token budgets are identical. A future curriculum-isolation test should replay the same pregenerated per-length examples in different orders.','',
    '## Next decision','',
    'If the recipe improves acquisition, confirm on more fresh seeds before treating it as reliable, then compare short- and longer-context training under matched token budgets. If failures persist, prioritize optimization/task acquisition rather than assigning failures to attention dilution.','',
    'Concrete follow-ups: complete the missing constant-learning-rate plus length-curriculum condition using controlled example replay; then compare retrieval-only pretraining against joint retrieval/composition training, tracking tokens-to-acquisition on validation data. These are hypotheses to test, not established causes. Defer strong long-context mechanism claims until the short-context control learns reliably.','',
    '## Files and reproduction','',
    '`protocol.json` records the prespecified rule and source hashes. `selection-frozen.json` records development-only selection. `results.json` contains all evaluation correctness vectors. Per-run directories contain checkpoints with model, optimizer, generator and torch RNG states.','',
    'Run `python stabilize_causal_training.py`, then `python analyze_stability.py`. Completed per-run results are reused; use a separate copy/run directory for a new protocol. Development baseline reuse requires the original seed-11/33/44 update-1200 checkpoints under `causal-run/checkpoints/`.','',
    'To resume an intermediate recipe checkpoint, run `python resume_stability.py CHECKPOINT --until 1200 --output resumed.pt`. The original length and learning-rate schedules are preserved. This does not rebuild evaluation summaries.','',
    f'Validated {cells} evaluation cells and {checkpoint_count} checkpoints; all training budgets match exactly.']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
