"""Validate results and produce a report, paired comparisons, and sampled boundaries."""
import json
import math
from pathlib import Path

OUT=Path(__file__).resolve().parent
d=json.loads((OUT/'robustness-results.json').read_text())
records=d['records']
index={tuple(r[k] for k in ['seed','regime','step','n','position','difficulty','variant','task']):r for r in records}
assert len(index)==len(records)==5*2*2*4*3*3*4*2
for r in records:
    assert len(r['correct'])==128
    assert abs(sum(r['correct'])/128-r['accuracy'])<1e-7
    assert all(math.isfinite(r[k]) for k in ['accuracy','retrieval','mass','entropy','read_loss'])
    assert all(0<=r[k]<=1.00001 for k in ['accuracy','retrieval','mass','entropy','both_values_correct'])
    if r['task']=='two':
        assert r['both_values_correct']<=r['accuracy']+1e-7
    if r['variant'] in ['scale','combined']:
        other='baseline' if r['variant']=='scale' else 'position'
        key=tuple(r[k] if k!='variant' else other for k in ['seed','regime','step','n','position','difficulty','variant','task'])
        assert r['retrieval']==index[key]['retrieval']
for s in d['protocol']['seeds']:
    for regime in ['penalty','no_penalty']:
        ls=[x for x in d['training_losses'] if x['seed']==s and x['regime']==regime]
        assert ls[-1]['loss']<ls[0]['loss']

def ci(xs):
    mean=sum(xs)/len(xs)
    half=2.776*(sum((x-mean)**2 for x in xs)/4/5)**.5
    return mean,mean-half,mean+half

paired=[]
for regime in ['penalty','no_penalty']:
    for step in [200,600]:
        for n in [16,64,256,1024]:
            for difficulty in ['random','similar','near_duplicate']:
                for task in ['one','two']:
                    for variant in ['scale','position','combined']:
                        deltas=[]; wins=losses=0
                        for seed in d['protocol']['seeds']:
                            diffs=[]
                            for pos in ['beginning','middle','end']:
                                a=index[(seed,regime,step,n,pos,difficulty,'baseline',task)]
                                b=index[(seed,regime,step,n,pos,difficulty,variant,task)]
                                diffs.append(b['accuracy']-a['accuracy'])
                                wins+=sum(y>x for x,y in zip(a['correct'],b['correct']))
                                losses+=sum(y<x for x,y in zip(a['correct'],b['correct']))
                            deltas.append(sum(diffs)/3)
                        m,lo,hi=ci(deltas)
                        paired.append(dict(regime=regime,step=step,n=n,difficulty=difficulty,task=task,variant=variant,delta=m,ci95=[lo,hi],seed_deltas=deltas,paired_wins=wins,paired_losses=losses))
(OUT/'paired-comparisons.json').write_text(json.dumps(paired,indent=2))
def get(regime,n,diff,task,variant):
    return next(x for x in d['summary'] if x['regime']==regime and x['step']==600 and x['n']==n and x['difficulty']==diff and x['task']==task and x['variant']==variant)
lines=['# Controlled attention robustness experiment','',
'## Protocol','',
'Five training seeds; two training regimes (imposed middle penalty 2 or 0); 600 optimizer updates; training length 16; 32 value classes. Shared-weight reader answers two independently supervised queries. The two-fact answer is their fixed sum modulo 32, not learned multi-hop reasoning. Training uses random distractors; confusable distractors are an evaluation distribution shift. Query is a noisy copy of its target key, not semantic paraphrase.','',
'Four evaluations use identical checkpoint weights and held-out examples: penalty 2 / scale 1 (baseline), penalty 2 / length scaling, penalty 0 / scale 1 (position), penalty 0 / length scaling (combined). Scaling is max(1, log(n)/log(16)). Thus baseline always includes the imposed penalty, even for the model trained without it.','',
'128 examples per length × position × difficulty cell, shared across all seeds and interventions. Three positions; four lengths; three difficulty levels. Similar and near-duplicate conditions insert four distractors per target at uniformly sampled non-target positions, with Gaussian noise SD 0.6 and 0.15 before normalization. Targets receive query noise SD 0.05.','',
'Intervals are descriptive 95% Student-t intervals across five training-seed means, averaged over three positions. They capture training variability on this fixed test set, not new-test-set uncertainty; comparisons are exploratory and not corrected for multiplicity.','',
'## Final accuracy: model trained with penalty','',
'| Records | Distractors | Task | Baseline | Scaling | Position | Combined |',
'|---:|---|---|---:|---:|---:|---:|']
for n in [16,64,256,1024]:
    for diff in ['random','similar','near_duplicate']:
        for task in ['one','two']:
            vals=[get('penalty',n,diff,task,v)['accuracy'] for v in ['baseline','scale','position','combined']]
            lines.append(f'| {n} | {diff} | {task} | '+' | '.join(f'{100*x:.1f}%' for x in vals)+' |')
lines+=['','## Paired effects at 1024 records, two-fact task','',
'| Training regime | Distractors | Intervention | Accuracy difference (percentage points) | 95% interval |',
'|---|---|---|---:|---|']
for p in paired:
    if p['step']==600 and p['n']==1024 and p['task']=='two':
        lines.append(f"| {p['regime']} | {p['difficulty']} | {p['variant']} | {100*p['delta']:+.2f} | [{100*p['ci95'][0]:+.2f}, {100*p['ci95'][1]:+.2f}] |")
lines+=['','## Exact two-value accuracy at 1024 records','',
'Model trained with penalty; requires both decoded values to be correct, excluding accidental modular-sum successes.','',
'| Distractors | Baseline | Scaling | Position | Combined |','|---|---:|---:|---:|---:|']
for diff in ['random','similar','near_duplicate']:
    cells=[]
    for v in ['baseline','scale','position','combined']:
        rs=[r for r in records if r['regime']=='penalty' and r['step']==600 and r['n']==1024 and r['task']=='two' and r['difficulty']==diff and r['variant']==v]
        cells.append(f"{100*sum(r['both_values_correct'] for r in rs)/len(rs):.1f}%")
    lines.append('| '+diff+' | '+' | '.join(cells)+' |')
lines+=['','## Position breakdown: near duplicates at 1024 records','',
'Final checkpoint, penalty-trained model, two-fact modular answer accuracy.','',
'| Evidence position | Baseline | Scaling | Position | Combined |','|---|---:|---:|---:|---:|']
for pos in ['beginning','middle','end']:
    cells=[]
    for v in ['baseline','scale','position','combined']:
        rs=[r for r in records if r['regime']=='penalty' and r['step']==600 and r['n']==1024 and r['task']=='two' and r['difficulty']=='near_duplicate' and r['position']==pos and r['variant']==v]
        cells.append(f"{100*sum(r['accuracy'] for r in rs)/len(rs):.1f}%")
    lines.append('| '+pos+' | '+' | '.join(cells)+' |')
lines+=['',
'The imposed penalty becomes a useful location prior for endpoint targets when confusable distractors are scattered throughout the context. Removing it can hurt endpoint accuracy dramatically. The three-position average must not be read as location-independent retrieval ability. Scaling preserves evidence rankings and can amplify this useful or harmful prior. These are properties of this synthetic setup, not evidence that positional bias generally helps.']
lines+=['','## Sampled reliability boundary','',
'Lengths whose lower descriptive 95% seed interval reaches 90% answer accuracy. These are sampled passing lengths, not interpolated limits or a guarantee of monotonic performance.','',
'| Training regime | Distractors | Task | Baseline | Scaling | Position | Combined |',
'|---|---|---|---|---|---|---|']
for regime in ['penalty','no_penalty']:
    for diff in ['random','similar','near_duplicate']:
        for task in ['one','two']:
            cells=[]
            for v in ['baseline','scale','position','combined']:
                good=[str(n) for n in [16,64,256,1024] if get(regime,n,diff,task,v)['ci95'][0]>=.9]
                cells.append(', '.join(good) or 'None')
            lines.append(f'| {regime} | {diff} | {task} | '+' | '.join(cells)+' |')
lines+=['','## Validation and limits','',
f'Validated all {len(records):,} evaluation records: complete factorial coverage, finite bounded metrics, answer counts consistent with accuracy, training loss reduction in all ten runs, unchanged evidence rankings under positive scaling. Runtime: '+f"{d['elapsed_seconds']:.1f} seconds.",'',
'This is a single attention readout with feed-forward decoder, not a causal language model. The position penalty is imposed rather than learned. Two-read queries are supplied explicitly. Modular sums can be correct through cancelling read errors; evidence retrieval is therefore reported separately in raw results. Three tested positions can miss other positional failures. No production-model claims follow.','',
'## Next experiment','',
'Use a small causal transformer with genuine positional encodings and no imposed middle penalty. Preserve paired evaluation, add independently resampled held-out sets, multiple-fact exact-match metrics, and validation-selected attention scales. Test whether the gains survive semantic matching and learned composition before claiming a general fix.']
(OUT/'robustness-report.md').write_text('\n'.join(lines),encoding='utf-8')
print('Validated',len(records),'records')
print('\n'.join(lines[lines.index('## Paired effects at 1024 records, two-fact task'):lines.index('## Sampled reliability boundary')]))
