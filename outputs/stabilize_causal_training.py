"""Validation-selected stabilization, followed by fresh-seed confirmation.

Three development seeds compare baseline, warmup/cosine, and a length curriculum.
Selection is frozen before confirmation on five unseen training seeds and new
test populations. All settings use the same architecture, updates and token count.
"""
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from causal_transformer_experiment import Model, batch, verify

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'stability-run'
DEVELOPMENT=[11,33,44]
CONFIRMATION=[61,72,83,94,105]
SETTINGS=['baseline','warmup_cosine','curriculum']
POSITIONS=[0,.25,.5,.75,1]
STEPS=1200
SAMPLE_COUNT=128

def schedule(setting,step):
    n=[4,8,16][(step-1)%3] if setting!='curriculum' else [4,8,16][min(2,(step-1)//400)]
    if setting=='baseline':
        lr=.002
    elif step<=100:
        lr=.002*step/100
    else:
        lr=.0002+.0018*(1+math.cos(math.pi*(step-100)/1100))/2
    return n,lr

def atomic_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    temp.replace(path)

def checkpoint(path,model,opt,g,seed,setting,step,token_count):
    temp=path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),batch_rng=g.get_state(),torch_rng=torch.get_rng_state(),seed=seed,setting=setting,step=step,tokens_seen=token_count),temp)
    temp.replace(path)

@torch.no_grad()
def evaluate(model,population,lengths):
    rows=[]
    model.eval()
    for n in lengths:
        for position in POSITIONS:
            for replicate in [0,1]:
                g=torch.Generator().manual_seed(population+replicate*100000+n*100+round(position*10))
                cs=[];ct=[];both=[];single_loss=two_loss=0.
                for _ in range(SAMPLE_COUNT//32):
                    tokens,labels,_=batch(n,32,g,position)
                    reads,composition,_=model(tokens)
                    cs+=(reads[:,0].argmax(-1)==labels[:,0]).int().tolist()
                    ct+=(composition.argmax(-1)==labels.sum(-1)%8).int().tolist()
                    both+=(reads.argmax(-1)==labels).all(-1).int().tolist()
                    single_loss+=F.cross_entropy(reads[:,0],labels[:,0],reduction='sum').item()
                    two_loss+=F.cross_entropy(composition,labels.sum(-1)%8,reduction='sum').item()
                rows.append(dict(n=n,position=position,replicate=replicate,single=sum(cs)/len(cs),two=sum(ct)/len(ct),both_reads=sum(both)/len(both),single_loss=single_loss/SAMPLE_COUNT,two_loss=two_loss/SAMPLE_COUNT,correct_single=cs,correct_two=ct))
    return rows

def task_scores(rows):
    return {f'{n}-{task}':sum(r[task] for r in rows if r['n']==n)/sum(r['n']==n for r in rows) for n in [8,16] for task in ['single','two']}

def run(seed,setting,stage,population,lengths,start_time):
    directory=OUT/stage/f'{setting}-seed-{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    result_file=directory/'results.json'
    if result_file.exists():
        return json.loads(result_file.read_text())
    torch.manual_seed(seed)
    model=Model()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    g=torch.Generator().manual_seed(seed+1000)
    logs=[];tokens_seen=0
    reused=stage=='development' and setting=='baseline'
    if reused:
        state=torch.load(ROOT/f'causal-run/checkpoints/seed-{seed}-step-1200.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        g.set_state(state['batch_rng']);torch.set_rng_state(state['torch_rng'])
        tokens_seen=832000
        checkpoint(directory/'step-1200.pt',model,optimizer,g,seed,setting,1200,tokens_seen)
    else:
        for step in range(1,STEPS+1):
            n,lr=schedule(setting,step)
            for group in optimizer.param_groups:group['lr']=lr
            tokens,labels,_=batch(n,32,g)
            reads,composition,_=model(tokens)
            read_loss=F.cross_entropy(reads.flatten(0,1),labels.flatten())
            two_loss=F.cross_entropy(composition,labels.sum(-1)%8)
            loss=read_loss+two_loss
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            tokens_seen+=tokens.numel()
            if step%200==0:
                logs.append(dict(step=step,n=n,lr=lr,read_loss=read_loss.item(),two_loss=two_loss.item(),tokens_seen=tokens_seen))
                print(f'{stage} {setting} seed {seed} step {step}: read {read_loss.item():.3f}, sum {two_loss.item():.3f} ({time.time()-start_time:.0f}s)',flush=True)
            if step%400==0:
                checkpoint(directory/f'step-{step}.pt',model,optimizer,g,seed,setting,step,tokens_seen)
    assert tokens_seen==832000
    rows=evaluate(model,population,lengths)
    scores=task_scores(rows)
    acquired=all(value>=.95 for value in scores.values())
    result=dict(seed=seed,setting=setting,stage=stage,reused_baseline_checkpoint=reused,tokens_seen=tokens_seen,steps=STEPS,scores=scores,acquired=acquired,minimum_score=min(scores.values()),mean_score=sum(scores.values())/4,training=logs,evaluation=rows)
    atomic_json(result_file,result)
    print(f'FINISHED {stage} {setting} seed {seed}: acquired={acquired}, minimum={min(scores.values()):.3f}',flush=True)
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(development_seeds=DEVELOPMENT,confirmation_seeds=CONFIRMATION,development_seed_note='Previously observed seeds, including two known unsuccessful seeds; only confirmation seeds are fresh.',settings=SETTINGS,steps=STEPS,batch=32,tokens_per_run=832000,architecture='Unchanged 126608-parameter causal RoPE transformer',penalty='none',attention_scaling='off',objective='read CE + learned composition CE',success='At least 95% single-fact AND composition accuracy at BOTH 8 and 16 records, averaging 5 positions x 2 populations x 128 examples',selection='Maximize development seed success count, then minimum per-seed/task/length accuracy, then mean accuracy; ties retain SETTINGS order',development_population=17000000,confirmation_population=21000000,confirmation_lengths=[8,16,32,64],torch_version=str(torch.__version__),python_version=platform.python_version(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_source_sha256=hashlib.sha256((ROOT/'causal_transformer_experiment.py').read_bytes()).hexdigest())
    for setting in SETTINGS:
        assert sum((2*schedule(setting,s)[0]+3)*32 for s in range(1,1201))==832000
    assert not set(DEVELOPMENT)&set(CONFIRMATION)
    atomic_json(OUT/'protocol.json',protocol)
    verify();start=time.time();development=[]
    for setting in SETTINGS:
        for seed in DEVELOPMENT:
            development.append(run(seed,setting,'development',17000000,[8,16],start))
    rankings=[]
    for setting in SETTINGS:
        rs=[r for r in development if r['setting']==setting]
        rankings.append(dict(setting=setting,success_count=sum(r['acquired'] for r in rs),worst_score=min(r['minimum_score'] for r in rs),mean_score=sum(r['mean_score'] for r in rs)/len(rs)))
    winner=max(rankings,key=lambda r:(r['success_count'],r['worst_score'],r['mean_score']))['setting']
    frozen=dict(winner=winner,rankings=rankings,selection_time_seconds=time.time()-start,confirmation_started=False)
    atomic_json(OUT/'selection-frozen.json',frozen)
    print('FROZEN SELECTION '+json.dumps(frozen),flush=True)
    confirmation=[]
    for seed in CONFIRMATION:
        # Both settings are evaluated with matched seeds and exact token budgets.
        for setting in dict.fromkeys(['baseline',winner]):
            confirmation.append(run(seed,setting,'confirmation',21000000,[8,16,32,64],start))
            atomic_json(OUT/'progress.json',dict(winner=winner,completed_confirmation_runs=len(confirmation),development=development,confirmation=confirmation))
    payload=dict(protocol=protocol,selection=frozen,development=development,confirmation=confirmation,elapsed_seconds=time.time()-start)
    atomic_json(OUT/'results.json',payload)
    print('COMPLETE '+json.dumps(dict(winner=winner,confirmation=[dict(seed=r['seed'],setting=r['setting'],acquired=r['acquired'],minimum_score=r['minimum_score']) for r in confirmation],elapsed_seconds=time.time()-start)),flush=True)

if __name__=='__main__':main()
