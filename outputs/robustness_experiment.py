"""CPU toy attention robustness experiment. Run from the outputs directory.

Two-fact task = two shared-weight reads followed by fixed modulo-32 addition.
This tests compositional retrieval, not learned multi-hop reasoning.
"""
import json
import math
import time
from pathlib import Path
import torch
from torch import nn

torch.set_num_threads(2)
OUT = Path(__file__).resolve().parent
SEEDS = [11, 22, 33, 44, 55]
LENGTHS = [16, 64, 256, 1024]
POSITIONS = ['beginning', 'middle', 'end']
LEVELS = ['random', 'similar', 'near_duplicate']
VARIANTS = ['baseline', 'scale', 'position', 'combined']
SAMPLES = 128

class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(16, 32, bias=False)
        self.k = nn.Linear(16, 32, bias=False)
        self.v = nn.Linear(32, 32, bias=False)
        self.ff = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
        self.out = nn.Linear(32, 32)

    def forward(self, keys, values, query, penalty=2., scale=1.):
        scores = torch.einsum('bqd,bnd->bqn', self.q(query), self.k(keys))/math.sqrt(32)
        p = torch.linspace(0, 1, keys.shape[1])
        attention = ((scores-penalty*4*p*(1-p))*scale).softmax(-1)
        read = torch.einsum('bqn,bnd->bqd', attention, self.v(values))
        return self.out(read+self.ff(read)), attention

def make_batch(n, size, g, position=None, level='random'):
    keys = nn.functional.normalize(torch.randn(size,n,16,generator=g),dim=-1)*4
    ids = torch.randint(32,(size,n),generator=g)
    rows = torch.arange(size)
    if position is None:
        first = torch.randint(n,(size,),generator=g)
        second = (first+torch.randint(1,n,(size,),generator=g))%n
    else:
        start = {'beginning':0,'middle':n//2-1,'end':n-2}[position]
        first, second = torch.full((size,),start), torch.full((size,),start+1)
    targets = torch.stack([first,second],dim=1)
    clean = keys[rows[:,None],targets].clone()
    # Four confusable distractors per target; disjoint from both target slots.
    if level != 'random':
        noise = .6 if level == 'similar' else .15
        free = (torch.arange(n)[None,:] != first[:,None]) & (torch.arange(n)[None,:] != second[:,None])
        placement = torch.rand(size,n,generator=g).masked_fill(~free,-1)
        indices = placement.topk(8,dim=1).indices
        for q in range(2):
            for j in range(4):
                slot = indices[:,q*4+j]
                keys[rows,slot] = nn.functional.normalize(clean[:,q]+noise*torch.randn(size,16,generator=g),dim=-1)*4
                ids[rows,slot] = (ids[rows,targets[:,q]]+torch.randint(1,32,(size,),generator=g))%32
    query = clean+.05*torch.randn(size,2,16,generator=g)
    labels = ids[rows[:,None],targets]
    return keys, nn.functional.one_hot(ids,32).float(), query, targets, labels

@torch.no_grad()
def evaluate(model, seed, regime, step):
    records=[]
    for ni,n in enumerate(LENGTHS):
        for pi,pos in enumerate(POSITIONS):
            for li,level in enumerate(LEVELS):
                # Identical held-out examples across seeds, regimes and checkpoints.
                g=torch.Generator().manual_seed(900000+ni*100+pi*10+li)
                batch=make_batch(n,SAMPLES,g,pos,level)
                keys,values,query,targets,labels=batch
                predictions={}
                for variant in VARIANTS:
                    penalty=0. if variant in ['position','combined'] else 2.
                    scale=max(1.,math.log(n)/math.log(16)) if variant in ['scale','combined'] else 1.
                    logits,attention=model(keys,values,query,penalty,scale)
                    pred=logits.argmax(-1)
                    selected=attention.argmax(-1)
                    predictions[variant]=selected
                    mass=attention.gather(-1,targets[:,:,None]).squeeze(-1)
                    entropy=-(attention*attention.clamp_min(1e-12).log()).sum(-1)/math.log(n)
                    for task in ['one','two']:
                        correct=pred[:,0]==labels[:,0] if task=='one' else pred.sum(-1)%32==labels.sum(-1)%32
                        retrieval=selected[:,0]==targets[:,0] if task=='one' else (selected==targets).all(-1)
                        loss=nn.functional.cross_entropy(logits[:,0],labels[:,0]).item() if task=='one' else nn.functional.cross_entropy(logits.flatten(0,1),labels.flatten()).item()
                        records.append(dict(seed=seed,regime=regime,step=step,n=n,position=pos,difficulty=level,variant=variant,task=task,accuracy=correct.float().mean().item(),both_values_correct=(pred==labels).all(-1).float().mean().item(),retrieval=retrieval.float().mean().item(),mass=mass[:,0].mean().item() if task=='one' else mass.mean().item(),entropy=entropy[:,0].mean().item() if task=='one' else entropy.mean().item(),read_loss=loss,correct=correct.int().tolist()))
                assert torch.equal(predictions['baseline'],predictions['scale']), 'Scaling changed rank'
                assert torch.equal(predictions['position'],predictions['combined']), 'Scaling changed rank'
    return records

def summarize(records):
    groups={}
    for r in records:
        key=tuple(r[k] for k in ['regime','step','n','difficulty','variant','task'])
        groups.setdefault(key,[]).append(r)
    summaries=[]
    for key,rs in groups.items():
        seed_means=[sum(r['accuracy'] for r in rs if r['seed']==s)/3 for s in SEEDS]
        mean=sum(seed_means)/5
        sem=(sum((x-mean)**2 for x in seed_means)/4/5)**.5
        summaries.append(dict(zip(['regime','step','n','difficulty','variant','task'],key),accuracy=mean,ci95=[max(0,mean-2.776*sem),min(1,mean+2.776*sem)],seed_accuracy=seed_means))
    return summaries

def main():
    start=time.time(); records=[]; losses=[]
    for seed in SEEDS:
        for regime in ['penalty','no_penalty']:
            torch.manual_seed(seed)
            model=Reader()
            optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
            g=torch.Generator().manual_seed(seed+1000)
            for step in range(1,601):
                keys,values,query,_,labels=make_batch(16,64,g)
                logits,_=model(keys,values,query,2. if regime=='penalty' else 0.)
                loss=nn.functional.cross_entropy(logits.flatten(0,1),labels.flatten())
                optimizer.zero_grad();loss.backward();optimizer.step()
                if step in [1,200,600]:
                    losses.append(dict(seed=seed,regime=regime,step=step,loss=loss.item()))
                if step in [200,600]:
                    model.eval();records.extend(evaluate(model,seed,regime,step));model.train()
            torch.save(model.state_dict(),OUT/f'weights-{seed}-{regime}.pt')
            print(f'{seed} {regime}: {time.time()-start:.1f}s',flush=True)
    payload=dict(protocol=dict(seeds=SEEDS,lengths=LENGTHS,positions=POSITIONS,difficulties=LEVELS,samples_per_cell=SAMPLES,training_length=16,training_steps=600,batch_size=64,classes=32,ci='Student t interval across five seed means; positions averaged; shared held-out set',task='one read or two independent reads followed by fixed modulo-32 addition',selection='No sweep selected on evaluation results; fixed scaling log(n)/log(16).'),training_losses=losses,summary=summarize(records),records=records,elapsed_seconds=time.time()-start)
    (OUT/'robustness-results.json').write_text(json.dumps(payload),encoding='utf-8')
    print('Complete',round(time.time()-start,1),flush=True)

if __name__=='__main__':
    main()
