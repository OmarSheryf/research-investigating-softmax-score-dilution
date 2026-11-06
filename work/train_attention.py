import json
import math
import torch
from torch import nn

torch.set_num_threads(2)

class TinyRetriever(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.q = nn.Linear(16, 32, bias=False)
        self.k = nn.Linear(16, 32, bias=False)
        self.v = nn.Linear(8, 32, bias=False)
        self.ff = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
        self.out = nn.Linear(32, 8)

    def forward(self, keys, values, query):
        scores = torch.einsum('bd,bnd->bn', self.q(query), self.k(keys)) / math.sqrt(32)
        n = keys.shape[1]
        position = torch.linspace(0, 1, n)
        # Deliberate experimental confound, not an emergent architectural effect.
        if self.mode != 'position':
            scores = scores - 2 * 4 * position * (1-position)
        if self.mode == 'scale':
            scores = scores * max(1, math.log(n)/math.log(16))
        attn = scores.softmax(-1)
        read = torch.einsum('bn,bnd->bd', attn, self.v(values))
        return self.out(read + self.ff(read)), attn

def batch(n, size, generator, pos=None):
    keys = torch.randn(size, n, 16, generator=generator)
    keys = nn.functional.normalize(keys, dim=-1) * 4
    ids = torch.randint(8, (size,n), generator=generator)
    values = nn.functional.one_hot(ids, 8).float()
    target = torch.randint(n, (size,), generator=generator) if pos is None else torch.full((size,), round(pos*(n-1)))
    query = keys[torch.arange(size), target] + .05*torch.randn(size,16,generator=generator)
    return keys, values, query, target, ids[torch.arange(size), target]

def evaluate(model):
    result = []
    model.eval()
    with torch.no_grad():
        for n in [16,64,256]:
            for position in [0,.5,1]:
                g = torch.Generator().manual_seed(9000+n+round(position*10))
                keys, values, query, target, labels = batch(n,256,g,position)
                logits, attention = model(keys,values,query)
                loss = nn.functional.cross_entropy(logits,labels).item()
                correct = (logits.argmax(-1)==labels).float().mean().item()
                retrieve = (attention.argmax(-1)==target).float().mean().item()
                mass = attention[torch.arange(256),target].mean().item()
                entropy = (-(attention*attention.clamp_min(1e-9).log()).sum(-1)/math.log(n)).mean().item()
                result.append(dict(n=n,pos=position,loss=round(loss,4),accuracy=round(correct,4),retrieval=round(retrieve,4),mass=round(mass,4),entropy=round(entropy,4)))
    model.train()
    return result

def main():
    runs=[]
    for mode in ['baseline','scale','position']:
        torch.manual_seed(42)
        model=TinyRetriever(mode)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
        g=torch.Generator().manual_seed(1234)
        checkpoints=[dict(step=0,metrics=evaluate(model))]
        for step in range(1,401):
            keys,values,query,target,labels=batch(16,64,g)
            logits,_=model(keys,values,query)
            loss=nn.functional.cross_entropy(logits,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step%50==0:
                checkpoints.append(dict(step=step,metrics=evaluate(model)))
        runs.append(dict(mode=mode,parameters=sum(p.numel() for p in model.parameters()),checkpoints=checkpoints))
        print(mode, checkpoints[-1]['metrics'][0],flush=True)
    with open('work/attention-training.json','w') as f:
        json.dump(dict(seed=42,trainLength=16,batchSize=64,evalSize=256,runs=runs),f)

if __name__=='__main__':
    main()
