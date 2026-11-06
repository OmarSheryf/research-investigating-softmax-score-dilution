import json,math,time,argparse,platform
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(2)
D,H,L=64,4,3
KEYS,VALUES,SEP=128,8,136
TRAIN=[4,8,16]
LENGTHS=[8,16,32,64]
POSITIONS=[0,.25,.5,.75,1]
OUT=Path(__file__).resolve().parent/'causal-run'
SEEDS=[11,22,33,44,55]

class Attention(nn.Module):
 def __init__(self):
  super().__init__()
  self.qkv=nn.Linear(D,3*D,bias=False)
  self.out=nn.Linear(D,D,bias=False)
  self.register_buffer('freq',1/(10000**(torch.arange(0,16,2).float()/16)))
 def rotate(self,x):
  phase=torch.arange(x.shape[-2])[:,None]*self.freq[None,:]
  a,b=x[...,0::2],x[...,1::2]
  return torch.stack((a*phase.cos()-b*phase.sin(),a*phase.sin()+b*phase.cos()),-1).flatten(-2)
 def forward(self,x,c=0.,inspect=False):
  b,t,_=x.shape
  q,k,v=self.qkv(x).view(b,t,3,H,16).permute(2,0,3,1,4).unbind(0)
  q,k=self.rotate(q),self.rotate(k)
  scale=1+c*(torch.arange(1,t+1).float().log()/math.log(35)-1).clamp_min(0)
  q=q*scale[None,None,:,None]
  if inspect:
   attn=(q@k.transpose(-1,-2)/4).masked_fill(~torch.ones(t,t,dtype=torch.bool).tril(),float('-inf')).softmax(-1)
   y=attn@v
  else:
   y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
   attn=None
  return self.out(y.transpose(1,2).reshape(b,t,D)),attn
class Block(nn.Module):
 def __init__(self):
  super().__init__()
  self.n1=nn.LayerNorm(D);self.n2=nn.LayerNorm(D);self.attn=Attention()
  self.ff=nn.Sequential(nn.Linear(D,128),nn.GELU(),nn.Linear(128,D))
 def forward(self,x,c,inspect=False):
  y,a=self.attn(self.n1(x),c,inspect);x=x+y
  return x+self.ff(self.n2(x)),a
class Model(nn.Module):
 def __init__(self):
  super().__init__()
  self.embed=nn.Embedding(137,D)
  self.blocks=nn.ModuleList([Block() for _ in range(L)])
  self.norm=nn.LayerNorm(D);self.read=nn.Linear(D,VALUES)
  self.compose=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,VALUES))
 def hidden(self,tokens,c=0.,inspect=False):
  x=self.embed(tokens);a=None
  for i,block in enumerate(self.blocks): x,a=block(x,c,inspect and i==L-1)
  return self.norm(x),a
 def forward(self,tokens,c=0.,inspect=False):
  x,a=self.hidden(tokens,c,inspect);q=x[:,-2:]
  return self.read(q),self.compose(q.flatten(1)),a

def batch(n,size,g,position=None,dispersion='far'):
 keys=torch.rand(size,KEYS,generator=g).argsort(-1)[:,:n]
 vals=torch.randint(VALUES,(size,n),generator=g)
 if position is None:
  first=torch.randint(n,(size,),generator=g)
  second=(first+torch.randint(1,n,(size,),generator=g))%n
 else:
  first=torch.full((size,),round(position*(n-1)))
  second=(first+(1 if dispersion=='near' else n//2))%n
 target=torch.stack((first,second),-1);rows=torch.arange(size)[:,None]
 tokens=torch.empty(size,2*n+3,dtype=torch.long)
 tokens[:,:2*n:2]=keys;tokens[:,1:2*n:2]=vals+KEYS
 tokens[:,2*n]=SEP;tokens[:,-2:]=keys[rows,target]
 return tokens,vals[rows,target],target

@torch.no_grad()
def evaluate(m,c,n,p,rep,dispersion='far',validation=False,size=128):
 g=torch.Generator().manual_seed((7000000 if validation else 9000000)+rep*10000+n*100+round(p*10)+(dispersion=='near'))
 totals={k:0. for k in ['single','two','both_reads','mass','entropy','single_loss','two_loss']}
 cs=[];ct=[]
 for start in range(0,size,32):
  count=min(32,size-start);tokens,labels,target=batch(n,count,g,p,dispersion)
  reads,compose,a=m(tokens,c,True)
  single=reads[:,0].argmax(-1)==labels[:,0]
  two=compose.argmax(-1)==labels.sum(-1)%VALUES
  both=(reads.argmax(-1)==labels).all(-1)
  attn=a[:,:,-2,:].mean(1);slot=2*target[:,0]+1
  mass=attn.gather(-1,slot[:,None]).squeeze(-1)
  entropy=-(attn*attn.clamp_min(1e-12).log()).sum(-1)/math.log(tokens.shape[1]-1)
  for name,vec in [('single',single),('two',two),('both_reads',both),('mass',mass),('entropy',entropy)]: totals[name]+=vec.float().sum().item()
  totals['single_loss']+=F.cross_entropy(reads[:,0],labels[:,0],reduction='sum').item()
  totals['two_loss']+=F.cross_entropy(compose,labels.sum(-1)%VALUES,reduction='sum').item()
  cs+=single.int().tolist();ct+=two.int().tolist()
 return dict(n=n,position=p,replicate=rep,dispersion=dispersion,coefficient=c,**{k:v/size for k,v in totals.items()},correct_single=cs,correct_two=ct)

def verify():
 torch.manual_seed(987);m=Model().eval();g=torch.Generator().manual_seed(321)
 tokens,labels,target=batch(8,4,g);rows=torch.arange(4)[:,None]
 assert torch.equal(tokens[:,-2:],tokens[:,:16:2][rows,target])
 assert torch.equal(labels,tokens[:,1:16:2][rows,target]-KEYS)
 assert torch.tensor([len(torch.unique(row))==8 for row in tokens[:,:16:2]]).all()
 altered=tokens.clone();altered[:,-1]=(altered[:,-1]+1)%KEYS
 with torch.no_grad():
  for c in [0.,1.]:
   a,_=m.hidden(tokens,c);b,_=m.hidden(altered,c)
   assert torch.allclose(a[:,:-1],b[:,:-1],atol=1e-6),'Causal leakage'
  a,_=m.hidden(tokens);b,attention=m.hidden(tokens,inspect=True)
  assert torch.allclose(a,b,atol=2e-5),'Diagnostic mismatch'
  assert (attention.triu(1)==0).all()
  assert torch.allclose(attention.sum(-1),torch.ones_like(attention.sum(-1)),atol=1e-5)
 print('Preflight passed: causal masking, target integrity, unique keys, diagnostic equivalence.',flush=True)

def interval(xs):
 mean=sum(xs)/len(xs)
 half=2.776*(sum((x-mean)**2 for x in xs)/4/5)**.5
 return [round(mean,6),round(mean-half,6),round(mean+half,6)]


def save_checkpoint(m,opt,g,seed,step):
 path=OUT/'checkpoints'/f'seed-{seed}-step-{step}.pt'
 path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix('.tmp')
 torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),batch_rng=g.get_state(),torch_rng=torch.get_rng_state(),seed=seed,step=step,architecture=dict(width=D,heads=H,layers=L),training_lengths=TRAIN),temp)
 temp.replace(path)
 return str(path)

def resume_checkpoint(path,steps):
 state=torch.load(path,map_location='cpu',weights_only=True)
 m=Model();m.load_state_dict(state['model'])
 opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.01);opt.load_state_dict(state['optimizer'])
 g=torch.Generator();g.set_state(state['batch_rng']);torch.set_rng_state(state['torch_rng'])
 for step in range(state['step']+1,steps+1):
  tokens,labels,_=batch(TRAIN[(step-1)%3],32,g)
  reads,two,_=m(tokens)
  loss=F.cross_entropy(reads.flatten(0,1),labels.flatten())+F.cross_entropy(two,labels.sum(-1)%VALUES)
  opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
  if step%200==0 or step==steps:
   print('Saved resumed checkpoint:',save_checkpoint(m,opt,g,state['seed'],step),flush=True)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 verify();start=time.time();records=[];training=[];selection=[]
 for seed in SEEDS:
  torch.manual_seed(seed);m=Model();opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.01)
  g=torch.Generator().manual_seed(seed+1000)
  for step in range(1,1201):
   n=TRAIN[(step-1)%3];tokens,labels,_=batch(n,32,g)
   reads,two,_=m(tokens)
   read_loss=F.cross_entropy(reads.flatten(0,1),labels.flatten())
   sum_loss=F.cross_entropy(two,labels.sum(-1)%VALUES);loss=read_loss+sum_loss
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
   if step%200==0:
    save_checkpoint(m,opt,g,seed,step)
    training.append(dict(seed=seed,step=step,read_loss=round(read_loss.item(),5),two_loss=round(sum_loss.item(),5)))
    print(f'seed {seed} step {step}: read {read_loss.item():.3f}, sum {sum_loss.item():.3f}, elapsed {time.time()-start:.0f}s',flush=True)
   if step in [400,1200]:
    m.eval()
    for nt in LENGTHS:
     for p in POSITIONS:
      for rep in [0,1]:
       for c in [0.,1.]: records.append(dict(seed=seed,step=step,**evaluate(m,c,nt,p,rep)))
    m.train()
  m.eval();validation=[]
  for c in [0.,.5,1.]:
   ms=[evaluate(m,c,n,p,0,validation=True,size=64) for n in [16,32] for p in [0,.5,1]]
   validation.append(dict(coefficient=c,score=sum((r['single']+r['two'])/2 for r in ms)/len(ms)))
  selected=max(validation,key=lambda x:x['score'])['coefficient']
  selection.append(dict(seed=seed,validation=validation,selected=selected))
  if selected==.5:
   for n in LENGTHS:
    for p in POSITIONS:
     for rep in [0,1]: records.append(dict(seed=seed,step=1200,**evaluate(m,.5,n,p,rep)))
  for n in [16,64]:
   for p in POSITIONS:
    for rep in [0,1]:
     for c in sorted(set([0.,1.,selected])): records.append(dict(seed=seed,step=1200,**evaluate(m,c,n,p,rep,dispersion='near')))
  (OUT/'progress.json').write_text(json.dumps(dict(training=training,selection=selection,records=records)),encoding='utf-8')
  print(f'Seed {seed} complete; selected coefficient {selected}.',flush=True)
 # All aggregation is paired by seed; replicas and positions averaged within seed.
 summary=[];paired=[];position_summary=[];dispersion_summary=[]
 def meanmetric(seed,step,n,c,metric,dispersion='far',position=None,replicate=None):
  rs=[r for r in records if r['seed']==seed and r['step']==step and r['n']==n and r['coefficient']==c and r['dispersion']==dispersion and (position is None or r['position']==position) and (replicate is None or r['replicate']==replicate)]
  assert rs
  return sum(r[metric] for r in rs)/len(rs)
 selected_by_seed={r['seed']:r['selected'] for r in selection}
 for step in [400,1200]:
  for n in LENGTHS:
   for metric in ['single','two','both_reads','mass','entropy']:
    for variant in (['baseline','scale'] if step==400 else ['baseline','scale','selected']):
     coeff=lambda s: 0. if variant=='baseline' else 1. if variant=='scale' else selected_by_seed[s]
     xs=[meanmetric(s,step,n,coeff(s),metric) for s in SEEDS]
     summary.append(dict(step=step,n=n,metric=metric,variant=variant,interval=interval(xs),seed_values=[round(x,6) for x in xs]))
     if variant!='baseline':
      ds=[meanmetric(s,step,n,coeff(s),metric)-meanmetric(s,step,n,0.,metric) for s in SEEDS]
      paired.append(dict(step=step,n=n,metric=metric,variant=variant,interval=interval(ds),seed_deltas=[round(x,6) for x in ds]))
   for p in POSITIONS:
    for metric in ['single','two']:
     for c in [0.,1.]:
      xs=[meanmetric(s,step,n,c,metric,position=p) for s in SEEDS]
      position_summary.append(dict(step=step,n=n,position=p,metric=metric,coefficient=c,interval=interval(xs)))
 for n in [16,64]:
  for metric in ['single','two','both_reads']:
   for c in [0.,1.]:
    for disp in ['near','far']:
     xs=[meanmetric(s,1200,n,c,metric,dispersion=disp) for s in SEEDS]
     dispersion_summary.append(dict(n=n,metric=metric,coefficient=c,dispersion=disp,interval=interval(xs)))
 # Compare independent test populations without treating seed/example pairs as independent trials.
 replicate_summary=[]
 for rep in [0,1]:
  for n in LENGTHS:
   for metric in ['single','two']:
    for c in [0.,1.]:
     xs=[meanmetric(s,1200,n,c,metric,replicate=rep) for s in SEEDS]
     replicate_summary.append(dict(replicate=rep,n=n,metric=metric,coefficient=c,interval=interval(xs)))
 for r in records:
  assert len(r['correct_single'])==len(r['correct_two'])==128
  assert abs(sum(r['correct_single'])/128-r['single'])<1e-6
  assert abs(sum(r['correct_two'])/128-r['two'])<1e-6
  assert all(math.isfinite(r[k]) for k in ['single','two','both_reads','mass','entropy','single_loss','two_loss'])
  assert all(0<=r[k]<=1.00001 for k in ['single','two','both_reads','mass','entropy'])
 payload=dict(protocol=dict(seeds=SEEDS,train_lengths=TRAIN,test_lengths=LENGTHS,positions=POSITIONS,steps=1200,batch=32,samples_per_cell=128,test_replicates=2,width=D,heads=H,layers=L,parameters=sum(p.numel() for p in m.parameters()),position_encoding='RoPE base 10000',penalty='none',objective='read CE + learned modulo-8 composition CE',ci='descriptive t interval across five training-seed means; fixed two independent test populations',storage='Checkpoints, optimizer and RNG states saved every 200 updates',torch_version=str(torch.__version__),python_version=platform.python_version()),training=training,selection=selection,summary=summary,paired=paired,position_summary=position_summary,dispersion_summary=dispersion_summary,replicate_summary=replicate_summary,validated_records=len(records),elapsed_seconds=round(time.time()-start,1))
 (OUT/'results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
 (OUT/'raw-evaluations.json').write_text(json.dumps(records),encoding='utf-8')
 print('RESULT_JSON '+json.dumps(payload,separators=(',',':')),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser()
 parser.add_argument('--resume',type=Path)
 parser.add_argument('--steps',type=int,default=1400)
 args=parser.parse_args()
 if args.resume:
  if args.steps<=torch.load(args.resume,map_location='cpu',weights_only=True)['step']:
   parser.error('--steps must exceed checkpoint step')
  resume_checkpoint(args.resume,args.steps)
 else:
  main()

