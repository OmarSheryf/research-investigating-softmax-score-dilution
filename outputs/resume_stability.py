"""Resume an intermediate checkpoint using its original 1200-update recipe."""
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from causal_transformer_experiment import Model, batch
from stabilize_causal_training import checkpoint, schedule

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--until',type=int,default=1200)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if not saved['step']<args.until<=1200:
        parser.error('--until must exceed the saved step and remain within this 1200-update protocol')
    if args.output.resolve()==args.checkpoint.resolve():
        parser.error('--output must differ from the input checkpoint')
    model=Model();model.load_state_dict(saved['model'])
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    opt.load_state_dict(saved['optimizer'])
    g=torch.Generator();g.set_state(saved['batch_rng']);torch.set_rng_state(saved['torch_rng'])
    tokens_seen=saved['tokens_seen']
    for step in range(saved['step']+1,args.until+1):
        n,lr=schedule(saved['setting'],step)
        for group in opt.param_groups:group['lr']=lr
        tokens,labels,_=batch(n,32,g)
        reads,composition,_=model(tokens)
        loss=F.cross_entropy(reads.flatten(0,1),labels.flatten())+F.cross_entropy(composition,labels.sum(-1)%8)
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        tokens_seen+=tokens.numel()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    checkpoint(args.output,model,opt,g,saved['seed'],saved['setting'],args.until,tokens_seen)
    print(str(args.output))

if __name__=='__main__':main()
