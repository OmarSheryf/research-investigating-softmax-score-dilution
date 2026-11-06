"""Validate checkpoints and replay 400 updates of the selected recipe exactly."""
import json
import torch
from torch import nn
from torch.nn import functional as F
from stabilize_causal_training import OUT, schedule
from causal_transformer_experiment import Model, batch

def main():
    results=json.loads((OUT/'results.json').read_text())
    winner=results['selection']['winner']
    files=sorted(OUT.glob('*/*/step-*.pt'))
    for path in files:
        state=torch.load(path,map_location='cpu',weights_only=True)
        model=Model();model.load_state_dict(state['model'])
        assert all(torch.isfinite(value).all() for value in model.state_dict().values())
        assert all(k in state for k in ['optimizer','batch_rng','torch_rng','tokens_seen'])
    directory=OUT/'confirmation'/f'{winner}-seed-61'
    saved=torch.load(directory/'step-800.pt',map_location='cpu',weights_only=True)
    target=torch.load(directory/'step-1200.pt',map_location='cpu',weights_only=True)
    model=Model();model.load_state_dict(saved['model'])
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    opt.load_state_dict(saved['optimizer'])
    g=torch.Generator();g.set_state(saved['batch_rng']);torch.set_rng_state(saved['torch_rng'])
    tokens_seen=saved['tokens_seen']
    for step in range(801,1201):
        n,lr=schedule(winner,step)
        for group in opt.param_groups:group['lr']=lr
        tokens,labels,_=batch(n,32,g)
        reads,composition,_=model(tokens)
        loss=F.cross_entropy(reads.flatten(0,1),labels.flatten())+F.cross_entropy(composition,labels.sum(-1)%8)
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        tokens_seen+=tokens.numel()
    max_difference=max((value-target['model'][key]).abs().max().item() for key,value in model.state_dict().items())
    assert max_difference==0,max_difference
    assert torch.equal(g.get_state(),target['batch_rng'])
    assert torch.equal(torch.get_rng_state(),target['torch_rng'])
    assert tokens_seen==target['tokens_seen']==832000
    payload=dict(checkpoints_loaded=len(files),setting=winner,seed=61,from_step=800,to_step=1200,max_parameter_difference=max_difference,rng_states_match=True,token_count_matches=True)
    (OUT/'checkpoint-validation.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload))

if __name__=='__main__':main()
