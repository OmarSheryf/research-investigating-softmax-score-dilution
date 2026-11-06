"""Check every saved state and reproduce updates 1001-1200 from seed 11."""
import json
import torch
from torch import nn
from torch.nn import functional as F
from causal_transformer_experiment import Model, batch, OUT, TRAIN

def main():
    files=sorted((OUT/'checkpoints').glob('seed-*-step-*.pt'))
    assert len(files)==30
    for file in files:
        state=torch.load(file,map_location='cpu',weights_only=True)
        assert all(k in state for k in ['model','optimizer','batch_rng','torch_rng','seed','step'])
        model=Model();model.load_state_dict(state['model'])
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    saved=torch.load(OUT/'checkpoints/seed-11-step-1000.pt',map_location='cpu',weights_only=True)
    target=torch.load(OUT/'checkpoints/seed-11-step-1200.pt',map_location='cpu',weights_only=True)
    model=Model();model.load_state_dict(saved['model'])
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    opt.load_state_dict(saved['optimizer'])
    generator=torch.Generator();generator.set_state(saved['batch_rng'])
    torch.set_rng_state(saved['torch_rng'])
    for step in range(1001,1201):
        tokens,labels,_=batch(TRAIN[(step-1)%3],32,generator)
        reads,composition,_=model(tokens)
        loss=F.cross_entropy(reads.flatten(0,1),labels.flatten())+F.cross_entropy(composition,labels.sum(-1)%8)
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    differences={k:(v-target['model'][k]).abs().max().item() for k,v in model.state_dict().items()}
    max_difference=max(differences.values())
    assert max_difference==0, f'Resume diverged: {max_difference}'
    assert torch.equal(generator.get_state(),target['batch_rng'])
    assert torch.equal(torch.get_rng_state(),target['torch_rng'])
    result=dict(checkpoints_loaded=len(files),resume_seed=11,resume_from=1000,resume_to=1200,max_parameter_difference=max_difference,exact_resume=True,data_rng_matches=True,torch_rng_matches=True)
    (OUT/'checkpoint-validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__=='__main__':main()
