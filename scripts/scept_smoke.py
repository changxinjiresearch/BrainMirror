"""Offline reproducible SCEPT smoke, synthetic voxels only; no BraTS download."""
import argparse
import json
from pathlib import Path
import torch
from brainmirror.scept import (TransformerVIB3D, IJEPAPretrainer,
    transfer_pretrained_encoder, Region, explain_region)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='runs/scept_alpha_smoke.json')
    args=parser.parse_args()
    torch.manual_seed(240216)
    torch.set_num_threads(2)
    a=TransformerVIB3D(dim=16,heads=4,depth=2,patch=4,latent_dim=8)
    x=torch.rand(1,4,16,16,16)
    pre=IJEPAPretrainer(a,momentum=.9)
    opt=torch.optim.Adam(list(a.encoder_parameters())+list(pre.predictor.parameters())+[pre.mask_token],lr=0.001)
    pre_loss=pre.pretrain_step(x,opt)
    # Independent downstream models, same architecture, distinct initialisation.
    model_a=TransformerVIB3D(dim=16,heads=4,depth=2,patch=4,latent_dim=8)
    model_b=TransformerVIB3D(dim=16,heads=4,depth=2,patch=4,latent_dim=8)
    transfer_pretrained_encoder(pre.student,model_a)
    roi=torch.zeros(16,16,16,dtype=torch.bool)
    roi[4:12,4:12,4:12]=True
    region=Region(4,12,4,12,4,12)
    output={'label':'engineering_synthetic_only','pretrain_loss':pre_loss,
            'ijepa_style':explain_region(model_a,x,region,roi),
            'scratch':explain_region(model_b,x,region,roi)}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps({'status':'PASS','output':str(path),
                      'traced_modules':len(output['scratch']['modules'])}))

if __name__=='__main__':main()
