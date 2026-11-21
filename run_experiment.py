"""Train a recorded configuration in a new output directory; test only explicitly."""
import argparse
import json
from pathlib import Path
from config import Config, BASE
from train import fit, seed_everything, select_device

RUNS={'MyNet_Baseline':'baseline/mynet','ResNet18':'baseline/resnet18','VGG16':'baseline/vgg16',
      **{n:f'optimization/{n}' for n in ['H1','H2','H3','S1','S2']},
      **{n:f'feature_experiments/{n}' for n in ['F1','F2','S3']}}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=RUNS,default='S1')
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,default=BASE/'runs')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--final-test',action='store_true',help='Evaluate the fixed checkpoint on val/ after model selection is frozen')
    args=p.parse_args()
    output=args.output_root.resolve()
    if output==BASE.resolve() or BASE/'results'==output or BASE/'results' in output.parents:
        p.error('Choose a new output directory outside archived results.')
    cfg=json.loads((BASE/'results'/RUNS[args.experiment]/'config.json').read_text(encoding='utf-8'))
    cfg.update(data_root=str(args.data_root.resolve()),output_root=str(output),device=args.device,pretrained=False)
    config=Config(**cfg).validate(); seed_everything(config)
    checkpoint=fit(config)
    print(f'Trained checkpoint: {checkpoint}')
    if args.final_test:
        from evaluate import evaluate_final_test
        evaluate_final_test(checkpoint,confirm_final=True,device=args.device)

if __name__=='__main__':main()
