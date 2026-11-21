"""根目录运行 python submission/main.py；submission 内运行 python main.py。

默认 smoke 只做轻量测试。正式训练必须显式选择 --mode train。
例如：python submission/main.py --mode train --model mynet --epochs 30
超参数默认值及输入特征实验选项统一位于 config.py。
最终测试仅在全部实验设计冻结后，通过 evaluate.evaluate_final_test 调用。
"""
import argparse
import sys
import torch
from config import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['smoke', 'train'], default='smoke')
    parser.add_argument('--data-root', help='Imagewoof root containing train/ and val/')
    parser.add_argument('--output-root', help='Directory for new checkpoints and results')
    parser.add_argument('--model', choices=['resnet18', 'vgg16', 'mynet'])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)
    parser.add_argument('--run-name', help='固定运行目录；同名实验从 last.pt 恢复')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--optimizer', choices=['sgd', 'adam', 'adamw', 'rmsprop'])
    parser.add_argument('--scheduler', choices=['none', 'step', 'cosine'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--image-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--pretrained', action='store_true', default=None,
                        help='显式启用后 torchvision 可能联网下载权重')
    args = vars(parser.parse_args())
    mode = args.pop('mode')
    config = Config(**{k: v for k, v in args.items() if v is not None}).validate()
    print(f'Python: {sys.executable}\nPyTorch: {torch.__version__}; '
          f'CUDA runtime: {torch.version.cuda}; available: {torch.cuda.is_available()}', flush=True)
    if mode == 'smoke':
        from smoke import run_smoke
        run_smoke(config)
    else:
        from train import fit
        print(fit(config))


if __name__ == '__main__':
    main()
