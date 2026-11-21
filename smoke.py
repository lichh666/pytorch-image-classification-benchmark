"""可重复运行的轻量检查；不读取最终测试图片，不留下权重或测试图表。"""
import gc
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import torch
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from config import Config
from dataset import build_internal_datasets, build_transform, make_loader
from models import create_model, parameter_count
from train import seed_everything, select_device, create_optimizer, create_scheduler, train_one_epoch, fit
from evaluate import evaluate_internal, evaluate_final_test, metrics_from_confusion, plot_results


def run_smoke(config=None):
    config = replace(config or Config(), pretrained=False, batch_size=2, epochs=2, workers=0)
    seed_everything(config)
    device = select_device(config.device)
    print(f'SMOKE device={device}; CUDA available={torch.cuda.is_available()}', flush=True)
    train, valid, split = build_internal_datasets(config)
    train2, valid2, split2 = build_internal_datasets(config)
    assert split == split2
    assert not set(train.indices) & set(valid.indices)
    assert set(train.indices) | set(valid.indices) == set(range(9025))
    assert train.dataset.class_to_idx == valid.dataset.class_to_idx
    # 仅枚举官方 val 的文件名/标签，确认数量和映射；禁止读取或推断其图像
    official = ImageFolder(Path(config.data_root) / 'val')
    assert len(official) == 3929 and official.class_to_idx == split['class_to_idx']
    print(f'DATA train={len(train)} internal_validation={len(valid)} final_test={len(official)} classes=10', flush=True)
    x, y = next(iter(make_loader(Subset(train, [0, 1]), config, device)))
    assert x.shape == (2, 3, config.image_size, config.image_size)
    for overrides in ({'normalize': False}, {'grayscale': True}, {'channel': 0}, {'image_size': 96}):
        cfg = replace(config, **overrides)
        im = train.dataset.loader(train.dataset.samples[0][0])
        tensor = build_transform(cfg)(im)
        assert tensor.shape == (3, cfg.image_size, cfg.image_size) and torch.isfinite(tensor).all()
    for name in ['resnet18', 'vgg16', 'mynet']:
        cfg = replace(config, model=name, optimizer='sgd', momentum=0.0)
        model = create_model(cfg).to(device)
        model.eval()
        with torch.no_grad():
            assert model(torch.randn(2, 3, cfg.image_size, cfg.image_size, device=device)).shape == (2, 10)
        optimizer = create_optimizer(model, cfg)
        scaler = torch.amp.GradScaler('cuda', init_scale=cfg.amp_init_scale,
                                    enabled=cfg.amp and device.type == 'cuda')
        # 每种网络仅对两张真实 train 图片更新一次，绝非完整 Epoch
        result = train_one_epoch(model, make_loader(Subset(train, [0, 1]), cfg, device),
                                 optimizer, device, scaler, cfg.amp)
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(gradients) == sum(p.requires_grad for p in model.parameters())
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().sum() > 0 for g in gradients)
        validation = evaluate_internal(model, make_loader(Subset(valid, [0, 1]), cfg, device),
                                       device, 10, cfg.amp)
        assert sum(map(sum, validation['confusion_matrix'])) == 2
        print(f'MODEL {name} parameters={parameter_count(model):,} output=[2,10] '
              f'loss={result["loss"]:.5f} gradients=finite internal_eval=OK', flush=True)
        if name == 'mynet':
            # 在临时目录检查保存/恢复和绘图，退出后自动清理
            with tempfile.TemporaryDirectory(prefix='imagewoof_smoke_') as temp:
                path = Path(temp) / 'best.pt'
                torch.save({'model': model.state_dict(), 'config': cfg.to_dict(), 'split': split}, path)
                saved = torch.load(path, map_location='cpu', weights_only=True)
                restored = create_model(Config(**saved['config']))
                restored.load_state_dict(saved['model'])
                model.eval(); restored.eval()
                with torch.no_grad():
                    assert torch.allclose(model.cpu()(x.cpu()), restored(x.cpu()), atol=1e-6)
                history = [{'epoch': 1, 'train_loss': result['loss'], 'train_accuracy': result['accuracy'],
                            'validation_loss': validation['loss'], 'validation_accuracy': validation['accuracy']}]
                plot_results(history, validation, split['classes'], temp)
                assert (Path(temp) / 'curves.png').is_file() and (Path(temp) / 'confusion_matrix.png').is_file()
                try:
                    evaluate_final_test(path)
                except ValueError:
                    pass
                else:
                    raise AssertionError('最终测试接口未阻止未确认的调用')
                # 用两张图片替代完整数据加载器，检查训练调度、最佳权重与日志联动。
                # 仅执行一次参数更新，不遍历完整的原始训练子集。
                tiny_cfg = replace(cfg, epochs=1, output_root=temp)
                with patch('train.build_internal_datasets', return_value=(
                        Subset(train, [0, 1]), Subset(valid, [0, 1]), split)):
                    best = fit(tiny_cfg)
                state = torch.load(best, map_location='cpu', weights_only=True)
                assert state['epoch'] == 1 and state['parameters'] == parameter_count(restored)
                assert state['split'] == split
                assert len(list((Path(temp) / 'results').glob('*/history.json'))) == 1
            for opt in ('sgd', 'adam', 'rmsprop'):
                for schedule in ('none', 'step', 'cosine'):
                    c = replace(cfg, optimizer=opt, scheduler=schedule)
                    o = create_optimizer(restored, c)
                    s = create_scheduler(o, c)
                    o.zero_grad(); restored(x.cpu()).sum().backward(); o.step()
                    if s:
                        s.step()
        del model, optimizer, gradients, scaler
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    measured = metrics_from_confusion(torch.tensor([[2, 1], [0, 1]]))
    assert abs(measured['accuracy'] - 0.75) < 1e-12
    assert abs(measured['precision_macro'] - 0.75) < 1e-12
    assert abs(measured['recall_macro'] - 5 / 6) < 1e-12
    assert str(select_device('cpu')) == 'cpu'
    with patch('torch.cuda.is_available', return_value=True):
        assert str(select_device('auto')) == 'cuda'
    with patch('torch.cuda.is_available', return_value=False):
        assert str(select_device()) == 'cpu'
        try:
            select_device('cuda')
        except RuntimeError:
            pass
        else:
            raise AssertionError('不可用 CUDA 未报错')
    print('PASS: imports, stratification/reproducibility, disjoint split, transforms, '
          'all models forward/backward, optimizers/schedulers, metrics, checkpoint, plots, device fallback. '
          'No final-test inference; temporary files removed.', flush=True)
