"""内部验证与最终测试用途分开；指标的混淆矩阵行是真值、列是预测。"""
import json
from pathlib import Path
import torch
from torch import nn


def metrics_from_confusion(matrix):
    cm = matrix.to(torch.float64)
    tp = cm.diag()
    precision = tp / cm.sum(0).clamp_min(1)
    recall = tp / cm.sum(1).clamp_min(1)
    return {'accuracy': (tp.sum() / cm.sum().clamp_min(1)).item(),
            'precision_macro': precision.mean().item(), 'recall_macro': recall.mean().item(),
            'precision_per_class': precision.tolist(), 'recall_per_class': recall.tolist(),
            'confusion_matrix': matrix.tolist()}


@torch.inference_mode()
def _evaluate(model, loader, device, num_classes, amp=False, predictions=None):
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    total_loss, count = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == 'cuda'):
            logits = model(x)
            loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError('评估损失不是有限数值')
        count += len(y)
        total_loss += loss.item() * len(y)
        predicted = logits.argmax(1)
        if predictions is not None:
            # 最终测试保留有序标签，后续复算指标无需再次执行模型。
            predictions.extend(zip(y.cpu().tolist(), predicted.cpu().tolist()))
        bins = (y * num_classes + predicted).cpu()
        cm += torch.bincount(bins, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    if count == 0:
        raise ValueError('评估数据为空')
    return {'loss': total_loss / count, **metrics_from_confusion(cm)}


def evaluate_internal(model, loader, device, num_classes=10, amp=False):
    return _evaluate(model, loader, device, num_classes, amp)


def plot_results(history, metrics, classes, folder, dataset_name='内部验证集'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if history:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, key in zip(axes, ('loss', 'accuracy')):
            for split in ('train', 'validation'):
                ax.plot([r['epoch'] for r in history], [r[f'{split}_{key}'] for r in history],
                        label='训练集' if split == 'train' else '内部验证集')
            ax.set(xlabel='训练轮次（Epoch）', ylabel='交叉熵损失' if key == 'loss' else '准确率',
                   title='损失变化' if key == 'loss' else '准确率变化')
            ax.legend()
        fig.tight_layout(); fig.savefig(folder / 'curves.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 7))
    matrix = metrics['confusion_matrix']
    im = ax.imshow(matrix, cmap='Blues')
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes, xlabel='预测类别', ylabel='真实类别',
           title=f'{dataset_name}混淆矩阵')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(folder / 'confusion_matrix.png', dpi=160); plt.close(fig)


def evaluate_final_test(checkpoint, *, confirm_final=False, device='auto'):
    """明确确认实验设计冻结后，读取最佳权重及其原始预处理配置。"""
    if not confirm_final:
        raise ValueError('最终测试仅在模型选择和调参全部结束后允许执行')
    from config import Config
    from dataset import build_final_test_dataset, make_loader
    from models import create_model
    from train import select_device
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = Config(**saved['config'])
    config.pretrained = False  # 已有完整权重，不重新下载
    config.device = device
    device = select_device(device)
    model = create_model(config).to(device)
    model.load_state_dict(saved['model'])
    ds = build_final_test_dataset(config, saved['split']['class_to_idx'])
    metrics = _evaluate(model, make_loader(ds, config, device), device, config.num_classes, config.amp)
    folder = Path(config.output_root) / 'results' / (config.run_name or Path(checkpoint).parent.name) / 'final_test'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    plot_results([], metrics, ds.classes, folder, dataset_name='官方最终测试集')
    return metrics
