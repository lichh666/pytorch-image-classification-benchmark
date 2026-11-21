"""统一训练、按样本加权梯度累积、Epoch 边界原子检查点与恢复。"""
import os
import random
import time
import json
import csv
import gc
from itertools import islice
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch import nn
from dataset import build_internal_datasets, make_loader
from models import create_model, parameter_count
from evaluate import evaluate_internal, plot_results
from config import Config


def normalized_config(values):
    """旧检查点缺省的新字段补默认值；tuple/list 的 JSON 表示视为相同配置。"""
    return json.loads(json.dumps(Config(**values).to_dict()))


def select_device(request='auto'):
    if request not in ('auto', 'cpu', 'cuda'): raise ValueError('未知设备')
    if request == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA 不可用')
    return torch.device('cuda' if request != 'cpu' and torch.cuda.is_available() else 'cpu')


def seed_everything(config):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config.deterministic
    torch.use_deterministic_algorithms(config.deterministic, warn_only=config.deterministic_warn_only)


def create_optimizer(model, config):
    options = dict(lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == 'sgd': return torch.optim.SGD(model.parameters(), momentum=config.momentum, **options)
    if config.optimizer == 'adam': return torch.optim.Adam(model.parameters(), betas=config.adam_betas, eps=config.adam_eps, **options)
    if config.optimizer == 'adamw': return torch.optim.AdamW(model.parameters(), betas=config.adam_betas, eps=config.adam_eps, **options)
    if config.optimizer == 'rmsprop': return torch.optim.RMSprop(model.parameters(), momentum=config.momentum, **options)
    raise ValueError('未知优化器')


def create_scheduler(optimizer, config):
    if config.scheduler == 'none': return None
    if config.scheduler == 'cosine': return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.cosine_t_max or config.epochs)
    if config.scheduler == 'step': return torch.optim.lr_scheduler.StepLR(optimizer, config.step_size, config.gamma)
    raise ValueError('未知学习率调度器')


def rng_state():
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [n[0], n[1].tolist(), n[2], n[3], n[4]],
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']; np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']: torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def atomic_save(state, path):
    temporary = path.with_suffix('.tmp'); torch.save(state, temporary); os.replace(temporary, path)


def write_json(value, path):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


def train_one_epoch(model, loader, optimizer, device, scaler, amp=False, accumulation_steps=1):
    model.train()
    loss_sum, correct, count, updates, overflow_retries = 0.0, 0, 0, 0, 0
    iterator = iter(loader)
    while group := list(islice(iterator, accumulation_steps)):
        # 按实际样本数归一化，末尾不足累积窗口也不额外缩小梯度。
        total = sum(len(y) for _, y in group)
        initial_rng = rng_state()
        buffers = {name: b.clone() for name, b in model.named_buffers()}
        for attempt in range(6):
            optimizer.zero_grad(set_to_none=True)
            local_loss, local_correct = 0.0, 0
            for x, y in group:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=amp and device.type == 'cuda'):
                    logits = model(x); loss = nn.functional.cross_entropy(logits, y)
                if not torch.isfinite(loss): raise FloatingPointError('训练损失 NaN/Inf；停止而不跳过样本')
                scaler.scale(loss * (len(y) / total)).backward()
                local_loss += loss.detach().item() * len(y)
                local_correct += (logits.detach().argmax(1) == y).sum().item()
            previous_scale = scaler.get_scale()
            if not scaler.is_enabled() and not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('非 AMP 梯度 NaN/Inf')
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale() >= previous_scale: break
            # 溢出时优化器没有更新；降低 scale、恢复随机状态和 BN 后重放同一窗口。
            overflow_retries += 1
            print(f'AMP_OVERFLOW retry={attempt+1} scale={scaler.get_scale()}', flush=True)
            restore_rng(initial_rng)
            with torch.no_grad():
                for name, b in model.named_buffers(): b.copy_(buffers[name])
        else: raise FloatingPointError('AMP 降低缩放后仍溢出')
        loss_sum += local_loss; correct += local_correct; count += total; updates += 1
        if updates % 100 == 0: print(f'PROGRESS samples={count}/{len(loader.dataset)} loss={loss_sum/count:.5f}', flush=True)
    if not count: raise ValueError('训练数据为空')
    return {'loss': loss_sum / count, 'accuracy': correct / count,
            'optimizer_updates': updates, 'overflow_retries': overflow_retries, 'samples': count}


def fit(config):
    config.validate(); seed_everything(config); device = select_device(config.device)
    training, validation, split = build_internal_datasets(config)
    train_loader = make_loader(training, config, device, True); val_loader = make_loader(validation, config, device)
    run = config.run_name or f'{config.model}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    checkpoint_dir = Path(config.output_root) / 'checkpoints' / run
    result_dir = Path(config.output_root) / 'results' / run
    checkpoint_dir.mkdir(parents=True, exist_ok=True); result_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = checkpoint_dir / 'last.pt', checkpoint_dir / 'best.pt'
    summary_path = result_dir / 'summary.json'
    if summary_path.exists() and json.loads(summary_path.read_text(encoding='utf-8')).get('completed'):
        saved_config = json.loads((result_dir / 'config.json').read_text(encoding='utf-8'))
        assert normalized_config(saved_config) == normalized_config(config.to_dict()), '已完成配置不同，禁止覆盖'
        assert best_path.exists() and last_path.exists()
        print(f'SKIP_COMPLETED {config.model}', flush=True); return best_path
    model = create_model(config).to(device)
    optimizer = create_optimizer(model, config); scheduler = create_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler('cuda', init_scale=config.amp_init_scale, enabled=config.amp and device.type == 'cuda')
    history, best_accuracy, best_metrics, best_epoch, elapsed_before = [], -1.0, None, 0, 0.0
    if last_path.exists():
        state = torch.load(last_path, map_location='cpu', weights_only=True)
        assert normalized_config(state['config']) == normalized_config(config.to_dict()) and state['split'] == split, '恢复配置或划分不匹配'
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        if scheduler: scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler']); history = state['history']; best_accuracy = state['best_accuracy']
        best_epoch, best_metrics = state['best_epoch'], state['best_metrics']; elapsed_before = state['elapsed_seconds']
        train_loader.generator.set_state(state['train_generator']); val_loader.generator.set_state(state['val_generator'])
        restore_rng(state['rng']); del state; gc.collect()
        print(f'RESUME {config.model} epoch={len(history)+1}', flush=True)
    write_json(config.to_dict(), result_dir / 'config.json'); write_json(split, result_dir / 'split.json')
    count = parameter_count(model)
    print(f'START {config.model} device={device} parameters={count} batch={config.batch_for(device)} accumulation={config.accumulation_steps}', flush=True)
    session_start = time.perf_counter()
    for epoch in range(len(history) + 1, config.epochs + 1):
        start = time.perf_counter()
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
        lr = optimizer.param_groups[0]['lr']
        if config.scheduler == 'cosine':
            expected = config.lr * (1 + np.cos(np.pi * (epoch-1)/(config.cosine_t_max or config.epochs))) / 2
            assert abs(lr - expected) < 1e-10, '学习率调度异常'
        trained = train_one_epoch(model, train_loader, optimizer, device, scaler, config.amp, config.accumulation_steps)
        validated = evaluate_internal(model, val_loader, device, config.num_classes, config.amp)
        if device.type == 'cuda': torch.cuda.synchronize()
        assert trained['samples'] == len(training) and sum(map(sum, validated['confusion_matrix'])) == len(validation)
        assert all(np.isfinite(v) for v in [trained['loss'], validated['loss'], validated['accuracy']])
        if scheduler: scheduler.step()  # 完成一轮训练和内部验证后调度一次
        if validated['accuracy'] > best_accuracy:
            best_accuracy, best_metrics, best_epoch = validated['accuracy'], validated, epoch
            atomic_save({'model': model.state_dict(), 'config': config.to_dict(), 'split': split,
                         'epoch': epoch, 'validation': validated, 'parameters': count}, best_path)
        history.append({'epoch': epoch, 'lr': lr, 'train_loss': trained['loss'], 'train_accuracy': trained['accuracy'],
                        'validation_loss': validated['loss'], 'validation_accuracy': validated['accuracy'],
                        'validation_precision_macro': validated['precision_macro'], 'validation_recall_macro': validated['recall_macro'],
                        'optimizer_updates': trained['optimizer_updates'], 'overflow_retries': trained['overflow_retries'],
                        'amp_scale': scaler.get_scale(), 'best_epoch': best_epoch, 'best_validation_accuracy': best_accuracy,
                        'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else 0,
                        'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20 if device.type=='cuda' else 0,
                        'epoch_seconds': time.perf_counter() - start})
        # last.pt 是恢复历史的事实来源；写文件失败时原检查点不受影响。
        atomic_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict() if scheduler else None, 'scaler': scaler.state_dict(),
                     'config': config.to_dict(), 'split': split, 'epoch': epoch, 'history': history,
                     'best_accuracy': best_accuracy, 'best_epoch': best_epoch, 'best_metrics': best_metrics,
                     'elapsed_seconds': elapsed_before + time.perf_counter() - session_start,
                     'rng': rng_state(), 'train_generator': train_loader.generator.get_state(),
                     'val_generator': val_loader.generator.get_state()}, last_path)
        write_json(history, result_dir / 'history.json')
        with (result_dir / 'history.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
        print(f'EPOCH {config.model} '+json.dumps(history[-1]), flush=True)
    elapsed = elapsed_before + time.perf_counter() - session_start
    best = torch.load(best_path, map_location='cpu', weights_only=True)
    model.load_state_dict(best['model']); del best
    checked = evaluate_internal(model, val_loader, device, config.num_classes, config.amp)
    assert checked['confusion_matrix'] == best_metrics['confusion_matrix'], '最佳权重指标不一致'
    write_json(best_metrics, result_dir / 'internal_validation.json')
    plot_results(history, best_metrics, split['classes'], result_dir)
    summary = {'completed': True, 'actual_epochs': len(history), 'stop_reason': '达到统一最大 Epoch',
               'parameters': count, 'training_seconds': elapsed,
               'time_definition': '训练、内部验证及检查点写入墙钟时间；不含最终自检和绘图；恢复时不计停机',
               'device': str(device), 'micro_batch_size': config.batch_for(device), 'accumulation_steps': config.accumulation_steps,
               'effective_batch_size': config.batch_for(device)*config.accumulation_steps,
               'tail_effective_batch_size': len(training) % (config.batch_for(device)*config.accumulation_steps),
               'torch': str(torch.__version__), 'best_internal_validation': best_metrics, 'best_epoch': best_epoch,
               'weight_reload_metrics_verified': True, 'overflow_retries': sum(r['overflow_retries'] for r in history),
               'final_test_used': False}
    write_json(summary, summary_path)
    print(f'COMPLETE {config.model} best_epoch={best_epoch} accuracy={best_accuracy:.6f} seconds={elapsed:.1f}', flush=True)
    return best_path
