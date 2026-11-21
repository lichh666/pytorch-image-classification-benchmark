"""所有实验参数集中在这里；默认不下载权重。"""
from dataclasses import dataclass, asdict
from pathlib import Path

BASE = Path(__file__).resolve().parent


@dataclass
class Config:
    model: str = 'resnet18'
    data_root: str = str(BASE / 'data' / 'imagewoof2-160')
    output_root: str = str(BASE / 'runs')
    pretrained: bool = False
    num_classes: int = 10
    seed: int = 42
    split_seed: int = 42  # 独立于训练随机种子，跨模型保持划分一致
    val_fraction: float = 0.2
    image_size: int = 160
    augment: bool = True
    crop_scale: tuple = (0.7, 1.0)
    flip_probability: float = 0.5
    normalize: bool = True
    grayscale: bool = False  # 灰度复制为三通道，便于公平比较网络
    channel: int | None = None  # 可仅保留某一颜色通道，再复制为三通道
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)  # 固定常数，不从最终测试集估计
    batch_size: int | None = None
    accumulation_steps: int = 1
    run_name: str | None = None  # baseline/模型名；相同名称自动从 last.pt 恢复
    epochs: int = 30
    optimizer: str = 'sgd'
    lr: float = 0.01
    momentum: float = 0.9
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    weight_decay: float = 0.0005
    scheduler: str = 'cosine'  # none / step / cosine
    cosine_t_max: int | None = None  # 默认等于epochs；截短实验可保持原来的学习率轨迹
    step_size: int = 10
    gamma: float = 0.1
    device: str = 'auto'
    amp: bool = True  # 仅 CUDA 启用；CPU 使用 float32
    amp_init_scale: float = 1024.0  # 避免随机初始化网络首次 FP16 反向即溢出
    workers: int = 0  # Windows 下保守默认值；正式实验可试 2
    cpu_threads: int = 4
    deterministic: bool = True
    # VGG16 的 CUDA 自适应池化反向不提供确定性实现；保留结构并明确警告。
    # 需要严格排查非确定性算子时，可将此项设为 False，遇到算子会报错。
    deterministic_warn_only: bool = True
    channels: tuple = (32, 64, 128, 192)
    blocks_per_stage: int | tuple = 2
    head_channels: int | None = None  # 默认保留原结构；宽度实验可固定头部为基线的96通道
    activation: str = 'relu'
    dropout: float = 0.25

    def batch_for(self, device):
        # RTX 4050 6 GB：实际完成 AMP 前向、反向和 SGD 更新后选择的保守值。
        return self.batch_size or ({'vgg16': 2}.get(self.model, 16)
                                   if device.type == 'cuda' else
                                   {'vgg16': 2}.get(self.model, 8))

    def validate(self):
        if self.model not in ('resnet18', 'vgg16', 'mynet'):
            raise ValueError('未知模型')
        if not 0 < self.val_fraction < 1 or self.image_size < 32:
            raise ValueError('划分比例应在 (0,1)，输入尺寸至少为 32')
        if self.epochs < 1 or self.lr <= 0 or (self.batch_size is not None and self.batch_size < 1):
            raise ValueError('epochs、lr、batch_size 必须为正')
        if self.grayscale and self.channel is not None:
            raise ValueError('灰度和单通道实验请分别进行')
        if self.channel not in (None, 0, 1, 2):
            raise ValueError('channel 应为 0、1 或 2')
        depths = ([self.blocks_per_stage] * len(self.channels)
                  if isinstance(self.blocks_per_stage, int) else self.blocks_per_stage)
        if self.num_classes != 10 or not self.channels or len(depths) != len(self.channels) or any(d < 1 for d in depths):
            raise ValueError('检查类别数量与 MyNet 结构配置')
        if self.head_channels is not None and self.head_channels < 2:
            raise ValueError('头部通道数至少为2')
        if self.adam_eps <= 0 or len(self.adam_betas) != 2 or not all(0 <= b < 1 for b in self.adam_betas):
            raise ValueError('检查 Adam 系列优化器参数')
        if any(width < 2 for width in self.channels) or not 0 <= self.dropout < 1:
            raise ValueError('MyNet 通道至少为 2，Dropout 应在 [0,1)')
        if self.workers < 0 or self.cpu_threads < 1 or self.step_size < 1:
            raise ValueError('检查 workers、cpu_threads 和 step_size')
        if self.amp_init_scale <= 0:
            raise ValueError('AMP 初始缩放必须为正')
        if self.accumulation_steps < 1:
            raise ValueError('梯度累积步数必须为正')
        if self.cosine_t_max is not None and self.cosine_t_max < self.epochs:
            raise ValueError('余弦周期不能短于本次训练预算')
        if not 0 < self.crop_scale[0] <= self.crop_scale[1] <= 1 or not 0 <= self.flip_probability <= 1:
            raise ValueError('检查随机裁剪范围和翻转概率')
        return self

    def to_dict(self):
        return asdict(self)
