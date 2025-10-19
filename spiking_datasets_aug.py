import platform
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import brainstate
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

__all__ = [
    'load_dataset',
]


# ===============================
# 增强工具：支持 [T, C] 张量
# ===============================

@dataclass
class Compose:
    transforms: List[Callable[[torch.Tensor, Any], Tuple[torch.Tensor, Any]]]

    def __call__(self, x: torch.Tensor, y: Any) -> Tuple[torch.Tensor, Any]:
        for t in self.transforms:
            x, y = t(x, y)
        return x, y


@dataclass
class RandomTimeShiftT:
    """沿时间维( dim=0 )做随机平移；mode='wrap' 循环移位, 'pad' 补零, 'trim' 截断+补零"""
    max_shift: int
    mode: str = "wrap"
    p: float = 1.0

    def __call__(self, x: torch.Tensor, y: Any):
        if self.max_shift <= 0 or torch.rand(()) > self.p:
            return x, y
        T = x.shape[0]
        shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)))
        if shift == 0:
            return x, y
        if self.mode == "wrap":
            x2 = torch.roll(x, shifts=shift, dims=0)
        elif self.mode == "pad":
            x2 = torch.zeros_like(x)
            if shift > 0:
                x2[shift:] = x[:T - shift]
            else:
                x2[:T + shift] = x[-shift:]
        elif self.mode == "trim":
            if shift > 0:
                core = x[:T - shift]
                pad = torch.zeros((shift, x.shape[1]), dtype=x.dtype)
                x2 = torch.cat([pad, core], dim=0)
            else:
                core = x[-shift:]
                pad = torch.zeros((-shift, x.shape[1]), dtype=x.dtype)
                x2 = torch.cat([core, pad], dim=0)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return x2, y


@dataclass
class ChannelJitter:
    """
    通道近邻扰动：以概率 rate 将若干 spike 从通道 c 挪到 c±k（k∈[1,max_delta]）。
    注意：对二值脉冲张量按“移动”处理（保持总脉冲数不变）。
    """
    rate: float = 0.02  # 扰动比例（0~1）
    max_delta: int = 1  # 最大近邻距离
    p: float = 1.0

    def __call__(self, x: torch.Tensor, y: Any):
        if self.rate <= 0 or torch.rand(()) > self.p:
            return x, y
        T, C = x.shape
        nz = (x > 0).nonzero(as_tuple=False)  # [N, 2] (t,c)
        N = nz.shape[0]
        if N == 0:
            return x, y
        k = int(max(1, round(self.rate * N)))
        sel = torch.randperm(N)[:k]
        idx = nz[sel]  # [k, 2]
        # 随机选择 ±delta（不为0）
        delta = torch.randint(1, self.max_delta + 1, (k,))
        sign = torch.randint(0, 2, (k,)) * 2 - 1  # ±1
        shift = (delta * sign).clamp(min=-self.max_delta, max=self.max_delta)
        tgt_c = (idx[:, 1] + shift).clamp(0, C - 1)
        out = x.clone()
        out[idx[:, 0], idx[:, 1]] = 0.0
        out[idx[:, 0], tgt_c] = 1.0
        return out, y


@dataclass
class TimeScale:
    """
    时间尺度压缩/伸缩：围绕时间中心缩放索引 t' = round((t - T/2)*scale + T/2)
    scale ∈ [min_scale, max_scale]（例如 0.9~1.1）。保持输出长度不变（溢出裁剪）。
    """
    min_scale: float = 0.9
    max_scale: float = 1.1
    p: float = 0.0

    def __call__(self, x: torch.Tensor, y: Any):
        if self.p <= 0 or torch.rand(()) > self.p:
            return x, y
        scale = float(torch.empty(()).uniform_(self.min_scale, self.max_scale))
        if abs(scale - 1.0) < 1e-6:
            return x, y
        T, C = x.shape
        center = (T - 1) / 2.0
        nz = (x > 0).nonzero(as_tuple=False)  # [N, 2]
        if nz.numel() == 0:
            return x, y
        t = nz[:, 0].to(torch.float32)
        c = nz[:, 1]
        t_new = torch.round((t - center) * scale + center).to(torch.long).clamp(0, T - 1)
        out = torch.zeros_like(x)
        out[t_new, c] = 1.0
        return out, y


@dataclass
class NormalizeSpikeCount:
    """
    （可选）脉冲数归一化：子采样到 target_total；若 allow_upsample=True 可做复制式上采样（带微抖动）。
    """
    target_total: Optional[int] = None
    allow_upsample: bool = False
    jitter: int = 1
    p: float = 0.0

    def __call__(self, x: torch.Tensor, y: Any):
        if self.p <= 0:
            return x, y
        events = (x > 0).to(x.dtype)
        N = int(events.sum().item())
        if N == 0:
            return x, y
        tgt = self.target_total if self.target_total is not None else N
        if tgt <= 0 or tgt == N:
            return events, y
        T, C = x.shape
        idx = events.nonzero(as_tuple=False)  # (N,2) (t,c)
        if tgt < N:
            keep = torch.randperm(N)[:tgt]
            sel = idx[keep]
            out = torch.zeros_like(x)
            out[sel[:, 0], sel[:, 1]] = 1.0
            return out, y
        else:
            if not self.allow_upsample:
                return events, y
            add_n = tgt - N
            dup = idx[torch.randint(0, N, (add_n,))]
            if self.jitter > 0:
                jt = torch.randint(-self.jitter, self.jitter + 1, (add_n,))
                t_new = (dup[:, 0] + jt).clamp(0, T - 1)
                dup = torch.stack([t_new, dup[:, 1]], dim=1)
            all_idx = torch.cat([idx, dup], dim=0)
            out = torch.zeros_like(x)
            out[all_idx[:, 0], all_idx[:, 1]] = 1.0
            return out, y


# ---------- 批级 MixUp / Blend（可同类约束） ----------

def _to_one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    oh = torch.zeros((y.shape[0], num_classes), dtype=torch.float32, device=y.device)
    oh.scatter_(1, y.view(-1, 1).long(), 1.0)
    return oh


class MixupCollate:
    """
    在 collate 阶段做样本混合（blend/mixup）：
    - alpha>0 启用，lambda~Beta(alpha, alpha)
    - same_class=True 时只在同类内部随机配对，不足时退回普通 mixup
    - 输出标签为 soft（one-hot混合）
    """

    def __init__(self, base_collate: Callable, num_classes: int,
                 alpha: float = 0.0, p: float = 1.0, same_class: bool = False):
        self.base_collate = base_collate
        self.num_classes = num_classes
        self.alpha = alpha
        self.p = p
        self.same_class = same_class

    def __call__(self, batch):
        xs, ys = zip(*batch)  # xs: [N, T, C] tensors, ys: int
        # 先做基础堆叠/补齐（与你原来的 generate_batch 一致）
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)  # [B,T,C]
        ys = torch.tensor(ys, dtype=torch.long)

        if self.alpha <= 0.0 or torch.rand(()) > self.p or xs.shape[0] < 2:
            return xs, ys  # 不混合则保留原始 int 标签

        B = xs.shape[0]
        device = xs.device
        if self.same_class:
            # 按类别分桶，在桶内随机打乱并配对；若桶小于2则回退全局配对
            perm = torch.empty(B, dtype=torch.long, device=device)
            used = torch.zeros(B, dtype=torch.bool, device=device)
            success = True
            for cls in ys.unique():
                idx = (ys == cls).nonzero(as_tuple=False).view(-1)
                if idx.numel() < 2:
                    success = False
                    break
                perm_sub = idx[torch.randperm(idx.numel(), device=device)]
                perm[idx] = perm_sub
                used[idx] = True
            if not success or not used.all():
                perm = torch.randperm(B, device=device)
        else:
            perm = torch.randperm(B, device=device)

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B,)).to(device)
        lam_x = lam.view(B, 1, 1)
        lam_y = lam.view(B, 1)

        xs_mix = lam_x * xs + (1 - lam_x) * xs[perm]
        y1 = _to_one_hot(ys, self.num_classes)
        y2 = _to_one_hot(ys[perm], self.num_classes)
        ys_mix = lam_y * y1 + (1 - lam_y) * y2
        return xs_mix, ys_mix


# ===============================
# 原数据集（加入 per-sample transform）
# ===============================

class SpikingDataset(Dataset):
    """
    Dataset class for the Spiking Heidelberg Digits (SHD) or
    Spiking Speech Commands (SSC) dataset.

    Arguments
    ---------
    dataset_name : str
        Name of the dataset, either shd or ssc.
    data_folder : str
        Path to folder containing the dataset (h5py file).
    split : str
        Split of the SHD dataset, must be either "train" or "test".
    nb_steps : int
        Number of time steps for the generated spike trains.
    transform : Callable[[Tensor, Any], (Tensor, Any)]
        Per-sample transform applied on dense [T, C] spikes.
    """

    def __init__(
        self,
        dataset_name: str,
        data_folder: str,
        split: str,
        nb_steps: int = 100,
        transform: Optional[Callable[[torch.Tensor, Any], Tuple[torch.Tensor, Any]]] = None,
    ):
        # Fixed parameters
        self.device = "cpu"  # to allow pin memory
        self.nb_steps = nb_steps
        self.nb_units = 700
        self.transform = transform

        # Read data from h5py file
        filename = f"{data_folder}/{dataset_name}_{split}.h5"
        self.h5py_file = h5py.File(filename, "r")
        self.firing_times = self.h5py_file["spikes"]["times"]
        max_time = 1.4
        self.max_time = max_time
        self.time_bins = np.linspace(0, self.max_time, num=self.nb_steps)
        self.units_fired = self.h5py_file["spikes"]["units"]
        self.labels = np.array(self.h5py_file["labels"], dtype=np.int_)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 栅格化为稀疏 [T, C]
        times = np.digitize(self.firing_times[index], self.time_bins)
        units = self.units_fired[index]
        x_idx = torch.LongTensor(np.array([times, units])).to(self.device)
        x_val = torch.FloatTensor(np.ones(len(times))).to(self.device)
        x_size = torch.Size([self.nb_steps, self.nb_units])  # [T, C]
        x = torch.sparse_coo_tensor(x_idx, x_val, x_size).to(self.device).to_dense()
        y = int(self.labels[index])

        # 逐样本数据增强（若配置）
        if self.transform is not None:
            x, y = self.transform(x, y)
        return x, y

    def generate_batch(self, batch):
        xs, ys = zip(*batch)
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        ys = torch.LongTensor(ys).to(self.device)
        return xs, ys


# We need to stack the batch elements
def _numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [_numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


# ===============================
# Loader 入口：通过 args 控制增强
# ===============================

def _build_train_transform(args) -> Optional[Callable]:
    # 默认开启：时间平移 shift
    ts_max = getattr(args, 'aug_shift_max', 20)  # 默认 20 个 time-steps
    ts_p = getattr(args, 'aug_shift_p', 1.0)  # 默认每样本都平移
    ts_mode = getattr(args, 'aug_shift_mode', 'wrap')  # 默认循环移位

    # 其余增强默认关闭
    cj_rate = getattr(args, 'aug_channel_jitter_rate', 0.0)
    cj_k = getattr(args, 'aug_channel_jitter_max_delta', 1)
    cj_p = getattr(args, 'aug_channel_jitter_p', 0.0)

    sc_p = getattr(args, 'aug_timescale_p', 0.0)
    sc_min = getattr(args, 'aug_timescale_min', 0.9)
    sc_max = getattr(args, 'aug_timescale_max', 1.1)

    norm_p = getattr(args, 'aug_norm_p', 0.0)
    norm_tgt = getattr(args, 'aug_norm_target_total', None)
    norm_up = getattr(args, 'aug_norm_allow_upsample', False)
    norm_jt = getattr(args, 'aug_norm_jitter', 1)

    ops = []
    if ts_max > 0 and ts_p > 0:
        ops.append(RandomTimeShiftT(max_shift=int(ts_max), mode=ts_mode, p=float(ts_p)))
    if cj_rate > 0 and cj_p > 0:
        ops.append(ChannelJitter(rate=float(cj_rate), max_delta=int(cj_k), p=float(cj_p)))
    if sc_p > 0:
        ops.append(TimeScale(min_scale=float(sc_min), max_scale=float(sc_max), p=float(sc_p)))
    if norm_p > 0:
        ops.append(NormalizeSpikeCount(
            target_total=None if norm_tgt is None else int(norm_tgt),
            allow_upsample=bool(norm_up),
            jitter=int(norm_jt),
            p=float(norm_p),
        ))
    return Compose(ops) if ops else None


def _make_collate(train_dataset: SpikingDataset, args, is_train: bool) -> Callable:
    if not is_train or not args.use_augm:
        return train_dataset.generate_batch

    # 默认开启：blend/mixup（同类混合）
    mix_alpha = getattr(args, 'aug_blend_alpha', 0.2)  # Beta(0.2, 0.2)
    mix_p = getattr(args, 'aug_blend_p', 1.0)  # 每个 batch 都混合
    same_cls = getattr(args, 'aug_blend_same_class', True)  # 同类内配对

    if mix_alpha <= 0.0:
        return train_dataset.generate_batch

    # 使用我们自定义的 MixupCollate（内部自带 pad_sequence）
    def _collate(batch):
        xs, ys = zip(*batch)
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        ys = torch.LongTensor(ys)
        if torch.rand(()) > mix_p or xs.shape[0] < 2:
            return xs, ys
        # 同步逻辑与 MixupCollate 类一致（为简洁直接写在闭包里）
        B = xs.shape[0]
        device = xs.device
        if same_cls:
            perm = torch.empty(B, dtype=torch.long, device=device)
            used = torch.zeros(B, dtype=torch.bool, device=device)
            success = True
            for cls in ys.unique():
                idx = (ys == cls).nonzero(as_tuple=False).view(-1)
                if idx.numel() < 2:
                    success = False
                    break
                perm[idx] = idx[torch.randperm(idx.numel(), device=device)]
                used[idx] = True
            if not success or not used.all():
                perm = torch.randperm(B, device=device)
        else:
            perm = torch.randperm(B, device=device)
        lam = torch.distributions.Beta(float(mix_alpha), float(mix_alpha)).sample((B,)).to(device)
        xs = lam.view(B, 1, 1) * xs + (1 - lam).view(B, 1, 1) * xs[perm]
        # 软标签
        y1 = torch.zeros((B, 20), dtype=torch.float32, device=device)  # SHD 20类
        y1.scatter_(1, ys.view(-1, 1), 1.0)
        y2 = y1[perm]
        ys = lam.view(B, 1) * y1 + (1 - lam).view(B, 1) * y2
        return xs, ys

    return _collate


def load_shd_data(args):
    # 构建 per-sample transform（仅训练集）
    train_transform = _build_train_transform(args)
    test_transform = None

    train_dataset = SpikingDataset('shd', args.data_folder, 'train', args.data_length, transform=train_transform)
    test_dataset = SpikingDataset('shd', args.data_folder, 'test', args.data_length, transform=test_transform)

    train_collate = _make_collate(train_dataset, args, is_train=True)
    test_collate = _make_collate(test_dataset, args, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=train_collate,
        shuffle=True,
        num_workers=0 if platform.system() == 'Windows' else args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=test_collate,
        shuffle=False,
        num_workers=0 if platform.system() == 'Windows' else args.num_workers,
    )
    return brainstate.util.DotDict(
        {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'in_shape': 700,
            'out_shape': 20,
            'input_process': lambda x: x,
        }
    )


def load_dataset(args):
    if args.dataset_name == 'shd':
        return load_shd_data(args)
    else:
        raise ValueError(f'Unknown dataset name: {args.dataset_name}')
