# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import tonic
import torch
import torchvision
from tonic import DiskCachedDataset, SlicedDataset
from tonic.collation import PadTensors
from tonic.slicers import SliceByTime
from torch.utils.data import DataLoader
from torchvision.transforms import RandomPerspective, RandomResizedCrop, RandomRotation

data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), './data/'))
cache_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), './data/cache/'))


class ChannelShift:
    """Shift flattened spatial channels left/right with zero padding."""

    def __init__(self, max_shift: int = 4, p: float = 0.5, generator: Optional[torch.Generator] = None):
        self.max_shift = int(max_shift)
        self.p = float(p)
        self.gen = generator

    def _rand(self) -> torch.Tensor:
        if self.gen is None:
            return torch.rand(())
        return torch.rand((), generator=self.gen)

    def _randint(self, low: int, high: int) -> int:
        if self.gen is None:
            return int(torch.randint(low, high, ()).item())
        return int(torch.randint(low, high, (), generator=self.gen).item())

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_shift <= 0 or self.p <= 0:
            return x
        if self._rand().item() > self.p:
            return x

        orig_shape = x.shape
        if x.ndim < 2:
            return x

        flat = x.reshape(orig_shape[0], -1)
        shift = self._randint(-self.max_shift, self.max_shift + 1)
        if shift == 0:
            return x
        T, C = flat.shape
        shift = max(min(shift, C), -C)
        if shift > 0:
            pad = flat.new_zeros(T, shift)
            shifted = torch.cat([pad, flat[:, :-shift]], dim=1)
        else:
            s = abs(shift)
            pad = flat.new_zeros(T, s)
            shifted = torch.cat([flat[:, s:], pad], dim=1)
        return shifted.reshape(orig_shape)


class BlendSameClass:
    """Blend samples with same-class partners inside a batch."""

    def __init__(
        self,
        p: float = 0.5,
        union_clip: bool = True,
        alpha: float = 0.5,
        generator: Optional[torch.Generator] = None
    ):
        self.p = float(p)
        self.union_clip = bool(union_clip)
        self.alpha = float(alpha)
        self.gen = generator

    def __call__(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.p <= 0.0 or X.size(0) <= 1:
            return X, y

        unique = y.unique(sorted=False)
        for label in unique.tolist():
            idx = (y == label).nonzero(as_tuple=False).squeeze(1)
            n = idx.numel()
            if n < 2:
                continue
            partner = idx[torch.roll(torch.arange(n), shifts=1)]
            mask = torch.rand(n, generator=self.gen) < self.p if self.gen is not None else torch.rand(n) < self.p
            if not mask.any():
                continue
            src = idx[mask]
            tgt = partner[mask]
            if self.union_clip:
                X[src] = torch.clamp(X[src] + X[tgt], 0, 1)
            else:
                X[src] = self.alpha * X[src] + (1.0 - self.alpha) * X[tgt]
        return X, y


def _build_cache_transform(args, train: bool, data_max: float):
    ops: List[Callable[[Any], Any]] = [
        lambda x: x / data_max,
        lambda x: torch.as_tensor(x, dtype=torch.float32),
    ]
    if train and getattr(args, 'use_augm', False):
        max_shift = int(getattr(args, 'aug_chan_shift_max', 0))
        p = float(getattr(args, 'aug_chan_shift_p', 0.0))
        if max_shift > 0 and p > 0:
            gen = torch.Generator()
            if getattr(args, 'aug_seed', None) is not None:
                gen.manual_seed(int(args.aug_seed))
            ops.append(ChannelShift(max_shift=max_shift, p=p, generator=gen))
        ops.extend(
            [
                RandomResizedCrop(
                    tonic.datasets.DVSGesture.sensor_size[:-1],
                    scale=(getattr(args, 'aug_rrc_scale_min', 0.6), getattr(args, 'aug_rrc_scale_max', 1.0)),
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                ),
                RandomPerspective(distortion_scale=getattr(args, 'aug_perspective_distortion', 0.5)),
                RandomRotation(getattr(args, 'aug_rotation_deg', 25)),
            ]
        )
    ops.append(
        lambda x: x.reshape(x.shape[0], -1)  # flatten spatial dimensions
    )
    return torchvision.transforms.Compose(ops)


def _make_collate_fn(blend: Optional[BlendSameClass], train: bool) -> Callable:
    base_collate = PadTensors(batch_first=True)

    def _collate(batch: List[Tuple[torch.Tensor, Any]]):
        X, y = base_collate(batch)
        if train and blend is not None:
            X, y = blend(X, y)
        return X, y

    return _collate


def _build_blend(args) -> Optional[BlendSameClass]:
    if not getattr(args, 'use_augm', False):
        return None
    p = float(getattr(args, 'blend_same_class_p', 0.0))
    if p <= 0:
        return None
    gen = torch.Generator()
    if getattr(args, 'aug_seed', None) is not None:
        gen.manual_seed(int(args.aug_seed) + 101)
    return BlendSameClass(
        p=p,
        union_clip=bool(getattr(args, 'blend_union_clip', False)),
        alpha=float(getattr(args, 'blend_alpha', 0.5)),
        generator=gen,
    )


def get_dvs128_data_v1(args):
    sensor_size = tonic.datasets.DVSGesture.sensor_size
    transform = tonic.transforms.ToFrame(
        sensor_size=sensor_size,
        time_window=args.frame_time * 1000,
        include_incomplete=False
    )
    tr_str = 'toframe'

    dataset = tonic.datasets.DVSGesture(save_to=data_folder, train=True, transform=None, target_transform=None)

    min_time_window = 1.7 * 1e6  # 1.7 s
    overlap = 0
    metadata_path = f'_{min_time_window}_{overlap}_{args.frame_time}_' + tr_str
    slicer_by_time = SliceByTime(
        time_window=min_time_window,
        overlap=overlap,
        include_incomplete=False
    )
    train_dataset_timesliced = SlicedDataset(
        dataset,
        slicer=slicer_by_time,
        transform=transform,
        metadata_path=None
    )

    data_max = 19.0  # commented to save time, re calculate if min_time_window changes
    print(f'Max train value: {data_max}')
    post_cache_transform = _build_cache_transform(args, train=True, data_max=data_max)
    blend = _build_blend(args)
    train_collate = _make_collate_fn(blend, train=True)

    train_cached_dataset = DiskCachedDataset(
        train_dataset_timesliced,
        transform=post_cache_transform,
        cache_path=os.path.join(cache_folder, 'diskcache_train' + metadata_path)
    )

    train_dataset = DataLoader(
        train_cached_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_collate,
        num_workers=args.num_workers,
    )

    print(f"Loaded train dataset with {len(train_dataset.dataset)} samples")

    test_dataset = tonic.datasets.DVSGesture(save_to=data_folder, train=False, transform=None, target_transform=None)

    min_time_window = 1.7 * 1e6  # 1.7 s
    overlap = 0  #
    slicer_by_time = SliceByTime(
        time_window=min_time_window,
        overlap=overlap,
        include_incomplete=False
    )
    os.makedirs(cache_folder, exist_ok=True)
    metadata_path = f'_{min_time_window}_{overlap}_{args.frame_time}_' + tr_str
    test_dataset_timesliced = SlicedDataset(
        test_dataset,
        slicer=slicer_by_time,
        transform=transform,
        metadata_path=None
    )

    data_max = 19.5  # commented to save time, re calculate if min_time_window changes
    print(f'Max test value: {data_max}')
    test_post_transform = _build_cache_transform(args, train=False, data_max=data_max)

    cached_test_dataset_time = DiskCachedDataset(
        test_dataset_timesliced,
        transform=test_post_transform,
        cache_path=os.path.join(cache_folder, 'diskcache_test' + metadata_path)
    )
    test_collate = _make_collate_fn(None, train=False)
    cached_test_dataloader_time = DataLoader(
        cached_test_dataset_time,
        batch_size=args.batch_size,
        collate_fn=test_collate,
        drop_last=False,
        num_workers=args.num_workers,
    )

    print(f"Loaded test dataset with {len(test_dataset)} samples")

    import brainstate
    return brainstate.util.DotDict(
        {
            'train_loader': train_dataset,
            'test_loader': cached_test_dataloader_time,
            'in_shape': np.prod(tonic.datasets.DVSGesture.sensor_size),
            'num_classes': 11,
        }
    )


def add_gesture_augmentation_args(parser):
    parser.add_argument('--frame-time', type=int, default=25, help='Time in ms to collect events into each frame')
    args, _ = parser.parse_known_args()
    if getattr(args, 'use_augm', False):
        group = parser.add_argument_group('Data Augmentation (DVS Gesture)')
        group.add_argument('--aug-seed', type=int, default=None, help='Seed for gesture augmentation RNGs.')
        group.add_argument('--aug-chan-shift-max', type=int, default=4,
                           help='Maximum absolute channel shift on flattened frames.')
        group.add_argument('--aug-chan-shift-p', type=float, default=0.5,
                           help='Probability to shift a sample.')
        group.add_argument('--aug-rrc-scale-min', type=float, default=0.6,
                           help='Minimum scale for RandomResizedCrop when using augmentation.')
        group.add_argument('--aug-rrc-scale-max', type=float, default=1.0,
                           help='Maximum scale for RandomResizedCrop when using augmentation.')
        group.add_argument('--aug-perspective-distortion', type=float, default=0.5,
                           help='Distortion scale for RandomPerspective.')
        group.add_argument('--aug-rotation-deg', type=float, default=25.0,
                           help='Max absolute rotation in degrees.')
        group.add_argument('--blend-same-class-p', type=float, default=0.0,
                           help='Probability to blend with same-class partner in a batch.')
        group.add_argument('--blend-union-clip', action='store_true',
                           help='Use union-clip (binary) blending instead of alpha mix.')
        group.add_argument('--blend-alpha', type=float, default=0.5,
                           help='Alpha for continuous blending when not using union-clip.')
    return parser


if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num_workers', type=int, default=0)
    argparser.add_argument('--batch-size', type=int, default=256)
    argparser.add_argument('--use-augm', type=bool, default=True)
    add_gesture_augmentation_args(argparser)
    args = argparser.parse_args()

    data = get_dvs128_data_v1(args)

    print('Train dataset:')
    for batch in data['train_loader']:
        print('input shape = ', batch[0].shape, 'target shape = ', batch[1].shape)
    print()
