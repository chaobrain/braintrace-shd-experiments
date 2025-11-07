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
import torch
import tonic
from tonic.collation import PadTensors
from tonic.datasets import NMNIST
from torch.utils.data import DataLoader

cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), './data'))


class ChannelShift:
    """Shift flattened channels left/right with zero padding."""

    def __init__(self, max_shift: int = 4, p: float = 0.5, generator: Optional[torch.Generator] = None):
        self.max_shift = int(max_shift)
        self.p = float(p)
        self.gen = generator

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_shift <= 0 or self.p <= 0:
            return x

        if torch.rand((), generator=self.gen if self.gen is not None else None).item() > self.p:
            return x

        orig_shape = x.shape
        if x.ndim < 2:
            return x
        flat = x.reshape(orig_shape[0], -1)
        shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (), generator=self.gen).item())
        if shift == 0:
            return x
        T, C = flat.shape
        shift = max(min(shift, C), -C)
        if shift > 0:
            pad = flat.new_zeros(T, shift)
            shifted = torch.cat([pad, flat[:, :-shift]], dim=1)
        else:
            shift = abs(shift)
            pad = flat.new_zeros(T, shift)
            shifted = torch.cat([flat[:, shift:], pad], dim=1)
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
            mask = torch.rand(n, generator=self.gen) < self.p
            if not mask.any():
                continue
            src = idx[mask]
            tgt = partner[mask]
            if self.union_clip:
                X[src] = torch.clamp(X[src] + X[tgt], 0, 1)
            else:
                X[src] = self.alpha * X[src] + (1.0 - self.alpha) * X[tgt]
        return X, y


def _build_per_sample_transform(args, train: bool, sensor_size) -> tonic.transforms.Compose:
    ops: List[Callable[[torch.Tensor], torch.Tensor]] = [
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=args.data_length),
        lambda x: x.reshape(x.shape[0], -1),
    ]
    if train and getattr(args, 'use_augm', False):
        max_shift = int(getattr(args, 'aug_chan_shift_max', 0))
        p = float(getattr(args, 'aug_chan_shift_p', 0.0))
        if max_shift > 0 and p > 0:
            gen = torch.Generator()
            if getattr(args, 'aug_seed', None) is not None:
                gen.manual_seed(int(args.aug_seed))
            ops.append(ChannelShift(max_shift=max_shift, p=p, generator=gen))
    return tonic.transforms.Compose(ops)


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
        gen.manual_seed(int(args.aug_seed) + 1)
    return BlendSameClass(
        p=p,
        union_clip=bool(getattr(args, 'blend_union_clip', False)),
        alpha=float(getattr(args, 'blend_alpha', 0.5)),
        generator=gen,
    )


def get_nmnist_data(args, first_saccade_only=True):
    # The Neuromorphic-MNIST (N-MNIST) dataset consists of 10 classes of handwritten digits (0-9)
    # recorded by a Dynamic Vision Sensor (DVS). The dataset contains 60k training and 10k test samples.

    in_shape = NMNIST.sensor_size
    out_shape = 10
    train_transform = _build_per_sample_transform(args, train=True, sensor_size=in_shape)
    test_transform = _build_per_sample_transform(args, train=False, sensor_size=in_shape)

    train_set = NMNIST(
        save_to=cache_dir,
        train=True,
        transform=train_transform,
        first_saccade_only=first_saccade_only,
    )
    test_set = NMNIST(
        save_to=cache_dir,
        train=False,
        transform=test_transform,
        first_saccade_only=first_saccade_only,
    )

    blend = _build_blend(args)
    train_collate = _make_collate_fn(blend, train=True)
    test_collate = _make_collate_fn(None, train=False)

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=train_collate,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=test_collate,
        num_workers=args.num_workers,
    )

    import brainstate
    return brainstate.util.DotDict(
        {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'in_shape': np.prod(in_shape),
            'out_shape': np.prod(out_shape),
        }
    )


def add_nmnist_augmentation_args(parser):
    args, _ = parser.parse_known_args()
    if getattr(args, 'use_augm', False):
        group = parser.add_argument_group('Data Augmentation (N-MNIST)')
        group.add_argument('--aug-seed', type=int, default=None, help='Seed for NMNIST augmentation RNGs.')
        group.add_argument('--aug-chan-shift-max', type=int, default=4,
                           help='Maximum absolute channel shift (flattened feature dim).')
        group.add_argument('--aug-chan-shift-p', type=float, default=0.5,
                           help='Probability to apply channel shift to a sample.')
        group.add_argument('--blend-same-class-p', type=float, default=0.0,
                           help='Probability to blend with a same-class partner in a batch.')
        group.add_argument('--blend-union-clip', action='store_true',
                           help='Use union-clip (binary) blending instead of alpha mix.')
        group.add_argument('--blend-alpha', type=float, default=0.5,
                           help='Alpha for continuous blending when not using union-clip.')
    return parser


if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_length', type=int, default=100)
    argparser.add_argument('--num_workers', type=int, default=0)
    argparser.add_argument('--batch-size', type=int, default=256)
    argparser.add_argument('--use-augm', type=bool, default=True)
    args = argparser.parse_args()

    data = get_nmnist_data(args)

    print('Train dataset:')
    for batch in data['train_loader']:
        print('input shape = ', batch[0].shape, 'target shape = ', batch[1].shape)
    print()


