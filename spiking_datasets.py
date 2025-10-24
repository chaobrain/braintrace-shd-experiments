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


import platform
from typing import Any, Callable, Optional, Tuple

import brainstate
import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

__all__ = [
    'load_shd_data',
    'add_data_augment_args',
]


class SpikingDataset(Dataset):
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
        xs = pad_sequence(xs, batch_first=True)
        ys = torch.LongTensor(ys).to(self.device)
        return xs, ys


def add_data_augment_args(parser):
    pass


def load_shd_data(args):
    train_dataset = SpikingDataset('shd', args.data_folder, 'train', args.data_length)
    test_dataset = SpikingDataset('shd', args.data_folder, 'test', args.data_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=train_dataset.generate_batch,
        shuffle=True,
        num_workers=0 if platform.system() == 'Windows' else args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=train_dataset.generate_batch,
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
