import platform
import random
from typing import Any, Callable, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

__all__ = [
    'load_shd_data',
    'SHDAugmentation',
]


class SHDAugmentation:
    """Data augmentation techniques for Spiking Heidelberg Digits dataset"""

    def __init__(
        self,
        temporal_jitter_std: float = 3.0,  # ms
        time_warp_factor: Tuple[float, float] = (0.9, 1.1),
        spike_drop_prob: float = 0.1,
        noise_spike_prob: float = 0.02,
        channel_shift_range: int = 3,
        channel_mask_ratio: float = 0.15,
        spike_time_noise_std: float = 2.0,  # ms
        mixup_alpha: float = 0.2,
        augment_prob: float = 0.5,
    ):
        self.temporal_jitter_std = temporal_jitter_std
        self.time_warp_factor = time_warp_factor
        self.spike_drop_prob = spike_drop_prob
        self.noise_spike_prob = noise_spike_prob
        self.channel_shift_range = channel_shift_range
        self.channel_mask_ratio = channel_mask_ratio
        self.spike_time_noise_std = spike_time_noise_std
        self.mixup_alpha = mixup_alpha
        self.augment_prob = augment_prob

    def __call__(self, x: torch.Tensor, y: Any) -> Tuple[torch.Tensor, Any]:
        if random.random() > self.augment_prob:
            return x, y

        # Apply 2-3 random augmentations
        augmentations = [
            self.temporal_jitter,
            self.time_warp,
            self.spike_dropout,
            self.add_noise_spikes,
            self.channel_shift,
            self.channel_mask,
            self.spike_time_perturbation,
        ]

        # Randomly select 2-3 augmentations
        num_augs = random.randint(2, 3)
        selected_augs = random.sample(augmentations, num_augs)

        for aug_fn in selected_augs:
            if random.random() < 0.5:  # Apply each selected augmentation with 50% probability
                x = aug_fn(x)

        return x, y

    def temporal_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Add temporal jittering to spike times"""
        if x.sum() == 0:  # No spikes to jitter
            return x

        # Convert dense to sparse representation
        spike_indices = torch.nonzero(x, as_tuple=True)
        if len(spike_indices[0]) == 0:
            return x

        # Add jitter to time indices
        time_indices = spike_indices[0].float()
        jitter = torch.randn_like(time_indices) * self.temporal_jitter_std
        time_indices = (time_indices + jitter).clamp(0, x.shape[0] - 1).long()

        # Reconstruct spike train
        new_x = torch.zeros_like(x)
        new_x[time_indices, spike_indices[1]] = 1.0

        return new_x

    def time_warp(self, x: torch.Tensor) -> torch.Tensor:
        """Stretch or compress spike train temporally"""
        if x.sum() == 0:
            return x

        factor = random.uniform(*self.time_warp_factor)
        T, C = x.shape

        # Reshape for interpolation
        x_reshaped = x.unsqueeze(0).unsqueeze(0)  # [1, 1, T, C]

        # Calculate new time dimension
        new_T = int(T * factor)

        # Interpolate
        x_warped = F.interpolate(
            x_reshaped,
            size=(new_T, C),
            mode='nearest'
        ).squeeze()

        # Pad or truncate to original length
        if new_T > T:
            x_warped = x_warped[:T, :]
        elif new_T < T:
            pad_length = T - new_T
            x_warped = F.pad(x_warped, (0, 0, 0, pad_length))

        # Ensure binary spikes
        x_warped = (x_warped > 0.5).float()

        return x_warped

    def spike_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly drop spikes"""
        if x.sum() == 0:
            return x

        # Create dropout mask
        spike_mask = torch.rand_like(x) > self.spike_drop_prob
        return x * spike_mask.float()

    def add_noise_spikes(self, x: torch.Tensor) -> torch.Tensor:
        """Add random noise spikes"""
        noise_mask = torch.rand_like(x) < self.noise_spike_prob
        # Only add noise where there are no existing spikes
        noise_spikes = noise_mask.float() * (1 - x)
        return torch.clamp(x + noise_spikes, 0, 1)

    def channel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Shift spike patterns across frequency channels"""
        if x.sum() == 0:
            return x

        shift = random.randint(-self.channel_shift_range, self.channel_shift_range)
        if shift == 0:
            return x

        return torch.roll(x, shifts=shift, dims=1)

    def channel_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly mask frequency channels"""
        T, C = x.shape
        num_mask = int(C * self.channel_mask_ratio)

        # Random channels to mask
        mask_indices = torch.randperm(C)[:num_mask]
        x_masked = x.clone()
        x_masked[:, mask_indices] = 0

        return x_masked

    def spike_time_perturbation(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to individual spike times"""
        if x.sum() == 0:
            return x

        spike_indices = torch.nonzero(x, as_tuple=True)
        if len(spike_indices[0]) == 0:
            return x

        # Add noise to time indices
        time_indices = spike_indices[0].float()
        noise = torch.randn_like(time_indices) * self.spike_time_noise_std
        time_indices = (time_indices + noise).clamp(0, x.shape[0] - 1).long()

        # Reconstruct spike train
        new_x = torch.zeros_like(x)
        new_x[time_indices, spike_indices[1]] = 1.0

        return new_x


class MixupAugmentation:
    """Mixup augmentation for spike trains"""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.dataset = None

    def set_dataset(self, dataset):
        """Set reference to dataset for sampling"""
        self.dataset = dataset

    def __call__(self, x: torch.Tensor, y: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup augmentation

        Returns:
            Augmented spike train and one-hot encoded mixed label
        """
        if self.dataset is None or random.random() > 0.5:
            # Return one-hot encoded label without mixup
            y_onehot = torch.zeros(20)  # 20 classes for SHD
            y_onehot[y] = 1.0
            return x, y_onehot

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Sample another example from dataset
        idx = random.randint(0, len(self.dataset) - 1)
        x2, y2 = self.dataset[idx]
        y2 = torch.as_tensor(y2, dtype=torch.long)

        # Ensure x2 has same shape as x
        if x2.shape[0] != x.shape[0]:
            # Pad or truncate x2
            if x2.shape[0] > x.shape[0]:
                x2 = x2[:x.shape[0]]
            else:
                pad_length = x.shape[0] - x2.shape[0]
                x2 = F.pad(x2, (0, 0, 0, pad_length))

        # Mix spike trains
        x_mixed = lam * x + (1 - lam) * x2

        # Threshold to maintain sparsity
        x_mixed = (x_mixed > 0.5).float()

        # Mix labels (one-hot encoding)
        y_mixed = torch.zeros(20)
        y_mixed[y] = lam
        y_mixed[y2] = 1 - lam

        return x_mixed, y_mixed


class SpikingDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        data_folder: str,
        split: str,
        nb_steps: int = 100,
        transform: Optional[Callable[[torch.Tensor, Any], Tuple[torch.Tensor, Any]]] = None,
        enable_mixup: bool = False,
    ):
        # Fixed parameters
        self.device = "cpu"  # to allow pin memory
        self.nb_steps = nb_steps
        self.nb_units = 700
        self.transform = transform
        self.enable_mixup = enable_mixup

        # Setup mixup if enabled
        if self.enable_mixup and split == 'train':
            self.mixup = MixupAugmentation(alpha=0.2)
            self.mixup.set_dataset(self)
        else:
            self.mixup = None

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
        # Rasterize to sparse [T, C]
        times = np.digitize(self.firing_times[index], self.time_bins)
        units = self.units_fired[index]
        x_idx = torch.LongTensor(np.array([times, units])).to(self.device)
        x_val = torch.FloatTensor(np.ones(len(times))).to(self.device)
        x_size = torch.Size([self.nb_steps, self.nb_units])  # [T, C]
        x = torch.sparse_coo_tensor(x_idx, x_val, x_size).to(self.device).to_dense()
        y = int(self.labels[index])

        # Apply standard augmentations
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Apply mixup if enabled (for training)
        if self.mixup is not None and self.training:
            x, y = self.mixup(x, y)

        return x, y

    def generate_batch(self, batch):
        xs, ys = zip(*batch)
        xs = pad_sequence(xs, batch_first=True)

        # Handle both integer labels and one-hot encoded labels (from mixup)
        if isinstance(ys[0], torch.Tensor):
            # Mixup case - stack one-hot vectors
            ys = torch.stack(ys).to(self.device)
        else:
            # Normal case - convert to LongTensor
            ys = torch.LongTensor(ys).to(self.device)

        return xs, ys

    def set_training(self, mode: bool):
        """Set training mode for dataset"""
        self.training = mode


def load_shd_data(args):
    # Create augmentation transform for training
    train_augmentation = SHDAugmentation(
        temporal_jitter_std=3.0,
        time_warp_factor=(0.9, 1.1),
        spike_drop_prob=0.1,
        noise_spike_prob=0.02,
        channel_shift_range=3,
        channel_mask_ratio=0.15,
        spike_time_noise_std=2.0,
        mixup_alpha=0.2,
        augment_prob=args.augment_prob
    )

    # Enable mixup if specified in args
    enable_mixup = args.enable_mixup

    train_dataset = SpikingDataset(
        'shd',
        args.data_folder,
        'train',
        args.data_length,
        transform=train_augmentation,
        enable_mixup=enable_mixup
    )
    train_dataset.set_training(True)

    test_dataset = SpikingDataset(
        'shd',
        args.data_folder,
        'test',
        args.data_length,
        transform=None,  # No augmentation for test set
        enable_mixup=False
    )
    test_dataset.set_training(False)

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
        collate_fn=test_dataset.generate_batch,
        shuffle=False,
        num_workers=0 if platform.system() == 'Windows' else args.num_workers,
    )

    import brainstate
    return brainstate.util.DotDict(
        {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'in_shape': 700,
            'out_shape': 20,
            'input_process': lambda x: x,
        }
    )


def add_data_augment_args(parser):
    args, _ = parser.parse_known_args()
    if args.use_augm:
        parser.add_argument('--augment_prob', type=float, default=0.5)
        parser.add_argument('--enable_mixup', action='store_true')


# Example usage with configuration
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='./data/SHD')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_length', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--augment_prob', type=float, default=0.5)
    parser.add_argument('--enable_mixup', action='store_true', default=True)

    args = parser.parse_args()

    # Load data with augmentation
    data = load_shd_data(args)

    # Test loading a batch
    for x, y in data.train_loader:
        print(f"Batch shape: {x.shape}, Labels shape: {y.shape}")
        print(f"Sparsity: {(x > 0).float().mean():.3f}")
        break
