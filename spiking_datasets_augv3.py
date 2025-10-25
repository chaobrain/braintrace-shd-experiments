import copy
import platform
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import brainstate

__all__ = [
    "load_shd_data",
    "add_data_augment_args",
]


@dataclass
class AugmentationConfig:
    random_shift: Optional[int] = None
    random_dilate: Optional[Tuple[float, float]] = None
    id_jitter: Optional[float] = None
    blend_probs: Optional[List[float]] = None
    target_size: Optional[int] = None
    normalise_spike_number: bool = False
    seed: Optional[int] = None

    def enabled(self) -> bool:
        return any([
            self.random_shift is not None,
            self.random_dilate is not None,
            self.id_jitter is not None,
            self.blend_probs is not None,
            self.normalise_spike_number,
        ])


def _copy_events(events: Sequence[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
    return [
        {
            "x": np.asarray(ev["x"], dtype=np.int64).copy(),
            "t": np.asarray(ev["t"], dtype=np.float64).copy(),
        }
        for ev in events
    ]


def normalise_spike_number(events: List[Dict[str, np.ndarray]], rng: np.random.Generator) -> List[Dict[str, np.ndarray]]:
    if not events:
        return []

    min_count = min((len(ev["t"]) for ev in events if len(ev["t"]) > 0), default=0)
    if min_count == 0:
        return events

    out = []
    for ev in events:
        times = ev["t"]
        units = ev["x"]
        if len(times) <= min_count:
            out.append({"t": times.copy(), "x": units.copy()})
            continue
        keep_prob = min_count / len(times)
        mask = rng.random(len(times)) < keep_prob
        out.append({"t": times[mask], "x": units[mask]})
    return out


def random_shift(
    events: List[Dict[str, np.ndarray]],
    rng: np.random.Generator,
    max_shift: int,
    num_input: int,
) -> List[Dict[str, np.ndarray]]:
    out = []
    for ev in events:
        shift = int(rng.uniform(-max_shift, max_shift))
        units = ev["x"].astype(np.int64) + shift
        mask = (units >= 0) & (units < num_input)
        out.append({"x": units[mask], "t": ev["t"][mask]})
    return out


def id_jitter(
    events: List[Dict[str, np.ndarray]],
    rng: np.random.Generator,
    sigma: float,
    num_input: int,
) -> List[Dict[str, np.ndarray]]:
    out = []
    for ev in events:
        if len(ev["x"]) == 0:
            out.append({"x": ev["x"].copy(), "t": ev["t"].copy()})
            continue
        shift = np.round(rng.standard_normal(len(ev["x"])) * sigma).astype(np.int64)
        units = ev["x"].astype(np.int64) + shift
        mask = (units >= 0) & (units < num_input)
        out.append({"x": units[mask], "t": ev["t"][mask]})
    return out


def random_dilate(
    events: List[Dict[str, np.ndarray]],
    rng: np.random.Generator,
    min_factor: float,
    max_factor: float,
    trial_ms: float,
) -> List[Dict[str, np.ndarray]]:
    out = []
    log_min = np.log(min_factor)
    log_max = np.log(max_factor)
    for ev in events:
        if len(ev["t"]) == 0:
            out.append({"x": ev["x"].copy(), "t": ev["t"].copy()})
            continue
        factor = np.exp(rng.uniform(log_min, log_max))
        times = ev["t"] * factor
        mask = times < trial_ms
        out.append({"x": ev["x"][mask], "t": times[mask]})
    return out


def blend(
    samples: Sequence[Dict[str, np.ndarray]],
    probs: Sequence[float],
    rng: np.random.Generator,
    num_input: int,
    trial_ms: float,
) -> Dict[str, np.ndarray]:
    samples = copy.deepcopy(samples)
    mean_x = np.zeros(len(samples))
    mean_t = np.zeros(len(samples))

    for idx, sample in enumerate(samples):
        if len(sample["x"]) > 0:
            mean_x[idx] = np.mean(sample["x"])
            mean_t[idx] = np.mean(sample["t"])
        else:
            mean_x[idx] = 0.0
            mean_t[idx] = 0.0

    overall_mean_x = np.mean(mean_x)
    overall_mean_t = np.mean(mean_t)

    for idx, sample in enumerate(samples):
        sample["x"] = sample["x"] + int(overall_mean_x - mean_x[idx])
        sample["t"] = sample["t"] + int(overall_mean_t - mean_t[idx])

    new_x: List[int] = []
    new_t: List[float] = []
    for idx, sample in enumerate(samples):
        if len(sample["t"]) == 0:
            continue
        keep_mask = rng.uniform(0.0, 1.0, len(sample["t"])) < probs[idx]
        new_x.extend(sample["x"][keep_mask])
        new_t.extend(sample["t"][keep_mask])

    if not new_t:
        return {"x": np.array([], dtype=np.int64), "t": np.array([], dtype=np.float64)}

    new_x_arr = np.array(new_x, dtype=np.int64)
    new_t_arr = np.array(new_t, dtype=np.float64)
    order = np.argsort(new_t_arr)
    new_x_arr = new_x_arr[order]
    new_t_arr = new_t_arr[order]

    mask = (new_t_arr >= 0.0) & (new_t_arr < trial_ms)
    new_t_arr = new_t_arr[mask]
    new_x_arr = new_x_arr[mask]

    mask = (new_x_arr >= 0) & (new_x_arr < num_input)
    new_t_arr = new_t_arr[mask]
    new_x_arr = new_x_arr[mask]

    return {"x": new_x_arr, "t": new_t_arr}


def blend_dataset(
    events: List[Dict[str, np.ndarray]],
    labels: Sequence[int],
    rng: np.random.Generator,
    probs: Sequence[float],
    target_size: int,
    num_input: int,
    trial_ms: float,
) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
    events_array = np.array(events, dtype=object)
    labels_array = np.array(labels, dtype=np.int64)
    n_current = len(events_array)
    if target_size <= n_current:
        return list(events_array), labels_array.tolist()

    new_events: List[Dict[str, np.ndarray]] = []
    new_labels: List[int] = []
    for _ in range(target_size - n_current):
        idx = rng.integers(0, n_current)
        label = labels_array[idx]
        same_class_idx = np.where(labels_array == label)[0]
        if len(same_class_idx) == 0:
            continue
        picked = rng.integers(0, len(same_class_idx), len(probs))
        samples = [copy.deepcopy(events_array[same_class_idx[i]]) for i in picked]
        blended = blend(samples, probs, rng, num_input, trial_ms)
        new_events.append(blended)
        new_labels.append(int(label))

    combined_events = list(events_array) + new_events
    combined_labels = labels_array.tolist() + new_labels
    return combined_events, combined_labels


class EventPropSpikingDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        data_folder: str,
        split: str,
        nb_steps: int = 100,
        transform: Optional[Callable[[torch.Tensor, Any], Tuple[torch.Tensor, Any]]] = None,
        aug_config: Optional[AugmentationConfig] = None,
    ):
        self.device = "cpu"
        self.nb_steps = nb_steps
        self.nb_units = 700
        self.transform = transform
        self.aug_config = aug_config or AugmentationConfig()

        filename = f"{data_folder}/{dataset_name}_{split}.h5"
        self.h5py_file = h5py.File(filename, "r")
        self.firing_times = self.h5py_file["spikes"]["times"]
        self.units_fired = self.h5py_file["spikes"]["units"]
        self.labels = np.array(self.h5py_file["labels"], dtype=np.int64)

        self.max_time = 1.4
        self.trial_ms = self.max_time * 1000.0
        self.bin_size_ms = self.trial_ms / float(self.nb_steps)

        self._base_seed = self.aug_config.seed
        self._rng = np.random.default_rng(self._base_seed)

        self.original_events = self._load_events()
        self.current_events: List[Dict[str, np.ndarray]] = []
        self.current_labels: List[int] = []
        self.refresh_epoch()

    def _load_events(self) -> List[Dict[str, np.ndarray]]:
        events: List[Dict[str, np.ndarray]] = []
        for idx in range(len(self.labels)):
            times = np.asarray(self.firing_times[idx], dtype=np.float64) * 1000.0
            units = np.asarray(self.units_fired[idx], dtype=np.int64)
            mask = (units >= 0) & (units < self.nb_units)
            times = times[mask]
            units = units[mask]
            events.append({"t": times, "x": units})
        return events

    def refresh_epoch(self, epoch_seed: Optional[int] = None) -> None:
        if not self.aug_config.enabled():
            self.current_events = _copy_events(self.original_events)
            self.current_labels = self.labels.tolist()
            return

        if epoch_seed is not None and self._base_seed is not None:
            rng = np.random.default_rng(self._base_seed + epoch_seed)
        else:
            rng = self._rng

        events = _copy_events(self.original_events)
        labels = self.labels.tolist()

        if self.aug_config.normalise_spike_number:
            events = normalise_spike_number(events, rng)

        if self.aug_config.blend_probs is not None:
            target = self.aug_config.target_size or len(events)
            events, labels = blend_dataset(
                events,
                labels,
                rng,
                self.aug_config.blend_probs,
                target,
                self.nb_units,
                self.trial_ms,
            )

        if self.aug_config.random_shift is not None:
            events = random_shift(events, rng, self.aug_config.random_shift, self.nb_units)

        if self.aug_config.random_dilate is not None:
            events = random_dilate(
                events,
                rng,
                self.aug_config.random_dilate[0],
                self.aug_config.random_dilate[1],
                self.trial_ms,
            )

        if self.aug_config.id_jitter is not None:
            events = id_jitter(events, rng, self.aug_config.id_jitter, self.nb_units)

        self.current_events = events
        self.current_labels = labels

    def __len__(self) -> int:
        return len(self.current_events)

    def _rasterize(self, index: int) -> torch.Tensor:
        event = self.current_events[index]
        times = event["t"]
        units = event["x"]
        if len(times) == 0:
            return torch.zeros((self.nb_steps, self.nb_units), dtype=torch.float32)

        bin_indices = np.floor(times / self.bin_size_ms).astype(np.int64)
        bin_indices = np.clip(bin_indices, 0, self.nb_steps - 1)
        units = np.clip(units, 0, self.nb_units - 1)

        coords = np.vstack((bin_indices, units))
        values = np.ones(len(bin_indices), dtype=np.float32)
        tensor = torch.sparse_coo_tensor(coords, values, size=(self.nb_steps, self.nb_units))
        return tensor.to_dense()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        x = self._rasterize(index)
        y = int(self.current_labels[index])

        if self.transform is not None:
            x, y = self.transform(x, y)

        return x, y

    def generate_batch(self, batch):
        xs, ys = zip(*batch)
        xs = pad_sequence(xs, batch_first=True)
        ys = torch.LongTensor(ys)
        return xs, ys


def _build_aug_config(args) -> AugmentationConfig:
    cfg = AugmentationConfig()
    cfg.random_shift = getattr(args, "aug_random_shift", None)

    min_factor = getattr(args, "aug_random_dilate_min", None)
    max_factor = getattr(args, "aug_random_dilate_max", None)
    if min_factor is not None and max_factor is not None:
        cfg.random_dilate = (min_factor, max_factor)

    cfg.id_jitter = getattr(args, "aug_id_jitter_sigma", None)

    blend_probs = getattr(args, "aug_blend_probs", None)
    if blend_probs is not None:
        cfg.blend_probs = blend_probs
        cfg.target_size = getattr(args, "aug_target_size", None)

    cfg.normalise_spike_number = getattr(args, "aug_normalise_spike_num", False)
    cfg.seed = getattr(args, "aug_seed", None)
    return cfg


def load_shd_data(args):
    use_augm = getattr(args, "use_augm", False)
    aug_config = _build_aug_config(args) if use_augm else AugmentationConfig()

    test_aug_config = AugmentationConfig(
        normalise_spike_number=aug_config.normalise_spike_number,
        seed=aug_config.seed,
    )

    train_dataset = EventPropSpikingDataset(
        "shd",
        args.data_folder,
        "train",
        args.data_length,
        transform=None,
        aug_config=aug_config,
    )

    test_dataset = EventPropSpikingDataset(
        "shd",
        args.data_folder,
        "test",
        args.data_length,
        transform=None,
        aug_config=test_aug_config,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=train_dataset.generate_batch,
        shuffle=True,
        num_workers=0 if platform.system() == "Windows" else args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=test_dataset.generate_batch,
        shuffle=False,
        num_workers=0 if platform.system() == "Windows" else args.num_workers,
    )

    return brainstate.util.DotDict(
        {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "in_shape": 700,
            "out_shape": 20,
            "input_process": lambda x: x,
        }
    )


def add_data_augment_args(parser):
    parser.add_argument(
        "--aug_random_shift",
        type=int,
        default=None,
        help="Maximum absolute channel shift (integer).",
    )
    parser.add_argument(
        "--aug_random_dilate_min",
        type=float,
        default=None,
        help="Minimum dilation factor for random_dilate (exclusive use when both min and max provided).",
    )
    parser.add_argument(
        "--aug_random_dilate_max",
        type=float,
        default=None,
        help="Maximum dilation factor for random_dilate.",
    )
    parser.add_argument(
        "--aug_id_jitter_sigma",
        type=float,
        default=None,
        help="Standard deviation for neuron ID jittering.",
    )
    parser.add_argument(
        "--aug_blend_probs",
        type=lambda s: [float(x) for x in s.split(",") if x],
        default=None,
        help="Comma separated probabilities for blend augmentation.",
    )
    parser.add_argument(
        "--aug_target_size",
        type=int,
        default=None,
        help="Target dataset size after blend augmentation.",
    )
    parser.add_argument(
        "--aug_normalise_spike_num",
        action="store_true",
        help="Enable spike count normalisation before augmentations.",
    )
    parser.add_argument(
        "--aug_seed",
        type=int,
        default=None,
        help="Base random seed for augmentation pipeline.",
    )
