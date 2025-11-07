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


def load_dataset(args):
    if args.dataset == 'shd':
        from dataset_shd import load_shd_data
        return load_shd_data(args)
    elif args.dataset == 'nmnist':
        from dataset_mnist import get_nmnist_data
        return get_nmnist_data(args)
    elif args.dataset == 'gesture':
        from dataset_gesture import get_dvs128_data_v1
        return get_dvs128_data_v1(
            args,
            split=args.train_val_split,
            augmentation=args.augmentation,
        )
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')


def add_data_augment_args(parser):
    args, _ = parser.parse_known_args()
    if args.dataset == 'shd':
        from dataset_shd import add_shd_augmentation_args
        parser = add_shd_augmentation_args(parser)
    elif args.dataset == 'gesture':
        from dataset_gesture import add_gesture_augmentation_args
        parser = add_gesture_augmentation_args(parser)
    elif args.dataset == 'nmnist':
        from dataset_mnist import add_nmnist_augmentation_args
        parser = add_nmnist_augmentation_args(parser)
    return parser
