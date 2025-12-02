#!/usr/bin/env bash
set -euo pipefail

cd ..
device=0

for s in boxcar relu gaussian multi_gaussian sigmoid; do
  python offline_main.py --model_type RLIF --nb_epochs 100 --nb_hiddens 1024 --normalization batchnorm --lr 0.005 --lr_step_size 5 --devices $device --new_exp_folder bptt-surrogate --nb_layers 3 --surrogate "$s"
  python offline_main.py --model_type RadLIF --nb_epochs 100 --nb_hiddens 1024 --normalization none --lr 0.005 --lr_step_size 5 --devices $device --new_exp_folder bptt-surrogate --nb_layers 3 --surrogate "$s"
done
