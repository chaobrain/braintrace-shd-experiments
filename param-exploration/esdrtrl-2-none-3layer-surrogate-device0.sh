#!/usr/bin/env bash
set -euo pipefail

cd ..

device=0

for surrogate in boxcar relu gaussian multi_gaussian sigmoid; do
python online_main.py --model_type RLIF --nb_epochs 100 --method esd-rtrl --nb_hiddens 1024 --devices "$device" --normalization none --lr 0.01 --surrogate $surrogate --lr_step_size 5 --etrace_decay 0.93 --nb_layers 3 --state_init rand --pdrop 0.1 --new_exp_folder esdrtrl-surrogate
python online_main.py --model_type RadLIF --nb_epochs 100 --method esd-rtrl --nb_hiddens 1024 --devices "$device" --normalization none --lr 0.01 --surrogate $surrogate --lr_step_size 5 --etrace_decay 0.97 --nb_layers 3 --state_init rand --pdrop 0.1 --new_exp_folder esdrtrl-surrogate
done

