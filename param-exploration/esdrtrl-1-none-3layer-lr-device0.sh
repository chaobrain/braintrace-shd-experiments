#!/usr/bin/env bash
set -euo pipefail

cd ..

device=0

for lr in 0.02 0.01 0.005 0.002 0.001
do
python online_main.py --model_type RLIF --nb_epochs 100 --method esd-rtrl --nb_hiddens 1024 --devices "$device" --normalization none --lr $lr --lr_step_size 5 --etrace_decay 0.93 --nb_layers 3 --state_init rand --pdrop 0.1 --new_exp_folder esdrtrl-lr
python online_main.py --model_type RadLIF --nb_epochs 100 --method esd-rtrl --nb_hiddens 1024 --devices "$device" --normalization none --lr $lr --lr_step_size 5 --etrace_decay 0.97 --nb_layers 3 --state_init rand --pdrop 0.1 --new_exp_folder esdrtrl-lr
done

