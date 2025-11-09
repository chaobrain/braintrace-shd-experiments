#!/usr/bin/env bash
set -euo pipefail

cd ..

device=0

for ed in 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99
do
python online_main.py --model_type RLIF --nb_epochs 100 --method esd-rtrl --nb_hiddens 1024 --devices "$device" --normalization none --lr 0.01 --lr_step_size 5 --etrace_decay $ed --nb_layers 3 --state_init rand --pdrop 0.1 --new_exp_folder esdrtrl-etrace-decay
python online_main.py --model_type RadLIF --nb_epochs 100 --method esd-rtrl --nb_hiddens 1024 --devices "$device" --normalization none --lr 0.01 --lr_step_size 5 --etrace_decay $ed --nb_layers 3 --state_init rand --pdrop 0.1 --new_exp_folder esdrtrl-etrace-decay
done

