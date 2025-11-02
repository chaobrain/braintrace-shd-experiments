
Without data augmentation


```bash

python online_main.py --model_type LIF --nb_epochs 150 --method esd-rtrl --nb_hiddens 1024 --normalization none --lr 0.01 --lr_step_size 10 --etrace_decay 0.90 --nb_layers 3 --surrogate gaussian  --state_init zero --pdrop 0.3
python online_main.py --model_type RLIF --nb_epochs 150 --method esd-rtrl --nb_hiddens 1024 --normalization none --lr 0.01 --lr_step_size 40 --etrace_decay 0.91 --nb_layers 3 --surrogate gaussian  --state_init rand --pdrop 0.1

python online_main.py --model_type RadLIF --nb_epochs 200 --method esd-rtrl --nb_hiddens 1024 --normalization none --lr 0.01 --lr_step_size 10 --etrace_decay 0.98 --nb_layers 3 --surrogate gaussian  --state_init rand --pdrop 0.1

```


With data augmentation


```bash

python online_main.py --model_type LIF --nb_epochs 150 --method esd-rtrl --nb_hiddens 1024 --normalization none --lr 0.01 --etrace_decay 0.9 --nb_layers 3 --surrogate gaussian --lr_step_size 10 --state_init zero --pdrop 0.1 --use_augm 1 --aug_random_shift 40 --aug_random_dilate_min 0.95 --aug_random_dilate_max 1.2 --aug_id_jitter_sigma 4 --aug_blend_probs 0.33,0.33,0.33 --aug_target_size 18000 --aug_step_freq 5
python online_main.py --model_type RLIF --nb_epochs 200 --method esd-rtrl --nb_hiddens 1024 --normalization none --lr 0.01 --etrace_decay 0.95 --nb_layers 3 --surrogate gaussian  --use_augm 1 --aug_random_shift 40 --pdrop 0 --use_augm 1 --aug_random_shift 40 --aug_random_dilate_min 0.95 --aug_random_dilate_max 1.2 --aug_id_jitter_sigma 4
python online_main.py --model_type RLIF --nb_epochs 200 --method esd-rtrl --nb_hiddens 1024 --normalization none --lr 0.01 --etrace_decay 0.95 --nb_layers 3 --surrogate gaussian  --use_augm 1 --aug_random_shift 40 --pdrop 0 --use_augm 1 --aug_random_shift 40 --aug_random_dilate_min 0.95 --aug_random_dilate_max 1.2 --aug_id_jitter_sigma 4 --aug_blend_probs 0.33,0.33,0.33   --aug_target_size 18000  --aug_step_freq 5


```


