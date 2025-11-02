
Without data augmentation:

```bash

python offline_main.py --model_type LIF --nb_epochs 100 --nb_hiddens 1024 --normalization none --lr 0.005 --devices 1 --new_exp_folder bptt-surrogate --nb_layers 3 --surrogate sigmoid
python offline_main.py --model_type RLIF --nb_epochs 100 --nb_hiddens 1024 --normalization none --lr 0.005 --devices 1 --new_exp_folder bptt-surrogate --nb_layers 3 --surrogate sigmoid
python offline_main.py --model_type adLIF --nb_epochs 100 --nb_hiddens 1024 --normalization none --lr 0.005 --devices 1 --new_exp_folder bptt-surrogate --nb_layers 3 --surrogate sigmoid
python offline_main.py --model_type RadLIF --nb_epochs 100 --nb_hiddens 1024 --normalization none --lr 0.005 --devices 1 --new_exp_folder bptt-surrogate --nb_layers 3 --surrogate sigmoid

```


With data augmentation:

```bash


python offline_main.py --model_type LIF --nb_epochs 150 --nb_hiddens 1024 --normalization none --lr 0.01 --devices 0 --new_exp_folder bptt-augm-shift-blend --nb_layers 3 --surrogate sigmoid --use_augm 1 --lr_step_size 20 --aug_random_shift 40 --aug_random_dilate_min 0.95 --aug_random_dilate_max 1.2 --aug_id_jitter_sigma 4 --aug_blend_probs 0.33,0.33,0.33   --aug_target_size 18000  --aug_step_freq 5
python offline_main.py --model_type RLIF --nb_epochs 150 --nb_hiddens 1024 --normalization none --lr 0.01 --devices 0 --new_exp_folder bptt-augm-shift-blend --nb_layers 3 --surrogate sigmoid --use_augm 1 --lr_step_size 20 --aug_random_shift 40 --aug_random_dilate_min 0.95 --aug_random_dilate_max 1.2 --aug_id_jitter_sigma 4 --aug_blend_probs 0.33,0.33,0.33   --aug_target_size 18000  --aug_step_freq 5
python offline_main.py --model_type adLIF --nb_epochs 150 --nb_hiddens 1024 --normalization none --lr 0.01 --devices 0 --new_exp_folder bptt-augm-shift-blend --nb_layers 3 --surrogate sigmoid --use_augm 1 --lr_step_size 20 --aug_random_shift 40 --aug_random_dilate_min 0.95 --aug_random_dilate_max 1.2 --aug_id_jitter_sigma 4 --aug_blend_probs 0.33,0.33,0.33   --aug_target_size 18000  --aug_step_freq 5
python offline_main.py --model_type RadLIF --nb_epochs 150 --nb_hiddens 1024 --normalization none --lr 0.01 --devices 0 --new_exp_folder bptt-augm-shift-blend --nb_layers 3 --surrogate sigmoid --use_augm 1 --lr_step_size 20 --aug_random_shift 40 --aug_random_dilate_min 0.95 --aug_random_dilate_max 1.2 --aug_id_jitter_sigma 4 --aug_blend_probs 0.33,0.33,0.33   --aug_target_size 18000  --aug_step_freq 5


```



