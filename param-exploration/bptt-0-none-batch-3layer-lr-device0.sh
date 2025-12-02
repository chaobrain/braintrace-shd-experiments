
cd ..
device=0

for b in batchnorm none
do
for lr in 0.02 0.01 0.005 0.002 0.001
do
  python offline_main.py --model_type RLIF --nb_epochs 100  --nb_hiddens 1024 --normalization $b --momentum 0.99  --lr $lr --lr_step_size 5 --devices "$device" --new_exp_folder bptt-batch-lr  --nb_layers 3
  python offline_main.py --model_type RadLIF --nb_epochs 100  --nb_hiddens 1024 --normalization $b --momentum 0.99  --lr $lr --lr_step_size 5 --devices "$device" --new_exp_folder bptt-batch-lr  --nb_layers 3
done
done

