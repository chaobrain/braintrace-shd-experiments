```bash


python offline_main.py --model_type LIF --dataset_name shd  --nb_epochs 100 --method bptt --nb_hiddens 1024 --lr 0.02 --devices 0  --normalization batchnorm  --nb_layers 3 --pdrop 0.2

```

```bash


python online_main.py --model_type LIF --dataset_name shd  --nb_epochs 100 --method esd-rtrl --nb_hiddens 1024 --lr 0.02 --devices 0 --etrace_decay 0.88 --normalization layernorm  --nb_layers 3 --pdrop 0.2

```



