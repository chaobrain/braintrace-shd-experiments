# ``BrainScale`` experiments on spiking neural networks



## Requirements


```bash

pip install BrainX[cuda12]
pip install BrainX[cuda13]
pip install tonic
pip install h5py matplotlib msgpack prettytable numpy -U
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

```

## Experiments

The experiments are implemented in the `param-exploration` folder. Run the experiments as follows:

```bash
bash param-exploration/bptt-0-none-batch-3layer-lr-device0.sh
```

