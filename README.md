# ``BrainScale`` experiments on spiking neural networks



## Requirements


```bash

pip install BrainX[cuda12]==2025.10.20
pip install BrainX[cuda13]==2025.10.20
pip install h5py matplotlib msgpack prettytable tonic 
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

```


## Datasets

Download SHD dataset from the following link and place it in the `data/SHD/` folder:

- https://zenkelab.org/datasets/

With file structure as follows:

```
data/
└── SHD
    ├── shd_train.h5
    └── shd_test.h5
```


