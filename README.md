# Dancing with Noise: Advancing Generative Speech Enhancement with Distribution Augmentation
The implementation of Dancing with Noise: Advancing Generative Speech Enhancement with Distribution Augmentation.
## Environment Requirements
```
# create virtual environment
conda create --name DANS python=3.9.0

# activate environment
conda activate DANS

# install required packages
pip install -r requirements_py39.txt
```
## How to train
python train.py --log_dir <path_to_model> --base_dir <path_to_dataset>
## How to evaluate
python enhancement.py --test_dir <path_to_noisy> --enhanced_dir <path_to_enhanced> --ckpt <path_to_model_checkpoint>

python calc_metrics.py --clean_dir <path_to_clean> --noisy_dir <path_to_noisy> --enhanced_dir <path_to_enhanced>
## Folder Structure
```
.
├── calc_metrics.py
├── enhancement.py
├── README.md
├── requirements_py39.txt
├── dans
│   ├── backbones
│   │   ├── __init__.py
│   │   ├── ncsnpp.py
│   │   ├── ncsnpp_utils
│   │   │   ├── layerspp.py
│   │   │   ├── layers.py
│   │   │   ├── normalization.py
│   │   │   ├── op
│   │   │   │   ├── fused_act.py
│   │   │   │   ├── fused_bias_act.cpp
│   │   │   │   ├── fused_bias_act_kernel.cu
│   │   │   │   ├── __init__.py
│   │   │   │   ├── upfirdn2d.cpp
│   │   │   │   ├── upfirdn2d_kernel.cu
│   │   │   │   └── upfirdn2d.py
│   │   │   ├── up_or_down_sampling.py
│   │   │   └── utils.py
│   │   └── shared.py
│   ├── data_module.py
│   ├── model.py
│   ├── sampling
│   │   ├── correctors.py
│   │   ├── __init__.py
│   │   └── predictors.py
│   ├── sdes.py
│   └── util
│       ├── inference.py
│       ├── other.py
│       ├── registry.py
│       ├── semp.py
│       └── tensors.py
└── train.py
```

