## Introduction

This repository is a fork of the [Lerobots GitHub project](https://github.com/huggingface/lerobot), introducing a standard Behavioral Cloning (BC) benchmark model. The goal is to provide a simple and consistent baseline for comparing against state-of-the-art methods already implemented in Lerobots.

### Model Architecture

The BC model adopts the same feature extractor backend as the Diffusion Policy (DP), extending it with a straightforward multi-layer perceptron (MLP) policy head. This head outputs an **action chunk** based on historical image and robot state data.

![bc_arch](https://github.com/user-attachments/assets/b2ddea0a-75b0-4410-958e-371544e80caf)

**Key notations:**
- `t`: current time step  
- `n`: historical lookback window  
- `I[t−n:t]`: sequence of RGB camera images over the lookback window  
- `o[t−n:t]`: sequence of robot observations (e.g., joint positions)  
- `a[t−n:t+k+n]`: predicted sequence of actions over a future horizon `k`

### Training Objective

The model is trained using Mean Squared Error (MSE) loss with L2 regularization to mitigate overfitting.

### Benchmark Evaluation

We evaluate our BC model on the **PushT** task, comparing it against:
- Action Chunking Transformer (ACT)  
- Diffusion Policy (DP)  
- Vector-Quantized Behavior Transformer (VQBeT)  

The results are visualized below:
![results](https://github.com/user-attachments/assets/6ee061df-9fa0-444e-8fb9-5e715bc80b1a)


### Code Additions and Modifications

The following updates were made to integrate the BC model into the Lerobots codebase:

```
.
├── lerobot
|   └── common
|       ├── policies
|       |    └── bc
|       |        ├── modeling_bc.py        # code for the BC model
|       |        └── configuration_bc.py   # config for setting BC model parameters
|       └── factory.py                     # updated to register BC model
└── demo
    ├── demo_methods.py                    # runner scripts to evaluate BC, ACT, DP and VQBeT 
    └── make_figures.py                    # helper script to make evaluation plots
```

## Installation
We include a brief installation guide here, adapted from [Lerobots](https://github.com/huggingface/lerobot). For full details please visit the main page of their github directly.
Download source code:
```bash
git clone https://github.com/GregHydeDartmouth/lerobot_bc_and_demo.git
cd lerobot_bc_and_demo
```

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html):
```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

When using `miniconda`, install `ffmpeg` in your environment:
```bash
conda install ffmpeg -c conda-forge
```

> **NOTE:** This usually installs `ffmpeg 7.X` for your platform compiled with the `libsvtav1` encoder. If `libsvtav1` is not supported (check supported encoders with `ffmpeg -encoders`), you can:
>  - _[On any platform]_ Explicitly install `ffmpeg 7.X` using:
>  ```bash
>  conda install ffmpeg=7.1.1 -c conda-forge
>  ```
>  - _[On Linux only]_ Install [ffmpeg build dependencies](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#GettheDependencies) and [compile ffmpeg from source with libsvtav1](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#libsvtav1), and make sure you use the corresponding ffmpeg binary to your install with `which ffmpeg`.

Install 🤗 LeRobot:
```bash
pip install -e .
```

> **NOTE:** If you encounter build errors, you may need to install additional dependencies (`cmake`, `build-essential`, and `ffmpeg libs`). On Linux, run:
`sudo apt-get install cmake build-essential python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev pkg-config`. For other systems, see: [Compiling PyAV](https://pyav.org/docs/develop/overview/installation.html#bring-your-own-ffmpeg)

For simulations, 🤗 LeRobot comes with gymnasium environments that can be installed as extras:
- [aloha](https://github.com/huggingface/gym-aloha)
- [xarm](https://github.com/huggingface/gym-xarm)
- [pusht](https://github.com/huggingface/gym-pusht)

For instance, to install 🤗 LeRobot with aloha and pusht, use:
```bash
pip install -e ".[aloha, pusht]"
```
