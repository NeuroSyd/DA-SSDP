# Synchrony-Gated Plasticity with Dopamine Modulation for Spiking Neural Networks

Official PyTorch implementation of **DA-SSDP** (Dopamine-modulated Spike-Synchrony-Dependent Plasticity), proposed in

> **Synchrony-Gated Plasticity with Dopamine Modulation for Spiking Neural Networks**  
> Transactions on Machine Learning Research (TMLR)

DA-SSDP is a light-weight, synchrony-based plasticity rule that runs **during training only**.  
It plugs into deep spiking transformers such as **SpikingResformer** and adds a local update on top of surrogate backpropagation, without changing the forward architecture or inference cost.

The training scripts and configs follow the original **SpikingResformer** codebase.

---

## 1. Installation

numpy==1.24.4
Pillow==11.2.1
PyYAML==5.4.1
PyYAML==6.0.2
spikingjelly==0.0.0.0.15
thop==0.1.1.post2209072238
timm==1.0.15
torch==1.12.1
torchvision==0.13.1


---

## 2. Datasets

### 2.1 ImageNet-1K

Organize ImageNet in the standard way:

    /path/to/imagenet
    |-- train
    |   |-- n01440764
    |   |-- n01443537
    |   `-- ...
    `-- val
        |-- n01440764
        |-- n01443537
        `-- ...

## 3. Training recipes

The training pipeline (optimizer, schedules, augmentations) is the same as in SpikingResformer.  
DA-SSDP is enabled or disabled through the YAML config.

### 3.1 ImageNet with DA-SSDP (SpikingResformer-S backbone)

Example: distributed training on 8 GPUs

    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc-per-node=8 \
        main.py \
        -- configs/main/spikingresformer_l.yaml \
        --data-path /path/to/imagenet \
        --output-dir /path/to/output_da_ssdp

Baseline SpikingResformer-L (no DA-SSDP) with the same setup:

    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc-per-node=8 \
        main.py \
        --configs/main/spikingresformer_l.yaml \
        --data-path /path/to/imagenet \
        --output-dir /path/to/output_baseline


---

### 3.2 Direct training on CIFAR-10 / CIFAR-100

Example: direct training on CIFAR-10 with a tiny backbone plus DA-SSDP

    python main.py \
        --configs/direct_training/cifar10.yaml \
        --data-path /path/to/cifar100 \


Example: CIFAR-100

    python main.py \
        --configs/direct_training/cifar100.yaml \
        --data-path /path/to/cifar100 \
        --output-dir /path/to/output_cifar100


---

### 3.3 Transfer learning from CIFAR10-DVS

Example: fine-tune an ImageNet-pretrained SpikingResformer-S + DA-SSDP on CIFAR10-DVS

    python main.py \
        --configs/transfer/cifar10dvs.yaml \
        --data-path /path/to/cifar10 \
        --output-dir /path/to/output_cifar10_ft \
        --transfer /path/to/imagenet_checkpoint.pth


---


