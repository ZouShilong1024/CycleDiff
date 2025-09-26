# 🌟 CycleDiff [![arxiv paper](https://img.shields.io/badge/Project-Page-green)](https://zoushilong1024.github.io/CycleDiff.github.io/)	[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2508.06625)

<br> 
<div align=center>
    <img src="./assets/Cyclediff_2-archi.png" align="middle", width=900>
</div>
<br>

- [🌟 CycleDiff 	](#-cyclediff-)
  - [🛠 Installation](#-installation)
  - [🚀 Train CycleDiff from scratch](#-train-cyclediff-from-scratch)
    - [0. prepare dataset](#0-prepare-dataset)
    - [1. train VAE](#1-train-vae)
    - [2. train ldm](#2-train-ldm)
    - [3. train cycle translator](#3-train-cycle-translator)
  - [Test CycleDiff](#test-cyclediff)
  - [🙏 Acknowledgement](#-acknowledgement)
  - [📜 License](#-license)


## 🛠 Installation
```bash
conda create -n cyclediff python=3.9
conda activate CycleDiff
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirement.txt --no-deps
pip install -e git+https://github.com/toshas/torch-fidelity.git@v0.3.0#egg=torch-fidelity
```

<!-- ## 🔥 News -->
<!-- - 06-30-2025: Release pre-trained cat-to-dog image translation model. See USAGE.md for usage examples. -->

## 🚀 Train CycleDiff from scratch
### 0. prepare dataset
The training and testing dataset structure should look like:
```
datasetA2B
|-- train
|   |-- class_A
|   |   |-- 0.png
|   |   |-- 1.png
|   |   |-- ...
|   |-- class_B
|   |   |-- 0.png
|   |   |-- 1.png
|   |   |-- ...
|-- test
|   |-- class_A
|   |   |-- 0.png
|   |   |-- 1.png
|   |   |-- ...
|   |-- class_B
|   |   |-- 0.png
|   |   |-- 1.png
|   |   |-- ...
```
### 1. train VAE
```bash
accelerate launch train_vae.py --cfg ./configs/datasetA2B/{class_A}_ae_kl_256x256_d4.yaml
accelerate launch train_vae.py --cfg ./configs/datasetA2B/{class_B}_ae_kl_256x256_d4.yaml
```
### 2. train ldm
```bash
accelerate launch train_uncond_ldm.py --cfg ./configs/datasetA2B/{class_A}_ddm_const4_ldm_unet6_114.yaml
accelerate launch train_uncond_ldm.py --cfg ./configs/datasetA2B/{class_A}_ddm_const4_ldm_unet6_114.yaml
```
### 3. train cycle translator
```bash
accelerate launch train_uncond_ldm_cycle_C_discriminator_timestep_Adam_ode_2.py --cfg ./configs/afhq_cat2dog/cat_ddm_const4_ldm_unet6_114.yaml
```

## Test CycleDiff
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch translation_uncond_ldm_cycle_C_discriminator_generator_timestep_ode_2.py --cfg ./configs/summer2winter/translation_C_disc_timestep_ode_2.yaml
```

<!-- ## I. Cycle train on `Animal Faces-HQ v2 dataset (AFHQv2)`
### 1. prepare dataset
You can download the [AFHQv2 dataset](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq) as the following command:
```shell
bash download.sh afhq-v2-dataset
```
The training and testing dataset structure should look like:
```
afhqv2
|-- train
|   |-- cat
|   |   |-- flickr_cat_000002.png
|   |   |-- flickr_cat_000003.png
|   |   |-- ...
|   |-- dog
|   |   |-- flickr_dog_000002.png
|   |   |-- flickr_dog_000003.png
|   |   |-- ...
|-- test
|   |-- cat
|   |   |-- flickr_cat_000008.png
|   |   |-- flickr_cat_000011.png
|   |   |-- ...
|   |-- dog
|   |   |-- flickr_dog_000043.png
|   |   |-- flickr_dog_000045.png
|   |   |-- ...
```
### 2. train vae
```shell
accelerate launch train_vae.py --cfg ./configs/afhq_cat2dog/cat_ae_kl_256x256_d4.yaml
accelerate launch train_vae.py --cfg ./configs/afhq_cat2dog/dog_ae_kl_256x256_d4.yaml
```

### 3. train ldm
```shell
accelerate launch train_uncond_ldm.py --cfg ./configs/afhq_cat2dog/cat_ddm_const4_ldm_unet6_114.yaml
accelerate launch train_uncond_ldm.py --cfg ./configs/afhq_cat2dog/dog_ddm_const4_ldm_unet6_114.yaml
```
### 4. cycle train ldm
```shell
accelerate launch train_uncond_ldm_cycle_C_discriminator.py --cfg ./configs/afhq_cat2dog/translation_C_disc.yaml
python train_uncond_ldm_cycle_C_discriminator.py --cfg ./configs/afhq_cat2dog/translation_C_disc.yaml
```

### 5. train codebook
```shell
python generate_latent_gt.py --cfg=./configs/afhq_cat2dog/train_codebook.yaml
```
### 6. translation trained model, inclued two generators, two discriminators and fineturned ldm model
```shell
python translation_uncond_ldm_cycle_C_discriminator.py --cfg ./configs/afhq_cat2dog/translation_C_disc.yaml
```
### 7. calculate fid score
```shell
fidelity -g 0 -f -i -b {batch_size} --input1 {predict/images/path} --input2 {groundtruth/images/path}
```
## II. Cycle train on `horse2zebra dataset`

## III. Train your own CycleDiff Model
### 1. Data preparation

### 2. Model Training -->

## 🙏 Acknowledgement
Our Code is based on []()

## 📜 License
