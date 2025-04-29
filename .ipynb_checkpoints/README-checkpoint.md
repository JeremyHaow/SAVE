## Requirements

Install the environment as follows:

```python
# create conda environment
conda create -n SAFE -y python=3.9
conda activate SAFE
# install pytorch 
pip install torch==2.2.1 torchvision==0.17.1
# install other dependencies
pip install -r requirements.txt
```

## Directory structure

<details>
<summary> You should organize the above data as follows: </summary>

```
data/datasets
|-- train_ForenSynths
|   |-- train
|   |   |-- car
|   |   |-- cat
|   |   |-- chair
|   |   |-- horse
|   |-- val
|   |   |-- car
|   |   |-- cat
|   |   |-- chair
|   |   |-- horse
|-- test1_ForenSynths/test
|   |-- biggan
|   |-- cyclegan
|   |-- deepfake
|   |-- gaugan
|   |-- progan
|   |-- stargan
|   |-- stylegan
|   |-- stylegan2
|-- test2_Self-Synthesis/test
|   |-- AttGAN
|   |-- BEGAN
|   |-- CramerGAN
|   |-- InfoMaxGAN
|   |-- MMDGAN
|   |-- RelGAN
|   |-- S3GAN
|   |-- SNGAN
|   |-- STGAN
|-- test3_Ojha/test
|   |-- dalle
|   |-- glide_100_10
|   |-- glide_100_27
|   |-- glide_50_27
|   |-- guided          # Also known as ADM.
|   |-- ldm_100
|   |-- ldm_200
|   |-- ldm_200_cfg
|-- test4_GenImage/test
|   |-- ADM
|   |-- BigGAN
|   |-- Glide
|   |-- Midjourney
|   |-- stable_diffusion_v_1_4
|   |-- stable_diffusion_v_1_5
|   |-- VQDM
|   |-- wukong
|-- test5_DiTFake/test
|   |-- FLUX.1-schnell
|   |-- PixArt-Sigma-XL-2-1024-MS
|   |-- stable-diffusion-3-medium-diffusers
```
</details>

## Training

```
bash scripts/train.sh
```

This script enables training with 4 GPUs, you can specify the number of GPUs by setting `GPU_NUM`.

## Inference

```
bash scripts/eval.sh
```