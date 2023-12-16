# CoSoAdapt
Decentralized Domain Adaptation via Collaborative Aggregation and Source-Free Refinement for Foggy Scene Understanding

> **Abstract**:
> Semantic scene understanding emerges as a challenging task under adverse weather conditions such as fog, due to cognition uncertainty caused by visibility degradation. Although most advances have been made in natural photo-modality, existing methods rely heavily on dense pixel annotation and image collection. In contrast, virtual graphic-modality provides freely generated label and photo-realistic synthesized data. Hence, attaining cross-modality knowledge is desirable in fog adaptation. Motivated by privacy-preserving protocols, this work propose a novel decentralized domain adaptation (DDA) approach dubbed as CoSoAdapt that unifies collaborative pre-training and source-free fine-tuning. Our highlights consist of: (1) Concerning privacy and efficiency in decentralized learning, we develop federated self-weighted learning to optimize domain-invariant content encoder for central aggregation and domain-specific attribute encoder for central assembling. Note that disentanglement enables lightweight communication while holding data locally, simultaneously the assembled modality codes impute missing modality without reference image to avoid catastrophic forgetting on source domain. (2) Regarding fine-granularity and compactness in unsupervised adaptation, we first elaborate an unpaired modality translation model called DisCoGAN, which combines contrastive learning and cycle consistency. We further explore a prototypical knowledge based regularization adaptation model termed ProRA, which relaxes soft adversarial learning for joint distribution alignment and structure clustering for pseudo label refinement. Experimental results demonstrate efficacy of our CoSoAdapt that achieves 55.26% and 38.87% mean intersection-over-union (mIoU) on foggy scene benchmark (i.e., Foggy Cityscapes) and unseen datasets (i.e., Foggy Driving and Foggy Zurich), outperforming state-of-the-art DDA methods for semantic segmentation.
<p align="center">
<img src="assets/fig1.png" width="700px"/></p>

## Overview
### Q&A 1: What is the contribution?
This work studies a privacy-preserving problem definition named decentralized domain adaptation (DDA) and contributes a state-of-the-art DDA method termed CoSoAdapt for benchmarks in semantic foggy scene understanding (SFSU). Highlights are listed by: (1) A disentanglement and contrastive-learning based unpaired translation model called DisCoGAN is proposed to transfer domain invariance in collaborative aggregation. (2) A prototypical-knowledge based regularization adaptation model called ProRA is proposed to align joint distribution and denoise pseudo label in source-free refinement.

### Q&A 2: Why is the contribution significant?
This work achieves privacy-preserving knowledge transfer in perspectives of federated domain adaptation and source domain-free adaptation. Impacts are listed by: (1) Concerning challenges of privacy and efficiency, federated self-weighted learning conquers uneven convergence and obtains lightweight communication. (2) Regarding limitations of fine-granularity and compactness, DisCoGAN transfers domain-specific attribute and preserves domain-invariant content, meanwhile ProRA aligns joint distribution and learns compact structure. (3) CoSoAdapt contributes a state-of-the-art model for DDA community, which reaches 55.26% mIoU on SFSU benchmark.

## Usage

### Dependencies
Clone this repo and install required packages:
```
git clone https://github.com/jxthyatt/DDA-CoSoAdapt.git
conda create -n cosoadapt python=3.8
conda activate cosoadapt
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### Example commands
To train DisCoGAN between "GTA5" domain and "Cityscapes" domain, run the following script:
```
python train_discogan.py \
--name gta2cty \
--dataroot ./data/gta2cty \
--phase train \
--batch_size 16 \
--resize_size 512 \
--n_ep 400 \
--lambda_rec 10 \
--lambda_cls 1.0 \
--lambda_cls_G 5.0 \
--lambda_cl_attr 1.0 \
--lambda_cl_cont 1.0 \
--num_domains 2 \
--img_save_freq 10 \
--model_save_freq 20 \
--gpu 0
```




