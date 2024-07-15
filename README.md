# CoSoAdapt
Federated Hallucination Translation and Source-free Regularization Adaptation in Decentralized Domain Adaptation for Foggy Scene Understanding

> **Abstract**:
> Semantic foggy scene understanding (SFSU) emerges a challenging task under out-of-domain distribution (OD) due to uncertain cognition caused by degraded visibility. With the strong assumption of data centralization, unsupervised domain adaptation (UDA) reduces vulnerability under OD scenario. Whereas, enlarged domain gap and growing privacy concern heavily challenge conventional UDA. Motivated by gap decomposition and data decentralization, we establish a decentralized domain adaptation (DDA) framework called Translate thEn Adapt (abbr. TEA) for privacy preservation. Our highlights lie in: (1) Regarding federated hallucination translation, a Disentanglement and Contrastive-learning based Generative Adversarial Network (abbr. DisCoGAN) is proposed to impose contrastive prior and disentangle latent space in cycle-consistent translation. To yield domain hallucination, client minimizes cross-entropy of local classifier but maximizes entropy of global model to train translator. (2) Regarding source-free regularization adaptation, a Prototypical-knowledge based Regularization Adaptation (abbr. ProRA) is presented to align joint distribution in output space. Soft adversarial learning relaxes binary label to rectify inter-domain discrepancy and inner-domain divergence. Structure clustering and entropy minimization drive intra-class features closer and inter-class features apart. Extensive experiments exhibit efficacy of our TEA which achieves 55.26% or 46.25% mIoU in adaptation from GTA5 to Foggy Cityscapes or Foggy Zurich, outperforming other DDA methods for SFSU.
<p align="center">
<img src="assets/fig1.png" width="700px"/></p>

## Overview
### Q&A 1: What is the contribution？
- We propose a federated hallucination translation approach called DisCoGAN under decentralized source data scenario. It designs a multi-player game where image translator fools global model and local classifier retains the semantics of translated data. Specifically, contrastive learning is imposed to disentangle domain-invariant content space and domain-specific attribute space in cycle-consistent translation. To yield hallucination of unseen domain, each client minimizes local cross-entropy of local classifier but maximizes entropy of global model to train translator.
- We present a source-free regularization adaptation method called ProRA under decentralized target data scenario. It performs joint distribution alignment in both domain and class to regularize output space. Specially, soft adversarial learning relaxes hard (binary) label to rectify inter-domain discrepancy for under-aligned instance and inner-domain divergence for over-aligned instance. Class structure clustering refines underlying compactness and prototypical entropy minimization alleviates over-adaptation, which drives intra-class features closer and inter-class features apart.
- We establish a privacy-preserving decentralized domain adaptation framework called TEA to unify DisCoGAN and ProRA. It bridges virtual-to-real gap and clear-to-foggy gap, in addition to handle domain shift under data decentralization scenario. Experiment results exhibit our efficacy that achieves 55.26% or 51.82% in mIoU in adaptation from GTA5 or SYNTHIA to (seen) Foggy Cityscapes, and 46.25% or 36.40% in mIoU to (seen) Foggy Zurich, and 42.24% or 41.31% in mIoU to (unseen) Foggy Driving, which exceeds other methods for SFSU. Code will be released at URL.

### Q&A 2: Why is the contribution significant?
This work explores a fresh problem definition termed as DDA and builds a privacy-preserving adaptation framework dubbed as TEA, which handles domain shift for SFSU under data decentralization scenario. Specifically, a federated hallucination translation module called DisCoGAN is proposed to design a minimax game where image translator maximizes entropy of global model, while local classifier minimizes cross-entropy to ensure controllable hallucination. Besides, a source-free regularization adaptation module called ProRA is presented to align joint distribution in output space, comprised of soft adversarial learning, class structure clustering, and prototypical entropy minimization. Experiment results exhibit efficacy of our TEA that reaches 55.26% or 51.82% in mIoU on Foggy Cityscapes, and 25.85% or 24.03% in mIoU gain on Foggy Zurich, adapting from GTA5 or SYNTHIA.
Whereas, the DisCoGAN hinges on available reference images. Inspired by multimodal foundation models, we will engage in text-guided image style transfer to manipulate arbitrary style translation (e.g., fog, rain, snow, night) using only text description. Another limitation is that ProRA suffers from conflict between sparsity and accuracy. Further work will contain refinement strategy to ameliorate implicit uncertainty in prototypical knowledge.

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
To train DisCoGAN between [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) domain and [Cityscapes](https://www.cityscapes-dataset.com/) domain, run the following script:
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

To test DisCoGAN between [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) domain and [Cityscapes](https://www.cityscapes-dataset.com/) domain, run the following script:
```
python predict_discogan.py \
--name gta2cty \
--dataroot ./data/gta2cty \
--phase test \
--resume ./results/gta2cty/epoch_400.pth
--result_dir ./predict \
--resize_size 512 \
--num_domains 2 \
--gpu 0
```

To train ProRA between [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) domain and [Cityscapes](https://www.cityscapes-dataset.com/) domain, run the following script:
```
## Preliminary (source-only training)
python train_prora.py \
--name gta2ctylabv2_src \
--model_name deeplabv2 \
--stage source_only \
--freeze_bn \
--gan Vanilla \
--lr 2.5e-4 \
--adv 0.01 \
--src_rootpath datasets/GTA5

## Preliminary (prototypical knowledge)
python gen_proto.py \
--resume_path ./logs/gta2ctylabv2_src/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl \
--tgt_rootpath datasets/cityscapes

## Preliminary (soft adversarial label)
python generate_pseudo_label.py \
--name gta2ctylabv2_soft \
--soft \
--resume_path ./logs/gta2ctylabv2_src/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl \
--no_droplast \
--tgt_rootpath datasets/cityscapes

## Target adversarial learning
python train_prora.py \
--name gta2ctylabv2_softadv \
--stage stage1 \
--used_save_pseudo \
--ema \
--proto_rectify \
--moving_prototype \
--path_soft Pseudo/gta2ctylabv2_soft \
--resume_path ./logs/gta2ctylabv2_src/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl \
--proto_consistW 10 \
--rce \
--regular_w 0 \
--tgt_rootpath datasets/cityscapes

## Target structure denoising
python train_prora.py \
--name gta2ctylabv2_denoise \
--stage stage2 \
--used_save_pseudo \
--path_LP Pseudo/gta2ctylabv2_softadv \
--resume_path ./logs/gta2ctylabv2_softadv/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl \
--proto_pseudo 1 \
--kl_weight 1 \
--regular_w 10 \
--finetune \
--lr 1e-4 \
--tgt_rootpath datasets/cityscapes

## Target entropy minimization
python train_prora.py \
--name gta2ctylabv2_entropy \
--stage stage3 \
--used_save_pseudo \
--path_LP Pseudo/gta2ctylabv2_denoise \
--resume_path ./logs/gta2ctylabv2_denoise/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl \
--proto_pseudo 1 \
--kl_weight 0 \
--finetune \
--lr 6e-4 \
--student_init simclr \
--ema_bn \
--tgt_rootpath datasets/cityscapes
```

To test ProRA between [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) domain and [Cityscapes](https://www.cityscapes-dataset.com/) domain, run the following script:
```
python test_prora.py \
--name gta2ctylabv2_ft \
--student_init simclr \
--resume ./logs/gta2ctylabv2_entropy/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl
```

## Results
<p align="center">
<img src="assets/fig2.png" width="700px"/></p>
<p align="center">
<img src="assets/fig3.png" width="700px"/></p>

### More experimental details are comming soon!
