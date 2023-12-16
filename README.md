# CoSoAdapt
Decentralized Domain Adaptation via Collaborative Aggregation and Source-Free Refinement for Foggy Scene Understanding

<p align="center">
<img src="assets/fig1.png" width="600px"/></p>

## Q&A
### What is the contribution?
This work studies a privacy-preserving problem definition named decentralized domain adaptation (DDA) and contributes a state-of-the-art DDA method termed CoSoAdapt for benchmarks in semantic foggy scene understanding (SFSU). Highlights are listed by: (1) A disentanglement and contrastive-learning based unpaired translation model called DisCoGAN is proposed to transfer domain invariance in collaborative aggregation. (2) A prototypical-knowledge based regularization adaptation model called ProRA is proposed to align joint distribution and denoise pseudo label in source-free refinement.

### Why is the contribution significant?
This work achieves privacy-preserving knowledge transfer in perspectives of federated domain adaptation and source domain-free adaptation. Impacts are listed by: (1) Concerning challenges of privacy and efficiency, federated self-weighted learning conquers uneven convergence and obtains lightweight communication. (2) Regarding limitations of fine-granularity and compactness, DisCoGAN transfers domain-specific attribute and preserves domain-invariant content, meanwhile ProRA aligns joint distribution and learns compact structure. (3) CoSoAdapt contributes a state- of-the-art model for DDA community, which reaches 55.26% mIoU on SFSU benchmark.
