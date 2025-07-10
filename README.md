# GCIDM-DTransformer  
*A Multi-Level Spatiotemporal Feature Fusion Framework for Soft Sensor Modeling in Industrial Processes*  
*(Accepted at KDD 2025 Research Track)*  

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red)](https://pytorch.org/)
## Overview
Data-driven methods for soft sensor modeling have become increasingly pervasive in industrial applications. However, the high dimensionality, spatiotemporal dependencies, and noise in industrial data complicate the extraction of key features and the capture of global dependencies, making it difficult to model dynamic behaviors and resulting in limited predictive accuracy. To address these challenges, we propose a soft sensor modeling approach called CGIDM-DTransformer, which integrates multi-level spatiotemporal feature extraction. Specifically, we employ a channel-adaptive topology-refined graph convolution (CTR-GC) to model spatial correlations and enhance feature representations, followed by an inversion operation across variable dimensions to capture global dependencies. Based on these representations, we develop a derivative memory long short-term memory network (DM-LSTM) to model dynamic behaviors and improve sensitivity to temporal patterns. Additionally, we incorporate a multi-head differential attention (MHDA) mechanism into a Transformer architecture to optimize feature weighting and mitigate noise interference. Finally, We evaluate our method on two real-world industrial datasets, demonstrating its superior prediction accuracy and generalization performance compared to existing approaches. This showcases an efficient and reliable framework for soft sensor modeling.
<div  align="center">    
    <img src="./assets/framework.png" width=90%/>
</div>
<div  align="center">    
      Figure 1 :Architecture of the CGIDM-DTransformer Mode .
</div>

## Dependencies

| Environment | Version / Package                             |
|-------------|-----------------------------------------------|
| **Python**       | ≥ 3.9                                    |
| **PyTorch**      | ≥ 1.13.1                                 |
| **scikit-learn** | ≥ 1.3.2                                  |
| **scipy**        | ≥ 1.11.4                                 |
| **Others**  | numpy, tensorboard, openssl, oauthlib         |

[//]: # (```bash)

[//]: # (# Conda &#40;recommended&#41;)

[//]: # (conda env create -f env.yml)

[//]: # (conda activate gcidm)

## Dataset

| Name | #Samples | Sensors | Target |  
|------|---------:|---------|--------|
| Debutanizer | 2 394 | 7 | Butane | 
| SRU | 10 081 | 5 | SO2 | 

---

## Baselines

| `--setting` | Method |
|-------------|----------------------------------------------|
| `no`        | Vanilla backbone (no imbalance remedy)       |
| `reweight`  | Class-balanced loss re-weighting             |
| `upsampling`| Raw-domain over-sampling                     |
| `embed_up`  | Embed-SMOTE (latent over-sampling)           |
| `gcidm` _(default)_ | **GCIDM-DTransformer**               |

## Quick Example — Debutanizer
### 1 · Train
```bash
  # Train
python train.py --dataset 
