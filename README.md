# GCIDM-DTransformer  
*A Multi-Level Spatiotemporal Feature Fusion Framework for Soft Sensor Modeling in Industrial Processes*  
*(Accepted at KDD 2025 Research Track)*  

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red)](https://pytorch.org/)
---

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
