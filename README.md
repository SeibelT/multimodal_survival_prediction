# Enhancing Feature Encoders through Multi-Modal and Multitask Masked Autoencoding for Survival Prediction from Whole Slide Images

## Introduction
This repository contains the code and resources for my master thesis, "Enhancing Feature Encoders through Multi-Modal and Multitask Masked Autoencoding for Survival Prediction from Whole Slide Images". This work is an extension and enhancement of the model presented in the paper "Pan-cancer integrative histology-genomic analysis via multimodal deep learning" by Chen et al[[1]](#1), which serves as the baseline model for this thesis.

## Objective
The primary objective of this thesis is to 
explore earlier fusions in multimodal survival analysis models based on histology and genomics features. 

In the first step, the baseline architecture is adapted such that 
the encoded genomics feature vector is concatenated to the bag of feature vectors extracted from the WSI. 

The main experiment observes the fusion of both modalities on a tile level during the encoding. For this task, a ViT MAE tiny[[2]](#2) pre-trained on Imagenet1k is further fine-tuned on the given dataset. 

## Features
- **Baseline Model**: Reimplementation of the proposed model from the referenced paper as a starting point[[1]](#1).
- **Earlier Fusion**: Implement additional aggregation models allowing earlier fusion in the processing pipeline.
- **Enhanced Pre-training**: Introduction of multimodal masked autoencoding and a survival task for pre-training the feature encoder. The code for the masked auto encoding, such as the ViT-mae-tiny model weights, was copied and adapted from [[2]](#2).


## Installation 
1. It is recommended to use a virtual environment with Python version 3.10.12

2. The required libraries can be installed from the provided text file      
    ```pip install -r requirements.txt  ```
<table>
<tr>
</tr>
<tr>
<td>

- h5py 3.10.0
- matplotlib 3.8.2
- numpy 1.26.2
- pandas 2.1.4
- Pillow 10.1.0
- scikit-learn 1.3.2
- scikit-survival 0.22.2
- scipy 1.11.4
- pytorch-lightning 2.1.3

</td>
<td>

- seaborn 0.13.0
- timm 0.9.12
- torch 2.1.2
- torchaudio 2.1.2
- torchmetrics 1.2.1
- torchvision 0.16.2
- tqdm 4.66.1
- wandb 0.16.1

</td>
</tr>
</table>

## Usage
The **aggregation** experiments can be started with 
```wandb sweep <path/to/file>.yaml```
The configuration files should be constructed as those in the configs/finalrun_configs folders. 
To run a regular train/validation/test split, it is advised to use aggregation.py 
Meanwhile, the kfold_sweep.py allows running a k-fold cross-validation with multiple wandb-agents. 

To ***train*** an encoder:    
```python encoder.py --mode train --config <path/to/file>.yaml```

To ***encode*** the dataset into bags of feature vectors with a new encoder:    
```python encoder.py --mode encode --config <path/to/file>.yaml```


## Data
Data is provided by [The Cancer Genome Atlas(TCGA) Research Network](https://www.cancer.gov/tcga). More precisely, the two cohorts 
Uterine Corpus Endometrial Carcinoma (UCEC) and 
 Breast Invasive Carcinoma (BRCA) were used in the scope of this research. 

## Citation

<a id="1">[1]</a> 
Shaoru Wang et al  (2023). 
A Closer Look at Self-Supervised Lightweight Vision Transformers.     
arXiv preprint arXiv:2205.14443
https://github.com/wangsr126/mae-lite

<a id="2">[2]</a> 
R J Chen et al  (2023). 
Pan-cancer integrative histology-genomic analysis via multimodal deep learning     
arXiv preprint arXiv:2108.02278
https://github.com/mahmoodlab/PORPOISE


## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details.

For more information on the GNU General Public License v3.0, please visit [https://www.gnu.org/licenses/gpl-3.0.en.html](https://www.gnu.org/licenses/gpl-3.0.en.html).

