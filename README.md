# In-UCDS

Recommender systems are typically biased toward a small group of users, leading to severe unfairness in recommendation performance, i.e., User-Oriented Fairness (UOF) issue. The existing research on UOF is limited and fails to deal with the root cause of the UOF issue: the learning process between advantaged and disadvantaged users is unfair. To tackle this issue, we propose an In-processingUser Constrained Dominant Sets (In-UCDS) framework, which is a general framework that can be applied to any backbone recommendation model to achieve user-oriented fairness. We split In-UCDS into two stages, i.e., the UCDS modeling stage and thein-processing training stage. In the UCDS modeling stage, for each disadvantaged user, we extract a constrained dominant set (a user cluster) containing some advantaged users that are similar to it. In the in-processing training stage, we move the representations of disadvantaged users closer to their corresponding cluster by calculating a fairness loss. By combining the fairness loss with the original backbone model loss, we address the UOF issue and maintain the overall recommendation performance simultaneously. Comprehensive experiments on three real-world datasets demonstrate that In-UCDS outperforms the state-of-the-art methods, leading to a fairer model with better overall recommendation performance.

[In-processing User Constrained Dominant Sets for User-Oriented Fairness in Recommender Systems](https://dl.acm.org/doi/abs/10.1145/3581783.3613831)  
Zhongxuan Han, Chaochao Chen, Xiaolin Zheng, Weiming Liu, Jun Wang, Wenjie Cheng, Yuyuan Li  
MM '23: Proceedings of the 31st ACM International Conference on Multimedia

## Prerequisites 
* python==3.9.21
* numpy==1.23.0
* torch==1.13.0
* tqdm==4.64.0
* scikit-learn==0.24.2
* pandas==1.5.0
* matplotlib==3.5.1

## Getting Started
1. Clone this repo:  
```
   git clone https://github.com/hahhacx/In-UCDS.git  
   cd In-UCDS
``` 
2. Create a Virtual Environment  
```
   conda create -n UCDS python=3.9.21
   conda activate UCDS
```
3. Install all the dependencies  
```
   pip install -r requirements.txt
```

## Dataset
The complete source code, including datasets, is available on both [Baidu Netdisk](https://pan.baidu.com/s/1zNkoOw2R2PoFcepRvMIVzw?pwd=ppmb) and [Google Drive](https://drive.google.com/drive/folders/1L1tTwiRsuXU_pkDguPJA6BD_5IzJB0Fq?usp=sharing).

## Training 
Run the following command to start training:
```
python main.py
```

## Testing
The trained models (weights) are available at [here](https://drive.google.com/drive/folders/17D5sp5mdNOIgRAXohBHU5fBg6-T4BMlk?usp=sharing).Download it, replace the `logs` folder, select the recommendation model and dataset you want to evaluate, and run the following code:
```
python test.py
```

## Citation
If you find our code/models useful, please consider citing our paper: 
```
@inproceedings{han2023ucds,
author = {Han, Zhongxuan and Chen, Chaochao and Zheng, Xiaolin and Liu, Weiming and Wang, Jun and Cheng, Wenjie and Li, Yuyuan},
title = {In-processing User Constrained Dominant Sets for User-Oriented Fairness in Recommender Systems},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
pages = {6190–6201},
year = {2023},
series = {MM '23}
}
```
