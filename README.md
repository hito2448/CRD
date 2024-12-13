# CRD

PyTorch Implementation of "Multimodal Industrial Anomaly Detection by Crossmodal Reverse Distillation".
[paper](https://arxiv.org/abs/2412.08949)

## 1. Environment
Create a new conda environment firstly.
```
conda create -n CRD python=3.8
conda activate CRD
pip install -r requirements.txt
```

## 2. Prepare Data
###  MVTec 3D AD Dataset
Download MVTec 3D AD dataset from [MVTec 3D AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad/). 
Unzip the file to `./data/`.
```
|--data
    |-- mvtec_3d_anomaly_detection
        |-- bagel
            |-- train
            |-- validation
            |-- test
        |-- ...
```

###  Eyecandies Dataset
Download Eyecandies dataset from [Eyecandies](https://eyecan-ai.github.io/eyecandies). 
Unzip the file to `./data/`.
```
|--data
    |-- Eyecandies
        |-- CandyCane
            |-- train
            |-- val
            |-- test_private
            |-- test_public
        |-- ...
```
And run the preprocessing to maintain consistency with the previous methods 
```bash
python ./utils/preprocessing.py
```
```
|--data
    |-- Eyecandies_preprocessed
        |-- CandyCane
            |-- train
            |-- validation
            |-- test
        |-- ...
```

## 3.Train and Test
To get the training and inference results, simply execute the following command.

For MVTec 3D AD Dataset:
```bash
python train_MVTec3D.py
```

For Eyecandies Dataset:
```bash
python train_Eyecandies.py
```

## Citation
If you think this work is helpful to you, please consider citing our paper.
```
@article{crd2024,
  title={Multimodal Industrial Anomaly Detection by Crossmodal Reverse Distillation},
  author={Xinyue Liu and Jianyuan Wang and Biao Leng and Shuo Zhang},
  journal={arXiv preprint arXiv:2412.08949},
  year={2024}
}
```
