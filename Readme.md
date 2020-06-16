# FER-Trainer-master

A CNN based pytorch implementation on facial expression recognition.
## 简介
本项为人脸表情识别的训练工程代码,The Facial Expression Recognition master.

本项目文件结构如下：
1. `models`，包含常用网络的backbone，方便快速开发验证.
2. `loss`，包含centerloss 与 island loss.
3. `transforms`，训练时图像增强抖动的代码模块.
4. `utils`，一些工能性代码模块：mixup，cutmix，grad_clip等.
5. `tools`，可能用到的小工具.

获取方式：
```
git clone https://github.com/alexuechao/FER-Trainer-master.git
```
## labels ##
labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

## Dependencies ##
- Python >=2.7
- Pytorch >=0.2.0
- h5py (Preprocessing)
- sklearn (plot confusion matrix)

## base_train.py ##
- Base trianer coding for FER, include trian, val and test.
- python run_fer_train.py config/config.json to begain training.

### dataloader.py ###
- the dataloader of fer

### plot confusion matrix ###
- python run_plot_matrix.py to get the matric of fer classes.

###      Accurary      ###
- datasets：    raf-db ;       Test_acc：   84%+   <Br/>
- datasets：   affectnet ;     Test_acc：   73%+   <Br/>
- datasets:    fer2013 ;       Test_acc：   73.86%

## Concat

This repo is currently maintained by [xuechao.shi](shixuechao@meetvr.com.cn). If you have any questions, please contact him.
