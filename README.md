## MAML：Model-Agnostic Meta-Learning模型的Pytorch实现
---

## 目录
1. [所需环境 Environment](#所需环境)
2. [模型结构 Structure](#模型结构)
3. [注意事项 Attention](#注意事项)
4. [文件下载 Download](#文件下载)
5. [训练步骤 How2train](#训练步骤) 

## 所需环境
Python3.7
Pytorch>=1.7.0+cu110  
Numpy==1.19.5
Pillow==8.2.0
Opencv-contrib-python==4.5.1.48
CUDA 11.0+
Pandas==1.2.4
Matplotlib==3.2.2

## 模型结构
MAML  
<p align="center">
----------------------------------------------------------------
<p align="center">
        Layer (type)               Output Shape         Param #
<p align="center">
<p align="center">
            Conv2d-1           [-1, 64, 26, 26]           1,792
<p align="center">
     BatchNormal2d-2           [-1, 64, 26, 26]             128
<p align="center">
            Conv2d-3          [-1, 128, 11, 11]          73,856
<p align="center">
     BatchNormal2d-4          [-1, 128, 11, 11]             256
<p align="center">
            Conv2d-5            [-1, 256, 4, 4]         295,168
<p align="center">
     BatchNormal2d-6            [-1, 256, 4, 4]             512
<p align="center">
            Linear-7                   [-1, 20]           5,140
<p align="center">
================================================================
<p align="center">
Total params: 376,852
<p align="center">
Trainable params: 376,852
<p align="center">
Non-trainable params: 0
<p align="center">
----------------------------------------------------------------
<p align="center">
Input size (MB): 0.01
<p align="center">
Forward/backward pass size (MB): 0.96
<p align="center">
Params size (MB): 1.44
<p align="center">
Estimated Total Size (MB): 2.41
<p align="center">
----------------------------------------------------------------

## 注意事项
1. MAML结构适用于小样本模型训练，为避免过学习，模型不应设计过重
2. Pytorch无法实现Parameter对象的直接赋值。需手动计算基于support_task的meta_model梯度下降过程，并存储梯度，再结合query_task重新实现前向推理
3. 添加正则化机制，防止过拟合
4. 数据路径、训练参数均位于config.py

## 文件下载    
使用Omniglot Dataset
链接：https://pan.baidu.com/s/13T1Qs4NZL8NS4yoxCi-Qyw  
提取码：sets  
下载解压后放置于config.py中设置的路径即可。

## 训练步骤
运行train.py

