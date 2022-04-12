# -*- coding: UTF-8 -*-
'''
@Project ：MAML_Pytorch
@File    ：config.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch

# ===generator===
root_path = "C:\\DATASET\\Omniglot Dataset\\images_background\\Alphabet_of_the_Magi"  # "文件根目录(绝对路径)"
task_nums = 50
threshold = 5
single_task_size = 5
batch_size = 16
input_size = (28, 28)

# ===training===
Epoches = 100
device = torch.device('cuda') if torch.cuda.is_available() else None
meta_lr = 1e-3
sub_lr = 1e-3
weight_decay = 5e-4
per_sample_interval = 5
ckpt_path = '.\\saved\\checkpoint'
sample_path = '.\\sample'
font_path = '.\\font\\simhei.ttf'
sample_path = '.\\sample\\Batch{:d}.jpg'
load_ckpt = False
