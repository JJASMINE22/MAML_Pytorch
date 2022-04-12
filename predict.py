# -*- coding: UTF-8 -*-
'''
@Project ：MAML_Pytorch
@File    ：predict.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from net.networks import CreateModel
from _utils.utils import Generator

if __name__ == '__main__':

    data_gen = Generator(root_path=cfg.root_path,
                         task_nums=cfg.task_nums,
                         single_task_size=cfg.single_task_size,
                         batch_size=cfg.batch_size,
                         input_size=cfg.input_size,
                         threshold=cfg.threshold,
                         device=cfg.device)

    Maml = CreateModel().to(cfg.device)
    try:
        ckpt = torch.load(cfg.ckpt_path + "\\maml.pth.tar")
        Maml.load_state_dict(ckpt['state_dict'])
        print("model successfully loaded, support loss {:.3f}, query loss {:3f}".format(ckpt['support_loss'],
                                                                                        ckpt['query_loss']))
    except:
        raise ("please enter the right params path")

    validate_gen = data_gen.generate(training=False)
    for i in range(data_gen.get_val_len()):
        sources, targets = next(validate_gen)
        index = np.arange(targets.size(0))
        np.random.shuffle(index)

        logits = Maml(sources)
        predictions = logits.cpu().detach().numpy().argmax(axis=-1)[index]
        targets = targets.cpu().numpy()[index]
        plt.scatter(np.arange(targets.__len__()), targets, marker='x',
                    color='r', label='target')
        plt.scatter(np.arange(predictions.__len__()), predictions, marker='^',
                    color='g', label='prediction')
        plt.grid()
        plt.legend(loc='upper right', fontsize='x-small')
        plt.show()

