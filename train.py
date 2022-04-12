# -*- coding: UTF-8 -*-
'''
@Project ：MAML_Pytorch
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''

import torch
import config as cfg
from torch import nn
from maml import MAML
from _utils.utils import Generator

if __name__ == '__main__':

    data_gen = Generator(root_path=cfg.root_path,
                         task_nums=cfg.task_nums,
                         single_task_size=cfg.single_task_size,
                         batch_size=cfg.batch_size,
                         input_size=cfg.input_size,
                         threshold=cfg.threshold,
                         device=cfg.device)

    Maml = MAML(learning_rate={'meta_lr': cfg.meta_lr,
                               'sub_lr': cfg.sub_lr},
                load_ckpt=cfg.load_ckpt,
                ckpt_path=cfg.ckpt_path + "模型文件",
                weight_decay=cfg.weight_decay,
                device=cfg.device)

    train_gen = data_gen.generate(training=True)
    validate_gen = data_gen.generate(training=False)

    for epoch in range(cfg.Epoches):

        for i in range(cfg.task_nums):
            support_sources, support_targets, query_sources, query_targets = next(train_gen)
            Maml.calculate_per_loss(support_sources, support_targets, query_sources, query_targets)
        Maml.train(cfg.task_nums)

        print('Epoch{:0>3d}\n'
              'support loss {:.3f}\n'
              'query loss {:.3f}\n'
              'support acc {:.2f}%\n'
              'query acc {:.2f}%'.format(epoch+1,
                                           Maml.support_loss / cfg.task_nums,
                                           Maml.query_loss / cfg.task_nums,
                                           Maml.support_acc / cfg.task_nums * 100,
                                           Maml.query_acc / cfg.task_nums * 100))

        torch.save({'state_dict': Maml.model.state_dict(),
                    'support_loss': Maml.support_loss / cfg.task_nums,
                    'query_loss': Maml.query_loss / cfg.task_nums,
                    'support_acc': Maml.support_acc / cfg.task_nums,
                    'query_acc': Maml.query_acc / cfg.task_nums},
                   cfg.ckpt_path + '\\Epoch{:0>3d}_support_loss{:.3f}_query_loss{:.3f}.pth.tar'.format(
                       epoch + 1, Maml.support_loss / cfg.task_nums, Maml.query_loss / cfg.task_nums))
        Maml.support_loss, Maml.query_loss = 0, 0
        Maml.support_acc, Maml.query_acc = 0, 0

        for i in range(data_gen.get_val_len()):
            sources, targets = next(validate_gen)
            Maml.validate(sources, targets)
            if not (i+1) % cfg.per_sample_interval:
                Maml.generate_sample(sources, (i+1))

        print('validate loss is {:.5f}\n'
              'validate acc is {:.3f}%\n'.format(Maml.val_loss / data_gen.get_val_len(),
                                                 Maml.val_acc / data_gen.get_val_len()*100))

        Maml.val_loss = 0
        Maml.val_acc = 0
