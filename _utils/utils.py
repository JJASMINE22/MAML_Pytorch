# -*- coding: UTF-8 -*-
'''
@Project ：MAML_Pytorch
@File    ：utils.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import os
import cv2
import torch
import numpy as np
import pandas as pd


class Generator:
    """
    data generator, image recognition
    """
    def __init__(self,
                 root_path: str,
                 task_nums: int,
                 single_task_size: int,
                 batch_size: int,
                 input_size: tuple,
                 threshold: int,
                 device=None):

        self.root_path = root_path
        self.task_nums = task_nums
        self.total_dirs = np.array(os.listdir(self.root_path))
        self.target_size = self.total_dirs.__len__()
        assert single_task_size < self.target_size

        self.single_task_size = single_task_size
        self.batch_size = batch_size
        self.input_size = input_size
        self.threshold = threshold
        self.device = device
        self.tasks_idx = self.assign_task()

    def assign_task(self):
        """assignment of tasks"""
        while True:
            tasks_index = np.array([np.random.choice(self.target_size,
                                                       self.single_task_size,
                                                       replace=False)
                                    for i in range(self.task_nums)])
            total_index = np.reshape(tasks_index, [-1]).tolist()

            if all([*map(lambda x: total_index.count(x) > self.threshold,
                         np.arange(self.target_size))]):
                return tasks_index

    def get_val_len(self):

        return self.total_dirs.__len__()

    def image_preprocess(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.input_size,
                           interpolation=cv2.INTER_CUBIC)
        image = np.array(image) / 127.5 - 1
        image = np.clip(image, -1., 1.)
        image = np.transpose(image, [2, 0, 1])

        return image

    def generate(self, training=True):
        """
        training process divides the data into support、query tasks
        """
        while True:
            if training:
                task_files = np.array([[np.random.choice(os.listdir(os.path.join(self.root_path, task_dir)),
                                                         self.batch_size//2, replace=False)
                                        for i, task_dir in enumerate(self.total_dirs[task_idx])]
                                       for task_idx in self.tasks_idx])

                sources, targets = [], []
                for i, task_idx in enumerate(self.tasks_idx):
                    for j, task_dir in enumerate(self.total_dirs[task_idx]):
                        for file in task_files[i, j]:
                            image_path = os.path.join(os.path.join(self.root_path, task_dir), file)
                            image = self.image_preprocess(image_path)

                            sources.append(image)
                            targets.append(self.total_dirs.tolist().index(task_dir))
                    index = np.arange(sources.__len__())
                    np.random.shuffle(index)
                    assign_sources, assign_targets = sources.copy(), targets.copy()
                    assign_sources = np.array(assign_sources)[index]
                    assign_targets = np.array(assign_targets)[index]
                    support_sources = torch.tensor(assign_sources[:self.batch_size*self.single_task_size//4],
                                                   dtype=torch.float32)
                    query_sources = torch.tensor(assign_sources[self.batch_size*self.single_task_size//4:],
                                                 dtype=torch.float32)
                    support_targets = torch.tensor(assign_targets[:self.batch_size*self.single_task_size//4],
                                                   dtype=torch.long)
                    query_targets = torch.tensor(assign_targets[self.batch_size*self.single_task_size//4:],
                                                 dtype=torch.long)
                    if self.device:
                        support_sources = support_sources.to(self.device)
                        query_sources = query_sources.to(self.device)
                        support_targets = support_targets.to(self.device)
                        query_targets = query_targets.to(self.device)
                    sources.clear()
                    targets.clear()
                    yield support_sources, support_targets, query_sources, query_targets

            else:
                sources, targets = [], []
                files = np.array([np.random.choice(os.listdir(os.path.join(self.root_path, _)),
                                                   self.batch_size, replace=False) for _ in self.total_dirs])
                for i in range(files.shape[-1]):
                    for j, dir in enumerate(self.total_dirs):
                        image_path = os.path.join(os.path.join(self.root_path, dir), files[j, i])
                        image = self.image_preprocess(image_path)

                        sources.append(image)
                        targets.append(self.total_dirs.tolist().index(dir))

                        if np.equal(len(sources), self.batch_size):
                            annotation_sources = torch.tensor(np.array(sources.copy()), dtype=torch.float32)
                            annotation_targets = torch.tensor(np.array(targets.copy()), dtype=torch.long)
                            if self.device:
                                annotation_sources = annotation_sources.to(self.device)
                                annotation_targets = annotation_targets.to(self.device)
                            sources.clear()
                            targets.clear()
                            yield annotation_sources, annotation_targets
