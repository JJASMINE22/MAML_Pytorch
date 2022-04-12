# -*- coding: UTF-8 -*-
'''
@Project ：MAML_Pytorch
@File    ：maml.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from PIL import Image, ImageFont, ImageDraw
from net.networks import CreateModel
import config as cfg

class MAML:
    def __init__(self,
                 learning_rate: dict,
                 load_ckpt: bool,
                 ckpt_path: str,
                 weight_decay: float,
                 device=None):
        self.device = device
        self.weight_decay = weight_decay
        self.sub_lr = learning_rate['sub_lr']
        self.meta_lr = learning_rate['meta_lr']

        self.model = CreateModel()
        if self.device:
            self.model = self.model.to(self.device)
        if load_ckpt:
            try:
                ckpt = torch.load(ckpt_path)
                self.model.load_state_dict(ckpt['state_dict'])
                print("model successfully loaded, NLL loss {:.3f}".format(ckpt['loss']))
            except:
                raise ("please enter the right params path")

        self.loss_func = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(lr=self.meta_lr,
                                          params=self.model.parameters())

        self.total_loss = 0
        self.support_loss, self.query_loss = 0, 0
        self.support_acc, self.query_acc = 0, 0
        self.val_loss, self.val_acc = 0, 0

    def calculate_per_loss(self, support_src, support_tgt, query_src, query_tgt):
        # calculate the loss of the support part, obtain the gradient of the meta model
        self.optimizer.zero_grad()
        support_logits = self.model(support_src)
        support_loss = self.loss_func(support_logits, support_tgt)
        for weight in self.model.get_weights():
            support_loss += self.weight_decay * torch.sum(torch.square(weight))
        support_loss.backward()

        # manually update the params of the meta model
        query_logits = self.manual_forward(query_src)
        # calculate the loss of the query part, obtain the gradient of the meta model
        query_loss = self.loss_func(query_logits, query_tgt)
        for weight in self.model.get_weights():
            query_loss += self.weight_decay * torch.sum(torch.square(weight))

        self.total_loss += query_loss

        self.support_loss += support_loss.data.item()
        self.query_loss += query_loss.data.item()

        self.support_acc += np.equal(support_logits.argmax(dim=-1).cpu().detach().numpy(),
                                     support_tgt.cpu().numpy()).sum()/support_src.size(0)
        self.query_acc += np.equal(query_logits.argmax(dim=-1).cpu().detach().numpy(),
                                      query_tgt.cpu().numpy()).sum()/query_src.size(0)

    def manual_gradient_update(self):
        """
        unable to assign Parameter to (with gradients) tensor
        store manually
        """
        self.manul_params = list()
        self.manul_params.append(self.model.conv1.weight - self.sub_lr * self.model.conv1.weight.grad)
        self.manul_params.append(self.model.conv1.bias - self.sub_lr * self.model.conv1.bias.grad)

        self.manul_params.append(self.model.bn1.weight - self.sub_lr * self.model.bn1.weight.grad)
        self.manul_params.append(self.model.bn1.bias - self.sub_lr * self.model.bn1.bias.grad)

        self.manul_params.append(self.model.conv2.weight - self.sub_lr * self.model.conv2.weight.grad)
        self.manul_params.append(self.model.conv2.bias - self.sub_lr * self.model.conv2.bias.grad)

        self.manul_params.append(self.model.bn2.weight - self.sub_lr * self.model.bn2.weight.grad)
        self.manul_params.append(self.model.bn2.bias - self.sub_lr * self.model.bn2.bias.grad)

        self.manul_params.append(self.model.conv3.weight - self.sub_lr * self.model.conv3.weight.grad)
        self.manul_params.append(self.model.conv3.bias - self.sub_lr * self.model.conv3.bias.grad)

        self.manul_params.append(self.model.bn3.weight - self.sub_lr * self.model.bn3.weight.grad)
        self.manul_params.append(self.model.bn3.bias - self.sub_lr * self.model.bn3.bias.grad)

        self.manul_params.append(self.model.linear.weight - self.sub_lr * self.model.linear.weight.grad)
        self.manul_params.append(self.model.linear.bias - self.sub_lr * self.model.linear.bias.grad)

    def manual_forward(self, source):
        """
        reproduce the forward passing
        """
        self.manual_gradient_update()

        x = torch.conv2d(source, weight=self.manul_params[0], bias=self.manul_params[1])

        x = (x - x.mean(dim=[0, -2, -1], keepdim=True)) / torch.sqrt(x.var(dim=[0, -2, -1], keepdim=True) + 1e-5)
        x = torch.add(self.manul_params[2].view([1, -1, 1, 1]) * x, self.manul_params[3].view([1, -1, 1, 1]))

        x = F.leaky_relu(x)
        x = torch.dropout(x, p=0.3, train=True)
        x = torch.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = torch.conv2d(x, weight=self.manul_params[4], bias=self.manul_params[5])

        x = (x - x.mean(dim=[0, -2, -1], keepdim=True)) / torch.sqrt(x.var(dim=[0, -2, -1], keepdim=True) + 1e-5)
        x = torch.add(self.manul_params[6].view([1, -1, 1, 1]) * x, self.manul_params[7].view([1, -1, 1, 1]))

        x = F.leaky_relu(x)
        x = torch.dropout(x, p=0.3, train=True)
        x = torch.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = torch.conv2d(x, weight=self.manul_params[8], bias=self.manul_params[9])

        x = (x - x.mean(dim=[0, -2, -1], keepdim=True)) / torch.sqrt(x.var(dim=[0, -2, -1], keepdim=True) + 1e-5)
        x = torch.add(self.manul_params[10].view([1, -1, 1, 1]) * x, self.manul_params[11].view([1, -1, 1, 1]))

        x = F.leaky_relu(x)
        x = torch.dropout(x, p=0.3, train=True)
        x = torch.max_pool2d(x, kernel_size=x.size(-1), stride=1)

        x = x.squeeze()

        x = torch.matmul(x, self.manul_params[12].transpose(1, 0))
        x = torch.add(x, self.manul_params[-1])

        output = F.log_softmax(x, dim=-1)

        return output

    def train(self, task_num):

        self.optimizer.zero_grad()
        total_loss = self.total_loss/task_num
        total_loss.backward()
        self.optimizer.step()
        self.total_loss = 0

    def validate(self, source, target):

        val_logits = self.model(source)
        val_loss = self.loss_func(val_logits, target)
        for weight in self.model.get_weights():
            val_loss += self.weight_decay * torch.sum(torch.square(weight))
        self.val_loss += val_loss.data.item()
        self.val_acc += np.equal(val_logits.argmax(dim=-1).cpu().detach().numpy(),
                                 target.cpu().numpy()).sum()/source.size(0)

    def generate_sample(self, source, batch):
        """
        Drawing and labeling
        """
        logit = self.model(source).detach().argmax(dim=-1)

        index = np.random.choice(logit.size(0), 1)

        source = source[index].squeeze().permute(1, 2, 0).cpu().numpy()
        text = logit[index].squeeze().cpu().numpy()
        text = str(text)

        image = Image.fromarray(np.uint8((source + 1) * 127.5))

        font = ImageFont.truetype(font=cfg.font_path,
                                  size=np.floor(3e-1 * image.size[1] + 0.5).astype('int'))
        draw = ImageDraw.Draw(image)

        draw.text(np.array([image.size[0] // 3, image.size[1] // 2]),
                  text, fill=(255, 0, 0), font=font)

        del draw

        image.save(cfg.sample_path.format(batch), quality=95, subsampling=0)