import math
import torch
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from layers.AMS import AMS
    from layers.Layer import WeightGenerator, CustomLinear
    from layers.RevIN import RevIN
    from functools import reduce
    from operator import mul
    print("exp_main！")
except ImportError as e:
    print(f"导入失败：{e}")




class PathFormerModel(nn.Module):
    def __init__(self, configs):
        super(PathFormerModel, self).__init__()
        self.layer_nums = configs.layer_nums  # 设置pathway的层数
        self.num_nodes = configs.num_nodes
        self.pre_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.k = configs.k
        self.num_experts_list = configs.num_experts_list
        self.patch_size_list = configs.patch_size_list
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.residual_connection = configs.residual_connection
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(num_features=configs.num_nodes, affine=False, subtract_last=False)

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.AMS_lists = nn.ModuleList()
        self.device = torch.device('cuda:{}'.format(configs.gpu))
        self.batch_norm = configs.batch_norm

        for num in range(self.layer_nums):
            self.AMS_lists.append(
                AMS(self.seq_len, self.seq_len, self.num_experts_list[num], self.device, k=self.k,
                    num_nodes=self.num_nodes, patch_size=self.patch_size_list[num], noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number=num + 1, residual_connection=self.residual_connection, batch_norm=self.batch_norm))
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pre_len)
        )

    def forward(self, x):

        balance_loss = 0
        # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')     # 输入 x 和模式 'norm' 进行归一化
        out = self.start_fc(x.unsqueeze(-1))    #增加一个维度


        batch_size = x.shape[0]

        for layer in self.AMS_lists:
            out, aux_loss = layer(out)        # layer(out) 调用了当前 AMS 层的 forward 方法，将上一层的输出 out 作为输入传递给该 AMS 层。
            balance_loss += aux_loss

        out = out.permute(0,2,1,3).reshape(batch_size, self.num_nodes, -1)
        out = self.projections(out).transpose(2, 1)

        # denorm
        if self.revin:
            out = self.revin_layer(out, 'denorm')

        return out, balance_loss


