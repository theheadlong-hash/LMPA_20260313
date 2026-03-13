import math
import torch
import torch.nn as nn
from torch.nn import init
import time
import torch.nn.functional as F
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    # from collections import OrderedDict
    from layers.Embedding import *
    from layers.Embedding import positional_encoding
    print("Layer！")
except ImportError as e:
    print(f"导入失败：{e}")





class Transformer_Layer(nn.Module):
    def __init__(self, device, d_model, d_ff, num_nodes, patch_nums, patch_size, dynamic, factorized, layer_number, batch_norm):
        super(Transformer_Layer, self).__init__()
        self.device = device
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.dynamic = dynamic
        self.patch_nums = patch_nums
        self.patch_size = patch_size
        self.layer_number = layer_number
        self.batch_norm = batch_norm


        ##intra_patch_attention
        self.intra_embeddings = nn.Parameter(torch.rand(self.patch_nums, 1, 1, self.num_nodes, 16),
                                                requires_grad=True)
        self.embeddings_generator = nn.ModuleList([nn.Sequential(*[
            nn.Linear(16, self.d_model)]) for _ in range(self.patch_nums)])
        self.intra_d_model = self.d_model
        self.intra_patch_attention = Intra_Patch_Attention(self.intra_d_model, factorized=factorized)
        self.weights_generator_distinct = WeightGenerator(self.intra_d_model, self.intra_d_model, mem_dim=16, num_nodes=num_nodes,
                                                          factorized=factorized, number_of_weights=2)
        self.weights_generator_shared = WeightGenerator(self.intra_d_model, self.intra_d_model, mem_dim=None, num_nodes=num_nodes,
                                                        factorized=False, number_of_weights=2)
        self.intra_Linear = nn.Linear(self.patch_nums, self.patch_nums*self.patch_size)



        ##inter_patch_attention
        self.stride = patch_size
        # patch_num = int((context_window - cut_size) / self.stride + 1)

        self.inter_d_model = self.d_model * self.patch_size
        ##inter_embedding
        self.emb_linear = nn.Linear(self.inter_d_model, self.inter_d_model)
        # Positional encoding
        self.W_pos = positional_encoding(pe='zeros', learn_pe=True, q_len=self.patch_nums, d_model=self.inter_d_model)
        n_heads = self.d_model
        d_k = self.inter_d_model // n_heads
        d_v = self.inter_d_model // n_heads
        print(f"n_heads: {n_heads}")
        print(f"d_k: {d_k}")
        print(f"d_v: {d_v}")

        self.inter_patch_attention = Inter_Patch_Attention(self.inter_d_model, self.inter_d_model, n_heads, d_k, d_v, attn_dropout=0,
                                          proj_dropout=0.1, res_attention=False)


        ##Normalization
        self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(self.d_model), Transpose(1,2))
        self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(self.d_model), Transpose(1,2))

        ##FFN
        self.d_ff = d_ff
        self.dropout = nn.Dropout(0.1)
        self.ff = nn.Sequential(nn.Linear(self.d_model, self.d_ff, bias=True),
                                nn.GELU(),
                                nn.Dropout(0.2),
                                nn.Linear(self.d_ff, self.d_model, bias=True))

    def forward(self, x):

        new_x  = x
        # x [批次大小(动态变化) x 时间步长 x 特征数量 x 每个特征的维度 ]=[27, 96, 7, 16]
        # print(f"1x shape: {x.shape}")
        batch_size = x.size(0)
        intra_out_concat = None

        weights_shared, biases_shared = self.weights_generator_shared()
        weights_distinct, biases_distinct = self.weights_generator_distinct()

        ####intra Attention#####
        for i in range(self.patch_nums):
            t = x[:, i * self.patch_size:(i + 1) * self.patch_size, :, :]  # x=[27, 96, 7, 16]
            # t [批次大小(动态变化) x 当前片的时间步长 x 特征数量 x 每个特征的维度 ]=[27, 16, 7, 16] patch_size=96/16=6
            # 从96时间步长中 提取出一个patch的时间长度 i=0 就是16，i=1 就是17-32...i是索引
            # print(f"2t shape: {t.shape}")

            # 先获取第 i 个内部嵌入
            current_intra_embedding = self.intra_embeddings[i]
            # 前文定义的：self.intra_embeddings = nn.Parameter(torch.rand(self.patch_nums, 1, 1, self.num_nodes, 16)
            # 当使用 nn.Parameter 包装一个张量时，这个张量会被自动添加到模型的参数列表中，并且在模型训练过程中会被自动更新。

            # 使用第 i 个嵌入生成器对当前嵌入进行处理——将每个片的嵌入从 16 维向量映射到 self.d_model 维（给定参数也是16）
            generated_embedding = self.embeddings_generator[i](current_intra_embedding) # ( 1, 1, self.num_nodes, 16)

            # 将生成的嵌入扩展到批次大小
            intra_emb = generated_embedding.expand(batch_size, -1, -1, -1)
            # intra_emb [批次大小（1批有27个样本、(动态变化)） x 时间步长 x 特征数量 x 每个特征的维度 ]=[27, 1, 7, 16]
            # 生成的嵌入在时间维度上只有一个步骤，目的是拼接后给原始的 t 张量添加一些额外的、与时间相关的特征信息
            # print(f"4intra_emb shape: {intra_emb.shape}")

            t = torch.cat([intra_emb, t], dim=1)
            # t [批次大小(动态变化) x 当前片的时间步长 x 特征数量 x 每个特征的维度 ]=[27, 16+1, 7, 16]
            # print(f"4t shape: {t.shape}")

            out, attention = self.intra_patch_attention(intra_emb, t, t, weights_distinct, biases_distinct, weights_shared,
                                               biases_shared)
            # out [批次大小(动态变化) x 可能是当前片经过注意力处理后的输出数量长 x 特征数量 x 每个特征的维度 ]=[27, 1, 7, 16]
            # attention [批次大小乘以头的数量 x 特征数量 x 可能表示注意力的输出数量x 与拼接后的 t 的时间步长一致，表示注意力作用的序列长度 ]=[27*2, 7, 1, 17]

            # print(f"5out shape: {out.shape}")
            # print(f"6attention shape: {attention .shape}")

            if intra_out_concat == None:
                intra_out_concat = out

            else:
                intra_out_concat = torch.cat([intra_out_concat, out], dim=1)

        intra_out_concat = intra_out_concat.permute(0,3,2,1)  # 27，16，7，1
        intra_out_concat = self.intra_Linear(intra_out_concat)  # 27，16，7，96
        # print(f"777intra_out_concat shape: {intra_out_concat.shape}")
        intra_out_concat = intra_out_concat.permute(0,3,2,1)    # 27，96，7，16
        # print(f"888intra_out_concat shape: {intra_out_concat.shape}")


        ####inter Attention######
        x = x.unfold(dimension=1, size=self.patch_size, step=self.stride)  # [b x patch_num x nvar x dim x patch_len]
        # print(f"7x shape: {x .shape}")

        x = x.permute(0, 2, 1, 3, 4)  # [b x nvar x patch_num x dim x patch_len ]
        b, nvar, patch_num, dim, patch_len = x.shape

        x = torch.reshape(x, (
        x.shape[0] * x.shape[1], x.shape[2], x.shape[3] * x.shape[-1]))  # [b*nvar, patch_num, dim*patch_len]
        print(f"8x shape: {x.shape}")
        x = self.emb_linear(x)
        x = self.dropout(x + self.W_pos)

        inter_out, attention = self.inter_patch_attention(Q=x, K=x, V=x)  # [b*nvar, patch_num, dim]
        print(f"9inter_out shape: {inter_out.shape}")

        inter_out = torch.reshape(inter_out, (b, nvar, inter_out.shape[-2], inter_out.shape[-1]))
        inter_out = torch.reshape(inter_out, (b, nvar, inter_out.shape[-2], self.patch_size, self.d_model))
        inter_out = torch.reshape(inter_out, (b, self.patch_size*self.patch_nums, nvar, self.d_model)) #[b, temporal, nvar, dim]

        out = new_x + intra_out_concat + inter_out
        if self.batch_norm:
            out = self.norm_attn(out.reshape(b*nvar, self.patch_size*self.patch_nums, self.d_model))
        ##FFN
        out = self.dropout(out)
        out = self.ff(out) + out
        if self.batch_norm:
            out = self.norm_ffn(out).reshape(b, self.patch_size*self.patch_nums, nvar, self.d_model)
        return out, attention



class CustomLinear(nn.Module):
    def __init__(self, factorized):
        super(CustomLinear, self).__init__()
        self.factorized = factorized

    def forward(self, input, weights, biases):
        if self.factorized:
            return torch.matmul(input.unsqueeze(3), weights).squeeze(3) + biases
        else:
            return torch.matmul(input, weights) + biases


class Intra_Patch_Attention(nn.Module):
    def __init__(self, d_model, factorized):
        super(Intra_Patch_Attention, self).__init__()
        self.head = 2

        if d_model % self.head != 0:
            raise Exception('Hidden size is not divisible by the number of attention heads')

        self.head_size = int(d_model // self.head)
        self.custom_linear = CustomLinear(factorized)

    def forward(self, query, key, value, weights_distinct, biases_distinct, weights_shared, biases_shared):
        batch_size = query.shape[0]

        key = self.custom_linear(key, weights_distinct[0], biases_distinct[0])
        value = self.custom_linear(value, weights_distinct[1], biases_distinct[1])
        query = torch.cat(torch.split(query, self.head_size, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_size, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_size, dim=-1), dim=0)

        query = query.permute((0, 2, 1, 3))
        key = key.permute((0, 2, 3, 1))
        value = value.permute((0, 2, 1, 3))



        attention = torch.matmul(query, key)
        attention /= (self.head_size ** 0.5)

        attention = torch.softmax(attention, dim=-1)

        x = torch.matmul(attention, value)
        x = x.permute((0, 2, 1, 3))
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)

        if x.shape[0] == 0:
            x = x.repeat(1, 1, 1, int(weights_shared[0].shape[-1] / x.shape[-1]))

        x = self.custom_linear(x, weights_shared[0], biases_shared[0])
        x = torch.relu(x)
        x = self.custom_linear(x, weights_shared[1], biases_shared[1])
        print(f"12x shape: {x.shape}")
        print(f"attention shape: {attention.shape}")

        return x, attention


class Inter_Patch_Attention(nn.Module):
    def __init__(self, d_model, out_dim, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0.,
                 proj_dropout=0., qkv_bias=True, lsa=False):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        # print(f"d_k: {d_k}")
        # print(f"d_v: {d_v}")
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout,
                                                  res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, out_dim), nn.Dropout(proj_dropout))


    def forward(self, Q, K=None, V=None, prev=None, key_padding_mask=None, attn_mask=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q
        # 如果 K 没有被传入（即为 None），那么 K 就会被赋值为 Q,在forward中没有被传入）
        # print(f"12Q shape: {Q.shape}")
        # print(f"12K shape: {K.shape}")
        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, Q.shape[1], self.n_heads, self.d_k).transpose(1,
                                                                                 2)  # q_s    : [bs x n_heads x q_len x d_k]  此处的q_len为patch_num
        k_s = self.W_K(K).view(bs, K.shape[1], self.n_heads, self.d_k).permute(0, 2, 3,
                                                                               1)  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, V.shape[1], self.n_heads, self.d_v).transpose(1,
                                                                                 2)  # v_s    : [bs x n_heads x q_len x d_v]
        print(f"13q_s shape: {q_s.shape}")
        print(f"13k_s shape: {k_s.shape}")
        print(f"13v_s shape: {v_s.shape}")

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        output = output.transpose(1, 2).contiguous().view(bs, Q.shape[1],
                                                          self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        return output, attn_weights


class ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q, k, v, prev=None, key_padding_mask=None, attn_mask=None):

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

        return output, attn_weights


class WeightGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, mem_dim, num_nodes, factorized, number_of_weights=4):
        super(WeightGenerator, self).__init__()
        #print('FACTORIZED {}'.format(factorized))
        self.number_of_weights = number_of_weights
        self.mem_dim = mem_dim
        self.num_nodes = num_nodes
        self.factorized = factorized
        self.out_dim = out_dim
        if self.factorized:
            self.memory = nn.Parameter(torch.randn(num_nodes, mem_dim), requires_grad=True).to('cpu')
            # self.memory = nn.Parameter(torch.randn(num_nodes, mem_dim), requires_grad=True).to('cuda:0')
            self.generator = self.generator = nn.Sequential(*[
                nn.Linear(mem_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 100)
            ])

            self.mem_dim = 10
            self.P = nn.ParameterList(
                [nn.Parameter(torch.Tensor(in_dim, self.mem_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
            self.Q = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.mem_dim, out_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
            self.B = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.mem_dim ** 2, out_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
        else:
            self.P = nn.ParameterList(
                [nn.Parameter(torch.Tensor(in_dim, out_dim), requires_grad=True) for _ in range(number_of_weights)])
            self.B = nn.ParameterList(
                [nn.Parameter(torch.Tensor(1, out_dim), requires_grad=True) for _ in range(number_of_weights)])
        self.reset_parameters()

    def reset_parameters(self):
        list_params = [self.P, self.Q, self.B] if self.factorized else [self.P]
        for weight_list in list_params:
            for weight in weight_list:
                init.kaiming_uniform_(weight, a=math.sqrt(5))

        if not self.factorized:
            for i in range(self.number_of_weights):
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.P[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.B[i], -bound, bound)

    def forward(self):
        if self.factorized:
            memory = self.generator(self.memory.unsqueeze(1))
            bias = [torch.matmul(memory, self.B[i]).squeeze(1) for i in range(self.number_of_weights)]
            memory = memory.view(self.num_nodes, self.mem_dim, self.mem_dim)
            weights = [torch.matmul(torch.matmul(self.P[i], memory), self.Q[i]) for i in range(self.number_of_weights)]
            return weights, bias
        else:
            return self.P, self.B



class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)
