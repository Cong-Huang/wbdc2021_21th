import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from lightgbm.sklearn import LGBMClassifier
from collections import defaultdict
import gc, pickle, os, time
import random

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names, combined_dnn_input
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers.interaction import FM, BiInteractionPooling
from deepctr_torch.layers.sequence import AttentionSequencePoolingLayer
from deepctr_torch.layers import DNN, concat_fun, InteractingLayer

import torch
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader,RandomSampler, SequentialSampler
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import warnings
from collections import Counter 
warnings.filterwarnings("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
pd.set_option('display.max_columns', None)



# MLP分类器
class DNN_head(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, hid_size, linear_hid_size=[128, 128]):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(hid_size)
        self.dense_1 = nn.Linear(hid_size, linear_hid_size[0])
        self.dense_2 = nn.Linear(linear_hid_size[0], linear_hid_size[1])  
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, feas):
        x = self.batch_norm(feas)
        x = self.relu(self.dense_1(x))
        # x = self.dropout(x)
        x = self.dense_2(x)
        return x


# DNN作为主编码器
class MMOE_DNN_v1(BaseModel):
    "2*p40 训练的，没加FM"
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, 
                 use_fm=False, use_din=False,
                 embed_dim=32,
                 dnn_hidden_units=(256, 128), 
                 l2_reg_linear=0.001, l2_reg_embedding=0.01, l2_reg_dnn=0.0, init_std=0.001, seed=1024,
                 dnn_dropout=0.5, dnn_activation='relu', dnn_use_bn=True, task='binary', device='cpu', gpus=None,
                 num_tasks=4, num_experts=16, expert_dim=32, 
                 ):
        super(MMOE_DNN_v1, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)
        
        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        
        if use_fm:
            self.fm = BiInteractionPooling()
        
        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                           use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_aux = DNN(embed_dim*2, dnn_hidden_units[-2:],
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                           use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            
        self.use_din = use_din
        if self.use_din:
            self.feedid_emb_din = self.embedding_dict.feedid
#             self.LSTM_din = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, num_layers=1,
#                                     batch_first=True, bidirectional=False)
            self.attention = AttentionSequencePoolingLayer(att_hidden_units=(64, 64),
                                                           embedding_dim=embed_dim,
                                                           att_activation='Dice',
                                                           return_score=False,
                                                           supports_masking=False,
                                                           weight_normalization=False)
            
        
        
        # 专家设置
        self.input_dim = dnn_hidden_units[-1]
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.num_tasks = num_tasks
        
        # expert-kernel
        self.expert_kernel = nn.Linear(self.input_dim, num_experts * expert_dim)
        
        # 每个任务的单独变换
#         self.gate_mlp = nn.ModuleList([DNN_head(232) for i in range(num_tasks)])
        # 每个任务的gate-kernel
        self.gate_kernels = nn.ModuleList([nn.Linear(self.input_dim, num_experts, bias=False)
                                           for i in range(num_tasks)])
        
        
        self.cls = nn.ModuleList([nn.Sequential(
                                    nn.Linear(self.expert_dim, 128), 
                                    nn.ReLU(),
                                    nn.Linear(128, 1)
                                  )
                                  for i in range(self.num_tasks)])
        
        self.gate_softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.to(device)
    
    
    def forward(self, X, fids=None, fids_length=None):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)   # [bs, n, emb_dim]
            fm_out = self.fm(fm_input)   # [bs, 1, emb_dim]
            fm_out = fm_out.squeeze(1)   # [bs, emb_dim]

        
        if self.use_din:
            fid_emb_query = self.feedid_emb_din(X[:, self.feature_index['feedid'][0]:self.feature_index['feedid'][1]].long())
            fid_emb_key = self.feedid_emb_din(fids)    # [bs, sl, emb_size]
#             fid_emb_key_lstm, _ = self.LSTM_din(fid_emb_key)   # [bs, sl, emb_size]
#             fid_emb_key = fid_emb_key + fid_emb_key_lstm
            fid_din = self.attention(fid_emb_query, fid_emb_key, fids_length)   #[bs, 1, emb_size]
            din_out = fid_din.squeeze(1)
        
        if self.use_dnn:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_out = self.dnn(dnn_input)   # [bs, dnn_hidden_units[-1]]
            
            if self.use_fm and self.use_din:
                aux_out = torch.cat([fm_out, din_out], dim=-1)
                aux_out = self.dnn_aux(aux_out)  #[bs, dnn_hidden_units[-1]]
                dnn_out = torch.cat([dnn_out, aux_out], dim=-1)
        
        # 每个mmoe的输出
        mmoe_outs = []
        expert_out = self.expert_kernel(dnn_out)  # [bs, num_experts * expert_dim]
        expert_out = expert_out.view(-1, self.expert_dim, self.num_experts)  # [bs, expert_dim, num_experts]
         
        for i in range(self.num_tasks):
#             gate_out_aux = self.gate_mlp[i](X[:, 13:245])
#             gate_input = torch.cat([dnn_out, gate_out_aux], dim=-1)
            gate_out = self.gate_kernels[i](dnn_out)  # [bs, num_experts]
            gate_out = self.gate_softmax(gate_out)     # [bs, num_experts]
            gate_out = gate_out.unsqueeze(1).expand_as(expert_out)  # [bs, expert_dim, num_experts]
            output = torch.sum(expert_out * gate_out, 2)   # [bs, expert_dim]
            mmoe_outs.append(output)
        
        task_outputs = []
        for idx, mmoe_out in enumerate(mmoe_outs):
            output = self.sigmoid(self.cls[idx](mmoe_out))
            task_outputs.append(output)
        
        return task_outputs


# DNN作为主编码器
class MMOE_DNN_v2(BaseModel):
    """4*v100训练的，包含FM层"""
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, 
                 use_fm=False, use_din=False,
                 embed_dim=32,
                 dnn_hidden_units=(256, 128), 
                 l2_reg_linear=0.001, l2_reg_embedding=0.01, l2_reg_dnn=0.0, init_std=0.001, seed=1024,
                 dnn_dropout=0.5, dnn_activation='relu', dnn_use_bn=True, task='binary', device='cpu', gpus=None,
                 num_tasks=4, num_experts=16, expert_dim=32, 
                 ):
        super(MMOE_DNN_v2, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)
        
        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        
        if use_fm:
            self.fm = BiInteractionPooling()
        
        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns) + embed_dim, dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                           use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
        
        
        # 专家设置
        self.input_dim = dnn_hidden_units[-1]
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.num_tasks = num_tasks
        
        # expert-kernel
        self.expert_kernel = nn.Linear(self.input_dim, num_experts * expert_dim)
        
        # 每个任务的gate-kernel
        self.gate_kernels = nn.ModuleList([nn.Linear(self.input_dim, num_experts, bias=False)
                                           for i in range(num_tasks)])
        
        
        self.cls = nn.ModuleList([nn.Sequential(
                                    nn.Linear(self.expert_dim, 256), 
                                    nn.ReLU(),
                                    nn.Dropout(p=0.2),
                                    nn.Linear(256, 1)
                                  )
                                  for i in range(self.num_tasks)])
        
        self.gate_softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.to(device)
    
    
    def forward(self, X, fids=None, fids_length=None):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)   # [bs, n, emb_dim]
            fm_out = self.fm(fm_input)   # [bs, 1, emb_dim]
            fm_out = fm_out.squeeze(1)   # [bs, emb_dim]
        
        if self.use_dnn:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            if self.use_fm:
                dnn_input = torch.cat([dnn_input, fm_out], dim=-1) 
            dnn_out = self.dnn(dnn_input)   # [bs, dnn_hidden_units[-1]]
        
        # 每个mmoe的输出
        mmoe_outs = []
        expert_out = self.expert_kernel(dnn_out)  # [bs, num_experts * expert_dim]
        expert_out = expert_out.view(-1, self.expert_dim, self.num_experts)  # [bs, expert_dim, num_experts]
         
        for i in range(self.num_tasks):
            gate_out = self.gate_kernels[i](dnn_out)  # [bs, num_experts]
            gate_out = self.gate_softmax(gate_out)     # [bs, num_experts]
            gate_out = gate_out.unsqueeze(1).expand_as(expert_out)  # [bs, expert_dim, num_experts]
            output = torch.sum(expert_out * gate_out, 2)   # [bs, expert_dim]
            mmoe_outs.append(output)
        
        task_outputs = []
        for idx, mmoe_out in enumerate(mmoe_outs):
            output = self.sigmoid(self.cls[idx](mmoe_out))
            task_outputs.append(output)
        
        return task_outputs
