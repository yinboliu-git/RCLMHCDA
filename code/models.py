import time

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, GCN2Conv, SAGEConv, GATConv, HGTConv, Linear,HANConv
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from globel_args import device
# HGTConv = HANConv
import copy


class HGT_init(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads, num_layers, data):

        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels) # -1表示根据输入的数据的维度自动调整

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            #  in_channels: Union[int, Dict[str, int]],
            conv = HGTConv(-1, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.fc = Linear(hidden_channels*2, 2)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data):
        # x_dict_, edge_index_dict = data['x_dict'], data['edge_dict']
        x_dict = x_dict_.copy()
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()
        all_list = []
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            all_list.append(x_dict.copy())

        for i,_ in x_dict_.items():
            x_dict[i] =torch.cat(tuple(x[i] for x in all_list), dim=1)

        Em = self.dropout(x_dict['n1'])
        Ed = self.dropout(x_dict['n2'])

        # E_out = torch.cat((Em,Ed), dim=0)

        return Em, Ed


class HGT(torch.nn.Module):
    def __init__(self, HGT_model,HGT_model_out):

        super().__init__()
        self.encode_hgt = HGT_model
        self.out_hgt = HGT_model_out

        self.fc_encode1 = torch.nn.Sequential(
            Linear(-1, 512),
            # torch.nn.ReLU(),
            Linear(512, 256)
        )

        self.fc_encode2 = torch.nn.Sequential(
            Linear(-1, 512),
            # torch.nn.ReLU(),
            Linear(512, 256)
        )

        self.fc_cl_encode1 = torch.nn.Sequential(
            Linear(-1, 512),
            # torch.nn.ReLU(),
            Linear(512, 256)
        )


        self.fc_cl_encode2 = torch.nn.Sequential(
            Linear(-1, 512),
            # torch.nn.ReLU(),
            Linear(512, 256)
        )

    def forward(self, data,edge_index):
        data_true = copy.deepcopy(data)

        Em_encode, Ed_encode = self.encode_hgt(data_true)

        # Em, Ed = self.fc_encode1(data_true['x_dict']['n1']), self.fc_encode2(data_true['x_dict']['n2'])
        # Em_encode, Ed_encode = self.fc_cl_encode1(Em_encode), self.fc_cl_encode2(Ed_encode)


        data_true['x_dict']['n1'] = torch.cat((data_true['x_dict']['n1'], Em_encode), dim=1)
        data_true['x_dict']['n2'] = torch.cat((data_true['x_dict']['n2'], Ed_encode), dim=1)

        Em, Ed = self.out_hgt(data_true)

        m_index = edge_index[0]
        d_index = edge_index[1]

        y = Em @ Ed.t()
        y_all = y
        # y = torch.cat((Em, Ed), dim=1)
        # y = self.fc(y)
        y = y[m_index, d_index].unsqueeze(-1)
        return y


class HGT_out_head(torch.nn.Module):
    def __init__(self, HGT_model):

        super().__init__()
        self.hgt = HGT_model

        self.fc1 = torch.nn.Sequential(
            Linear(-1, 256),
            # torch.nn.ReLU(),
            Linear(256, 128)
        )

        self.fc2 = torch.nn.Sequential(
            Linear(-1, 256),
            # torch.nn.ReLU(),
            Linear(256, 128)
        )

        self.mask_init = 0

    def forward(self, data, max_noise=0.01):
        mask_module = 'all'
        data_true = data

        if self.mask_init == 0:
            self.data_mask_init = data
            self.data_mask = self.new_mask_data(max_noise=max_noise,mask_module=mask_module)
            self.mask_init += 1

        Em_code, Ed_code = self.hgt(data_true)
        Em_mask_code, Ed_mask_code = self.hgt(self.data_mask)

        # y_true = Em_code @ Ed_code.t()
        # y_mask = Em_mask_code @ Ed_mask_code.t()
        # return self.fc1(y_true.unsqueeze(-1)),self.fc1(y_mask.unsqueeze(-1))

        # y = torch.cat((Em, Ed), dim=1)
        # y = self.fc(y)
        # y_true = self.fc1(Em_code) @ self.fc2(Ed_code.t())
        # y_mask = self.fc1(Em_mask_code) @ self.fc2(Ed_mask_code.t())
        return self.fc1(Em_code),self.fc1(Ed_code), self.fc2(Em_mask_code), self.fc2(Ed_mask_code)

    def new_mask_data(self, max_noise=0.01, mask_module='all'):
        self.data_mask = get_mask_data(copy.deepcopy(self.data_mask_init), max_noise,mask_module)
        return self.data_mask


def get_mask_data(data,max_noise, mask_module): # ctrl = all, only_node, only_edge, other

    if mask_module=='all' or mask_module=='only_node':
        for node_type, x in data['x_dict'].items():
            data['x_dict'][node_type] = get_node_noise(x,max_noise).to(device)

    if mask_module=='all' or mask_module=='only_edge':
        edge_matrix = data['edge_matrix']
        edge_matrix = get_edge_noise(edge_matrix,max_noise)

        edge_index_pos = np.column_stack(np.argwhere(edge_matrix != 0))
        edge_index_pos = torch.tensor(edge_index_pos, dtype=torch.long).to(device)
        data[('n1', 'e1', 'n2')].edge_index = edge_index_pos
        data[('n2', 'e1', 'n1')].edge_index = torch.flip(edge_index_pos, (0,)).to(device)
        edge_index_dict = {}
        for etype in data.edge_types:
            edge_index_dict[etype] = data[etype].edge_index
        data['edge_dict'] = edge_index_dict

    return data


def get_node_noise(x, noise_max=0.01):
    seed = torch.initial_seed()
    noise_seed = int(time.time())
    torch.manual_seed(noise_seed)

    shape = x.shape
    noise = torch.randn(shape).to(device) * noise_max # 噪声强度

    x_noise = x + noise

    torch.manual_seed(seed)
    return x_noise

def get_edge_noise(adj_matrix_init, noise_max=0.01):
    seed = torch.initial_seed()
    noise_seed = int(time.time())
    torch.manual_seed(noise_seed)

    adj_matrix = adj_matrix_init.copy()
    edge_index_pos = np.column_stack(np.argwhere(adj_matrix != 0))

    num_pos_edges_number = edge_index_pos.shape[1]
    selected_neg_edge_indices = torch.randint(high=edge_index_pos.shape[1], size=(int(num_pos_edges_number*noise_max)+1,),
                                              dtype=torch.long)

    edge_index_pos_noise = edge_index_pos[:, selected_neg_edge_indices]
    adj_matrix[edge_index_pos_noise[0],edge_index_pos_noise[1]] = 0
    adj_matrix_noise = adj_matrix

    torch.manual_seed(seed)
    return adj_matrix_noise


class DotProductAttention(torch.nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, query, key, value):
        # 计算注意力权重
        scores = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = torch.softmax(scores, dim=-1)

        # 使用注意力权重加权求和得到上下文向量
        context_vector = torch.matmul(attention_weights, value)

        return context_vector, attention_weights


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.encoder = Linear(-1, 256)
        self.decoder = Linear(256, 256)
        self.attention = DotProductAttention()

    def forward(self, encoder_input, decoder_input):
        # Encoder部分
        encoder_output = self.encoder(encoder_input)

        # Decoder部分
        decoder_output = self.decoder(decoder_input)

        # 注意力机制
        context_vector, attention_weights = self.attention(decoder_output, encoder_output, encoder_output)

        # 合并上下文向量和Decoder输出
        output = torch.cat([context_vector, decoder_output], dim=-1)

        return output, attention_weights