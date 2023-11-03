import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, GCN2Conv, SAGEConv, GATConv, HGTConv, Linear
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import roc_auc_score
import os
from utils import get_data
from train_model import CV_train, CL_train_init

def set_seed(seed):
    torch.manual_seed(seed)
    #进行随机搜索的这个要注释掉
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class Config:
    def __init__(self):
        self.datapath = './data/'
        self.save_file = './save_file/'

        self.kfold = 5
        self.maskMDI = False
        self.hidden_channels = 512   # 256 512
        self.num_heads = 4   # 4 8
        self.num_layers = 4   # 4 8
        self.self_encode_len = 256
        self.globel_random = 100
        self.other_args = {'arg_name': [], 'arg_value': []}
        self.cl_model_file = './model.pt'
        self.use_CL_model = True
        self.train_CL_model = True

        # 对比学习参数
        self.CL_epochs = 400
        self.CL_print_epoch = 20
        self.new_noise = 100
        self.CL_margin = 1.0
        self.CL_noise_max = 0.1

        # 解码参数
        self.epochs = 1000
        self.print_epoch = 20

def set_attr(config, param_search):
    param_grid = param_search
    param_keys = param_grid.keys()
    param_grid_list = list(ParameterGrid(param_grid))
    for param in param_grid_list:
        config.other_args = {'arg_name': [], 'arg_value': []}
        for keys in param_keys:
            setattr(config, keys, param[keys])
            config.other_args['arg_name'].append(keys)
            print(keys,param[keys])
            config.other_args['arg_value'].append(param[keys])
        yield config
    return 0

class Data_paths:
    def __init__(self):
        self.paths = './data/'
        self.md = self.paths + 'c_d.csv'
        self.mm = [self.paths + 'c_gs.csv', self.paths + 'c_ss.csv']
        self.dd = [self.paths + 'd_gs.csv', self.paths + 'd_ss.csv']

best_param_search = {
    'hidden_channels': [64,128,256,512],
    'num_heads': [4,8,16,32],
    'num_layers': [2,4,6,8],
    # 'CL_margin' :[0.5,1.0,1.5,2.0],
    'CL_noise_max' : [0.05,0.1,0.2,0.4],
}

# [0.940800391523292 0.9223040878351222 0.8813156127929688 0.874336051940918
#  0.9336199760437012 0.8149833679199219 0.8350184440612793 256.0 8.0 6.0
#  0.1 720.0]



if __name__ == '__main__':

    set_seed(521)
    param_search = best_param_search

    save_file = '10cv_data_1000'
    params_all = Config()
    param_generator = set_attr(params_all, param_search)
    data_list = []
    filepath = Data_paths()

    while True:
        try:
            params = next(param_generator)
        except:
            break

        data_tuple = get_data(file_pair=filepath, params=params)
        try:
            if params.train_CL_model == True:
                model_cl = CL_train_init(params, data_tuple) # 对比学习
            data_idx, auc_name = CV_train(params, data_tuple) # 交叉验证
            data_list.append(data_idx)
        except Exception as e:
            print('出错了...')

    if data_list.__len__() > 1:
        data_all = np.concatenate(tuple(x for x in data_list), axis=1)
    else:
        data_all = data_list[0]
    np.save(params_all.save_file + save_file + '.npy', data_all)
    data_all
    print(auc_name)

    data_idx = np.load(params_all.save_file + save_file + '.npy', allow_pickle=True)

    data_mean = data_idx[:, :, 2:].mean(0)
    idx_max = data_mean[:, 1].argmax()
    print()
    print('最大值为：')
    print(data_mean[idx_max, :])

    # data_all

# data_idx_test = np.load('./My_paper/paper7_CL/method1_CL/交叉验证/save_file/test_search.npy',allow_pickle=True)
