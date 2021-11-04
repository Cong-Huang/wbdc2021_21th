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

import argparse
import logging
import warnings
from collections import Counter 
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'model'))
from mmoe import MMOE_DNN_v1, MMOE_DNN_v2

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

print(BASE_DIR)



class MyDataset(Dataset):
    def __init__(self, df, 
                 sparse_cols, dense_cols, 
                 word2id_list, 
                 uid_2_emb=None, fid_2_emb=None, aid_2_emb=None,
                 manual_fea=None,
                 ):
        self.sparse_features = df[sparse_cols].values
        self.dense_features = df[dense_cols].values
        self.dates = df['date_'].values
        
        self.word2id_list = word2id_list
        
        self.uid_2_emb = uid_2_emb
        self.fid_2_emb = fid_2_emb
        self.aid_2_emb = aid_2_emb
        
        self.manual_fea = manual_fea
        self.mf_size = [41, 30, 33, 32, 32, 32, 32]
        self.df_len = df.shape[0]

    def __len__(self):
        return self.df_len

    def __getitem__(self, i):  
        # 标签信息，日期信息
        date_ = self.dates[i]
        
        # Sparse特征
        sparse_f = self.sparse_features[i]
        uid, fid, device, aid, bgm_song, bgm_singer, kw1, tag1 = [int(x) for x in sparse_f]
        
        # Dense特征
        dense_f = list(self.dense_features[i])
        ## munual_fea
        mf_list = [uid, fid, aid, bgm_song, bgm_singer, kw1, tag1]
        
        for idx, mf in enumerate(self.manual_fea):
            dense_f.extend(list(mf.get((mf_list[idx], date_), [0.0]*self.mf_size[idx])))
        
        # Embedding特征
        all_emb_f = list(self.uid_2_emb.get(uid, [0.0]*128))
        all_emb_f.extend(list(self.fid_2_emb.get(fid, [0.0]*576)))
        all_emb_f.extend(list(self.aid_2_emb.get(aid, [0.0]*64)))
        
        sparse_f = [self.word2id_list[idx].get(int(sparse_f[idx]), 1) for idx in range(len(sparse_f))]
        
        return torch.FloatTensor(sparse_f + dense_f + all_emb_f)



def predict(model, test_loader, device):
    model.eval()
    pred_1, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7 = [], [], [], [], [], [], []
    with torch.no_grad():
        for x in tqdm(test_loader):
            y_pred = model(x.to(device))
            pred_1.extend(y_pred[0].cpu().data.numpy().squeeze().tolist())
            pred_2.extend(y_pred[1].cpu().data.numpy().squeeze().tolist())
            pred_3.extend(y_pred[2].cpu().data.numpy().squeeze().tolist())
            pred_4.extend(y_pred[3].cpu().data.numpy().squeeze().tolist())
            pred_5.extend(y_pred[4].cpu().data.numpy().squeeze().tolist())
            pred_6.extend(y_pred[5].cpu().data.numpy().squeeze().tolist())
            pred_7.extend(y_pred[6].cpu().data.numpy().squeeze().tolist())
    return (pred_1, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7)


def get_loader(test, sparse_features, dense_features, word2id_list, 
               uid_2_emb, fid_2_emb, aid_2_emb, manual_fea):

    ## 构建test_loader
    test_dataset = MyDataset(test, sparse_features, dense_features,
                            word2id_list=word2id_list, 
                            uid_2_emb=uid_2_emb, fid_2_emb=fid_2_emb, aid_2_emb=aid_2_emb,
                            manual_fea=manual_fea)
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(test_dataset,
                           sampler=test_sampler, 
                           batch_size=20480, 
                           num_workers=14, 
                           pin_memory=True)
    print("test loader size {}".format(len(test_loader)))
    return test_loader


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--test_path", type=str, required=True, help="The test data file path")
    args = parser.parse_args()
    
    
    start_time = time.time()
    test = pd.read_csv(args.test_path)
    test['date_'] = 15
    print("test shape: {}".format(test.shape))
    # feed侧信息
    feed_info = pd.read_pickle("/home/tione/notebook/wbdc2021-semi/data/features/feed_info.pkl")
    feed_info.drop(columns=['all_keyword', 'all_tag'], inplace=True)
    print("feed info shape: {}".format(feed_info.shape))

    test = test.merge(feed_info, how='left', on=['feedid'])
    
    ## 特征列的定义
    play_cols = ['is_finish', 'play_times', 'stay_times', 'play', 'stay']
    y_list = ['read_comment', 'like', 'click_avatar', 'favorite', 'forward', 'comment', 'follow']

    ## 离散和连续特征
    sparse_features = ['userid', 'feedid', 'device', 'authorid', 'bgm_song_id', 
                       'bgm_singer_id', 'keyword1', 'tag1']
    dense_features = ['videoplayseconds', 'desc_ocr_same_rate', 'desc_len', 'asr_len', 'ocr_len']
    print("sparse_fea: {}, dense_fea: {}".format(len(sparse_features), len(dense_features)))
    
    
    fid_2_emb, uid_2_emb, aid_2_emb = pickle.load(open("/home/tione/notebook/wbdc2021-semi/data/features/fid_uid_aid_2_emb.pkl", 'rb'))
    ## 打印长度
    print('feedid', len(fid_2_emb), len(fid_2_emb[54042]))
    print('userid', len(uid_2_emb), len(uid_2_emb[0]))
    print('authorid', len(aid_2_emb), len(aid_2_emb[0]))

    emb_fea_nums = len(fid_2_emb[54042]) + len(uid_2_emb[0]) + len(aid_2_emb[0])
    print("embedding features nums: {}".format(emb_fea_nums))
    
    ## 手工特征
    manual_fea = pickle.load(open("/home/tione/notebook/wbdc2021-semi/data/features/singer_col_stat_feas_test.pkl", 'rb'))
    
    ## word2id list
    word2id_list = pickle.load(open("/home/tione/notebook/wbdc2021-semi/data/features/all_word2id.pkl", 'rb'))
    
    ## 构建test_loader
    test_loader = get_loader(test, sparse_features, dense_features, word2id_list, 
                             uid_2_emb, fid_2_emb, aid_2_emb, manual_fea)
    
    dense_fea_nums = sum([41, 30, 33, 32, 32, 32, 32])
    print("manual feature size {}".format(dense_fea_nums))
    print("all feature size {}".format(len(sparse_features) + len(dense_features) + dense_fea_nums + emb_fea_nums))
    
    ## 定义所有的特征列
    emb_size = 48
    actions =  ['read_comment', 'like', 'click_avatar', 'favorite', 'forward', 'comment', 'follow']
    new_dense_features = dense_features + ['dense_{}'.format(i) for i in range(dense_fea_nums)]
    print("dense_fea_nums: {}".format(len(new_dense_features)))
    # count #unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = [SparseFeat(feat, max(list(word2id_list[i].values()))+1, embedding_dim=emb_size)
                              for i, feat in enumerate(sparse_features)] +\
                            [DenseFeat(feat, 1) for feat in new_dense_features + 
                             ['emb_{}'.format(i) for i in range(emb_fea_nums)]]
    print("fixlen_feature nums: {}".format(len(fixlen_feature_columns)))
    # 所有特征列， dnn和linear都一样
    dnn_feature_columns = fixlen_feature_columns    # for DNN
    linear_feature_columns = fixlen_feature_columns   # for Embedding
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)   # all-特征名字
    print("Feature nums is {}".format(len(feature_names)))
    
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
    
    
    ## 定义模型，并开始推断  
    submit = test[['userid', 'feedid']]
    for col in actions:
        submit[col] = 0.0
    
    
    for model_date_ in [6, 7, 8, 9, 10, 11, 12, 13, 14]:
        model = MMOE_DNN_v1(linear_feature_columns=linear_feature_columns, 
                     dnn_feature_columns=dnn_feature_columns,
                     embed_dim=emb_size,
                     use_fm=False,
                     use_din=False,
                     dnn_use_bn=True,
                     dnn_hidden_units=(2048, 1024, 512, 256), 
                     init_std=0.0001, dnn_dropout=0.5, task='binary', 
                     l2_reg_embedding=1e-5, 
                     l2_reg_linear=0.0,
                     l2_reg_dnn=0.0, 
                     device=device,
                     num_tasks=7, num_experts=48, expert_dim=128)
        model.to(device)
        model.load_state_dict(torch.load("/home/tione/notebook/wbdc2021-semi/data/model/best_mmoe_model_date_is_{}.bin".format(model_date_)))
        model = torch.nn.DataParallel(model)
        
        test_preds = predict(model, test_loader, device)
        for i in range(len(actions)):
            submit[actions[i]] += np.round(test_preds[i], 10) * 1.0
        
    
    del test_loader
    gc.collect()
    test_loader = get_loader(test, sparse_features, dense_features, word2id_list, 
                             uid_2_emb, fid_2_emb, aid_2_emb, manual_fea)
    
    for model_idx in [6, 7, 8, 9, 10, 11, 12, 13, 14]:
        model = MMOE_DNN_v2(linear_feature_columns=linear_feature_columns, 
                     dnn_feature_columns=dnn_feature_columns,
                     embed_dim=emb_size,
                     use_fm=True,
                     use_din=False,
                     dnn_use_bn=True,
                     dnn_hidden_units=(2048, 1024, 512, 512), 
                     init_std=0.0001, dnn_dropout=0.5, task='binary', 
                     l2_reg_embedding=1e-5, 
                     l2_reg_linear=0.0,
                     l2_reg_dnn=0.0, 
                     device=device,
                     num_tasks=7, num_experts=32, expert_dim=256)
    
        model.to(device)
        model.load_state_dict(torch.load("/home/tione/notebook/wbdc2021-semi/data/model/best_mmoe_model_date_is_{}_v2.bin".format(model_idx)))
        model = torch.nn.DataParallel(model)
        
        test_preds = predict(model, test_loader, device)
        for i in range(len(actions)):
            submit[actions[i]] += np.round(test_preds[i], 10) * 1.0
    
    for col in submit.columns[2:]:
        submit[col] = np.round(submit[col] / (1.0 * 9 + 1.0 * 9), 8)
    submit.to_csv("/home/tione/notebook/wbdc2021-semi/data/submission/result.csv", index=None)
    print("time costed: {} (s)".format(round(time.time() - start_time, 6)))
    

if __name__ == "__main__":
    main()