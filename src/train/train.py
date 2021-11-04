import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from lightgbm.sklearn import LGBMClassifier
from collections import defaultdict
import gc, pickle, os, time
import random
from utils import reduce_mem, uAUC, get_logger

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



logger = get_logger("data/log/mmoe_0807.txt")
logger.info("start training...")


train = pd.read_pickle("data/features/train_v0.pkl").reset_index(drop=True)
test = pd.read_pickle("data/features/test_v0.pkl").reset_index(drop=True)
print(train.shape, test.shape)

df = pd.concat([train, test], ignore_index=True)
print(df.shape)

del train, test
gc.collect()
# print(df.info())

## 特征列的定义
play_cols = ['is_finish', 'play_times', 'stay_times', 'play', 'stay']
y_list = ['read_comment', 'like', 'click_avatar', 'favorite', 'forward', 'comment', 'follow']

## 离散和连续特征
sparse_features = ['userid', 'feedid', 'device', 'authorid', 'bgm_song_id', 
                   'bgm_singer_id', 'keyword1', 'tag1']
dense_features = [x for x in df.columns if x not in sparse_features + ['date_'] + play_cols + y_list]
logger.info("sparse_fea: {}, dense_fea: {}".format(len(sparse_features), len(dense_features)))


def get_word2id(df, col, is_del=True):
    cnt_dict0 = dict(Counter(df[col]))
    if is_del:
        cnt_dict = {k: v for k, v in cnt_dict0.items() if v >= 2}
        word2id = {k: (i+2) for i, k in enumerate(cnt_dict.keys())}
    else:
        word2id = {k: i for i, k in enumerate(cnt_dict0.keys())}
    print("{}, {} -> {}".format(col, len(cnt_dict0), len(word2id)))
    return word2id

userid_2_id = get_word2id(df, 'userid', is_del=True)
feedid_2_id = get_word2id(df, 'feedid', is_del=True)
device_2_id = get_word2id(df, 'device', is_del=False)
authorid_2_id = get_word2id(df, 'authorid', is_del=True)
bgm_song_id_2_id = get_word2id(df, 'bgm_song_id', is_del=True)
bgm_singer_id_2_id = get_word2id(df, 'bgm_singer_id', is_del=True)
keyword1_2_id = get_word2id(df, 'keyword1', is_del=True)
tag1_2_id = get_word2id(df, 'tag1', is_del=True)
print(len(userid_2_id), len(feedid_2_id), len(device_2_id), len(authorid_2_id), 
      len(bgm_song_id_2_id), len(bgm_singer_id_2_id), len(keyword1_2_id), len(tag1_2_id))


pickle.dump([userid_2_id, feedid_2_id, device_2_id, authorid_2_id,
             bgm_song_id_2_id, bgm_singer_id_2_id, keyword1_2_id, tag1_2_id], 
            open("data/features/all_word2id.pkl", 'wb'))



%%time

## feed侧的embedding
fid_kw_tag_word_emb = pd.read_pickle("data/features/fid_kw_tag_word_emb_final.pkl")
fid_mmu_emb = pd.read_pickle("data/features/fid_mmu_emb_final.pkl")
fid_w2v_emb = pd.read_pickle("data/features/fid_w2v_emb_final.pkl") 
fid_prone_emb = pd.read_pickle("data/features/fid_prone_emb_final.pkl")
print(fid_kw_tag_word_emb.shape, fid_mmu_emb.shape, fid_w2v_emb.shape, fid_prone_emb.shape)
## 合并
fid_2_emb_df = fid_kw_tag_word_emb
fid_2_emb_df = fid_2_emb_df.merge(fid_mmu_emb, how='left', on=['feedid'])
fid_2_emb_df = fid_2_emb_df.merge(fid_w2v_emb, how='left', on=['feedid'])
fid_2_emb_df = fid_2_emb_df.merge(fid_prone_emb, how='left', on=['feedid'])
fid_2_emb_df.fillna(0.0, inplace=True)
print("feedid: ", fid_2_emb_df.shape)

## userid侧的embedding
uid_2_emb_df = pd.read_pickle("data/features/uid_prone_emb_final.pkl")
print("userid: ", uid_2_emb_df.shape)

## authorid侧的embedding
aid_2_emb_df = pd.read_pickle("data/features/aid_prone_emb_final.pkl")
print("authorid: ", aid_2_emb_df.shape)


## 制作hash
fid_2_emb = {}
for line in (fid_2_emb_df.values):
    fid_2_emb[int(line[0])] = line[1:].astype(np.float32)   
uid_2_emb = {}
for line in (uid_2_emb_df.values):
    uid_2_emb[int(line[0])] = line[1:].astype(np.float32)
aid_2_emb = {}
for line in (aid_2_emb_df.values):
    aid_2_emb[int(line[0])] = line[1:].astype(np.float32)


## 删除，减少内存消耗
del fid_kw_tag_word_emb, fid_mmu_emb, fid_w2v_emb, fid_prone_emb
# del fid_2_emb_df, uid_2_emb_df, aid_2_emb_df
gc.collect()
gc.collect()

## 打印长度
print('feedid', len(fid_2_emb), len(fid_2_emb[54042]))
print('userid', len(uid_2_emb), len(uid_2_emb[0]))
print('authorid', len(aid_2_emb), len(aid_2_emb[0]))
# print('userid_date_', len(uiddate_2_bertemb), len(uiddate_2_bertemb[(0, 8)]))

emb_fea_nums = len(fid_2_emb[54042]) + len(uid_2_emb[0]) + len(aid_2_emb[0])
logger.info("embedding features nums: {}".format(emb_fea_nums))


manual_fea = pickle.load(open("data/features/singer_col_stat_feas.pkl", 'rb'))

# munual_fea = [userid_2_stat_fea, feedid_2_stat_fea, authorid_2_stat_fea, 
#               bgm_song_id_2_stat_fea, bgm_singer_id_2_stat_fea, 
#               keyword1_2_stat_fea, tag1_2_stat_fea]

# def get_emb_hash(df):
#     res = {}
#     for line in tqdm(df.values):
#         res[(int(line[0]), int(line[1]))] = line[2:].astype(np.float32)
#     return res

# manual_fea[0] = get_emb_hash(manual_fea[0])

for x in manual_fea:
    print(len(x), type(x))
    

class MyDataset(Dataset):
    def __init__(self, df, sparse_cols, dense_cols, labels, 
                 word2id_list, 
                 uid_2_emb=None, fid_2_emb=None, aid_2_emb=None,
                 uiddate_2_bertemb=None,
                 manual_fea=None,
                 uid_date_2_fid_hist=None, uid_date_2_aid_hist=None,
                 ):
        self.sparse_features = df[sparse_cols].values
        self.dense_features = df[dense_cols].values
        self.dates = df['date_'].values
        self.labels = df[labels].values
        
        self.word2id_list = word2id_list
        
        self.uid_2_emb = uid_2_emb
        self.fid_2_emb = fid_2_emb
        self.aid_2_emb = aid_2_emb
        self.uiddate_2_bertemb = uiddate_2_bertemb
        self.manual_fea = manual_fea
        self.mf_size = [41, 30, 33, 32, 32, 32, 32]
        self.df_len = df.shape[0]

    def __len__(self):
        return self.df_len

    def __getitem__(self, i):  
        # 标签信息，日期信息
        label = self.labels[i]
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
        
        return (
                torch.FloatTensor(sparse_f + dense_f + all_emb_f),
                torch.FloatTensor(label),
               )

    
word2id_list = pickle.load(open("data/features/all_word2id.pkl", 'rb'))

def get_loader(df, batch_size=20480, train_mode=True, n_cpu=14):
    ds = MyDataset(df, sparse_features, dense_features, labels=y_list,
                    word2id_list=word2id_list, 
                    uid_2_emb=uid_2_emb, fid_2_emb=fid_2_emb, aid_2_emb=aid_2_emb,
                    uiddate_2_bertemb=None,
                    manual_fea=manual_fea,
                    uid_date_2_fid_hist=None, uid_date_2_aid_hist=None)
    if train_mode:
        sampler = RandomSampler(ds)
    else:
        sampler = SequentialSampler(ds)
    my_loader = DataLoader(ds,
                           sampler=sampler, 
                           batch_size=batch_size, 
                           num_workers=n_cpu, 
                           pin_memory=True)
    return my_loader



dense_fea_nums = sum([41, 30, 33, 32, 32, 32, 32])
print(dense_fea_nums)
print(emb_fea_nums)
print(len(sparse_features), len(dense_features))
print(len(sparse_features) + len(dense_features) + dense_fea_nums + emb_fea_nums)


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
class MMOE_DNN(BaseModel):
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, 
                 use_fm=False, use_din=False,
                 embed_dim=32,
                 dnn_hidden_units=(256, 128), 
                 l2_reg_linear=0.001, l2_reg_embedding=0.01, l2_reg_dnn=0.0, init_std=0.001, seed=1024,
                 dnn_dropout=0.5, dnn_activation='relu', dnn_use_bn=True, task='binary', device='cpu', gpus=None,
                 num_tasks=4, num_experts=16, expert_dim=32, 
                 ):
        super(MMOE_DNN, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
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
    
    
# 打印模型参数
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def predict(model, test_loader, device):
    model.eval()
    pred_1, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7 = [], [], [], [], [], [], []
    with torch.no_grad():
        for x in tqdm(test_loader):
            y_pred = model(x[0].to(device))
            pred_1.extend(y_pred[0].cpu().data.numpy().squeeze().tolist())
            pred_2.extend(y_pred[1].cpu().data.numpy().squeeze().tolist())
            pred_3.extend(y_pred[2].cpu().data.numpy().squeeze().tolist())
            pred_4.extend(y_pred[3].cpu().data.numpy().squeeze().tolist())
            pred_5.extend(y_pred[4].cpu().data.numpy().squeeze().tolist())
            pred_6.extend(y_pred[5].cpu().data.numpy().squeeze().tolist())
            pred_7.extend(y_pred[6].cpu().data.numpy().squeeze().tolist())
    return (pred_1, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7)


def evaluate(model, valid_loader, val_x, device):
    actions = ['read_comment', 'like', 'click_avatar', 'favorite', 'forward', 'comment', 'follow']
    pred_ans = predict(model, valid_loader, device)
    uauc_list = []
    for i in range(len(actions)):
        assert len(val_x[actions[i]].values) == len(pred_ans[i]) == len(val_x['userid'].values)
        uauc_list.append(uAUC(val_x[actions[i]].values, pred_ans[i], val_x['userid'].values))
    return  round(np.average(uauc_list, weights=[4, 3, 2, 1, 1, 1, 1]), 6), uauc_list


def train_model(model, train_loader, valid_loader, valid,
                optimizer, epochs, device, model_save_file):
    train_bs = len(train_loader)
    best_score = 0.0
    
    patience = 0
    for epoch in range(epochs):
        logger.info("======= epoch {} ======".format(epoch+1))
        model.train()
        start_time = time.time()
        total_loss_sum = 0
        time.sleep(1.0)
        for idx, (out) in tqdm(enumerate(train_loader)):
            y = out[-1].to(device)
            y_pred = model(out[0].to(device))
            
            loss1 = F.binary_cross_entropy(y_pred[0].squeeze(), y[:, 0])
            loss2 = F.binary_cross_entropy(y_pred[1].squeeze(), y[:, 1])
            loss3 = F.binary_cross_entropy(y_pred[2].squeeze(), y[:, 2])
            loss4 = F.binary_cross_entropy(y_pred[3].squeeze(), y[:, 3])
            loss5 = F.binary_cross_entropy(y_pred[4].squeeze(), y[:, 4])
            loss6 = F.binary_cross_entropy(y_pred[5].squeeze(), y[:, 5])
            loss7 = F.binary_cross_entropy(y_pred[6].squeeze(), y[:, 6])
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
            
            reg_loss = model.module.get_regularization_loss()
            total_loss = loss + reg_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            total_loss_sum += total_loss.item()
            
            if (idx + 1) == train_bs:
                time.sleep(0.5)
                LR = optimizer.state_dict()['param_groups'][0]['lr']
                logger.info("Epoch {:03d} | Step {:04d} / {} | Loss {:.4f} | Reg Loss {:.4f}| LR {:.5f} | Time {:.4f}".format(
                     epoch+1, idx+1, train_bs, total_loss_sum/(idx+1), reg_loss.item(), LR,
                     time.time() - start_time))
        
        time.sleep(0.5)
        score, uAUC_list = evaluate(model, valid_loader, val_x, device)
        logger.info("Epoch:{} 结束，验证集uAUC = {}".format(epoch + 1, score))
        logger.info("uAUC list {}".format(uAUC_list))
        
        if score > best_score:
            best_score = score
            patience = 0
            model_to_save = model.module if hasattr(model,'module') else model
            torch.save(model_to_save.state_dict(),  "data/{}".format(model_save_file))
        else:
            patience += 1
        logger.info("Valid cur uAUC = {}, Valid best uAUC = {}, Cost Time {:.2f}".format(score, best_score, 
                                                                              time.time() - start_time))
        if patience >= 2:
            logger.info("Early Stopped! ")
            break
            
            
train = df[df['date_'] <= 14].reset_index(drop=True)
test = df[df['date_'] == 15].reset_index(drop=True)
del df
gc.collect()

# kf_way = fold.split(train)
print(train.shape, test.shape)

submit = pd.read_csv("/home/tione/notebook/wbdc2021/data/wedata/wechat_algo_data2/submit_demo_semi_a.csv")
for y in submit.columns[2:]:
    submit[y] = 0



for date_ in [14]:
    start_time = time.time()
    logger.info("********* train date_ is {} *********".format(date_))
    
    # For debug
    train_idx = train[train['date_'] != date_].index
    valid_idx = train[train['date_'] == date_].index
    
    ## 开始训练模型
    emb_size = 48
    actions =  ['read_comment', 'like', 'click_avatar', 'favorite', 'forward', 'comment', 'follow']
    
    # count #unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = [SparseFeat(feat, max(list(word2id_list[i].values()))+1, embedding_dim=emb_size) 
                              for i, feat in enumerate(sparse_features)] +\
                             [DenseFeat(feat, 1) for feat in dense_features + 
                              ['dense_{}'.format(i) for i in range(dense_fea_nums)] +
                              ['emb_{}'.format(i) for i in range(emb_fea_nums)]]
    logger.info("fixlen_fea nums: {}".format(len(fixlen_feature_columns)))
    
    # 所有特征列， dnn和linear都一样
    dnn_feature_columns = fixlen_feature_columns    # for DNN
    linear_feature_columns = fixlen_feature_columns   # for Embedding
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)   # all-特征名字
    logger.info("feature nums is {}".format(len(feature_names)))

    ## 验证集上的标签
    val_x = train.iloc[valid_idx][['userid'] + actions].reset_index(drop=True)
    for col in val_x.columns:
        val_x[col] = val_x[col].astype(np.int32)
    logger.info("valid df shape is {}".format(val_x.shape))
    
    # get 数据加载器
    train_loader = get_loader(train.iloc[train_idx].reset_index(drop=True))
    valid_loader = get_loader(train.iloc[valid_idx].reset_index(drop=True), train_mode=False)
    test_loader = get_loader(test, train_mode=False)
    
    logger.info("train_loader len {}, valid_loader len {}, test_loader len {},".format(len(train_loader), 
                                                                                       len(valid_loader),
                                                                                       len(test_loader)))
    
    # DEVICE
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        logger.info('cuda ready...')
        device = 'cuda:0'
    
    # 定义模型
    model = MMOE_DNN(linear_feature_columns=linear_feature_columns, 
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
    logger.info(get_parameter_number(model))

    ## 优化器和训练模型
    optimizer = optim.RMSprop(model.parameters(), lr=0.0015) 
    num_epochs = 15
    ## 模型并行
    model = torch.nn.DataParallel(model)
    
    model_save_file = 'best_mmoe_model_date_is_{}.bin'.format(date_)
    train_model(model, train_loader, valid_loader, val_x,
                optimizer, epochs=num_epochs, device=device, model_save_file=model_save_file)

    ## 加载最优模型, 小学习率继续预训练
    model.module.load_state_dict(torch.load("data/{}".format(model_save_file)))
    optimizer = optim.RMSprop(model.parameters(), lr=6e-5)
    num_epochs = 5
    train_model(model, train_loader, valid_loader, val_x, optimizer, epochs=num_epochs, device=device, 
                model_save_file=model_save_file)
    
    
    ## 取最优模型在验证集上进行验证
    logger.info("取最优模型在验证集上进行验证...")
    model.module.load_state_dict(torch.load("data/{}".format(model_save_file)))
    val_score, uauc_list = evaluate(model, valid_loader, val_x, device)
    logger.info("Valid best score is {}".format(val_score))
    logger.info(uauc_list)
    
    ## 对测试集进行预测
    test_preds = predict(model, test_loader, device)
    for i in range(len(actions)):
        submit[actions[i]] += np.round(test_preds[i], 8)
    logger.info("time costed: {}".format(round(time.time() - start_time, 6)))
    
    del train_loader, valid_loader, test_loader, model
    gc.collect()
    torch.cuda.empty_cache()
    
    





