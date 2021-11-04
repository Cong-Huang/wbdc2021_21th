import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import gc
import random
import time
from utils import reduce_mem, uAUC, ProNE, HyperParam
import logging
import pickle
from gensim.models import word2vec
from sklearn.decomposition import PCA, TruncatedSVD, SparsePCA
import networkx as nx
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer
import warnings
from joblib import Parallel, delayed

pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")


## 数据预处理，去除噪声
data_path = "/home/tione/notebook/wbdc2021/data/wedata/wechat_algo_data2/"

train = pd.read_csv(data_path + 'user_action.csv')
print(train.shape)
train.drop_duplicates(['userid', 'feedid'], inplace=True)
print(train.shape)

train['play'] = train['play'] / 1000.0
train['stay'] = train['stay'] / 1000.0
train['play'] = train['play'].apply(lambda x: min(x, 180.0))
train['stay'] = train['stay'].apply(lambda x: min(x, 180.0))

test_a = pd.read_csv(data_path + 'test_a.csv')
# test_b = pd.read_csv(data_path + 'test_b.csv')
# print(test_a.shape, test_b.shape)
print(test_a.shape)

feed_info = pd.read_csv(data_path + 'feed_info.csv')
feed_info['videoplayseconds'] = feed_info['videoplayseconds'].apply(lambda x: min(x, 60))
print("缺失值情况：\n", feed_info.isnull().sum())

## 填充缺失值
for col in ['description', 'ocr', 'asr', 'description_char', 'ocr_char', 'asr_char',
            'machine_keyword_list', 'manual_keyword_list', 'manual_tag_list', 'machine_tag_list']:
    feed_info[col] = feed_info[col].fillna('')
    
for col in ['bgm_song_id', 'bgm_singer_id']:
    feed_info[col] = feed_info[col].fillna(-1)
    
# reduce memory
train = reduce_mem(train, train.columns)
test_a = reduce_mem(test_a, test_a.columns)

train.to_pickle("../data/origin/user_action.pkl")
test_a.to_pickle("../data/origin/test_a.pkl")
feed_info.to_pickle("../data/origin/feed_info.pkl")



## User侧的GNN特征
y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
max_day = 15

## 读取训练集
train = pd.read_pickle('../data/origin/user_action.pkl')
test = pd.read_pickle('../data/origin/test_a.pkl')
test['date_'] = max_day
print(train.shape, test.shape)

## 合并处理
df = pd.concat([train, test], ignore_index=True)
print(df.shape)

feed_info = pd.read_pickle('../data/origin/feed_info.pkl')[['feedid', 'authorid', 'videoplayseconds']]
df = df.merge(feed_info, how='left', on=['feedid'])
df['play_times'] = df['play'] / df['videoplayseconds']

# df = df.sample(n=1000000).reset_index(drop=True)
print(df.shape)


def get_proNE_embedding(df, col1, col2, emb_size=64):
    ### userid-feedid二部图
    uid_lbl,qid_lbl = LabelEncoder(), LabelEncoder()
    df['new_col1'] = uid_lbl.fit_transform(df[col1])
    df['new_col2'] = qid_lbl.fit_transform(df[col2])
    new_uid_max = df['new_col1'].max() + 1
    df['new_col2'] += new_uid_max
    
    ## 构建图
    G = nx.Graph()
    G.add_edges_from(df[['new_col1','new_col2']].values)

    model = ProNE(G, emb_size=emb_size, n_iter=6, step=12) 
    features_matrix = model.fit(model.mat, model.mat)
    model.chebyshev_gaussian(model.mat, features_matrix,
                             model.step, model.mu, model.theta)
    ## 得到proNE的embedding
    emb = model.transform()

    ## for userid
    uid_emb = emb[emb['nodes'].isin(df['new_col1'])]
    uid_emb['nodes'] = uid_lbl.inverse_transform(uid_emb['nodes'])  # 得到原id
    uid_emb.rename(columns={'nodes' : col1}, inplace=True)
    for col in uid_emb.columns[1:]:
        uid_emb[col] = uid_emb[col].astype(np.float32)
    user_prone_emb = uid_emb[uid_emb.columns]
    user_prone_emb = user_prone_emb.reset_index(drop=True)
    user_prone_emb.columns = [col1] + ['prone_emb{}'.format(i) for i in range(emb_size)]


    ## for feedid
    fid_emb = emb[emb['nodes'].isin(df['new_col2'])]
    fid_emb['nodes'] = qid_lbl.inverse_transform(fid_emb['nodes'] - new_uid_max)  ## 还原需要减掉
    fid_emb.rename(columns={'nodes' : col2}, inplace=True)
    for col in fid_emb.columns[1:]:
        fid_emb[col] = fid_emb[col].astype(np.float32)
    feed_prone_emb = fid_emb[fid_emb.columns]
    feed_prone_emb = feed_prone_emb.reset_index(drop=True)
    feed_prone_emb.columns = [col2] + ['prone_emb{}'.format(i) for i in range(emb_size)]
    print(user_prone_emb.shape, feed_prone_emb.shape)
    return user_prone_emb, feed_prone_embdef get_proNE_embedding(df, col1, col2, emb_size=64):
    ### userid-feedid二部图
    uid_lbl,qid_lbl = LabelEncoder(), LabelEncoder()
    df['new_col1'] = uid_lbl.fit_transform(df[col1])
    df['new_col2'] = qid_lbl.fit_transform(df[col2])
    new_uid_max = df['new_col1'].max() + 1
    df['new_col2'] += new_uid_max
    
    ## 构建图
    G = nx.Graph()
    G.add_edges_from(df[['new_col1','new_col2']].values)

    model = ProNE(G, emb_size=emb_size, n_iter=6, step=12) 
    features_matrix = model.fit(model.mat, model.mat)
    model.chebyshev_gaussian(model.mat, features_matrix,
                             model.step, model.mu, model.theta)
    ## 得到proNE的embedding
    emb = model.transform()

    ## for userid
    uid_emb = emb[emb['nodes'].isin(df['new_col1'])]
    uid_emb['nodes'] = uid_lbl.inverse_transform(uid_emb['nodes'])  # 得到原id
    uid_emb.rename(columns={'nodes' : col1}, inplace=True)
    for col in uid_emb.columns[1:]:
        uid_emb[col] = uid_emb[col].astype(np.float32)
    user_prone_emb = uid_emb[uid_emb.columns]
    user_prone_emb = user_prone_emb.reset_index(drop=True)
    user_prone_emb.columns = [col1] + ['prone_emb{}'.format(i) for i in range(emb_size)]


    ## for feedid
    fid_emb = emb[emb['nodes'].isin(df['new_col2'])]
    fid_emb['nodes'] = qid_lbl.inverse_transform(fid_emb['nodes'] - new_uid_max)  ## 还原需要减掉
    fid_emb.rename(columns={'nodes' : col2}, inplace=True)
    for col in fid_emb.columns[1:]:
        fid_emb[col] = fid_emb[col].astype(np.float32)
    feed_prone_emb = fid_emb[fid_emb.columns]
    feed_prone_emb = feed_prone_emb.reset_index(drop=True)
    feed_prone_emb.columns = [col2] + ['prone_emb{}'.format(i) for i in range(emb_size)]
    print(user_prone_emb.shape, feed_prone_emb.shape)
    return user_prone_emb, feed_prone_emb


user_prone_emb1, feed_prone_emb = get_proNE_embedding(df[['userid', 'feedid']], 
                                                      col1='userid', col2='feedid', emb_size=64)
user_prone_emb2, auth_prone_emb = get_proNE_embedding(df[['userid', 'authorid']], 
                                                      col1='userid', col2='authorid', emb_size=64)

user_prone_emb2.columns = ['userid'] + ['prone_emb{}'.format(i) for i in range(64, 128)]
user_prone_emb = user_prone_emb1.merge(user_prone_emb2, how='left', on=['userid'])

print(user_prone_emb.shape, feed_prone_emb.shape, auth_prone_emb.shape)
user_prone_emb.to_pickle("../data/features/uid_prone_emb_final.pkl")
feed_prone_emb.to_pickle("../data/features/fid_prone_emb_final.pkl")
auth_prone_emb.to_pickle("../data/features/aid_prone_emb_final.pkl")


## 多模态特征

feed_emb = pd.read_csv("../data/origin/feed_embeddings.csv")
print(feed_emb.shape)
time.sleep(0.5)
feedid_list, emb_list = [], []
for line in tqdm(feed_emb.values):
    fid, emb = int(line[0]), [float(x) for x in line[1].split()]
    feedid_list.append(fid)
    emb_list.append(emb)

feedid_emb = np.array(emb_list, dtype=np.float32)
emb_size = 192

# feedid_emb = feedid_emb - feedid_emb.mean(0, keepdims=True)
# ss = StandardScaler()
# feedid_emb = ss.fit_transform(feedid_emb)
# print(feedid_emb.shape)

# pca = PCA(n_components=emb_size)
# fid_emb = pca.fit_transform(feedid_emb)

svd = TruncatedSVD(n_components=emb_size)
fid_emb = svd.fit_transform(feedid_emb)


print(fid_emb.shape)
fid_emb = fid_emb.astype(np.float32)

fid_mmu_emb = pd.concat([feed_emb[['feedid']],
                         pd.DataFrame(fid_emb, columns=['mmu_emb{}'.format(i) for i in range(emb_size)])], 
                        axis=1)

fid_mmu_emb.to_pickle("../data/features/fid_mmu_emb_final.pkl")

# print(svd.explained_variance_ratio_)
# print(np.cumsum(svd.explained_variance_ratio_))
print("方差信息保留：", svd.explained_variance_ratio_.sum())




## Feedid的word2vec特征
## 读取训练集
train = pd.read_pickle('../data/origin/user_action.pkl')
test = pd.read_pickle('../data/origin/test_a.pkl')
test['date_'] = 15
print(train.shape, test.shape)
## 合并处理
df = pd.concat([train, test], ignore_index=True)
print(df.shape)


feed_info = pd.read_pickle('../data/origin/feed_info.pkl')[['feedid', 'videoplayseconds']]
df = df.merge(feed_info, how='left', on=['feedid'])
df['play_times'] = df['play'] / df['videoplayseconds']


# 用户历史n天的 feedid序列
user_fid_list = []
n_day = 5
for target_day in range(6, 17):
    left, right = max(target_day - n_day, 1), target_day - 1
    tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)
    user_dict = tmp.groupby('userid')['feedid'].agg(list)
    user_fid_list.extend(user_dict.values.tolist())
    
#     tmp = tmp[tmp['play_times'] >= 1.0].reset_index(drop=True)
#     print(tmp.shape)
#     user_dict = tmp.groupby('userid')['feedid'].agg(list)
#     user_fid_list.extend(user_dict.values.tolist())
    print(target_day, left, right, len(user_dict))


## 训练word2vec
print("number of sentence {}".format(len(user_fid_list)))
emb_size = 128
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = word2vec.Word2Vec(user_fid_list, min_count=1, window=20,
                          vector_size=emb_size, sg=1, workers=14, epochs=10)
model.save("../data/w2v_model_128d.model")
# model = word2vec.Word2Vec.load("../data/w2v_model_128d.model")


## 将每个feedid的向量保存为pickle

feed_emb = pd.read_csv("../data/origin/feed_embeddings.csv")[['feedid']]
w2v_fid_mat = []
emb_size = 128
null_cnt = 0
for fid in tqdm(feed_emb.feedid.values):
    try:
        emb = model.wv[fid]
    except:
        emb = np.zeros(emb_size)
        null_cnt += 1
    w2v_fid_mat.append(emb)

print(null_cnt)
w2v_fid_mat = np.array(w2v_fid_mat, dtype=np.float32)

fid_w2v_emb = pd.concat([feed_emb, pd.DataFrame(w2v_fid_mat, 
                                                columns=['w2v_emb{}'.format(i) for i in range(emb_size)])], 
                        axis=1)

fid_w2v_emb.to_pickle("../data/features/fid_w2v_emb_final.pkl")




## Feed_info的预处理
feed_info = pd.read_pickle('../data/origin/feed_info.pkl')

manual_kw = feed_info['manual_keyword_list'].progress_apply(lambda x: x.split(';'))
machine_kw = feed_info['machine_keyword_list'].progress_apply(lambda x: x.split(';'))
manual_tag = feed_info['manual_tag_list'].progress_apply(lambda x: x.split(';'))
def func(x):
    if len(x) == 0:
        return ['-1']
    return [_.split()[0] for _ in x.split(';') if float(_.split()[1]) >= 0.5]
machine_tag = feed_info['machine_tag_list'].progress_apply(lambda x: func(x))

all_kw = []   # 关键词
assert len(manual_kw) == len(machine_kw)
for i in (range(len(manual_kw))):
    tmp = set(manual_kw[i] + machine_kw[i])
    tmp = [x.strip() for x in tmp if x != '' and x != '-1']
    if len(tmp) == 0:
        tmp = ['-1']
    all_kw.append(' '.join(tmp))

all_tag = []   # tag标签
assert len(manual_tag) == len(machine_tag)
for i in (range(len(manual_tag))):
    tmp = set(manual_tag[i] + machine_tag[i])
    tmp = [x.strip() for x in tmp if x != '' and x != '-1']
    if len(tmp) == 0:
        tmp = ['-1']
    all_tag.append(' '.join(tmp))
    
assert len(all_kw) == len(all_tag)

## 处理keyword
print("****** 处理keyword *******")
emb_size = 48
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1), min_df=20)
all_kw_mat = tfidf_vectorizer.fit_transform(all_kw)
kw1 = np.array(all_kw_mat.argmax(axis=1)).reshape(-1)

svd = TruncatedSVD(n_components=emb_size)
all_kw_mat = svd.fit_transform(all_kw_mat)
print(all_kw_mat.shape)
print("方差信息保留：", svd.explained_variance_ratio_.sum())


## 处理tag
print("****** 处理tag *******")
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1), min_df=10)
all_tag_mat = tfidf_vectorizer.fit_transform(all_tag)
tag1 = np.array(all_tag_mat.argmax(axis=1)).reshape(-1)

svd = TruncatedSVD(n_components=emb_size)
all_tag_mat = svd.fit_transform(all_tag_mat)
print(all_tag_mat.shape)
print("方差信息保留：", svd.explained_variance_ratio_.sum())


## 处理words
print("****** 处理words *******")
all_words = feed_info['description'] + ' ' + feed_info['ocr'] + ' ' + feed_info['asr']
all_words = [' '.join(x.split()[:100]) for x in all_words.values.tolist()]
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1), min_df=20)
all_words_mat = tfidf_vectorizer.fit_transform(all_words)

svd = TruncatedSVD(n_components=emb_size)
all_words_mat = svd.fit_transform(all_words_mat)
print(all_words_mat.shape)
print("方差信息保留：", svd.explained_variance_ratio_.sum())


## 处理chars
print("****** 处理chars *******")
all_chars = feed_info['description_char'] + ' ' + feed_info['ocr_char'] + ' ' + feed_info['asr_char']
all_chars = [' '.join(x.split()[:100]) for x in all_chars.values.tolist()]
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1), min_df=20)
all_chars_mat = tfidf_vectorizer.fit_transform(all_chars)

svd = TruncatedSVD(n_components=emb_size)
all_chars_mat = svd.fit_transform(all_chars_mat)
print(all_chars_mat.shape)
print("方差信息保留：", svd.explained_variance_ratio_.sum())


all_kw_mat = all_kw_mat.astype(np.float32)
all_tag_mat = all_tag_mat.astype(np.float32)
all_words_mat = all_words_mat.astype(np.float32)
all_chars_mat = all_chars_mat.astype(np.float32)

fid_kw_tag_word_emb = pd.concat([feed_info[['feedid']], 
                                pd.DataFrame(all_kw_mat, columns=['kw_emb{}'.format(i) for i in range(emb_size)]),
                                pd.DataFrame(all_tag_mat, columns=['tag_emb{}'.format(i) for i in range(emb_size)]), 
                                pd.DataFrame(all_words_mat, columns=['word_emb{}'.format(i) for i in range(emb_size)]), 
                                pd.DataFrame(all_chars_mat, columns=['char_emb{}'.format(i) for i in range(emb_size)]), 
                                ], axis=1)

fid_kw_tag_word_emb.to_pickle("../data/features/fid_kw_tag_word_emb_final.pkl")



## 相同字数占比, desc, ocr, asr字数
def funct(row):
    desc = row['description_char']
    ocr = row['ocr_char']
    desc, ocr = set(desc.split()), set(ocr.split())
    return len(desc & ocr) / (min(len(desc), len(ocr)) + 1e-8)

feed_info['desc_ocr_same_rate'] = feed_info.apply(lambda row: funct(row), axis=1)
feed_info['desc_len'] = feed_info['description_char'].apply(lambda x: len(x.split()))
feed_info['asr_len'] = feed_info['asr_char'].apply(lambda x: len(x.split()))
feed_info['ocr_len'] = feed_info['ocr_char'].apply(lambda x: len(x.split()))

feed_info['keyword1'] = kw1
feed_info['tag1'] = tag1
feed_info['all_keyword'] = all_kw
feed_info['all_tag'] = all_tag

# def get_tag_top1(x):
#     try:
#         tmp = sorted([(int(x_.split()[0]), float(x_.split()[1]))  for x_ in x.split(';') if len(x_) > 0], 
#                        key=lambda x: x[1], reverse=True)
#     except:
#         return 0
#     return tmp[0][0]
# feed_info['tag_m1'] =  feed_info['machine_tag_list'].apply(lambda x: get_tag_top1(x))

feed_info.drop(columns=['description', 'ocr', 'asr', 
                        'manual_keyword_list', 'machine_keyword_list', 
                        'manual_tag_list', 'machine_tag_list',
                        'description_char', 'ocr_char', 'asr_char'], inplace=True)
feed_info['bgm_song_id'] = feed_info['bgm_song_id'].astype(np.int32)
feed_info['bgm_singer_id'] = feed_info['bgm_singer_id'].astype(np.int32)
feed_info.to_pickle("../data/features/feed_info.pkl")



## CTR特征


## 读取训练集
train = pd.read_pickle('../data/origin/user_action.pkl')
print(train.shape)

## 读取测试集
test = pd.read_pickle('../data/origin/test_a.pkl')
test['date_'] = 15
print(test.shape)

## 合并处理
df = pd.concat([train, test], ignore_index=True)
print(df.shape)

del train, test
gc.collect()

# feed侧信息
feed_info = pd.read_pickle("../data/features/feed_info.pkl")
feed_info.drop(columns=['all_keyword', 'all_tag'], inplace=True)
print(feed_info.shape)

df = df.merge(feed_info, on='feedid', how='left')

## 是否观看完视频（其实不用严格按大于关系，也可以按比例，比如观看比例超过0.9就算看完）
df['is_finish'] = (df['play'] >= df['videoplayseconds']).astype('int8')
df['play_times'] = df['play'] / df['videoplayseconds']
df['stay_times'] = df['stay'] / df['videoplayseconds']



play_cols = ['is_finish', 'play_times', 'stay_times', 'play', 'stay']
y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
max_day = 15

cols = ['userid', 'feedid', 'authorid', 'bgm_song_id',
        'bgm_singer_id', 'tag1', 'keyword1', 'date_'] + y_list

df = reduce_mem(df, [col for col in df.columns.tolist() if col not in y_list])
df = df[cols]

for col in y_list:
    df[col] = df[col].astype(np.float32)
        
print(df.shape)
print(df.info())


## 统计历史n天的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）
n_day = 5
max_day = 15

all_stat_cols = [['userid'], ['feedid'], ['authorid'], 
                 ['bgm_song_id'], ['bgm_singer_id'], ['tag1'], ['keyword1'],
                 ['userid', 'tag1'], ['userid', 'keyword1'], ['userid', 'authorid']]


def get_ctr_fea(df, all_stat_cols):
    
    def in_func(stat_cols):
        f = '_'.join(stat_cols)
        print('======== ' + f + ' =========')
        stat_df = pd.DataFrame()
        for target_day in range(6, max_day + 1):
            left, right = max(target_day - n_day, 1), target_day - 1
        
            tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)
            tmp['date_'] = target_day
            
            g = tmp.groupby(stat_cols)
            feats = []
            
            for y in y_list:
                tmp['{}_{}day_{}_sum'.format(f, n_day, y)] = g[y].transform('sum')
                tmp['{}_{}day_{}_mean'.format(f, n_day, y)] = g[y].transform('mean')
#                 tmp['{}_{}day_{}'.format(f, n_day, y) + '_all_count'] = g[y].transform('count')
#                 tmp['{}_{}day_{}'.format(f, n_day, y) + '_label_count'] = g[y].transform('sum')
            
#                 HP = HyperParam(1, 1)
#                 HP.update_from_data_by_moment(tmp['{}_{}day_{}'.format(f, n_day, y) + '_all_count'].values, 
#                                           tmp['{}_{}day_{}'.format(f, n_day, y) + '_label_count'].values)
#                 tmp['{}_{}day_{}_ctr'.format(f, n_day, y)] = (tmp['{}_{}day_{}'.format(f, n_day, y) + '_label_count']
#                                                               + HP.alpha) / (tmp['{}_{}day_{}'.format(f, n_day, y) + '_all_count']
#                                                                              + HP.alpha + HP.beta)
        
                feats.extend(['{}_{}day_{}_sum'.format(f, n_day, y),
                              '{}_{}day_{}_mean'.format(f, n_day, y)])
            
            tmp = tmp[stat_cols + ['date_'] + feats].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)
            stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)
        
        stat_df = reduce_mem(stat_df, stat_df.columns)
        m, n = stat_df.shape
        stat_df.to_pickle("../data/features/ctr_feas/{}_{}_{}_{}days_ctr_fea.pkl".format(f, m, n, n_day))
        
    n_jobs = len(all_stat_cols)
    all_stat_df = Parallel(n_jobs=n_jobs)(delayed(in_func)(col) for col in all_stat_cols)      

tmp_res = get_ctr_fea(df, all_stat_cols[:7])
tmp_res = get_ctr_fea(df, all_stat_cols[7:])


## stat统计特征
%%time

## 读取训练集
train = pd.read_pickle('../data/origin/user_action.pkl')
print(train.shape)
    
## 读取测试集
test = pd.read_pickle('../data/origin/test_a.pkl')
test['date_'] = 15
print(test.shape)

## 合并处理
df = pd.concat([train, test], ignore_index=True)
print(df.shape)

del train, test
gc.collect()

# feed侧信息
feed_info = pd.read_pickle("../data/features/feed_info.pkl")
feed_info.drop(columns=['all_keyword', 'all_tag'], inplace=True)
print(feed_info.shape)

df = df.merge(feed_info, on='feedid', how='left')

## 是否观看完视频（其实不用严格按大于关系，也可以按比例，比如观看比例超过0.9就算看完）
df['is_finish'] = (df['play'] >= df['videoplayseconds']).astype('int32')
df['play_times'] = df['play'] / df['videoplayseconds']
df['stay_times'] = df['stay'] / df['videoplayseconds']

# df = reduce_mem(df, df.columns)

play_cols = ['is_finish', 'play_times', 'stay_times', 'play', 'stay']
y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
max_day = 15

cols = ['userid', 'feedid', 'authorid', 'bgm_song_id',
        'bgm_singer_id', 'tag1', 'keyword1', 'date_'] + play_cols
df = df[cols]
df = reduce_mem(df, [col for col in cols if col not in play_cols])

for col in play_cols:
    df[col] = df[col].astype(np.float32)

print(df.shape)
print(df.info())

gc.collect()


n_day = 5
max_day = 15
all_stat_cols = [['userid'], ['feedid'], ['authorid'], 
                 ['bgm_song_id'], ['bgm_singer_id'], ['tag1'], ['keyword1'],
                 ['userid', 'tag1'], ['userid', 'keyword1'], ['userid', 'authorid']]

def get_stat_fea(df, all_stat_cols):
    
    def in_func(stat_cols):
        f = '_'.join(stat_cols)
        print('======== ' + f + ' =========')
        stat_df = pd.DataFrame()
        for target_day in tqdm(range(6, max_day + 1)):
            left, right = max(target_day - n_day, 1), target_day - 1
        
            tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)
            tmp['date_'] = target_day
            tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count')
        
            g = tmp.groupby(stat_cols)
            tmp['{}_{}day_finish_rate'.format(f, n_day)] = g[play_cols[0]].transform('mean')  # 观看完成率
        
            # 特征列
            feats = ['{}_{}day_count'.format(f, n_day), '{}_{}day_finish_rate'.format(f, n_day)]
        
            for x in play_cols[1:]:
                for stat in ['max', 'mean', 'sum']:
                    tmp['{}_{}day_{}_{}'.format(f, n_day, x, stat)] = g[x].transform(stat)
                    feats.append('{}_{}day_{}_{}'.format(f, n_day, x, stat))
        
            tmp = tmp[stat_cols + ['date_'] + feats].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)
            stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)
        
        stat_df = reduce_mem(stat_df, stat_df.columns)
        
        m, n = stat_df.shape
        stat_df.to_pickle("../data/features/stat_feas/{}_{}_{}_{}day_stat_fea.pkl".format(f, m, n, n_day))

    n_jobs = len(all_stat_cols)
    all_stat_df = Parallel(n_jobs=n_jobs)(delayed(in_func)(col) for col in all_stat_cols)
    

get_stat_fea(df, all_stat_cols[:7])
get_stat_fea(df, all_stat_cols[7:])



## 全局统计特征

## 读取训练集
train = pd.read_pickle('../data/origin/user_action.pkl')
print(train.shape)
    
## 读取测试集
test = pd.read_pickle('../data/origin/test_a.pkl')
test['date_'] = 15
print(test.shape)

## 合并处理
df = pd.concat([train, test], ignore_index=True)
print(df.shape)

del train, test
gc.collect()

# feed侧信息
feed_info = pd.read_pickle("../data/features/feed_info.pkl")
feed_info.drop(columns=['all_keyword', 'all_tag'], inplace=True)
print(feed_info.shape)

df = df.merge(feed_info, on='feedid', how='left')

## 是否观看完视频（其实不用严格按大于关系，也可以按比例，比如观看比例超过0.9就算看完）
df['is_finish'] = (df['play'] >= df['videoplayseconds']).astype('int8')
df['play_times'] = df['play'] / df['videoplayseconds']
df['stay_times'] = df['stay'] / df['videoplayseconds']

df = reduce_mem(df, df.columns)

play_cols = ['is_finish', 'play_times', 'stay_times', 'play', 'stay', 'videoplayseconds']
y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
max_day = 15

cols = ['userid', 'feedid', 'authorid', 'bgm_song_id',
        'bgm_singer_id', 'tag1', 'keyword1', 'date_'] + play_cols
df = df[cols]
df = reduce_mem(df, [col for col in cols if col not in play_cols])

for col in play_cols:
    df[col] = df[col].astype(np.float32)


print(df.shape)
print(df.info())
gc.collect()

%%time
## 全局信息统计，包括曝光、偏好等，略有穿越，但问题不大，可以上分，只要注意不要对userid-feedid做组合统计就行

count_feas = []
for f in tqdm(['userid', 'feedid', 'authorid', 'tag1', 'keyword1', 'bgm_song_id', 'bgm_singer_id']):
    df[f + '_count_global'] = df[f].map(df[f].value_counts())


for f1, f2 in tqdm([
     ['userid', 'feedid'], ['userid', 'authorid'], ['userid', 'tag1'], ['userid', 'keyword1'], 
     ['userid', 'bgm_song_id'], ['userid', 'bgm_singer_id']]):
    df['{}_in_{}_nunique_global'.format(f1, f2)] = df.groupby(f2)[f1].transform('nunique')
    df['{}_in_{}_nuni_div_cnt_global'.format(f1, f2)] = df['{}_in_{}_nunique_global'.format(f1, f2)] / df[f2 + '_count_global']
    
    df['{}_in_{}_nunique_global'.format(f2, f1)] = df.groupby(f1)[f2].transform('nunique')
    df['{}_in_{}_nuni_div_cnt_global'.format(f2, f1)] = df['{}_in_{}_nunique_global'.format(f2, f1)] / df[f1 + '_count_global']
    

    
# for f1, f2 in tqdm([['userid', 'authorid'], ['userid', 'tag1'], ['userid', 'keyword1'],
#                     ['userid', 'bgm_song_id'], ['userid', 'bgm_singer_id'],]):
#     df['{}_{}_count'.format(f1, f2)] = df.groupby([f1, f2])['date_'].transform('count')
#     df['{}_in_{}_count_prop'.format(f1, f2)] = df['{}_{}_count'.format(f1, f2)] / (df[f2 + '_count'] + 1)
#     df['{}_in_{}_count_prop'.format(f2, f1)] = df['{}_{}_count'.format(f1, f2)] / (df[f1 + '_count'] + 1)

df['videoplayseconds_in_userid_mean_global'] = df.groupby('userid')['videoplayseconds'].transform('mean')
df['videoplayseconds_in_authorid_mean_global'] = df.groupby('authorid')['videoplayseconds'].transform('mean')
df['videoplayseconds_in_keyword1_mean_global'] = df.groupby('keyword1')['videoplayseconds'].transform('mean')
df['videoplayseconds_in_tag1_mean_global'] = df.groupby('tag1')['videoplayseconds'].transform('mean')
df['videoplayseconds_in_bgm_song_id_mean_global'] = df.groupby('bgm_song_id')['videoplayseconds'].transform('mean')
df['videoplayseconds_in_bgm_singer_id_mean_global'] = df.groupby('bgm_singer_id')['videoplayseconds'].transform('mean')


df['feedid_in_authorid_nunique_global'] = df.groupby('authorid')['feedid'].transform('nunique')

del df['userid_in_feedid_nuni_div_cnt_global'], df['feedid_in_userid_nuni_div_cnt_global']
gc.collect()


userid_global_fea_cols = ['userid', 'userid_count_global', 'feedid_in_userid_nunique_global', 
                          'authorid_in_userid_nunique_global', 'authorid_in_userid_nuni_div_cnt_global',
                          'tag1_in_userid_nunique_global', 'tag1_in_userid_nuni_div_cnt_global',
                          'keyword1_in_userid_nunique_global', 'keyword1_in_userid_nuni_div_cnt_global', 
                          'bgm_song_id_in_userid_nunique_global', 'bgm_song_id_in_userid_nuni_div_cnt_global', 
                          'bgm_singer_id_in_userid_nunique_global', 'bgm_singer_id_in_userid_nuni_div_cnt_global', 
                          'videoplayseconds_in_userid_mean_global'] 
feedid_global_fea_cols = ['feedid', 'feedid_count_global', 'userid_in_feedid_nunique_global']
authorid_global_fea_cols = ['authorid', 'authorid_count_global',
                            'userid_in_authorid_nunique_global', 'userid_in_authorid_nuni_div_cnt_global', 
                            'videoplayseconds_in_authorid_mean_global', 
                            'feedid_in_authorid_nunique_global']
tag1_global_fea_cols = ['tag1', 'tag1_count_global',
                        'userid_in_tag1_nunique_global', 'userid_in_tag1_nuni_div_cnt_global', 
                        'videoplayseconds_in_tag1_mean_global']
keyword1_global_fea_cols = ['keyword1', 'keyword1_count_global', 
                            'userid_in_keyword1_nunique_global', 'userid_in_keyword1_nuni_div_cnt_global',
                            'videoplayseconds_in_keyword1_mean_global']
bgm_song_id_global_fea_cols = ['bgm_song_id', 'bgm_song_id_count_global', 
                               'userid_in_bgm_song_id_nunique_global', 'userid_in_bgm_song_id_nuni_div_cnt_global', 
                               'videoplayseconds_in_bgm_song_id_mean_global']
bgm_singer_id_global_fea_cols = ['bgm_singer_id', 'bgm_singer_id_count_global',
                                 'userid_in_bgm_singer_id_nunique_global', 'userid_in_bgm_singer_id_nuni_div_cnt_global', 
                                 'videoplayseconds_in_bgm_singer_id_mean_global']
print(len(userid_global_fea_cols + feedid_global_fea_cols + authorid_global_fea_cols+ tag1_global_fea_cols
          + keyword1_global_fea_cols + bgm_song_id_global_fea_cols + bgm_singer_id_global_fea_cols))


userid_global_fea = df[userid_global_fea_cols]
userid_global_fea.drop_duplicates(inplace=True)
print('userid_global_fea', userid_global_fea.shape)

feedid_global_fea = df[feedid_global_fea_cols]
feedid_global_fea.drop_duplicates(inplace=True)
print('feedid_global_fea', feedid_global_fea.shape)

authorid_global_fea = df[authorid_global_fea_cols]
authorid_global_fea.drop_duplicates(inplace=True)
print('authorid_global_fea', authorid_global_fea.shape)


keyword1_global_fea = df[keyword1_global_fea_cols]
keyword1_global_fea.drop_duplicates(inplace=True)
print('keyword1_global_fea', keyword1_global_fea.shape)

tag1_global_fea = df[tag1_global_fea_cols]
tag1_global_fea.drop_duplicates(inplace=True)
print('tag1_global_fea', tag1_global_fea.shape)


bgm_song_id_global_fea = df[bgm_song_id_global_fea_cols]
bgm_song_id_global_fea.drop_duplicates(inplace=True)
print('bgm_song_id_global_fea', bgm_song_id_global_fea.shape)

bgm_singer_id_global_fea = df[bgm_singer_id_global_fea_cols]
bgm_singer_id_global_fea.drop_duplicates(inplace=True)
print('bgm_singer_id_global_fea', bgm_singer_id_global_fea.shape)


userid_global_fea.to_pickle("../data/features/global_feas/userid_global_fea.pkl")
feedid_global_fea.to_pickle("../data/features/global_feas/feedid_global_fea.pkl")
authorid_global_fea.to_pickle("../data/features/global_feas/authorid_global_fea.pkl")
keyword1_global_fea.to_pickle("../data/features/global_feas/keyword1_global_fea.pkl")
tag1_global_fea.to_pickle("../data/features/global_feas/tag1_global_fea.pkl")
bgm_song_id_global_fea.to_pickle("../data/features/global_feas/bgm_song_id_global_fea.pkl")
bgm_singer_id_global_fea.to_pickle("../data/features/global_feas/bgm_singer_id_global_fea.pkl")



## 整理所有的手工特征，并且进行归一化

userid_global_fea = pd.read_pickle("../data/features/global_feas/userid_global_fea.pkl")
feedid_global_fea = pd.read_pickle("../data/features/global_feas/feedid_global_fea.pkl")
authorid_global_fea = pd.read_pickle("../data/features/global_feas/authorid_global_fea.pkl")
keyword1_global_fea = pd.read_pickle("../data/features/global_feas/keyword1_global_fea.pkl")
tag1_global_fea = pd.read_pickle("../data/features/global_feas/tag1_global_fea.pkl")
bgm_song_id_global_fea = pd.read_pickle("../data/features/global_feas/bgm_song_id_global_fea.pkl")
bgm_singer_id_global_fea = pd.read_pickle("../data/features/global_feas/bgm_singer_id_global_fea.pkl")


keyword1_global_fea.loc[keyword1_global_fea['keyword1'] == 0, keyword1_global_fea.columns[1:]] = 0
bgm_song_id_global_fea.loc[bgm_song_id_global_fea['bgm_song_id'] == -1, bgm_song_id_global_fea.columns[1:]] = 0
bgm_singer_id_global_fea.loc[bgm_singer_id_global_fea['bgm_singer_id'] == -1, bgm_singer_id_global_fea.columns[1:]] = 0


def normalize(df, col):
    for col in col:
        x = df[col].astype(np.float32)
        x = np.log(x + 1.0)
        mms = MinMaxScaler()
        x = mms.fit_transform(x.values.reshape(-1, 1))
        df[col] = x.reshape(-1).astype(np.float16)
        df[col] = df[col].fillna(0.0)
    return df

userid_global_fea = normalize(userid_global_fea, userid_global_fea.columns[1:])
feedid_global_fea = normalize(feedid_global_fea, feedid_global_fea.columns[1:])
authorid_global_fea = normalize(authorid_global_fea, authorid_global_fea.columns[1:])
keyword1_global_fea = normalize(keyword1_global_fea, keyword1_global_fea.columns[1:])
tag1_global_fea = normalize(tag1_global_fea, tag1_global_fea.columns[1:])
bgm_song_id_global_fea = normalize(bgm_song_id_global_fea, bgm_song_id_global_fea.columns[1:])
bgm_singer_id_global_fea = normalize(bgm_singer_id_global_fea, bgm_singer_id_global_fea.columns[1:])



uid_ctr = pd.read_pickle("../data/features/ctr_feas/userid_1843213_16_5days_ctr_fea.pkl")
fid_ctr = pd.read_pickle("../data/features/ctr_feas/feedid_736591_16_5days_ctr_fea.pkl")
aid_ctr = pd.read_pickle("../data/features/ctr_feas/authorid_158013_16_5days_ctr_fea.pkl")
kw1_ctr = pd.read_pickle("../data/features/ctr_feas/keyword1_24661_16_5days_ctr_fea.pkl")
kw1_ctr.loc[kw1_ctr['keyword1'] == 0, kw1_ctr.columns[2:]] = 0
tag1_ctr = pd.read_pickle("../data/features/ctr_feas/tag1_2494_16_5days_ctr_fea.pkl")

bgm_song_ctr = pd.read_pickle("../data/features/ctr_feas/bgm_song_id_188669_16_5days_ctr_fea.pkl")
bgm_song_ctr.loc[bgm_song_ctr['bgm_song_id'] == -1, bgm_song_ctr.columns[2:]] = 0
bgm_singer_ctr = pd.read_pickle("../data/features/ctr_feas/bgm_singer_id_134256_16_5days_ctr_fea.pkl")
bgm_singer_ctr.loc[bgm_singer_ctr['bgm_singer_id'] == -1, bgm_singer_ctr.columns[2:]] = 0


def normalize(df, col):
    for col in col:
        x = df[col].astype(np.float32)
        x = np.log(x + 1.0)
        mms = MinMaxScaler()
        x = mms.fit_transform(x.values.reshape(-1, 1))
        df[col] = x.reshape(-1).astype(np.float16)
        df[col] = df[col].fillna(0.0)
    return df

uid_ctr = normalize(uid_ctr, uid_ctr.columns[2:])
fid_ctr = normalize(fid_ctr, fid_ctr.columns[2:])
aid_ctr = normalize(aid_ctr, aid_ctr.columns[2:])
kw1_ctr = normalize(kw1_ctr, kw1_ctr.columns[2:])
tag1_ctr = normalize(tag1_ctr, tag1_ctr.columns[2:])
bgm_song_ctr = normalize(bgm_song_ctr, bgm_song_ctr.columns[2:])
bgm_singer_ctr = normalize(bgm_singer_ctr, bgm_singer_ctr.columns[2:])

print(uid_ctr.shape, fid_ctr.shape, aid_ctr.shape, kw1_ctr.shape, tag1_ctr.shape, 
      bgm_song_ctr.shape, bgm_singer_ctr.shape)



uid_stat = pd.read_pickle("../data/features/stat_feas/userid_1843213_16_5day_stat_fea.pkl")
fid_stat = pd.read_pickle("../data/features/stat_feas/feedid_736591_16_5day_stat_fea.pkl")
aid_stat = pd.read_pickle("../data/features/stat_feas/authorid_158013_16_5day_stat_fea.pkl")
kw1_stat = pd.read_pickle("../data/features/stat_feas/keyword1_24661_16_5day_stat_fea.pkl")
kw1_stat.loc[kw1_stat['keyword1'] == 0, kw1_stat.columns[2:]] = 0
tag1_stat = pd.read_pickle("../data/features/stat_feas/tag1_2494_16_5day_stat_fea.pkl")

bgm_song_stat = pd.read_pickle("../data/features/stat_feas/bgm_song_id_188669_16_5day_stat_fea.pkl")
bgm_song_stat.loc[bgm_song_stat['bgm_song_id'] == -1, bgm_song_stat.columns[2:]] = 0
bgm_singer_stat = pd.read_pickle("../data/features/stat_feas/bgm_singer_id_134256_16_5day_stat_fea.pkl")
bgm_singer_stat.loc[bgm_singer_stat['bgm_singer_id'] == -1, bgm_singer_stat.columns[2:]] = 0


def normalize(df, col):
    for col in col:
        x = df[col].astype(np.float32)
        x = np.log(x + 1.0)
        mms = MinMaxScaler()
        x = mms.fit_transform(x.values.reshape(-1, 1))
        df[col] = x.reshape(-1).astype(np.float16)
        df[col] = df[col].fillna(0.0)
    return df

uid_stat = normalize(uid_stat, uid_stat.columns[2:])
fid_stat = normalize(fid_stat, fid_stat.columns[2:])
aid_stat = normalize(aid_stat, aid_stat.columns[2:])
kw1_stat = normalize(kw1_stat, kw1_stat.columns[2:])
tag1_stat = normalize(tag1_stat, tag1_stat.columns[2:])
bgm_song_stat = normalize(bgm_song_stat, bgm_song_stat.columns[2:])
bgm_singer_stat = normalize(bgm_singer_stat, bgm_singer_stat.columns[2:])




userid_stat_fea = uid_ctr.merge(uid_stat, how='left', on=['userid', 'date_'])
userid_stat_fea = userid_stat_fea.merge(userid_global_fea, how='left', on=['userid'])

feedid_stat_fea = fid_ctr.merge(fid_stat, how='left', on=['feedid', 'date_'])
feedid_stat_fea = feedid_stat_fea.merge(feedid_global_fea, how='left', on=['feedid'])

authorid_stat_fea = aid_ctr.merge(aid_stat, how='left', on=['authorid', 'date_'])
authorid_stat_fea = authorid_stat_fea.merge(authorid_global_fea, how='left', on=['authorid'])

keyword1_stat_fea = kw1_ctr.merge(kw1_stat, how='left', on=['keyword1', 'date_'])
keyword1_stat_fea = keyword1_stat_fea.merge(keyword1_global_fea, how='left', on=['keyword1'])

tag1_stat_fea = tag1_ctr.merge(tag1_stat, how='left', on=['tag1', 'date_'])
tag1_stat_fea = tag1_stat_fea.merge(tag1_global_fea, how='left', on=['tag1'])

bgm_song_id_stat_fea = bgm_song_ctr.merge(bgm_song_stat, how='left', on=['bgm_song_id', 'date_'])
bgm_song_id_stat_fea = bgm_song_id_stat_fea.merge(bgm_song_id_global_fea, how='left', on=['bgm_song_id'])

bgm_singer_id_stat_fea = bgm_singer_ctr.merge(bgm_singer_stat, how='left', on=['bgm_singer_id', 'date_'])
bgm_singer_id_stat_fea = bgm_singer_id_stat_fea.merge(bgm_singer_id_global_fea, how='left', on=['bgm_singer_id'])

print(userid_stat_fea.shape, feedid_stat_fea.shape, authorid_stat_fea.shape, 
      keyword1_stat_fea.shape, tag1_stat_fea.shape, 
      bgm_song_id_stat_fea.shape, bgm_singer_id_stat_fea.shape)


def get_emb_hash(df):
    res = {}
    for line in tqdm(df.values):
        res[(int(line[0]), int(line[1]))] = line[2:].astype(np.float32)
    return res

userid_2_stat_fea = get_emb_hash(userid_stat_fea)
feedid_2_stat_fea = get_emb_hash(feedid_stat_fea)
authorid_2_stat_fea = get_emb_hash(authorid_stat_fea)
bgm_song_id_2_stat_fea = get_emb_hash(bgm_song_id_stat_fea)
bgm_singer_id_2_stat_fea = get_emb_hash(bgm_singer_id_stat_fea)
keyword1_2_stat_fea = get_emb_hash(keyword1_stat_fea)
tag1_2_stat_fea = get_emb_hash(tag1_stat_fea)

pickle.dump([userid_stat_fea, feedid_2_stat_fea, authorid_2_stat_fea, 
             bgm_song_id_2_stat_fea, bgm_singer_id_2_stat_fea, 
             keyword1_2_stat_fea, tag1_2_stat_fea], 
            open("../data/features/singer_col_stat_feas.pkl", 'wb'))


## 读取训练集
train = pd.read_pickle('../data/origin/user_action.pkl')
train = train[train['date_'] >= 6].reset_index(drop=True)
print(train.shape)
    
## 读取测试集
test = pd.read_pickle('../data/origin/test_a.pkl')
test['date_'] = 15
print(test.shape)

df = pd.concat([train, test], ignore_index=True)
print(df.shape)
df = reduce_mem(df, df.columns)

del train, test
gc.collect()


# feed侧信息
feed_info = pd.read_pickle("../data/features/feed_info.pkl")
feed_info.drop(columns=['all_keyword', 'all_tag'], inplace=True)
feed_info = reduce_mem(feed_info, feed_info.columns)
print(feed_info.shape)


df = df.merge(feed_info, how='left', on=['feedid'])
df.drop(columns=['play', 'stay'], inplace=True)

play_cols = ['is_finish', 'play_times', 'stay_times', 'play', 'stay']
y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
max_day = 15

train = df[(df['date_'] >= 6) & (df['date_'] <= 14)].reset_index(drop=True)
test = df[df['date_'] == 15].reset_index(drop=True)
print(train.shape, test.shape)

del df
gc.collect()

train.to_pickle("../data/features/train_v0.pkl")
test.to_pickle("../data/features/test_v0.pkl")






