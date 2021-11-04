# coding: utf-8
import time
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from numba import njit
from scipy.stats import rankdata
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer
from sklearn.decomposition import TruncatedSVD,SparsePCA
from sklearn.model_selection import KFold,StratifiedKFold
import scipy.sparse
from scipy import linalg
from scipy.special import iv
import scipy.sparse as sp
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import networkx as nx

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,recall_score
import gc, time, os, sys, argparse
import warnings
# 导入贝叶斯平滑类
import numpy
import random
import scipy.special as special
import math
from math import log
import logging

warnings.filterwarnings('ignore')


def get_logger(log_path):
    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 第二步，创建一个handler，用于写入日志文件
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.INFO)
    # 第三步，再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # 第四步，定义handler的输出格式（时间，文件，行数，错误级别，错误提示）
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", '%m/%d/%Y %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 第五步，将logger添加到handler里面
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def reduce_mem(df, cols):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in tqdm(cols):
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df


@njit
def _auc(actual, pred_ranks):
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(pred_ranks[actual == 1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)


def fast_auc(actual, predicted):
    # https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/208031
    pred_ranks = rankdata(predicted)
    return _auc(actual, pred_ranks)


def uAUC(labels, preds, user_id_list):
    """Calculate user AUC"""
    user_pred = defaultdict(lambda: [])
    user_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        pred = preds[idx]
        truth = labels[idx]
        user_pred[user_id].append(pred)
        user_truth[user_id].append(truth)

    user_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = user_truth[user_id]
        flag = False
        # 若全是正样本或全是负样本，则flag为False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        user_flag[user_id] = flag

    total_auc = 0.0
    size = 0.0
    for user_id in user_flag:
        if user_flag[user_id]:
            auc = fast_auc(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
            total_auc += auc 
            size += 1.0
    user_auc = float(total_auc)/size
    return round(user_auc, 6)




class ProNE():
    def __init__(self, G, emb_size=100, step=10, theta=0.5, mu=0.2, n_iter=5, random_state=2021):
        self.G = G
        self.emb_size = emb_size
        self.G = self.G.to_undirected()
        self.node_number = self.G.number_of_nodes()
        self.random_state = random_state
        self.step = step
        self.theta = theta
        self.mu = mu
        self.n_iter = n_iter
        
        mat = scipy.sparse.lil_matrix((self.node_number, self.node_number))

        for e in tqdm(self.G.edges()):
            if e[0] != e[1]:
                mat[int(e[0]), int(e[1])] = 1
                mat[int(e[1]), int(e[0])] = 1
        self.mat = scipy.sparse.csr_matrix(mat)
        print(mat.shape)
        
    def get_embedding_rand(self, matrix):
        # Sparse randomized tSVD for fast embedding
        t1 = time.time()
        l = matrix.shape[0]
        smat = scipy.sparse.csc_matrix(matrix)  # convert to sparse CSC format
        print('svd sparse', smat.data.shape[0] * 1.0 / l ** 2)
        U, Sigma, VT = randomized_svd(smat, n_components=self.emb_size, 
                                      n_iter=self.n_iter, 
                                      random_state=self.random_state)
        U = U * np.sqrt(Sigma)
        U = preprocessing.normalize(U, "l2")
        print('sparsesvd time', time.time() - t1)
        return U

    def get_embedding_dense(self, matrix, emb_size):
        # get dense embedding via SVD
        t1 = time.time()
        U, s, Vh = linalg.svd(matrix, full_matrices=False, 
                              check_finite=False, overwrite_a=True)
        U = np.array(U)
        U = U[:, :emb_size]
        s = s[:emb_size]
        s = np.sqrt(s)
        U = U * s
        U = preprocessing.normalize(U, "l2")
        print('densesvd time', time.time() - t1)
        return U
    
    def fit(self, tran, mask):
        # Network Embedding as Sparse Matrix Factorization
        t1 = time.time()
        l1 = 0.75
        C1 = preprocessing.normalize(tran, "l1")
        neg = np.array(C1.sum(axis=0))[0] ** l1

        neg = neg / neg.sum()

        neg = scipy.sparse.diags(neg, format="csr")
        neg = mask.dot(neg)
        print("neg", time.time() - t1)

        C1.data[C1.data <= 0] = 1
        neg.data[neg.data <= 0] = 1

        C1.data = np.log(C1.data)
        neg.data = np.log(neg.data)

        C1 -= neg
        F = C1
        features_matrix = self.get_embedding_rand(F)
        return features_matrix
    
    def chebyshev_gaussian(self, A, a, order=10, mu=0.5, s=0.5):
        # NE Enhancement via Spectral Propagation
        print('Chebyshev Series -----------------')
        t1 = time.time()

        if order == 1:
            return a

        A = sp.eye(self.node_number) + A
        DA = preprocessing.normalize(A, norm='l1')
        L = sp.eye(self.node_number) - DA

        M = L - mu * sp.eye(self.node_number)

        Lx0 = a
        Lx1 = M.dot(a)
        Lx1 = 0.5 * M.dot(Lx1) - a

        conv = iv(0, s) * Lx0
        conv -= 2 * iv(1, s) * Lx1
        for i in range(2, order):
            Lx2 = M.dot(Lx1)
            Lx2 = (M.dot(Lx2) - 2 * Lx1) - Lx0
            #         Lx2 = 2*L.dot(Lx1) - Lx0
            if i % 2 == 0:
                conv += 2 * iv(i, s) * Lx2
            else:
                conv -= 2 * iv(i, s) * Lx2
            Lx0 = Lx1
            Lx1 = Lx2
            del Lx2
            print('Bessell time', i, time.time() - t1)
        mm = A.dot(a - conv)
        self.embeddings = self.get_embedding_dense(mm, self.emb_size)
        return self.embeddings
    
    def transform(self):
        if self.embeddings is None:
            print("Embedding is not train")
            return {}
        self.embeddings = pd.DataFrame(self.embeddings)
        self.embeddings.columns = ['ProNE_Emb_{}'.format(i) for i in range(len(self.embeddings.columns))]
        self.embeddings = self.embeddings.reset_index().rename(columns={'index' : 'nodes'}).sort_values(by=['nodes'],ascending=True).reset_index(drop=True)
        return self.embeddings
    
    
    


class HyperParam(object):
    def __init__(self, alpha, beta): # 先初始化alpha和beta
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        sample = numpy.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            #imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return I, C

    # 更新方式1
    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        '''estimate alpha, beta using fixed point iteration(类似EM估计)
        tries: 展示次数
        success: 点击次数
        iter_num: 迭代次数
        epsilon: 精度
        '''
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            # 当迭代稳定时，停止迭代
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        '''fixed point iteration'''
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        for i in range(len(tries)):
            sumfenzialpha += (special.digamma(success[i]+alpha) - special.digamma(alpha))
            sumfenzibeta += (special.digamma(tries[i]-success[i]+beta) - special.digamma(beta))
            sumfenmu += (special.digamma(tries[i]+alpha+beta) - special.digamma(alpha+beta))

        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)

    # 更新方式1
    def update_from_data_by_moment(self, tries, success):
        '''estimate alpha, beta using moment estimation(矩估计)
        tries: 展示次数
        success: 点击次数
        '''
        # 样本均值和样本方差
        mean, var = self.__compute_moment(tries, success)
        #print 'mean and variance: ', mean, var
        #self.alpha = mean*(mean*(1-mean)/(var+0.000001)-1)
        self.alpha = (mean+0.000001) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)
        #self.beta = (1-mean)*(mean*(1-mean)/(var+0.000001)-1)
        self.beta = (1.000001 - mean) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)

    def __compute_moment(self, tries, success):
        '''moment estimation(求样本均值和样本方差)'''
        ctr_list = []
        var = 0.0
        for i in range(len(tries)):
            ctr_list.append(float(success[i])/(tries[i] + 0.000000001))
        mean = sum(ctr_list)/len(ctr_list)
        for ctr in ctr_list:
            var += pow(ctr-mean, 2)

        return mean, var/(len(ctr_list)-1)