# **2021中国高校计算机大赛-微信大数据挑战赛Baseline**

本次比赛基于脱敏和采样后的数据信息，对于给定的一定数量到访过微信视频号“热门推荐”的用户，根据这些用户在视频号内的历史n天的行为数据，通过算法在测试集上预测出这些用户对于不同视频内容的互动行为（包括点赞、点击头像、收藏、转发等）的发生概率。 

本次比赛以多个行为预测结果的加权uAUC值进行评分。大赛官方网站：https://algo.weixin.qq.com/

## **1. 环境依赖**
- deepctr-torch==0.2.7
- gensim==4.0.1
- lightgbm==3.2.1
- networkx==2.5.1
- numba==0.53.1
- numpy==1.19.2
- pandas==1.1.0
- scipy==1.5.3

## **2. 目录结构**

```
./
├── README.md
├── requirements.txt, python package requirements 
├── init.sh, script for installing package requirements
├── train.sh, script for preparing train/inference data and training models, including pretrained models
├── inference.sh, script for inference 
├── src
│   ├── prepare, codes for preparing train/inference dataset
|       ├──fea.py  
|       ├──utils.py  
│   ├── model, codes for model architecture
|       ├──mmoe.py  
|   ├── train, codes for training
|       ├──train.py  
|       ├──utils.py 
|   ├── inference.py, main function for inference on test dataset
|   ├── evaluation.py, main function for evaluation 
├── data
│   ├── wedata, dataset of the competition
│       ├── wechat_algo_data1, preliminary dataset
│   ├── submission, prediction result after running inference.sh
│   ├── model, model files (e.g. tensorflow checkpoints)
```

## **3. 运行流程**
- 安装环境：sh init.sh (init.sh为空，直接使用envs里面的环境运行)
- 进入目录：cd /home/tione/notebook/wbdc2021-semi
- 数据准备和模型训练：sh train.sh
- 预测并生成结果文件：sh inference.sh /home/tione/notebook/wbdc2021/data/wedata/wechat_algo_data2/test_b.csv

## **4. 模型及特征**
- 模型：[MMOE](https://dl.acm.org/doi/10.1145/3219819.3220007)
- 参数：
    - batch_size: 20480
    - emded_dim: 48
    - num_epochs: 20
    - learning_rate: 0.001
- 特征：
    - sparse特征：userid, feedid, device, authorid, bgm_song_id, bgm_singer_id, keyword1, tag1
    - dense特征：ctr特征、统计特征、embedding特征（word2vec, proNE图特征, 多模态降维特征）
    
## **5. 算法性能**
- 资源配置：2*P40_48G显存_14核CPU_112G内存
- 预测耗时  
    - 总预测时长: 1788 s
    - 单个目标行为2000条样本的平均预测时长: 121 ms


## **6. 代码说明**
模型预测部分代码位置如下：

| 路径 | 行数 | 内容 |
| :--- | :--- | :--- |
| src/inference.py | 226 | `test_preds = predict(model, test_loader, device)`|

## **7. 相关文献**
* Ma J , Zhe Z , Yi X , et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts. ACM, 2018.