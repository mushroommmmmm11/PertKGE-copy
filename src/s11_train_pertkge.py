"""
PertKGE - 基于知识图谱嵌入的化学扰动靶点推断模型
主要功能：使用五折交叉验证训练知识图谱嵌入模型，用于化学化合物与蛋白质靶点的关系预测

模型架构: 基于DistMult的知识图谱嵌入模型
训练策略: 五折交叉验证 + 早停机制
评估指标: MRR、Hits@K等链接预测指标
"""

"""
target_inference_1:
python train_pertkge.py --cause_file "../processed_data/target_inference_1/cause.txt"\
                          --process_file "../processed_data/knowledge_graph/process.txt"\
                          --effect_file "../processed_data/target_inference_1/effect.txt"\
                          --test_file "../processed_data/target_inference_1/test.txt"\
                          --h_dim 300\
                          --margin 1.0\
                          --lr 1e-4\
                          --wd 1e-5\
                          --n_neg 100\
                          --mode 'reproduce'\
                          --batch_size 2048\
                          --warm_up 10\
                          --patients 5\
                          --load_processed_data\
                          --processed_data_file "../processed_data/target_inference_1/"\
                          --save_model_path "../best_model/target_inference_1/"\
                          --task "target_inference"\
                          --run_name "target_inference_1"
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 指定使用GPU设备
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 不使用GPU设备

from time import time
import pandas as pd
import numpy as np
import tqdm
import random
from collections import defaultdict
import argparse

import torch
import torch.nn as nn
from torch import cuda
import torch_npu
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss,DataLoader
from torchkge import KnowledgeGraph,DistMultModel,TransEModel,TransHModel
from torchkge.models.bilinear import HolEModel,ComplExModel

from utils import *
from model import *


def parse_args(args=None):
    """
    解析命令行参数
    返回: args对象和参数字典
    """
    parser = argparse.ArgumentParser(
        description='PertKG',
        usage='main.py [<args>] [-h | --help]'
    )
    # 数据文件参数
    parser.add_argument('--cause_file',default="../processed_data/deepce/cause.txt",
                       help="因果关系文件，包含化合物与生物过程的关系")
    parser.add_argument('--process_file',default="../processed_data/knowledge_graph/process.txt",
                       help="生物过程文件，包含生物过程之间的关系")
    parser.add_argument('--effect_file', default="../processed_data/deepce/effect.txt",
                       help="效应文件，包含生物过程与靶点的关系")
    parser.add_argument('--test_file',default="../processed_data/deepce/test.txt",
                       help="测试集文件，用于最终评估模型性能")
    
    # 模型训练参数
    parser.add_argument('--seed', type = int, default=43,
                       help="随机种子，确保结果可复现")
    parser.add_argument('--h_dim', type = int, default=300,
                       help="嵌入维度，实体和关系的向量维度")
    parser.add_argument('--margin', type = float, default=1.0,
                       help="Margin loss的边界参数，控制正负样本的间隔")
    parser.add_argument('--lr', type = float, default=1e-4,
                       help="学习率")
    parser.add_argument('--wd', type = float, default=1e-5,
                       help="权重衰减")
    parser.add_argument('--n_neg', type = int, default=100,
                       help="每个正样本对应的负样本数量")
    parser.add_argument('--batch_size', type = int, default=2048,
                       help="批处理大小")
    parser.add_argument('--warm_up', type = int, default=10,
                       help="预热轮数, 前warm_up轮不进行早停判断")
    parser.add_argument('--patients', type = int, default=5,
                       help="早停耐心值，连续patients轮验证集性能不提升则停止训练")
    parser.add_argument('--use_cuda', type = str, default='batch',
                       help="CUDA使用模式，控制GPU内存使用策略, 'batch'或'none'")
    parser.add_argument('--nepoch', type = int, default=100,
                       help="最大训练轮数")
    
    # 模型保存和数据处理参数
    parser.add_argument('--save_model', action='store_true', default=True,
                       help="是否保存模型")
    parser.add_argument('--save_model_path',default="../best_model/deepce_distmult_2/",
                       help="模型保存路径")
    parser.add_argument('--load_processed_data', action='store_true', default=False,
                       help="是否加载预处理数据，避免重复处理")
    parser.add_argument('--processed_data_file',default="../processed_data/deepce/",
                       help="预处理数据文件路径")
    
    
    parser.add_argument('--mode', default="reproduce", help = 'choose reproduce if user want to report testing results')  # test or not
    parser.add_argument('--task', default="target_inference", help="choose from ['target_inference', 'virtual_screening', 'unbiased_test']")
    parser.add_argument('--run_name', default="deepce_distmult", help="Name of the running.")

    # +++++++++++++++++
    return parser.parse_args(args),parser.parse_args(args).__dict__


def five_fold_cv(args):
    """
    五折交叉验证训练流程
    
    主要步骤：
    1. 数据准备和分割：读取数据文件并生成五折分割
    2. 知识图谱构建：构建训练用的知识图谱
    3. 模型训练和验证：每折独立训练模型并验证
    4. 测试和结果统计：在测试集上评估并统计最终结果
    
    返回：无，直接输出训练和测试结果
    """
    
    # read cause, process, effect, test file
    # 读取数据文件：因果关系、生物过程、效应关系和测试数据
    # h_cand: 候选头实体，t_cand: 候选尾实体
    cause, pertkg_wo_cause, test, ent2id, rel2id, pro2nc, h_cand, t_cand = read_files(args)
    
    # generate train\valid
    # 生成五折交叉验证的数据分割
    five_fold_train, five_fold_valid = generate_five_fold_files(args, cause)

    # 存储每折的结果
    results = []
    # 五折交叉验证循环
    for i in range(5):
        # load data to consrtuct kg
        print('split_{}!!!'.format(i))
        
        # logger
        # 初始化TensorBoard日志记录器
        train_logger = SummaryWriter('../outlog/{}'.format(args.run_name,i))

        # loading train and valid df
        # 获取当前折的训练集和验证集
        train = five_fold_train[i]
        valid = five_fold_valid[i]

        # 构建化学扰动知识图谱
        print('construct chemical perturbation profiles-based knowledge graph!!!')
        s1 = time()
        # 合并不含因果关系的KG和当前折的训练数据
        df = pd.concat([pertkg_wo_cause,train])
        # 打乱知识图谱三元组顺序，避免训练偏差
        df = df.sample(frac=1,random_state=42).reset_index(drop=True) # 打乱KG
        # 构建知识图谱对象
        kg = KnowledgeGraph(df,ent2ix=ent2id,rel2ix=rel2id)
        e1 = time()
        print(f"Total constructing time: {round(e1 - s1, 2)}s")
        print()

        print('split_{} traing now!!!'.format(i))
        # choose method
        # 初始化模型、损失函数和优化器
        # 选择DistMult作为知识图谱嵌入模型
        model = DistMultModel(args.h_dim, len(ent2id), len(rel2id))
         # 使用Margin Loss作为损失函数
        criterion = MarginLoss(args.margin)
        # GPU配置
        # if cuda.is_available():
        #     cuda.empty_cache()
        #     model.cuda()
        #     criterion.cuda()
        # 设备配置
        model = model.to(device)
        criterion = criterion.to(device)
        
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        # 负采样器：使用伯努利负采样策略
        kgsampler = BernoulliNegativeSampler(kg,n_neg=args.n_neg)
        # 数据加载器：批量加载知识图谱三元组
        kgloader = DataLoader(kg, batch_size=args.batch_size, use_cuda=False) # 手动加载到对应device
        

        # wo train
        # 训练前测试：验证模型初始性能
        _ = tester('DistMult',model,
                    args,
                    test,
                    ent2id,rel2id,
                    h_cand,t_cand,
                    args.task)

        # train
        # 模型训练循环
        best_mrr = 0
        patients = 0
        for epoch in range(args.nepoch):
            running_loss = 0.0
            model.train()
            for batch in tqdm.tqdm(kgloader):
                h, t, r = batch[0], batch[1], batch[2]
                h, t, r = h.to(device), t.to(device), r.to(device)
                # 生成负样本：通过替换头实体或尾实体
                n_h, n_t = kgsampler.corrupt_batch(h, t, r)
                n_h, n_t = n_h.to(device), n_t.to(device)

                optimizer.zero_grad()

                # forward + backward + optimize
                # 前向传播：计算正样本和负样本的得分
                pos, neg = model(h, t, r, n_h, n_t)
                loss = criterion(pos, neg)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            train_loss = running_loss
            print(
            'Epoch {} | train loss: {:.5f}'.format(epoch + 1,
                                                train_loss))  
            
            # eval
            # 验证阶段: 无偏评估器计算验证集指标
            MR,MRR,Hit10,Hit30,Hit100 = unbiased_evaluator('DistMult',model,
                                                    valid,
                                                    ent2id,rel2id,
                                                    pro2nc)
            # 记录验证指标到TensorBoard
            train_logger.add_scalar("Hits@100", Hit100, epoch+1)

            print('Epoch {} | valid:'.format(epoch + 1))
            print('MR {} | MRR: {} | Hits@10:{} | Hits@30:{} | Hits@100: {}'.format(MR,
                                                                            MRR,
                                                                            Hit10,
                                                                            Hit30,
                                                                            Hit100))
            
            # TEST
            # _ = tester('DistMult',model,
            #             args,
            #             test,
            #             ent2id,rel2id,
            #             h_cand,t_cand,
            #             args.task)
                
            # 早停机制: 在预热期后开始判断
            if epoch > args.warm_up:
                if MRR > best_mrr:  # MRR is used as ER metrics # 使用MRR作为早停指标
                    best_mrr = MRR
                    patients = 0
                    if args.save_model:
                            torch.save(model.state_dict(), args.save_model_path + "pertkg{}.pt".format(i))

                else:
                    patients += 1

                if patients >= args.patients:
                    break

        train_logger.flush() # 确保所有日志写入文件

        # 测试阶段: 在reproduce模式下进行最终测试
        if args.mode == 'reproduce':
            # report test metrics according to task
            print('split_{} testing now!!!'.format(i))
            model.load_state_dict(torch.load(args.save_model_path + "pertkg{}.pt".format(i), map_location=device))
            # 在测试集上评估模型性能
            metrics = tester('DistMult',model,
                            args,
                            test,
                            ent2id,rel2id,
                            h_cand,t_cand,
                            args.task)
            results.append(metrics) # 存储当前折的结果
            print('_'*50)
    
    # 统计最终结果：计算五折的平均值和标准差
    if args.mode == 'reproduce':
        # report mean±std
        print('report mean±std testing results using 5 trained model!!!')
        if args.task == 'target_inference':
            # 针对目标推断任务的结果统计
            df = pd.DataFrame(results, columns=['Top-10', 'Recall@10', 'Top-30', 'Recall@30', 'Top-100', 'Recall@100'])
            print(df.describe())
        
        # elif args.task == 'virtual_screening':
        #     print('because ef is varied across different target, so we count metrics like unbiased_test here. using inference file for ef metrics.')
        #     df = pd.DataFrame(results, columns=['Hits@10', 'Hits@30', 'Hits@100'])
        #     print(df.describe())

        # elif args.task == 'unbiased_test':
        #     df = pd.DataFrame(results, columns=['Hits@10', 'Hits@30', 'Hits@50'])
        #     print(df.describe())

        else:
            print('no testing metrics because task is not defined, plz run inference.ipynb to reload best_model for specific testing!!!')

        print('_'*50)

if __name__ == '__main__':
    """
    主程序入口
    执行完整的模型训练和评估流程
    
    流程：
    1. 解析命令行参数
    2. 设置随机种子确保可复现性
    3. 创建必要的目录结构
    4. 执行五折交叉验证训练
    5. 输出总运行时间
    """
    s = time()

    print('print args_dict!!!')
    args, args_dict = parse_args()
    print(args_dict) # 打印参数配置
    print('_'*50)

    # device
    if args.use_cuda == 'none':
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    elif torch.npu.isavailable():
        device = torch.device('npu')
        torch.npu.empty_cache()
    else:
        device = torch.device('cpu')
    print(f"model will run on {device}")
    
    set_seeds(args.seed) # 设置随机种子确保结果可复现
    # save model
    if args.save_model:
        if not os.path.exists(args.save_model_path):
                os.makedirs(args.save_model_path)
    # log
    if args.run_name:
        if not os.path.exists('../outlog/{}/'.format(args.run_name)):
                os.makedirs('../outlog/{}/'.format(args.run_name))

    print('traing and testing using five-fold cross validation stategy!!!')
    print('_'*50)
    five_fold_cv(args)

    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
