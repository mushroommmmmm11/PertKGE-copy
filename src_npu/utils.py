import os
from time import time
import pandas as pd
import numpy as np
import tqdm
import random
from collections import defaultdict
from scipy.stats import rankdata

import torch
from torch.nn.functional import normalize
from torch import nn, cat

global device

# general function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_rank(a):
    a = len(a) + 1 - rankdata(a)

    return a.tolist()



def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    elif device.type == 'npu':
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

# get ent2id,rel2id
def get_dictionaries(df):
    """Build entities or relations dictionaries.
    Parameters
    ----------
    df: `pandas.DataFrame`
        Data frame containing three columns [from, to, rel].
    ent: bool
        if True then ent2ix is returned, if False then rel2ix is returned.
    Returns
    -------
    dict: dictionary
        Either ent2ix or rel2ix.
    """
    tmp1 = list(set(df['from'].unique()).union(set(df['to'].unique())))
    tmp2 = list(df['rel'].unique())
    return {ent: i for i, ent in enumerate(sorted(tmp1))},{rel: i for i, rel in enumerate(sorted(tmp2))}

def get_cpi(df):
    # get postive dict
    c2p_dict = defaultdict(set)
    p2c_dict = defaultdict(set)
    for i in range(len(df)):
        c = df.iloc[i]['from']
        p = df.iloc[i]['to']
        c2p_dict[c].add(p)
        p2c_dict[p].add(c)
    
    return c2p_dict,p2c_dict

def split_into_five_sets(input_set):
    subsets = [[] for _ in range(5)]
    
    index = 0
    for element in input_set:
        subsets[index].append(element)
        index = (index + 1) % 5
    
    return subsets


def read_files(args):
    print('read input files!!!')
    s = time()

    # read files
    cause = pd.read_csv(args.cause_file,sep='\t',names=['from','rel','to']).drop_duplicates()
    effect = pd.read_csv(args.effect_file,sep='\t',names=['from','rel','to']).drop_duplicates()
    test = pd.read_csv(args.test_file,sep='\t',names=['from','rel','to']).drop_duplicates()

    # 检查process文件是否包含置信度列
    process_df = pd.read_csv(args.process_file, sep='\t', header=None)
    has_confidence = process_df.shape[1] == 4
    
    if has_confidence:
        # 如果process文件包含置信度，使用4列格式
        process = pd.read_csv(args.process_file, sep='\t', names=['from','rel','to','confidence']).drop_duplicates()
        print(f"检测到置信度数据，process文件包含{len(process)}个带置信度的三元组")
    else:
        # 否则使用3列格式
        process = pd.read_csv(args.process_file, sep='\t', names=['from','rel','to']).drop_duplicates()
        print(f"未检测到置信度数据，process文件包含{len(process)}个三元组")

    # generate processed data
    if args.load_processed_data:    
        ent2id = np.load(args.processed_data_file + 'ent2id.npy', allow_pickle=True).item()
        rel2id = np.load(args.processed_data_file + 'rel2id.npy', allow_pickle=True).item()
        pro2nc = np.load(args.processed_data_file + 'pro2nc.npy', allow_pickle=True).item()
        if has_confidence:
            confidence_mapping = np.load(args.processed_data_file + 'confidence_mapping.npy', allow_pickle=True).item()
        else:
            confidence_mapping = None
    else:
        # 1.ent2id, rel2id
        if has_confidence:
            # 对于带置信度的数据，使用专门的字典生成函数
            pertkg = pd.concat([cause, process[['from','rel','to']], effect])
            ent2id, rel2id = get_dictionaries(pertkg)
            # 创建置信度映射
            confidence_mapping = {}
            for _, row in process.iterrows():
                # 将基于名称的映射转换为基于索引的映射，提高训练时查找效率
                triple_key = (ent2id.get(row['from']), rel2id.get(row['rel']), ent2id.get(row['to']))
                confidence_mapping[triple_key] = row['confidence']
        else:
            pertkg = pd.concat([cause, process, effect])
            ent2id, rel2id = get_dictionaries(pertkg)
            confidence_mapping = None
            
        np.save(args.processed_data_file + 'ent2id.npy', ent2id)
        np.save(args.processed_data_file + 'rel2id.npy', rel2id)
        if confidence_mapping is not None:
            np.save(args.processed_data_file + 'confidence_mapping.npy', confidence_mapping)
        
        # 2.pro2nc
        comp_cand = {k for k,v in ent2id.items() if k.startswith('CID:')}
        _,p2c_dict = get_cpi(cause)
        pro = set(cause['to'])
        pro2nc = {}
        for x in pro:
            pro2nc[x] = random.sample(comp_cand - p2c_dict[x],k=3000)  # 3000 is trade of accuracy and evaluate speed
        np.save(args.processed_data_file + 'pro2nc.npy',pro2nc)

    pertkg_wo_cause = pd.concat([process[['from','rel','to']], effect])

    h_cand = [v for k,v in ent2id.items() if k.startswith('CID:')]
    t_cand = [v for k,v in ent2id.items() if k.startswith(('Protein:','TF:','RBP:'))]
    print('total {} compound and {} targets'.format(len(h_cand),len(t_cand)))

    e = time()
    print(f"reading time: {round(e - s, 2)}s")
    print('_'*50)

    return cause, pertkg_wo_cause, test, ent2id, rel2id, pro2nc, h_cand, t_cand, confidence_mapping


def get_confidence_weights_for_batch(heads, tails, relations, confidence_mapping, default_confidence=0.5):
    """
    为批次中的每个三元组获取置信度权重
    
    Args:
        heads (torch.Tensor): 头实体索引，shape: (batch_size)
        tails (torch.Tensor): 尾实体索引，shape: (batch_size)
        relations (torch.Tensor): 关系索引，shape: (batch_size)
        confidence_mapping (dict): 置信度映射字典，key为(head_idx, rel_idx, tail_idx)
        default_confidence (float): 默认置信度
        
    Returns:
        torch.Tensor: 置信度权重，shape: (batch_size)
    """
    batch_size = heads.size(0)
    device = heads.device
    
    if confidence_mapping is None:
        return torch.full((batch_size,), default_confidence, dtype=torch.float32, device=device)
    
    # 将索引转为CPU进行查表操作
    heads_cpu = heads.cpu().numpy()
    tails_cpu = tails.cpu().numpy()
    relations_cpu = relations.cpu().numpy()
    
    # 构建权重列表
    weights = []
    for i in range(batch_size):
        h_idx = int(heads_cpu[i])
        t_idx = int(tails_cpu[i])
        r_idx = int(relations_cpu[i])
        
        triple_key = (h_idx, r_idx, t_idx)
        confidence = confidence_mapping.get(triple_key, default_confidence)
        weights.append(confidence)
    
    return torch.tensor(weights, dtype=torch.float32, device=device)


class WeightedMarginLoss(nn.Module):
    """
    带置信度权重的合页损失函数
    
    损失计算公式：Loss_total = sum(s_i * max(0, margin - pos_score_i + neg_score_i))
    其中s_i是第i个三元组的置信度权重
    """
    
    def __init__(self, margin=1.0):
        """
        初始化加权合页损失
        
        Args:
            margin (float): 合页损失的边界值
        """
        super(WeightedMarginLoss, self).__init__()
        self.margin = margin
    
    def forward(self, pos_scores, neg_scores, confidence_weights=None):
        """
        计算加权合页损失
        
        Args:
            pos_scores (torch.Tensor): 正样本得分
                - 如果没有负样本: shape (batch_size,)
                - 如果有n_neg个负样本: shape (batch_size * n_neg,)，已repeat
            neg_scores (torch.Tensor): 负样本得分，shape: (batch_size * n_neg)
            confidence_weights (torch.Tensor, optional): 置信度权重，shape: (batch_size)
            
        Returns:
            torch.Tensor: 加权损失值
        """
        # 计算基础合页损失
        loss = torch.clamp(self.margin - pos_scores + neg_scores, min=0)
        
        # 如果提供了置信度权重，则只对正样本应用权重
        if confidence_weights is not None:
            # 确保权重在合理范围内
            confidence_weights = torch.clamp(confidence_weights, min=0.0, max=1.0)
            
            # pos_scores可能已经是(batch_size * n_neg)形状（被repeat过）
            # 也可能是(batch_size,)形状
            # neg_scores总是(batch_size * n_neg)
            n_repeat = neg_scores.size(0) // confidence_weights.size(0)
            expanded_weights = confidence_weights.repeat_interleave(n_repeat)
            
            weighted_loss = loss * expanded_weights
        else:
            # 如果没有置信度权重，使用默认权重1.0
            weighted_loss = loss
        
        return weighted_loss.mean()


def generate_five_fold_files(args, cause):
    print('generate five fold files for training and evaluating!!!')
    s = time()
    if args.load_processed_data:
        five_fold_train = []
        five_fold_valid = []
        for i in range(5):
            train = pd.read_csv(args.processed_data_file + 'train{}.txt'.format(i),sep='\t',names=['from','rel','to'])
            valid = pd.read_csv(args.processed_data_file + 'valid{}.txt'.format(i),sep='\t',names=['from','rel','to'])
            five_fold_train.append(train)
            five_fold_valid.append(valid)
    else:
        cause_comp = set(cause['from'])
        five_cause_comp_set = split_into_five_sets(cause_comp)
        
        five_fold_train = []
        five_fold_valid = []
        for i in range(5):
            train = cause[~cause['from'].isin(five_cause_comp_set[i])]
            valid = cause[cause['from'].isin(five_cause_comp_set[i])]
            five_fold_train.append(train)
            five_fold_valid.append(valid)
            # and save
            train.to_csv(args.processed_data_file + 'train{}.txt'.format(i),sep='\t',index=False,header=False)
            valid.to_csv(args.processed_data_file + 'valid{}.txt'.format(i),sep='\t',index=False,header=False)

    e = time()
    print(f"generating time: {round(e - s, 2)}s")
    print('_'*50)

    return five_fold_train, five_fold_valid


# metirc 
def count_mean_std(my_list):
    mean_value = np.mean(my_list)
    std_value = np.std(my_list)

    print("Mean:", mean_value)
    print("Standard Deviation:", std_value)

def count_metrics(df, k=10,metrics='top'):
    if metrics == 'top':
        comp = set(df['from'])
        c_top = []
        for x in comp:
            sdf = df[df['from'] == x]
            ssdf = sdf[sdf['rank']<= k]
            if len(ssdf)> 0 :
                c_top.append(1)

        return  len(c_top)/len(comp)

    elif metrics == 'recall':
        comp = set(df['from'])
        c_recall = []
        for x in comp:
            sdf = df[df['from'] == x]
            ssdf = sdf[sdf['rank']<= k]
            c_recall.append(len(ssdf)/len(sdf))

        return  np.mean(c_recall)

    elif metrics == 'virtual_screening':
        # TODO:add implement
        pass
    
    elif metrics == 'hit':
        ranks = df['rank']
        hit = sum(1 for x in ranks if x<= k) / len(ranks)

        return hit
    

def unbiased_evaluator(model_name,model,
                        df,
                        ent2id, rel2id, 
                        # ent_emb, rel_emb,
                        pro2nc):
    '''used for unbiased early stopping,
        hits@n was used as metircs
    '''


    if model_name == 'DistMult':
        model.normalize_parameters()
        model.eval()
        with torch.no_grad():
            ent_emb,rel_emb = model.get_embeddings() # (n_ent, emb_dim)

            h_ranks = []
            r_emb = rel_emb[rel2id['HAS_BINDING_TO']]
            for i in tqdm.tqdm(range(len(df))):
                c = df.iloc[i]['from']
                p = df.iloc[i]['to']
                
                #
                h_emb = ent_emb[ent2id[c]]
                t_emb = ent_emb[ent2id[p]]
                score = (h_emb * r_emb * t_emb).sum()
                
                # replace head
                h_cand = pro2nc[p]
                h_cand = [ent2id[x] for x in h_cand]
                h_cand_emb = ent_emb[h_cand]  # (3000,dim)
                h_score = (h_cand_emb * r_emb * t_emb).sum(dim=1)

                rank = int(sum(h_score >= score).cpu()) + 1
                h_ranks.append(rank)

    elif model_name == 'TransE':
        model.normalize_parameters()
        model.eval()
        with torch.no_grad():
            ent_emb,rel_emb = model.get_embeddings() # (n_ent, emb_dim)

            h_ranks = []
            r_emb = rel_emb[rel2id['HAS_BINDING_TO']]
            for i in tqdm.tqdm(range(len(df))):
                c = df.iloc[i]['from']
                p = df.iloc[i]['to']

                h_emb = ent_emb[ent2id[c]]
                t_emb = ent_emb[ent2id[p]]
                score = -model.dissimilarity(h_emb.reshape(1,-1) + r_emb.reshape(1,-1), t_emb.reshape(1,-1))

                # replace head
                h_cand = pro2nc[p]
                h_cand = [ent2id[x] for x in h_cand]
                h_cand_emb = ent_emb[h_cand]  # (3000,dim)
                n_decoys = len(h_cand)

                h_score = - model.dissimilarity(h_cand_emb + r_emb.repeat(n_decoys,1), t_emb.repeat(n_decoys,1))

                rank = int(sum(h_score >= score).cpu()) + 1
                h_ranks.append(rank)

    elif model_name == 'TransH':
        model.normalize_parameters()
        model.eval()
        with torch.no_grad():
            ent_emb = normalize(model.ent_emb.weight.data,p=2, dim=1)
            rel_emb = model.rel_emb.weight.data
            norm_vect = normalize(model.norm_vect.weight.data,p=2, dim=1)

            h_ranks = []
            r_emb = rel_emb[rel2id['HAS_BINDING_TO']]
            r_norm = norm_vect[rel2id['HAS_BINDING_TO']]
            for i in tqdm.tqdm(range(len(df))):
                c = df.iloc[i]['from']
                p = df.iloc[i]['to']
                h_emb = ent_emb[ent2id[c]]
                t_emb = ent_emb[ent2id[p]]

                score = - model.dissimilarity(h_emb - (h_emb * r_norm).sum() * r_norm + r_emb, 
                                    t_emb - (t_emb * r_norm).sum() * r_norm)

                # replace head
                h_cand = pro2nc[p]
                h_cand = [ent2id[x] for x in h_cand]
                h_cand_emb = ent_emb[h_cand]  # (3000,dim)
                n_decoys = len(h_cand)

                h_score = - model.dissimilarity(h_cand_emb - (h_cand_emb * r_norm).sum(dim=1).view(-1, 1) * r_norm + r_emb, 
                                    (t_emb - (t_emb * r_norm).sum() * r_norm).repeat(n_decoys,1))  

                rank = int(sum(h_score >= score).cpu()) + 1
                h_ranks.append(rank)

    elif model_name == 'ComplEx':
        model.normalize_parameters()
        model.eval()
        with torch.no_grad():
            re_ent_emb, im_ent_emb, re_rel_emb, im_rel_emb = model.get_embeddings()

            h_ranks = []
            r_re_emb = re_rel_emb[rel2id['HAS_BINDING_TO']]
            r_im_emb = im_rel_emb[rel2id['HAS_BINDING_TO']]
            for i in tqdm.tqdm(range(len(df))):
                c = df.iloc[i]['from']
                p = df.iloc[i]['to']
                h_re_emb = re_ent_emb[ent2id[c]] 
                h_im_emb = im_ent_emb[ent2id[c]]
                t_re_emb = re_ent_emb[ent2id[p]]  
                t_im_emb = im_ent_emb[ent2id[p]]

                score = (h_re_emb * (r_re_emb * t_re_emb + r_im_emb * t_im_emb) + h_im_emb * (r_re_emb * t_im_emb - r_im_emb * t_re_emb)).sum()

                # replace head
                h_cand = pro2nc[p]
                h_cand = [ent2id[x] for x in h_cand]
                h_cand_re_emb = re_ent_emb[h_cand]
                h_cand_im_emb = im_ent_emb[h_cand]
                n_decoys = len(h_cand)

                h_score = (h_cand_re_emb * (r_re_emb * t_re_emb + r_im_emb * t_im_emb) + 
                    h_cand_im_emb * (r_re_emb * t_im_emb - r_im_emb * t_re_emb)).sum(dim=1)

                rank = int(sum(h_score >= score).cpu()) + 1
                h_ranks.append(rank)

    elif model_name == 'ConvKB':
        model.normalize_parameters()
        model.eval()
        with torch.no_grad():
            ent_emb,rel_emb = model.get_embeddings() # (n_ent, emb_dim)

            h_ranks = []
            for i in tqdm.tqdm(range(len(df))):
                c = df.iloc[i]['from']
                p = df.iloc[i]['to']
                
                #
                h_emb = ent_emb[ent2id[c]].view(1,1,-1)
                r_emb = rel_emb[rel2id['HAS_BINDING_TO']].view(1,1,-1)
                t_emb = ent_emb[ent2id[p]].view(1,1,-1)
                concat = cat((h_emb, r_emb, t_emb), dim=1)
                score = model.output(model.convlayer(concat).reshape(1, -1))[:, 1]
                
                # replace head
                h_cand = pro2nc[p]
                h_cand = [ent2id[x] for x in h_cand]
                n_decoys = len(h_cand)

                h_cand_emb = ent_emb[h_cand].view(n_decoys,1,-1)
                r_emb = rel_emb[rel2id['HAS_BINDING_TO']].view(1,1,-1).repeat(n_decoys,1,1)
                t_emb = ent_emb[ent2id[p]].view(1,1,-1).repeat(n_decoys,1,1)
                concat = cat((h_cand_emb, r_emb, t_emb), dim=1)
                h_score = model.output(model.convlayer(concat).reshape(n_decoys, -1))[:, 1]

                rank = int(sum(h_score >= score).cpu()) + 1
                h_ranks.append(rank)

    else:
        model.normalize_parameters()
        model.eval()
        with torch.no_grad():

            h_ranks = []
            for i in tqdm.tqdm(range(len(df))):
                c = torch.LongTensor([ent2id[df.iloc[i]['from']]]).to(device)
                r = torch.LongTensor([rel2id[df.iloc[i]['rel']]]).to(device)
                p = torch.LongTensor([ent2id[df.iloc[i]['to']]]).to(device)

                score = model.scoring_function(c,p,r)

                # replace head
                h_cand = pro2nc[df.iloc[i]['to']]
                h_cand = [ent2id[x] for x in h_cand]

                n_decoys = len(h_cand)

                h_score = model.scoring_function(torch.LongTensor(h_cand).to(device),p.repeat(n_decoys),r.repeat(n_decoys))

                rank = int(sum(h_score >= score).cpu()) + 1
                h_ranks.append(rank)

    h_MR = np.mean(h_ranks)
    h_MRR = sum([1 / x for x in h_ranks]) / len(h_ranks)
    h_Hit10 = sum(1 for x in h_ranks if x<= 10) / len(h_ranks)
    h_Hit30 = sum(1 for x in h_ranks if x<= 30) / len(h_ranks)
    h_Hit100 = sum(1 for x in h_ranks if x<= 100) / len(h_ranks)


    return h_MR,h_MRR,h_Hit10,h_Hit30,h_Hit100

def tester(model_name,model,
           args,
           df,
           ent2id, rel2id, 
        #    ent_emb, rel_emb,
           h_cand, t_cand,
           task = 'target_inference',
           ):
    '''
    target inference
    '''

    if model_name == 'DistMult':
        model.normalize_parameters()
        model.eval()
        with torch.no_grad():
            ent_emb,rel_emb = model.get_embeddings() # (n_ent, emb_dim)

            metrics = []
            ranks = []
            r_emb = rel_emb[rel2id['HAS_BINDING_TO']]
            if task == 'target_inference':
                for i in tqdm.tqdm(range(len(df))):
                    c = df.iloc[i]['from']
                    p = df.iloc[i]['to']
                    
                    # to avoid ent not existing in kg
                    try:
                        h_emb = ent_emb[ent2id[c]]
                        t_emb = ent_emb[ent2id[p]]
                    except KeyError:
                        ranks.append(len(t_cand))
                        continue

                    score = (h_emb * r_emb * t_emb).sum()
                    
                    # replace head
                    t_cand_emb = ent_emb[t_cand]  # (2000,dim)
                    t_score = (h_emb * r_emb * t_cand_emb).sum(dim=1)

                    r = int(sum(t_score >= score).cpu())
                    ranks.append(r)
                
                ranks = [1 if x == 0 else x for x in ranks]  # ensure no zero rank 
                df['rank'] = ranks

                for k in [10,30,100]:
                    top = count_metrics(df, k, 'top')
                    recall = count_metrics(df, k, 'recall')
                    metrics.append(top)
                    metrics.append(recall)

                print('Top-10:{} | Top-30:{} | Top-100:{} | Recall@10:{} | Recall@30:{} | Recall@100:{}'.format(metrics[0],
                                                                                                                metrics[2],
                                                                                                                metrics[4],
                                                                                                                metrics[1],
                                                                                                                metrics[3],
                                                                                                                metrics[5]))
                
            elif task == 'virtual_screening':
                '''because ef is varied across different target,
                so we count metrics like unbiased_test here.
                '''
                # FIXME: to do VS intead of TI
                for i in tqdm.tqdm(range(len(df))):
                    c = df.iloc[i]['from']
                    p = df.iloc[i]['to']
                    
                    # to avoid ent not existing in kg
                    try:
                        h_emb = ent_emb[ent2id[c]]
                        t_emb = ent_emb[ent2id[p]]
                    except KeyError:
                        ranks.append(len(t_cand))
                        continue

                    score = (h_emb * r_emb * t_emb).sum()
                    
                    # replace head
                    t_cand_emb = ent_emb[t_cand]
                    t_score = (h_emb * r_emb * t_cand_emb).sum(dim=1)

                    r = int(sum(t_score >= score).cpu())
                    ranks.append(r)
                
                ranks = [1 if x == 0 else x for x in ranks]  # ensure no zero rank 
                df['rank'] = ranks

                for k in [10,30,100]:
                    hit = count_metrics(df, k, 'hit')
                    metrics.append(hit)

                print('Hits@10:{} | Hits@30:{} | Hits@100:{}'.format(metrics[0], 
                                                                    metrics[1],
                                                                    metrics[2]))
            
            elif task == 'unbiased_test':
                '''
                pos:neg = 1:1000
                '''

                decoys_dict = np.load(args.processed_data_file + 'decoys_pro_wocpi.npy', allow_pickle = True).item()

                for i in tqdm.tqdm(range(len(df))):
                    c = df.iloc[i]['from']
                    p = df.iloc[i]['to']
                    
                    # to avoid ent not existing in kg
                    try:
                        h_emb = ent_emb[ent2id[c]]
                        t_emb = ent_emb[ent2id[p]]
                    except KeyError:
                        ranks.append(len(t_cand))
                        continue

                    score = (h_emb * r_emb * t_emb).sum()
                    
                    # replace head
                    decoys_id = [ent2id[x] for x in decoys_dict[p]]
                    h_cand_emb = ent_emb[decoys_id]
                    h_score = (h_cand_emb * r_emb * t_emb).sum(dim=1)

                    r = int(sum(h_score >= score).cpu())
                    ranks.append(r)
                
                ranks = [1 if x == 0 else x for x in ranks]  # ensure no zero rank 
                df['rank'] = ranks

                for k in [10,30,50]:
                    hit = count_metrics(df, k, 'hit')
                    metrics.append(hit)

                print('Hits@10:{} | Hits@30:{} | Hits@50:{}'.format(metrics[0], 
                                                                    metrics[1],
                                                                    metrics[2]))

            return metrics

    elif model_name == 'TransE':
        model.normalize_parameters()
        model.eval()
        with torch.no_grad():
            ent_emb,rel_emb = model.get_embeddings() # (n_ent, emb_dim)

            metrics = []
            ranks = []
            r_emb = rel_emb[rel2id['HAS_BINDING_TO']]
            if task == 'target_inference':
                for i in tqdm.tqdm(range(len(df))):
                    c = df.iloc[i]['from']
                    p = df.iloc[i]['to']
                    
                    # to avoid ent not existing in kg
                    try:
                        h_emb = ent_emb[ent2id[c]]
                        t_emb = ent_emb[ent2id[p]]
                    except KeyError:
                        ranks.append(len(t_cand))
                        continue

                    score = - model.dissimilarity(h_emb.reshape(1,-1) + r_emb.reshape(1,-1), t_emb.reshape(1,-1))
                    # print(score)
                    # print(score.shape)
                    # replace head
                    t_cand_emb = ent_emb[t_cand]
                    n_decoys = len(t_cand)
                    
                    t_score = - model.dissimilarity(h_emb.repeat(n_decoys,1) + r_emb.repeat(n_decoys,1), t_cand_emb)
                    # print(t_score.shape)

                    r = int(sum(t_score >= score).cpu())
                    # print(r)
                    ranks.append(r)
                
                ranks = [1 if x == 0 else x for x in ranks]  # ensure no zero rank 
                df['rank'] = ranks

                for k in [10,30,100]:
                    top = count_metrics(df, k, 'top')
                    recall = count_metrics(df, k, 'recall')
                    metrics.append(top)
                    metrics.append(recall)

                print('Top-10:{} | Top-30:{} | Top-100:{} | Recall@10:{} | Recall@30:{} | Recall@100:{}'.format(metrics[0],
                                                                                                                metrics[2],
                                                                                                                metrics[4],
                                                                                                                metrics[1],
                                                                                                                metrics[3],
                                                                                                                metrics[5]))

            return metrics

    elif model_name == 'TransH':
        model.normalize_parameters()
        model.eval()
        with torch.no_grad():
            ent_emb = normalize(model.ent_emb.weight.data,p=2, dim=1)
            rel_emb = model.rel_emb.weight.data
            norm_vect = normalize(model.norm_vect.weight.data,p=2, dim=1)

            metrics = []
            ranks = []
            r_emb = rel_emb[rel2id['HAS_BINDING_TO']]
            r_norm = norm_vect[rel2id['HAS_BINDING_TO']]
            if task == 'target_inference':
                for i in tqdm.tqdm(range(len(df))):
                    c = df.iloc[i]['from']
                    p = df.iloc[i]['to']
                    
                    # to avoid ent not existing in kg
                    try:
                        h_emb = ent_emb[ent2id[c]]
                        t_emb = ent_emb[ent2id[p]]
                    except KeyError:
                        ranks.append(len(t_cand))
                        continue

                    score = - model.dissimilarity(h_emb - (h_emb * r_norm).sum() * r_norm + r_emb, 
                                    t_emb - (t_emb * r_norm).sum() * r_norm)
                    # print(score)
                    # print(score.shape)
                    # replace head
                    t_cand_emb = ent_emb[t_cand]
                    n_decoys = len(t_cand)
                    
                    t_score = - model.dissimilarity((h_emb - (h_emb * r_norm).sum() * r_norm + r_emb).repeat(n_decoys,1), 
                                    t_cand_emb - (t_cand_emb * r_norm).sum(dim=1).view(-1,1) * r_norm)  # 应该能自动广播的

                    r = int(sum(t_score >= score).cpu())
                    # print(r)
                    ranks.append(r)
                
                ranks = [1 if x == 0 else x for x in ranks]  # ensure no zero rank 
                df['rank'] = ranks

                for k in [10,30,100]:
                    top = count_metrics(df, k, 'top')
                    recall = count_metrics(df, k, 'recall')
                    metrics.append(top)
                    metrics.append(recall)

                print('Top-10:{} | Top-30:{} | Top-100:{} | Recall@10:{} | Recall@30:{} | Recall@100:{}'.format(metrics[0],
                                                                                                                metrics[2],
                                                                                                                metrics[4],
                                                                                                                metrics[1],
                                                                                                                metrics[3],
                                                                                                                metrics[5]))

            return metrics

    elif model_name == 'ComplEx':
        model.normalize_parameters()
        model.eval()
        with torch.no_grad():
            re_ent_emb, im_ent_emb, re_rel_emb, im_rel_emb = model.get_embeddings()

            metrics = []
            ranks = []
            r_re_emb = re_rel_emb[rel2id['HAS_BINDING_TO']]
            r_im_emb = im_rel_emb[rel2id['HAS_BINDING_TO']]
            if task == 'target_inference':
                for i in tqdm.tqdm(range(len(df))):
                    c = df.iloc[i]['from']
                    p = df.iloc[i]['to']
                    
                    # to avoid ent not existing in kg
                    try:
                        h_re_emb = re_ent_emb[ent2id[c]] 
                        h_im_emb = im_ent_emb[ent2id[c]]
                        t_re_emb = re_ent_emb[ent2id[p]]  
                        t_im_emb = im_ent_emb[ent2id[p]]
                    except KeyError:
                        ranks.append(len(t_cand))
                        continue

                    score = (h_re_emb * (r_re_emb * t_re_emb + r_im_emb * t_im_emb) + h_im_emb * (r_re_emb * t_im_emb - r_im_emb * t_re_emb)).sum()

                    t_cand_re_emb = re_ent_emb[t_cand]
                    t_cand_im_emb = im_ent_emb[t_cand]
                    n_decoys = len(t_cand)
                    
                    t_score = (h_re_emb * (r_re_emb * t_cand_re_emb + r_im_emb * t_cand_im_emb) + 
                                 h_im_emb * (r_re_emb * t_cand_im_emb - r_im_emb * t_cand_re_emb)).sum(dim=1)

                    r = int(sum(t_score >= score).cpu())
                    # print(r)
                    ranks.append(r)
                
                ranks = [1 if x == 0 else x for x in ranks]  # ensure no zero rank 
                df['rank'] = ranks

                for k in [10,30,100]:
                    top = count_metrics(df, k, 'top')
                    recall = count_metrics(df, k, 'recall')
                    metrics.append(top)
                    metrics.append(recall)

                print('Top-10:{} | Top-30:{} | Top-100:{} | Recall@10:{} | Recall@30:{} | Recall@100:{}'.format(metrics[0],
                                                                                                                metrics[2],
                                                                                                                metrics[4],
                                                                                                                metrics[1],
                                                                                                                metrics[3],
                                                                                                                metrics[5]))

            return metrics

    elif model_name == 'ConvKB':
        model.normalize_parameters()
        model.eval()
        with torch.no_grad():
            ent_emb,rel_emb = model.get_embeddings() # (n_ent, emb_dim)

            metrics = []
            ranks = []
            if task == 'target_inference':
                for i in tqdm.tqdm(range(len(df))):
                    c = df.iloc[i]['from']
                    p = df.iloc[i]['to']
                    
                    # to avoid ent not existing in kg
                    try:
                        h_emb = ent_emb[ent2id[c]]
                        t_emb = ent_emb[ent2id[p]]
                    except KeyError:
                        ranks.append(len(t_cand))
                        continue


                    h_emb = ent_emb[ent2id[c]].view(1,1,-1)
                    r_emb = rel_emb[rel2id['HAS_BINDING_TO']].view(1,1,-1)
                    t_emb = ent_emb[ent2id[p]].view(1,1,-1)
                    concat = cat((h_emb, r_emb, t_emb), dim=1)
                    score = model.output(model.convlayer(concat).reshape(1, -1))[:, 1]
                    # print(score)


                    # replace tail
                    n_decoys = len(t_cand)
                    h_emb = ent_emb[ent2id[c]].view(1,1,-1).repeat(n_decoys,1,1)
                    r_emb = rel_emb[rel2id['HAS_BINDING_TO']].view(1,1,-1).repeat(n_decoys,1,1)
                    t_cand_emb = ent_emb[t_cand].view(n_decoys,1,-1)
                    concat = cat((h_emb, r_emb, t_cand_emb), dim=1)

                    t_score = model.output(model.convlayer(concat).reshape(n_decoys, -1))[:, 1]
                    # print(t_score.shape)

                    r = int(sum(t_score >= score).cpu())
                    # print(r)
                    ranks.append(r)
                
                ranks = [1 if x == 0 else x for x in ranks]  # ensure no zero rank 
                df['rank'] = ranks

                for k in [10,30,100]:
                    top = count_metrics(df, k, 'top')
                    recall = count_metrics(df, k, 'recall')
                    metrics.append(top)
                    metrics.append(recall)

                print('Top-10:{} | Top-30:{} | Top-100:{} | Recall@10:{} | Recall@30:{} | Recall@100:{}'.format(metrics[0],
                                                                                                                metrics[2],
                                                                                                                metrics[4],
                                                                                                                metrics[1],
                                                                                                                metrics[3],
                                                                                                                metrics[5]))

            return metrics

    else:
        model.normalize_parameters()
        model.eval()
        with torch.no_grad():
            metrics = []
            ranks = []
            if task == 'target_inference':
                for i in tqdm.tqdm(range(len(df))):
                    try:
                        c = torch.LongTensor([ent2id[df.iloc[i]['from']]]).to(device)
                        r = torch.LongTensor([rel2id[df.iloc[i]['rel']]]).to(device)
                        p = torch.LongTensor([ent2id[df.iloc[i]['to']]]).to(device)
                    except KeyError:
                        ranks.append(len(t_cand))
                        continue

                    score = model.scoring_function(c,p,r)
                    # print(score)
                    # print(score.shape)
                    # replace head
                    n_decoys = len(t_cand)
                    
                    t_score = model.scoring_function(c.repeat(n_decoys),torch.LongTensor(t_cand).to(device),r.repeat(n_decoys))
                    # print(t_score.shape)

                    r = int(sum(t_score >= score).cpu())
                    # print(r)
                    ranks.append(r)
                
                ranks = [1 if x == 0 else x for x in ranks]  # ensure no zero rank 
                df['rank'] = ranks

                for k in [10,30,100]:
                    top = count_metrics(df, k, 'top')
                    recall = count_metrics(df, k, 'recall')
                    metrics.append(top)
                    metrics.append(recall)

                print('Top-10:{} | Top-30:{} | Top-100:{} | Recall@10:{} | Recall@30:{} | Recall@100:{}'.format(metrics[0],
                                                                                                                metrics[2],
                                                                                                                metrics[4],
                                                                                                                metrics[1],
                                                                                                                metrics[3],
                                                                                                                metrics[5]))

            return metrics



def inference(ent,
            ent2id, rel2id, 
            ent_emb, rel_emb,
            h_cand, t_cand,
            task = 'target_inference'):
    # NOTE:only for DistMult

    r_emb = rel_emb[rel2id['HAS_BINDING_TO']]

    if task == 'target_inference':
        try:
            h_emb = ent_emb[ent2id[ent]]
        except KeyError:
            raise ValueError("{} is not included in KG".format(ent))
        
        t_cand_emb = ent_emb[t_cand]  # (2000,dim)
        score = torch.sigmoid((h_emb * r_emb * t_cand_emb).sum(dim=1)).cpu().tolist()

    elif task == 'virtual_screening':
        t_emb = ent_emb[ent2id[ent]]
        h_cand_emb = ent_emb[h_cand]

        score = torch.sigmoid((h_cand_emb * r_emb * t_emb).sum(dim=1)).cpu().tolist()
    
    elif task == 'batch_target_inference':
        '''
        in this case, ent is list of target
        '''
        
        batch_size = 64 
        t_cand_emb = ent_emb[t_cand].unsqueeze(dim=0)   # (1,n,dim)

        score = []
        for i in tqdm.tqdm(range(0, len(ent), batch_size)):
            batch = ent[i:i+batch_size]  
            ent_id = [ent2id[x] for x in batch]
            h_emb = ent_emb[ent_id].unsqueeze(dim=1)  # (n,1,dim)

            score.append((h_emb * r_emb * t_cand_emb).sum(dim=-1).cpu())

        score = torch.sigmoid(torch.concat(score,dim=0)).tolist()

    elif task == 'batch_virtual_screening':
        '''
        in this case, ent is list of target
        '''
        
        batch_size = 64 
        h_cand_emb = ent_emb[h_cand].unsqueeze(dim=0)  # (1,n,dim)

        score = []
        for i in tqdm.tqdm(range(0, len(ent), batch_size)):
            batch = ent[i:i+batch_size]
            ent_id = [ent2id[x] for x in batch]
            t_emb = ent_emb[ent_id].unsqueeze(dim=1)  # (n,1,dim)

            score.append((h_cand_emb * r_emb * t_emb).sum(dim=-1).cpu())

        score = torch.sigmoid(torch.concat(score,dim=0)).tolist()
        
    return score
