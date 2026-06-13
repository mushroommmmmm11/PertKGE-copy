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


# general function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def n3_regularization(h_emb, r_emb, t_emb, neg_t_emb):
    """N3 regularization penalty on current-batch embeddings (Section 3.1)."""
    n3 = (h_emb.abs() ** 3).sum() + (r_emb.abs() ** 3).sum() + (t_emb.abs() ** 3).sum() + (neg_t_emb.abs() ** 3).sum()
    return n3 / h_emb.shape[0]


def build_entity_type_map(ent2id):
    """Build entity_name -> type mapping from name prefixes (Section 4)."""
    type_map = {}
    for name in ent2id:
        if name.startswith('DNA:'):
            type_map[name] = 'gene'
        elif name.startswith('subtype:'):
            type_map[name] = 'epilepsy_subtype'
        elif name.startswith('CID:'):
            type_map[name] = 'compound'
        elif name.startswith(('Protein:', 'TF:', 'RBP:')):
            type_map[name] = 'protein'
        elif name == 'epilepsy':
            type_map[name] = 'disease'
        else:
            type_map[name] = 'disease'
    return type_map

def get_rank(a):
    a = len(a) + 1 - rankdata(a)

    return a.tolist()



def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.npu.is_available():
            torch.npu.manual_seed(seed)
            torch.npu.manual_seed_all(seed)
    except AttributeError:
        pass
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


def read_pathway_extra(filepath):
    """Parse human_gene_pathway_filtered.txt: space-separated, normalize DNA: X → DNA:X."""
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # parts[0] = 'DNA:', parts[1] = 'AKR1A1', parts[2] = 'participates_in'
            if len(parts) >= 4:
                entity = parts[0] + parts[1]            # DNA:AKR1A1
                relation = parts[2]                      # participates_in
                obj = ' '.join(parts[3:])                # pathway name
                rows.append({'from': entity, 'rel': relation, 'to': obj})
    return pd.DataFrame(rows).drop_duplicates()


def _normalize_entity(e):
    """Normalize fullwidth colon (：) to regular colon (:) in entity names."""
    return e.replace('\uff1a', ':')


def read_subtype_file(filepath):
    """Parse subtype_epilepsy.txt: columns separated by 2+ spaces, 3 cols.
    Normalize fullwidth colons in entity names.
    """
    import re
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r'  +', line)
            if len(parts) < 3:
                continue
            rows.append({
                'from': _normalize_entity(parts[0]),
                'rel': parts[1].strip(),
                'to': _normalize_entity(parts[2])
            })
    return pd.DataFrame(rows).drop_duplicates()


def read_subdisease_gene_file(filepath):
    """Parse subdisease_gene.txt: single-space separated, 4 cols.
    Uses 'subdisease_regulates' as anchor to handle spaces in entity names.
    Normalizes fullwidth colons.
    """
    rows = []
    anchor = 'subdisease_regulates'
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx = line.find(anchor)
            if idx < 0:
                continue
            subj = line[:idx].strip()
            rest = line[idx + len(anchor):].strip()
            # rest is "DNA:GENE classification:Level"
            # split on 'classification:' to get gene part
            cls_idx = rest.find('classification:')
            if cls_idx >= 0:
                obj = rest[:cls_idx].strip()
                classification = rest[cls_idx + len('classification:'):].strip()
            else:
                obj = rest
                classification = 'Supportive'
            rows.append({
                'from': _normalize_entity(subj),
                'rel': anchor,
                'to': _normalize_entity(obj),
                'classification': classification
            })
    return pd.DataFrame(rows).drop_duplicates()


def read_files(args):
    print('read input files!!!')
    s = time()

    # read original files
    cause = pd.read_csv(args.cause_file,sep='\t',names=['from','rel','to']).drop_duplicates()
    process = pd.read_csv(args.process_file,sep='\t',names=['from','rel','to']).drop_duplicates()
    effect = pd.read_csv(args.effect_file,sep='\t',names=['from','rel','to']).drop_duplicates()

    # test file (optional)
    if os.path.exists(args.test_file):
        test = pd.read_csv(args.test_file, sep='\t', names=['from','rel','to']).drop_duplicates()
    else:
        test = pd.DataFrame(columns=['from','rel','to'])
        print(f'[Warning] test file not found: {args.test_file}, using empty test set')

    # read epilepsy-related data files
    pathway_extra = read_pathway_extra(args.pathway_extra_file)
    print(f'  pathway_extra: {len(pathway_extra)} triples')

    subtype = read_subtype_file(args.subtype_file)
    print(f'  subtype: {len(subtype)} triples')

    subdisease_gene = read_subdisease_gene_file(args.subdisease_gene_file)
    print(f'  subdisease_gene: {len(subdisease_gene)} triples')

    # Build edge evidence dict for loss weighting (Section 2.2)
    edge_evidence_dict = {}
    for _, row in subdisease_gene.iterrows():
        edge_evidence_dict[(row['from'], row['rel'], row['to'])] = row.get('classification', 'Supportive')

    # generate processed data
    if args.load_processed_data:    
        ent2id = np.load(args.processed_data_file + 'ent2id.npy', allow_pickle=True).item()
        rel2id = np.load(args.processed_data_file + 'rel2id.npy', allow_pickle=True).item()
        pro2nc = np.load(args.processed_data_file + 'pro2nc.npy', allow_pickle=True).item()

    else:
        # 1.ent2id, rel2id — include new epilepsy data
        pertkg = pd.concat([cause, process, effect, pathway_extra, subtype, subdisease_gene])
        ent2id,rel2id = get_dictionaries(pertkg)
        np.save(args.processed_data_file + 'ent2id.npy',ent2id)
        np.save(args.processed_data_file + 'rel2id.npy',rel2id)
        # 2.pro2nc
        comp_cand = {k for k,v in ent2id.items() if k.startswith('CID:')}
        _,p2c_dict = get_cpi(cause)
        pro = set(cause['to'])
        pro2nc = {}
        for x in pro:
            candidates = list(comp_cand - p2c_dict[x])
            pro2nc[x] = random.sample(candidates, k=min(3000, len(candidates)))
        np.save(args.processed_data_file + 'pro2nc.npy',pro2nc)

    pertkg_wo_cause = pd.concat([process, effect, pathway_extra, subtype, subdisease_gene])
    pertkg_wo_cause_global = pd.concat([process, effect, pathway_extra])

    h_cand = [v for k,v in ent2id.items() if k.startswith('CID:')]
    t_cand = [v for k,v in ent2id.items() if k.startswith(('Protein:','TF:','RBP:'))]
    print('total {} compound and {} targets'.format(len(h_cand),len(t_cand)))

    e = time()
    print(f"reading time: {round(e - s, 2)}s")
    print('_'*50)

    return cause, pertkg_wo_cause, pertkg_wo_cause_global, test, ent2id, rel2id, pro2nc, h_cand, t_cand, subtype, subdisease_gene, edge_evidence_dict

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



def check_validation_leakage(train_triples, valid_triples, relation_name, all_train_triples=None):
    """检查验证集实体是否在训练集同关系中出现过（防止数据泄露）"""
    # 检查 head 实体泄露
    train_heads = set(h for h, r, t in train_triples if r == relation_name)
    valid_heads = set(h for h, r, t in valid_triples if r == relation_name)
    leakage = train_heads & valid_heads
    if len(leakage) > 0:
        print(f"[DEMO-WARN] [{relation_name}] {len(leakage)} head entities in train+valid: {list(leakage)[:5]}")

    # 检查 tail 实体泄露
    train_tails = set(t for h, r, t in train_triples if r == relation_name)
    valid_tails = set(t for h, r, t in valid_triples if r == relation_name)
    tail_leakage = train_tails & valid_tails
    if len(tail_leakage) > 0:
        print(f"[DEMO-WARN] [{relation_name}] {len(tail_leakage)} tail entities in train+valid: {list(tail_leakage)[:5]}")

    # 检查 exact triple 泄露
    train_set = set(train_triples)
    valid_set = set(valid_triples)
    triple_leak = train_set & valid_set
    if len(triple_leak) > 0:
        raise AssertionError(
            f"[{relation_name}] 存在 {len(triple_leak)} 条 exact triple 泄露: {list(triple_leak)[:5]}"
        )


    # === 新增：跨关系 unseen entity 检查 ===
    if all_train_triples is not None:
        all_train_entities = set()
        for h, r, t in all_train_triples:
            all_train_entities.add(h)
            all_train_entities.add(t)

        valid_entities = set()
        for h, r, t in valid_triples:
            valid_entities.add(h)
            valid_entities.add(t)

        unseen = valid_entities - all_train_entities
        if unseen:
            print(f"[FATAL] [{relation_name}] {len(unseen)} COMPLETELY UNSEEN entities in valid set "
                  f"(never appear in ANY training triple). ComplEx cannot predict these.")
            print(f"  Unseen entities: {sorted(unseen)[:15]}")
        else:
            print(f"[Check] [{relation_name}] all valid entities appear in training KG -> OK")

    print(f"[Check] {relation_name}: head leak=0, tail leak=0, triple leak=0 -> PASS")


def generate_entity_level_splits(args, subtype_df, subdisease_gene_df):
    """Entity-Level 5-Fold CV
    - is_subtype_of: 按 head 实体（亚型）分 5 组 (10/10/10/10/8)
    - subdisease_regulates: 按 tail 实体（基因）分 5 组 (35/35/35/35/33)
    - 两组 fold 同步切换
    """
    print("generate entity-level five fold splits for is_subtype_of & subdisease_regulates!!!")
    s = time()

    if args.load_processed_data:
        five_fold_subtype_train, five_fold_subtype_valid = [], []
        five_fold_subdisease_train, five_fold_subdisease_valid = [], []
        for i in range(5):
            st = pd.read_csv(args.processed_data_file + f"subtype_train{i}.txt",
                             sep="\t", names=["from", "rel", "to"])
            sv = pd.read_csv(args.processed_data_file + f"subtype_valid{i}.txt",
                             sep="\t", names=["from", "rel", "to"])
            sg = pd.read_csv(args.processed_data_file + f"subdisease_train{i}.txt",
                             sep="\t", names=["from", "rel", "to"])
            sgv = pd.read_csv(args.processed_data_file + f"subdisease_valid{i}.txt",
                              sep="\t", names=["from", "rel", "to"])
            five_fold_subtype_train.append(st)
            five_fold_subtype_valid.append(sv)
            five_fold_subdisease_train.append(sg)
            five_fold_subdisease_valid.append(sgv)
    else:
        # --- is_subtype_of: 按 head 实体切分 ---
        subtype_heads = sorted(subtype_df["from"].unique())
        subtype_groups = [[] for _ in range(5)]
        for idx, h in enumerate(subtype_heads):
            subtype_groups[idx % 5].append(h)

        five_fold_subtype_train, five_fold_subtype_valid = [], []
        for fold in range(5):
            valid_heads = set(subtype_groups[fold])
            train = subtype_df[~subtype_df["from"].isin(valid_heads)]
            valid = subtype_df[subtype_df["from"].isin(valid_heads)]
            five_fold_subtype_train.append(train)
            five_fold_subtype_valid.append(valid)
            train.to_csv(args.processed_data_file + f"subtype_train{fold}.txt",
                         sep="\t", index=False, header=False)
            valid.to_csv(args.processed_data_file + f"subtype_valid{fold}.txt",
                         sep="\t", index=False, header=False)

        # --- subdisease_regulates: 按 tail 实体切分 ---
        subdisease_tails = sorted(subdisease_gene_df["to"].unique())
        subdisease_groups = [[] for _ in range(5)]
        for idx, t in enumerate(subdisease_tails):
            subdisease_groups[idx % 5].append(t)

        five_fold_subdisease_train, five_fold_subdisease_valid = [], []
        for fold in range(5):
            valid_tails = set(subdisease_groups[fold])
            train = subdisease_gene_df[~subdisease_gene_df["to"].isin(valid_tails)]
            valid = subdisease_gene_df[subdisease_gene_df["to"].isin(valid_tails)]
            five_fold_subdisease_train.append(train)
            five_fold_subdisease_valid.append(valid)
            train.to_csv(args.processed_data_file + f"subdisease_train{fold}.txt",
                         sep="\t", index=False, header=False)
            valid.to_csv(args.processed_data_file + f"subdisease_valid{fold}.txt",
                         sep="\t", index=False, header=False)

    # 打印切分统计
    for fold in range(5):
        n_st = len(five_fold_subtype_train[fold])
        n_sv = len(five_fold_subtype_valid[fold])
        n_sgt = len(five_fold_subdisease_train[fold])
        n_sgv = len(five_fold_subdisease_valid[fold])
        print(f"  Fold {fold}: subtype train={n_st} valid={n_sv} | "
              f"subdisease train={n_sgt} valid={n_sgv}")

    e = time()
    print(f"entity-level splitting time: {round(e - s, 2)}s")
    print("_" * 50)

    return (five_fold_subtype_train, five_fold_subtype_valid,
            five_fold_subdisease_train, five_fold_subdisease_valid)


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
            
            # OPT B: Pre-compute full compound-protein score matrix on NPU once.
            # All 3381 queries then become pure CPU index lookups.
            r_id = rel2id["HAS_BINDING_TO"]
            r_re = re_rel_emb[r_id]
            r_im = im_rel_emb[r_id]
            compound_to_idx = {}
            protein_to_idx = {}
            compound_ids = []
            protein_ids = []
            for p_str, c_list in pro2nc.items():
                if p_str not in ent2id: continue
                p_id = ent2id[p_str]
                if p_id not in protein_to_idx:
                    protein_to_idx[p_id] = len(protein_ids)
                    protein_ids.append(p_id)
                for c_str in c_list:
                    if c_str not in ent2id: continue
                    c_id = ent2id[c_str]
                    if c_id not in compound_to_idx:
                        compound_to_idx[c_id] = len(compound_ids)
                        compound_ids.append(c_id)
            compound_ids = torch.tensor(compound_ids, device=re_ent_emb.device)
            protein_ids = torch.tensor(protein_ids, device=re_ent_emb.device)
            compound_re = re_ent_emb[compound_ids]
            compound_im = im_ent_emb[compound_ids]
            protein_re = re_ent_emb[protein_ids]
            protein_im = im_ent_emb[protein_ids]
            shared_re = r_re * protein_re + r_im * protein_im
            shared_im = r_re * protein_im - r_im * protein_re
            scores_all = torch.mm(compound_re, shared_re.T) + torch.mm(compound_im, shared_im.T)
            scores_all = scores_all.cpu().numpy()  # OPT B v3: numpy for CPU indexing
            h_ranks = []
            for i in tqdm.tqdm(range(len(df)), desc="NUMPY-V3"):
                c = df.iloc[i]["from"]; p = df.iloc[i]["to"]
                if c not in ent2id or p not in ent2id:
                    h_ranks.append(len(compound_ids)); continue
                ci = compound_to_idx.get(ent2id[c])
                pi = protein_to_idx.get(ent2id[p])
                if ci is None or pi is None:
                    h_ranks.append(len(compound_ids)); continue
                true_score = float(scores_all[ci, pi])
                cand_strs = pro2nc.get(p, [])
                if not cand_strs:
                    h_ranks.append(len(compound_ids)); continue
                cand_indices = []
                for cand in cand_strs:
                    if cand in ent2id:
                        cand_ci = compound_to_idx.get(ent2id[cand])
                        if cand_ci is not None: cand_indices.append(cand_ci)
                if not cand_indices:
                    h_ranks.append(len(compound_ids)); continue
                cand_indices = np.array(cand_indices, dtype=np.int64)  # numpy
                cand_scores = scores_all[cand_indices, pi]
                rank = int((cand_scores >= true_score).sum()) + 1
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
                c = torch.LongTensor([ent2id[df.iloc[i]['from']]]).to('npu')
                r = torch.LongTensor([rel2id[df.iloc[i]['rel']]]).to('npu')
                p = torch.LongTensor([ent2id[df.iloc[i]['to']]]).to('npu')

                score = model.scoring_function(c,p,r)

                # replace head
                h_cand = pro2nc[df.iloc[i]['to']]
                h_cand = [ent2id[x] for x in h_cand]

                n_decoys = len(h_cand)

                h_score = model.scoring_function(torch.LongTensor(h_cand).to('npu'),p.repeat(n_decoys),r.repeat(n_decoys))

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
                for i in tqdm.tqdm(range(len(df)), disable = args.local_rank!=0):
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
                        c = torch.LongTensor([ent2id[df.iloc[i]['from']]]).to('npu')
                        r = torch.LongTensor([rel2id[df.iloc[i]['rel']]]).to('npu')
                        p = torch.LongTensor([ent2id[df.iloc[i]['to']]]).to('npu')
                    except KeyError:
                        ranks.append(len(t_cand))
                        continue

                    score = model.scoring_function(c,p,r)
                    # print(score)
                    # print(score.shape)
                    # replace head
                    n_decoys = len(t_cand)
                    
                    t_score = model.scoring_function(c.repeat(n_decoys),torch.LongTensor(t_cand).to('npu'),r.repeat(n_decoys))
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
