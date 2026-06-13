import os
import json
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from time import time
import pandas as pd
import numpy as np
import tqdm
import random
from collections import defaultdict
import argparse

import math
import torch
import torch.nn as nn
# from torch import cuda
import torch_npu
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader as PyTorchDataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss
from torchkge import KnowledgeGraph,DistMultModel,TransEModel,TransHModel
from torchkge.models.bilinear import HolEModel,ComplExModel

from utils import *
from model import *
from data_loader import TrainDataset, EVIDENCE_WEIGHTS, DistributedWeightedRandomSampler

def setup_distributed():
    """初始化分布式训练环境"""
    dist.init_process_group(backend='hccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.npu.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='PertKG',
        usage='main.py [<args>] [-h | --help]'
    )
    parser.add_argument('--cause_file',default="../processed_data/deepce/cause.txt")
    parser.add_argument('--process_file',default="../processed_data/knowledge_graph/process.txt")
    parser.add_argument('--effect_file', default="../processed_data/deepce/effect.txt")
    parser.add_argument('--test_file',default="../processed_data/deepce/test.txt")
    # New epilepsy data integration
    parser.add_argument('--pathway_extra_file', default="../kg数据3/human_gene_pathway_filtered.txt",
                        help='Additional DNA→pathway mappings (space-separated, normalizes to canonical entity format)')
    parser.add_argument('--subtype_file', default="../kg数据3/subtype_epilepsy.txt",
                        help='Epilepsy subtype hierarchy (subtype is_subtype_of epilepsy)')
    parser.add_argument('--subdisease_gene_file', default="../kg数据3/subdisease_gene.txt",
                        help='Subtype-disease gene associations (subtype subdisease_regulates DNA:gene classification:evidence)')
    parser.add_argument('--seed', type = int, default=43)
    parser.add_argument('--h_dim', type = int, default=300)
    parser.add_argument('--margin', type = float, default=1.0)
    parser.add_argument('--wd', type = float, default=1e-5)
    parser.add_argument('--n_neg', type = int, default=100)
    parser.add_argument('--batch_size', type = int, default=2048)
    parser.add_argument('--warm_up', type = int, default=10)
    parser.add_argument('--patients', type = int, default=5)
    parser.add_argument('--use_cuda', type = str, default='batch')
    parser.add_argument('--save_model', action='store_true', default=True)
    parser.add_argument('--save_model_path',default="../best_model/deepce_distmult_2/")
    parser.add_argument('--load_processed_data', action='store_true', default=False)
    parser.add_argument('--processed_data_file',default="../processed_data/deepce/")
    parser.add_argument('--mode', default="reproduce", help = 'choose reproduce if user want to report testing results')  # test or not
    parser.add_argument('--task', default="target_inference", help="choose from ['target_inference', 'virtual_screening', 'unbiased_test']")
    parser.add_argument('--run_name', default="deepce_distmult", help="Name of the running.")
    parser.add_argument('--distributed', action='store_true', default=False, help='Enable distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume training from checkpoint')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite existing checkpoint and start fresh')

    # +++++++++++++++++
    return parser.parse_args(args),parser.parse_args(args).__dict__


def save_checkpoint(save_path, model, optimizer, epoch, best_mrr, patients, val_metrics, is_main_process):
    """保存完整的训练状态到checkpoint"""
    if not is_main_process:
        return
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_mrr': best_mrr,
        'patients': patients,
        'val_metrics': val_metrics,
    }
    torch.save(state, save_path)
    print(f'[Checkpoint] Saved checkpoint to {save_path} (epoch {epoch+1})')


def load_checkpoint(save_path, model, optimizer, device):
    """从checkpoint恢复训练状态"""
    if not os.path.exists(save_path):
        return None
    state = torch.load(save_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    # scaler removed
    print(f'[Checkpoint] Loaded checkpoint from {save_path} (epoch {state["epoch"]+1})')
    return state


def get_progress_file(save_model_path):
    return os.path.join(save_model_path, 'progress.json')


def save_progress(progress_file, completed_splits, in_progress_split, is_main_process):
    """保存训练进度到JSON文件"""
    if not is_main_process:
        return
    data = {
        'completed_splits': completed_splits,
        'in_progress_split': in_progress_split,
    }
    with open(progress_file, 'w') as f:
        json.dump(data, f, indent=2)
    if in_progress_split >= 0:
        print(f'[Progress] Split {in_progress_split} in progress, completed: {completed_splits}')


def load_progress(progress_file):
    """加载训练进度"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {'completed_splits': [], 'in_progress_split': -1}



def softplus_loss(pos_score, neg_score, n_neg=100):
    """Softplus-based positive/negative sample loss (Section 3.3).

    Parameters
    ----------
    pos_score : torch.Tensor, shape (batch_size,) or (batch_size * n_neg,)
    neg_score : torch.Tensor, shape (batch_size * n_neg,)
    n_neg : int, number of negative samples per positive

    Returns
    -------
    torch.Tensor, shape (batch_size,)
        Per-sample loss = softplus(-pos) + mean(softplus(neg)).
    """
    pos_loss = torch.nn.functional.softplus(-pos_score)
    if n_neg > 1:
        neg_score = neg_score.view(-1, n_neg)
    neg_loss = torch.nn.functional.softplus(neg_score).mean(dim=1)
    if pos_loss.dim() > 0 and pos_loss.shape[0] != neg_loss.shape[0]:
        pos_loss = pos_loss.view(-1, n_neg).mean(dim=1)
    elif pos_loss.dim() == 0:
        pos_loss = pos_loss.unsqueeze(0)
    return pos_loss + neg_loss


def evaluate_mrr_complex_batched(model, triples, ent2id, rel2id, device):
    """Numpy-based batched MRR evaluation for ComplEx. Pulls all embeddings to CPU
    once, uses np.dot for all-entity scoring to avoid NPU per-index kernel overhead."""
    if not triples:
        return 0.0, 0.0, 0.0, []
    import numpy as np
    model.eval()
    with torch.no_grad():
        re_ent, im_ent, re_rel, im_rel = model.get_embeddings()
        re_ent_np = re_ent.cpu().numpy().astype(np.float32)
        im_ent_np = im_ent.cpu().numpy().astype(np.float32)
        re_rel_np = re_rel.cpu().numpy().astype(np.float32)
        im_rel_np = im_rel.cpu().numpy().astype(np.float32)

        h_ids, t_ids = [], []
        r_id = None
        for h_str, r_str, t_str in triples:
            try:
                h_ids.append(ent2id[h_str])
                t_ids.append(ent2id[t_str])
                if r_id is None:
                    r_id = rel2id[r_str]
            except KeyError:
                continue
        if not h_ids:
            return 0.0, 0.0, 0.0, []

        h_ids = np.array(h_ids, dtype=np.int64)
        t_ids = np.array(t_ids, dtype=np.int64)

        h_re = re_ent_np[h_ids]
        h_im = im_ent_np[h_ids]
        t_re = re_ent_np[t_ids]
        t_im = im_ent_np[t_ids]
        r_re = re_rel_np[r_id]
        r_im = im_rel_np[r_id]

        # True scores: Re(<h, r, conj(t)>)
        score_true = (h_re * (r_re * t_re + r_im * t_im) +
                      h_im * (r_re * t_im - r_im * t_re)).sum(axis=1)

        # Precompute relation-weighted entity matrix: (n_ent, dim)
        ent_re_w = re_ent_np * r_re + im_ent_np * r_im
        ent_im_w = im_ent_np * r_re - re_ent_np * r_im

        # All-entity scores via efficient np.dot
        scores_all = h_re.dot(ent_re_w.T) + h_im.dot(ent_im_w.T)

        ranks = (scores_all >= score_true[:, np.newaxis]).sum(axis=1)
        ranks = np.maximum(ranks, 1)

        mrr = float((1.0 / ranks).mean())
        h1 = float((ranks <= 1).mean())
        h3 = float((ranks <= 3).mean())

        return mrr, h1, h3, ranks.tolist()


def check_fewshot_gap(model, train_triples, valid_triples, ent2id, rel2id,
                      n_ent, device, epoch, gap_threshold=0.3):
    """Monitor train-valid MRR gap for few-shot relations (Section 5.2).
    Uses numpy-batched evaluate_mrr_complex_batched for CPU-speed evaluation."""
    flagged = []
    for relation in ['is_subtype_of', 'subdisease_regulates']:
        train_rel = [t for t in train_triples if t[1] == relation]
        valid_rel = [t for t in valid_triples if t[1] == relation]
        if not train_rel or not valid_rel:
            continue

        train_mrr, _, _, _ = evaluate_mrr_complex_batched(
            model, train_rel, ent2id, rel2id, device)
        valid_mrr, _, _, _ = evaluate_mrr_complex_batched(
            model, valid_rel, ent2id, rel2id, device)
        gap = train_mrr - valid_mrr

        print('[Epoch {}] {} | Train MRR: {:.4f} | Valid MRR: {:.4f} | Gap: {:.4f}'.format(
            epoch, relation, train_mrr, valid_mrr, gap))
        if gap > gap_threshold:
            print('  -> WARNING: {} gap={:.3f} > {}'.format(relation, gap, gap_threshold))
            flagged.append(relation)
    return flagged


def log_fewshot_report(model, valid_subtype_triples, valid_subdisease_triples,
                       ent2id, rel2id, n_ent, device, epoch):
    """Print per-epoch few-shot relation report (Section 5.3).
    Uses numpy-batched evaluate_mrr_complex_batched for CPU-speed evaluation."""
    subtype_mrr, subtype_h1, subtype_h3, _ = evaluate_mrr_complex_batched(
        model, valid_subtype_triples, ent2id, rel2id, device)
    subdis_mrr, subdis_h1, subdis_h3, _ = evaluate_mrr_complex_batched(
        model, valid_subdisease_triples, ent2id, rel2id, device)

    print('=== Epoch {} Few-Shot Relation Report ==='.format(epoch))
    print('is_subtype_of        | Valid MRR: {:.4f} | Hits@1: {:.4f} | Hits@3: {:.4f}'.format(
        subtype_mrr, subtype_h1, subtype_h3))
    print('subdisease_regulates | Valid MRR: {:.4f} | Hits@1: {:.4f} | Hits@3: {:.4f}'.format(
        subdis_mrr, subdis_h1, subdis_h3))
    print('=' * 50)
    return subtype_mrr, subdis_mrr
# Section 2.3: Dynamic decay strategy
DECAY_CONFIG = {
    "total_epochs": 150,
    "stage1_end_epoch": 60,
    "stage2_end_epoch": 120,
    "init_lr": 5e-4,
    "stage2_min_lr": 5e-5,
    "stage3_fine_tune_lr": 1e-5,
    "l2_reg_weight": 1e-4,
    "warmup_epochs": 5,
    "warmup_start_lr": 1e-6,
}


def get_stage(epoch):
    if epoch <= DECAY_CONFIG["stage1_end_epoch"]:
        return 1
    elif epoch <= DECAY_CONFIG["stage2_end_epoch"]:
        return 2
    else:
        return 3


def decay_sampling_weights(base_weights, epoch):
    stage = get_stage(epoch)
    if stage == 1:
        return base_weights
    elif stage == 2:
        progress = (epoch - DECAY_CONFIG["stage1_end_epoch"]) / (
            DECAY_CONFIG["stage2_end_epoch"] - DECAY_CONFIG["stage1_end_epoch"]
        )
        return [w + (1.0 - w) * progress for w in base_weights]
    else:
        return [1.0] * len(base_weights)


def get_lr_for_epoch(epoch):
    # Linear warmup for first few epochs
    if epoch <= DECAY_CONFIG["warmup_epochs"]:
        ratio = epoch / DECAY_CONFIG["warmup_epochs"]
        return DECAY_CONFIG["warmup_start_lr"] + ratio * (
            DECAY_CONFIG["init_lr"] - DECAY_CONFIG["warmup_start_lr"])
    stage = get_stage(epoch)
    if stage == 1:
        return DECAY_CONFIG["init_lr"]
    elif stage == 2:
        progress = (epoch - DECAY_CONFIG["stage1_end_epoch"]) / (
            DECAY_CONFIG["stage2_end_epoch"] - DECAY_CONFIG["stage1_end_epoch"]
        )
        return DECAY_CONFIG["stage2_min_lr"] + 0.5 * (
            DECAY_CONFIG["init_lr"] - DECAY_CONFIG["stage2_min_lr"]
        ) * (1 + math.cos(math.pi * progress))
    else:
        return DECAY_CONFIG["stage3_fine_tune_lr"]


def five_fold_cv(args):
    # read cause, process, effect, test file
    cause, pertkg_wo_cause, pertkg_wo_cause_global, test, ent2id, rel2id, pro2nc, h_cand, t_cand, subtype, subdisease_gene, edge_evidence_dict = read_files(args)
    
    # generate train\valid
    five_fold_train, five_fold_valid = generate_five_fold_files(args, cause)

    # Entity-Level 5-Fold CV for is_subtype_of & subdisease_regulates
    (five_fold_subtype_train, five_fold_subtype_valid,
     five_fold_subdisease_train, five_fold_subdisease_valid) = generate_entity_level_splits(args, subtype, subdisease_gene)

    # 判断是否为分布式训练
    is_distributed = args.distributed
    if is_distributed:
        local_rank = args.local_rank
        world_size = dist.get_world_size()
        is_main_process = local_rank == 0
    else:
        is_main_process = True

    results = []
    # ---------- 断点重续：进度追踪 ----------
    progress_file = get_progress_file(args.save_model_path)
    if args.resume and not args.overwrite:
        progress = load_progress(progress_file)
        completed_splits = progress['completed_splits']
        if is_main_process:
            print(f'[Resume] Resuming training, already completed splits: {completed_splits}')
    else:
        completed_splits = []
        if args.overwrite:
            if is_main_process:
                print('[Overwrite] Starting fresh training, cleared previous progress')
    if args.overwrite:
        save_progress(progress_file, [], -1, is_main_process)
    # ---------------------------------------
    for i in range(5):
        # 跳过已完成的 split
        if args.resume and i in completed_splits:
            if is_main_process:
                print(f'split_{i} already completed, skipping...')
            continue
        save_progress(progress_file, completed_splits, i, is_main_process)

        # load data to consrtuct kg
        if is_main_process:
            print('split_{}!!!'.format(i))
        
        # logger
        if is_main_process:
            train_logger = SummaryWriter('../outlog/{}_split{}'.format(args.run_name, i))
            batch_records = []  # 内存缓冲，epoch结束时一次性写盘
        else:
            train_logger = None

        # loading train and valid df
        train = five_fold_train[i]
        valid = five_fold_valid[i]

        if is_main_process:
            print('construct chemical perturbation profiles-based knowledge graph!!!')
        s1 = time()
        df = pd.concat([pertkg_wo_cause_global, train,
                        five_fold_subtype_train[i], five_fold_subdisease_train[i]])
        # 验证集污染检查 (entity-level CV)
        if is_main_process:
            print('Running validation leakage check...')
            train_triples = list(zip(df['from'], df['rel'], df['to']))
            valid_subtype_triples = list(zip(
                five_fold_subtype_valid[i]['from'],
                five_fold_subtype_valid[i]['rel'],
                five_fold_subtype_valid[i]['to']))
            valid_subdisease_triples = list(zip(
                five_fold_subdisease_valid[i]['from'],
                five_fold_subdisease_valid[i]['rel'],
                five_fold_subdisease_valid[i]['to']))
            check_validation_leakage(train_triples, valid_subtype_triples, 'is_subtype_of', all_train_triples=train_triples)
            check_validation_leakage(train_triples, valid_subdisease_triples, 'subdisease_regulates', all_train_triples=train_triples)
        df = df.sample(frac=1,random_state=42).reset_index(drop=True) # 打乱KG
        # Build TrainDataset with evidence weighting (Sections 2.1-2.2)
        dataset = TrainDataset(df, ent2id, rel2id, edge_evidence_dict)
        e1 = time()
        if is_main_process:
            print(f"Total constructing time: {round(e1 - s1, 2)}s")
            print()

        if is_main_process:
            print('split_{} traing now!!!'.format(i))
        # choose method
        fewshot_ids = [rel2id["is_subtype_of"], rel2id["subdisease_regulates"]]
        model = ComplExModelWithFewshotDropout(
            args.h_dim, len(ent2id), len(rel2id),
            fewshot_rel_ids=fewshot_ids, fewshot_dropout=0.5
        )
        torch.npu.empty_cache()
        model = model.to(device)
        
        # 分布式训练：使用DDP包装模型
        if is_distributed:
            model = DDP(model, device_ids=[torch.device(f'npu:{local_rank}')])
        
        # Separate entity/relation params for differential LR: rel_lr = 0.5 * ent_lr
        ent_params, rel_params, other_params = [], [], []
        for n, p in model.named_parameters():
            if 're_ent_emb' in n or 'im_ent_emb' in n:
                ent_params.append(p)
            elif 're_rel_emb' in n or 'im_rel_emb' in n:
                rel_params.append(p)
            else:
                other_params.append(p)
        optimizer = Adam([
            {'params': ent_params,  'lr': DECAY_CONFIG['init_lr'],       'lr_mult': 1.0},
            {'params': rel_params,  'lr': DECAY_CONFIG['init_lr'] * 0.5, 'lr_mult': 0.5},
            {'params': other_params,'lr': DECAY_CONFIG['init_lr'],       'lr_mult': 1.0},
        ], weight_decay=args.wd)
        # Manual cosine LR scheduling below (replaces CosineAnnealingLR)
        # AMP removed: scaler caused GradScaler scale explosion on NPU
        # Build entity type map and type-constrained neg sampler (Section 4)
        kg = KnowledgeGraph(df=df)
        kgsampler = BernoulliNegativeSampler(kg, n_neg=args.n_neg)
        
        # WeightedRandomSampler with initial base weights (Section 2.1)
        base_weights = dataset.get_sampling_weights()
        if is_distributed:
            wt_sampler = DistributedWeightedRandomSampler(
                weights=base_weights,
                num_samples=len(dataset),
                world_size=world_size,
                rank=local_rank,
                replacement=True)
        else:
            wt_sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(base_weights),
                num_samples=len(dataset), replacement=True)
        kgloader = PyTorchDataLoader(dataset, batch_size=args.batch_size,
                                     sampler=wt_sampler, shuffle=False)


        # wo train
        model_to_test = model.module if is_distributed else model
        _ = tester('ComplEx',model_to_test,
                    args,
                    test,
                    ent2id,rel2id,
                    h_cand,t_cand,
                    args.task)

        # train
        checkpoint_path = os.path.join(args.save_model_path, f'checkpoint_split{i}.pt')
        best_mrr = 0
        fewshot_patients = 0  # Section 5.2
        patients = 0
        start_epoch = 0
        val_metrics = {}
        if args.resume and os.path.exists(checkpoint_path):
            checkpoint_state = load_checkpoint(checkpoint_path, model, optimizer, device)
            if checkpoint_state is not None:
                best_mrr = checkpoint_state['best_mrr']
                patients = checkpoint_state['patients']
                start_epoch = checkpoint_state['epoch'] + 1
                val_metrics = checkpoint_state.get('val_metrics', {})
                if is_main_process:
                    print(f'[Resume] Resumed split_{i} from epoch {checkpoint_state["epoch"]+1}')
        for epoch in range(start_epoch, DECAY_CONFIG["total_epochs"]):
            # Sections 2.3 + 3.4: per-epoch sampling weight decay + LR scheduling
            decayed_w = decay_sampling_weights(base_weights, epoch + 1)
            if is_distributed:
                wt_sampler = DistributedWeightedRandomSampler(
                    weights=decayed_w,
                    num_samples=len(dataset),
                    world_size=world_size,
                    rank=local_rank,
                    replacement=True)
                wt_sampler.set_epoch(epoch)
            else:
                wt_sampler = WeightedRandomSampler(
                    weights=torch.DoubleTensor(decayed_w),
                    num_samples=len(dataset), replacement=True)
            kgloader = PyTorchDataLoader(dataset, batch_size=args.batch_size,
                                         sampler=wt_sampler, shuffle=False)

            stage = get_stage(epoch + 1)
            if stage == 1:
                for pg in optimizer.param_groups:
                    pg["lr"] = get_lr_for_epoch(epoch + 1) * pg.get("lr_mult", 1.0)
            elif stage == 2:
                # Manual per-group cosine decay (preserves lr_mult ratio)
                stage2_progress = (epoch + 1 - DECAY_CONFIG["stage1_end_epoch"]) / (
                    DECAY_CONFIG["stage2_end_epoch"] - DECAY_CONFIG["stage1_end_epoch"])
                cosine_factor = 0.5 * (1 + math.cos(math.pi * stage2_progress))
                for pg in optimizer.param_groups:
                    mult = pg.get("lr_mult", 1.0)
                    pg["lr"] = mult * (DECAY_CONFIG['stage2_min_lr'] + 
                        (DECAY_CONFIG['init_lr'] - DECAY_CONFIG['stage2_min_lr']) * cosine_factor)
            elif stage == 3:
                for pg in optimizer.param_groups:
                    pg["lr"] = DECAY_CONFIG["stage3_fine_tune_lr"] * pg.get("lr_mult", 1.0)

            running_loss = 0.0
            model.train()
            model.train()
            
            # 分布式训练：只有主进程显示进度条
            # 预计算总批次数，避免动态计算导致显示混乱
            total_batches = len(kgloader)
            pbar = tqdm.tqdm(kgloader, desc=f'Epoch {epoch+1}', total=total_batches, 
                           leave=True, disable=not is_main_process)
                
            for bi, batch in enumerate(pbar):
                h, r, t, neg_t, weight_batch = batch  # 5-tuple from TrainDataset
                # CPU-side negative sampling BEFORE .to(device) to avoid 6144 NPU?CPU .item() syncs per batch
                n_h, n_t = kgsampler.corrupt_batch(h, t, r)
                h, r, t = h.to(device), r.to(device), t.to(device)
                neg_t = neg_t.to(device)
                weight_batch = weight_batch.to(device)
                n_h, n_t = n_h.to(device), n_t.to(device)

                optimizer.zero_grad()

                # forward + backward
                pos, neg = model(h, t, r, n_h, n_t)

                # Section 3.3: softplus loss
                per_sample_loss = softplus_loss(pos, neg, n_neg=args.n_neg)

                # Sections 2.2 + 3.4: evidence-weighted loss with +1e-8 guard
                LAMBDA_N3 = 5e-3
                weighted_loss = (weight_batch * per_sample_loss).sum() / (weight_batch.sum() + 1e-8)

                # Section 3.1: N3 regularization on ComplEx real+imag embeddings
                model_core = model.module if is_distributed else model
                re_ent, im_ent, re_rel, im_rel = model_core.get_embeddings()
                n3_loss = (n3_regularization(re_ent[h], re_rel[r], re_ent[t], re_ent[neg_t]) +
                           n3_regularization(im_ent[h], im_rel[r], im_ent[t], im_ent[neg_t]))

                loss = weighted_loss + LAMBDA_N3 * n3_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()


                running_loss += loss.item()
                if is_main_process and (bi % 50 == 0 or bi == total_batches - 1):
                    batch_records.append(f'{epoch+1},{bi},{loss.item():.6f},{weighted_loss.item():.6f},{n3_loss.item():.6f},{pos.mean().item():.4f},{neg.mean().item():.4f}\n')
            
            # 关闭进度条
            if is_main_process and hasattr(pbar, 'close'):
                pbar.close()
            
            # 分布式训练时同步loss
            if is_distributed:
                loss_tensor = torch.tensor([running_loss], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                running_loss = loss_tensor.item() / world_size
            
            train_loss = running_loss
            if is_main_process:
                print(
                'Epoch {} | train loss: {:.5f}'.format(epoch + 1,
                                                    train_loss))  
            
            # 分布式训练：评估前同步所有进程，确保模型状态一致
            if is_distributed:
                dist.barrier()
            
            # eval
            if is_main_process:
                model_to_eval = model.module if is_distributed else model
                MR,MRR,Hit10,Hit30,Hit100 = unbiased_evaluator('ComplEx',model_to_eval,
                                                        valid,
                                                        ent2id,rel2id,
                                                        pro2nc)

                train_logger.add_scalar("Hits@100", Hit100, epoch+1)
                train_logger.add_scalar("MRR", MRR, epoch+1)
                train_logger.add_scalar("MR", MR, epoch+1)
                train_logger.add_scalar("train_loss", train_loss, epoch+1)

                print('Epoch {} | valid:'.format(epoch + 1))
                print('MR {} | MRR: {} | Hits@10:{} | Hits@30:{} | Hits@100: {}'.format(MR,
                                                                                MRR,
                                                                                Hit10,
                                                                                Hit30,
                                                                                Hit100))

                # Section 5.3: per-epoch few-shot relation report
                train_triples_list = list(zip(df['from'], df['rel'], df['to']))
                valid_subtype_list = list(zip(
                    five_fold_subtype_valid[i]['from'],
                    five_fold_subtype_valid[i]['rel'],
                    five_fold_subtype_valid[i]['to']))
                valid_subdisease_list = list(zip(
                    five_fold_subdisease_valid[i]['from'],
                    five_fold_subdisease_valid[i]['rel'],
                    five_fold_subdisease_valid[i]['to']))

                log_fewshot_report(model_to_eval,
                                   valid_subtype_list,
                                   valid_subdisease_list,
                                   ent2id, rel2id, len(ent2id), device, epoch + 1)

                # Section 5.2: few-shot gap monitoring
                flagged = check_fewshot_gap(
                    model_to_eval,
                    train_triples_list,
                    valid_subtype_list + valid_subdisease_list,
                    ent2id, rel2id, len(ent2id), device, epoch + 1)
                if flagged:
                    fewshot_patients += 1
                else:
                    fewshot_patients = 0

                # 每个 epoch 保存 checkpoint（断点重续用）
                val_metrics = {'MR': MR, 'MRR': MRR, 'Hits@10': Hit10, 'Hits@30': Hit30, 'Hits@100': Hit100}
                save_checkpoint(checkpoint_path, model, optimizer, epoch, best_mrr, patients, val_metrics, is_main_process)
            else:
                MR,MRR,Hit10,Hit30,Hit100 = 0,0,0,0,0
            
            # 分布式训练：评估后同步，确保主进程完成评估后所有进程一起继续
            if is_distributed:
                dist.barrier()
            
            # TEST
            # _ = tester('ComplEx',model,
            #             args,
            #             test,
            #             ent2id,rel2id,
            #             h_cand,t_cand,
            #             args.task)
                
            if epoch > args.warm_up:
                if MRR > best_mrr:  # MRR is used as ER metrics
                    best_mrr = MRR
                    patients = 0
                    if args.save_model and is_main_process:
                        # 分布式训练时保存原始模型
                        model_to_save = model.module if is_distributed else model
                        torch.save(model_to_save.state_dict(), args.save_model_path + "pertkg{}.pt".format(i))

                else:
                    patients += 1

                if patients >= args.patients:
                    print('Early stopping at epoch {} (CPI MRR plateau)'.format(epoch + 1))
                    break
                if fewshot_patients >= 3:
                    print('Early stopping at epoch {} (few-shot gap > 0.3 for 3 epochs)'.format(epoch + 1))
                    break

        if is_main_process:
            # 一次性写盘所有 per-batch 记录（消除训练中的 I/O 停顿）
            log_path = f'../pertkge_full/output/per_batch_log_split{i}.csv'
            with open(log_path, 'w') as f:
                f.write('epoch,batch,loss,wloss,n3_loss,pos_mean,neg_mean')

                f.writelines(batch_records)
            if train_logger:
                train_logger.flush()

        if args.mode == 'reproduce' and is_main_process:
            # report test metrics according to task
            print('split_{} testing now!!!'.format(i))
            model_to_load = model.module if is_distributed else model
            model_to_load.load_state_dict(torch.load(args.save_model_path + "pertkg{}.pt".format(i), map_location=device))

            metrics = tester('ComplEx',model_to_load,
                            args,
                            test,
                            ent2id,rel2id,
                            h_cand,t_cand,
                            args.task)
            results.append(metrics)
            # 更新进度：标记此 split 已完成
            completed_splits.append(i)
            save_progress(progress_file, completed_splits, -1, is_main_process)
            print('_'*50)
    
    if args.mode == 'reproduce':
        # report mean±std
        print('report mean±std testing results using 5 trained model!!!')
        if args.task == 'target_inference':
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
    s = time()

    # 检测是否使用分布式训练（通过环境变量RANK判断）
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        import sys
        args_distributed = ['--distributed'] + sys.argv[1:]
        args, args_dict = parse_args(args_distributed)
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        args_dict['local_rank'] = args.local_rank
        # 初始化分布式环境
        setup_distributed()
        is_distributed = True
    else:
        args, args_dict = parse_args()
        is_distributed = False

    print('print args_dict!!!')
    print(args_dict)
    print('_'*50)

    # 分布式训练时，每个进程使用不同的NPU设备
    device = torch.device(f'npu:{args.local_rank}')
    print(f"-- model will run on {device}")
    
    set_seeds(args.seed)
    # save model
    if args.save_model and (not is_distributed or args.local_rank == 0):
        if not os.path.exists(args.save_model_path):
                os.makedirs(args.save_model_path)
    # log
    if args.run_name and (not is_distributed or args.local_rank == 0):
        if not os.path.exists('../outlog/{}/'.format(args.run_name)):
                os.makedirs('../outlog/{}/'.format(args.run_name))

    if not is_distributed or args.local_rank == 0:
        print('traing and testing using five-fold cross validation stategy!!!')
        print('_'*50)
    
    five_fold_cv(args)

    # 清理分布式环境
    if is_distributed:
        cleanup_distributed()

    if not is_distributed or args.local_rank == 0:
        e = time()
        print(f"Total running time: {round(e - s, 2)}s")
