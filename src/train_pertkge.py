import os
import re
from time import time
import pandas as pd
import numpy as np
import tqdm
import random
import argparse

import torch
import torch_npu  
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader as PyTorchDataLoader
from torch.utils.tensorboard import SummaryWriter

# Distributed training libraries
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
from torchkge import KnowledgeGraph, DistMultModel
from utils import *

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='PertKG Distributed')
    parser.add_argument('--cause_file', default="../processed_data/deepce/cause.txt")
    parser.add_argument('--process_file', default="../processed_data/knowledge_graph/process.txt")
    parser.add_argument('--effect_file', default="../processed_data/deepce/effect.txt")
    parser.add_argument('--test_file', default="../processed_data/deepce/test.txt")
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--h_dim', type=int, default=300)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--n_neg', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4096) 
    parser.add_argument('--warm_up', type=int, default=10)
    parser.add_argument('--patients', type=int, default=5)
    parser.add_argument('--nepoch', type=int, default=100)
    parser.add_argument('--save_model_path', default="../best_model/target_inference_1/")
    parser.add_argument('--load_processed_data', action='store_true', default=False)
    parser.add_argument('--processed_data_file', default="../processed_data/target_inference_1/")
    parser.add_argument('--mode', default="reproduce")
    parser.add_argument('--task', default="target_inference")
    parser.add_argument('--run_name', default="dist_target_1")
    
    # Distributed parameter
    parser.add_argument("--local_rank", type=int, default=-1)
    
    return parser.parse_args(args), parser.parse_args(args).__dict__

def five_fold_cv(args):
    # Initialize distributed environment
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if local_rank == -1:
        local_rank = 0
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend='hccl')
    device = torch.device(f'npu:{local_rank}')

    cause, pertkg_wo_cause, test, ent2id, rel2id, pro2nc, h_cand, t_cand = read_files(args)
    five_fold_train, five_fold_valid = generate_five_fold_files(args, cause)

    for i in range(5):
        if local_rank == 0:
            print(f'--- Starting Split {i} ---')
        
        train_logger = SummaryWriter(f'../outlog/{args.run_name}_{i}') if local_rank == 0 else None
        train_df = five_fold_train[i]
        valid_df = five_fold_valid[i]

        df=pd.concat([pertkg_wo_cause, train_df]).sample(frac=1, random_state=42).reset_index(drop=True)

        kg = KnowledgeGraph(df, ent2ix=ent2id, rel2ix=rel2id)

        # 1. Force 32-bit precision and load model
        model = DistMultModel(args.h_dim, len(ent2id), len(rel2id)).to(torch.float32).to(device)
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
        
        criterion = MarginLoss(args.margin).to(torch.float32).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr * 8, weight_decay=args.wd) 

        # 2. Distributed sampler
        sampler = DistributedSampler(kg)
        kgloader = PyTorchDataLoader(kg, batch_size=args.batch_size, sampler=sampler)
        kgsampler = BernoulliNegativeSampler(kg, n_neg=args.n_neg)

        best_mrr = 0
        patients = 0
        
        for epoch in range(args.nepoch):
            sampler.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            
            pbar = tqdm.tqdm(kgloader, desc=f"Rank {local_rank} Epoch {epoch+1}") if local_rank == 0 else kgloader
            for batch in pbar:
                h, t, r = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                n_h, n_t = kgsampler.corrupt_batch(h, t, r)

                optimizer.zero_grad()
                pos, neg = model(h, t, r, n_h, n_t)
                loss = criterion(pos, neg)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if local_rank == 0:
                print(f"Epoch {epoch+1} Loss: {running_loss:.4f}")
                # Evaluation logic on Rank 0
                MR, MRR, Hit10, Hit30, Hit100 = unbiased_evaluator('DistMult', model.module, valid_df, ent2id, rel2id, pro2nc)
                print(f"Valid MRR: {MRR:.5f}")

                if MRR > best_mrr:
                    best_mrr = MRR
                    patients = 0
                    torch.save(model.module.state_dict(), os.path.join(args.save_model_path, f"best_fold_{i}.pt"))
                else:
                    patients += 1
                
                if patients >= args.patients: break

    dist.destroy_process_group()

if __name__ == '__main__':
    args, _ = parse_args()
    set_seeds(args.seed)
    five_fold_cv(args)
