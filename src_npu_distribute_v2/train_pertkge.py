import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from time import time
import pandas as pd
import numpy as np
import tqdm
import random
from collections import defaultdict
import argparse

import torch
import torch.nn as nn
# from torch import cuda
import torch_npu
from torch.optim import Adam
from torch.utils.data import Dataset
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
    parser.add_argument('--seed', type = int, default=43)
    parser.add_argument('--h_dim', type = int, default=300)
    parser.add_argument('--margin', type = float, default=1.0)
    parser.add_argument('--lr', type = float, default=1e-4)
    parser.add_argument('--wd', type = float, default=1e-5)
    parser.add_argument('--n_neg', type = int, default=100)
    parser.add_argument('--batch_size', type = int, default=2048)
    parser.add_argument('--warm_up', type = int, default=10)
    parser.add_argument('--patients', type = int, default=5)
    parser.add_argument('--use_cuda', type = str, default='batch')
    parser.add_argument('--nepoch', type = int, default=100)
    parser.add_argument('--save_model', action='store_true', default=True)
    parser.add_argument('--save_model_path',default="../best_model/deepce_distmult_2/")
    parser.add_argument('--load_processed_data', action='store_true', default=False)
    parser.add_argument('--processed_data_file',default="../processed_data/deepce/")
    parser.add_argument('--mode', default="reproduce", help = 'choose reproduce if user want to report testing results')  # test or not
    parser.add_argument('--task', default="target_inference", help="choose from ['target_inference', 'virtual_screening', 'unbiased_test']")
    parser.add_argument('--run_name', default="deepce_distmult", help="Name of the running.")
    parser.add_argument('--distributed', action='store_true', default=False, help='Enable distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')

    # +++++++++++++++++
    return parser.parse_args(args),parser.parse_args(args).__dict__


def five_fold_cv(args):
    # read cause, process, effect, test file
    cause, pertkg_wo_cause, test, ent2id, rel2id, pro2nc, h_cand, t_cand = read_files(args)
    
    # generate train\valid
    five_fold_train, five_fold_valid = generate_five_fold_files(args, cause)

    # 判断是否为分布式训练
    is_distributed = args.distributed
    if is_distributed:
        local_rank = args.local_rank
        world_size = dist.get_world_size()
        is_main_process = local_rank == 0
    else:
        is_main_process = True

    results = []
    for i in range(5):
        # load data to consrtuct kg
        if is_main_process:
            print('split_{}!!!'.format(i))
        
        # logger
        if is_main_process:
            train_logger = SummaryWriter('../outlog/{}_split{}'.format(args.run_name, i))
        else:
            train_logger = None

        # loading train and valid df
        train = five_fold_train[i]
        valid = five_fold_valid[i]

        if is_main_process:
            print('construct chemical perturbation profiles-based knowledge graph!!!')
        s1 = time()
        df = pd.concat([pertkg_wo_cause,train])
        df = df.sample(frac=1,random_state=42).reset_index(drop=True) # 打乱KG
        kg = KnowledgeGraph(df,ent2ix=ent2id,rel2ix=rel2id)
        e1 = time()
        if is_main_process:
            print(f"Total constructing time: {round(e1 - s1, 2)}s")
            print()

        if is_main_process:
            print('split_{} traing now!!!'.format(i))
        # choose method
        model = DistMultModel(args.h_dim, len(ent2id), len(rel2id))
        criterion = MarginLoss(args.margin)
        # if cuda.is_available():
        #     cuda.empty_cache()
        #     model.cuda()
        #     criterion.cuda()
        torch.npu.empty_cache()
        model = model.to(device)
        criterion = criterion.to(device)
        
        # 分布式训练：使用DDP包装模型
        if is_distributed:
            model = DDP(model, device_ids=[torch.device(f'npu:{local_rank}')])
        
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        kgsampler = BernoulliNegativeSampler(kg, n_neg=args.n_neg)
        
        # 分布式训练：使用 DistributedSampler 分配数据
        if is_distributed:
            sampler = DistributedSampler(kg, shuffle=True)
            kgloader = PyTorchDataLoader(kg, batch_size=args.batch_size, sampler=sampler)
        else:
            kgloader = PyTorchDataLoader(kg, batch_size=args.batch_size, shuffle=True)

        # wo train
        model_to_test = model.module if is_distributed else model
        _ = tester('DistMult',model_to_test,
                    args,
                    test,
                    ent2id,rel2id,
                    h_cand,t_cand,
                    args.task)

        # train
        best_mrr = 0
        patients = 0
        for epoch in range(args.nepoch):
            # 分布式训练：每个epoch设置sampler的epoch，确保shuffle一致
            if is_distributed:
                sampler.set_epoch(epoch)
            
            running_loss = 0.0
            model.train()
            
            # 分布式训练：只有主进程显示进度条
            # 预计算总批次数，避免动态计算导致显示混乱
            total_batches = len(kgloader)
            pbar = tqdm.tqdm(kgloader, desc=f'Epoch {epoch+1}', total=total_batches, 
                           leave=True, disable=not is_main_process)
                
            for batch in pbar:
                h, t, r = batch[0], batch[1], batch[2]
                h, t, r = h.to(device), t.to(device), r.to(device)
                n_h, n_t = kgsampler.corrupt_batch(h, t, r)
                n_h, n_t = n_h.to(device), n_t.to(device)

                optimizer.zero_grad()

                # forward + backward + optimize
                pos, neg = model(h, t, r, n_h, n_t)
                loss = criterion(pos, neg)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
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
                MR,MRR,Hit10,Hit30,Hit100 = unbiased_evaluator('DistMult',model_to_eval,
                                                        valid,
                                                        ent2id,rel2id,
                                                        pro2nc)

                train_logger.add_scalar("Hits@100", Hit100, epoch+1)

                print('Epoch {} | valid:'.format(epoch + 1))
                print('MR {} | MRR: {} | Hits@10:{} | Hits@30:{} | Hits@100: {}'.format(MR,
                                                                                MRR,
                                                                                Hit10,
                                                                                Hit30,
                                                                                Hit100))
            else:
                MR,MRR,Hit10,Hit30,Hit100 = 0,0,0,0,0
            
            # 分布式训练：评估后同步，确保主进程完成评估后所有进程一起继续
            if is_distributed:
                dist.barrier()
            
            # TEST
            # _ = tester('DistMult',model,
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
                    break

        if is_main_process and train_logger:
            train_logger.flush()

        if args.mode == 'reproduce' and is_main_process:
            # report test metrics according to task
            print('split_{} testing now!!!'.format(i))
            model_to_load = model.module if is_distributed else model
            model_to_load.load_state_dict(torch.load(args.save_model_path + "pertkg{}.pt".format(i), map_location=device))

            metrics = tester('DistMult',model_to_load,
                            args,
                            test,
                            ent2id,rel2id,
                            h_cand,t_cand,
                            args.task)
            results.append(metrics)
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
