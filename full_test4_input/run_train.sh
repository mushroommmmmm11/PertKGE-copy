#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/hccl/lib64:$LD_LIBRARY_PATH
export PATH=/root/miniconda3/envs/D910_PyTorch2.1/bin:$PATH
export PYTHONUNBUFFERED=1
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
cd /workspace/PertKGE/src_npu_zml2

tensorboard --logdir=/workspace/PertKGE/outlog --host=0.0.0.0 --port=6006 \
    > /tmp/tensorboard.log 2>&1 &

torchrun --nproc_per_node=8 --master_port=29500 train_pertkge.py \
    --distributed \
    --cause_file   ../full_test4/input/cause.txt \
    --process_file ../full_test4/input/process.txt \
    --effect_file  ../full_test4/input/effect.txt \
    --test_file    ../full_test4/input/test.txt \
    --pathway_extra_file ../full_test4/input/human_gene_pathway_filtered.txt \
    --subtype_file ../full_test4/input/subtype_epilepsy.txt \
    --subdisease_gene_file ../full_test4/input/subdisease_gene.txt \
    --h_dim 300 \
    --batch_size 2048 \
    --n_neg 100 \
    --patients 5 \
    --seed 43 \
    --run_name full_test4 \
    --save_model_path ../full_test4/output/model/ \
    --overwrite \
    > ../full_test4/output/train.log 2>&1 &