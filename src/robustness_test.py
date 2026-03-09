import torch
import random
import numpy as np
import pandas as pd
from torchkge.models import DistMultModel
from utils import read_files, tester, set_seeds

# --- 1. Path Configuration ---
WEIGHT_PATH = '../best_model/target_inference_1/best_fold_0.pt' 
PROCESSED_DATA = '../processed_data/target_inference_1/'

def run_multi_ratio_robustness():
    class Args:
        cause_file = "../processed_data/target_inference_1/cause.txt"
        process_file = "../processed_data/knowledge_graph/process.txt"
        effect_file = "../processed_data/target_inference_1/effect.txt"
        test_file = "../processed_data/target_inference_1/test.txt"
        load_processed_data = True
        processed_data_file = PROCESSED_DATA
        h_dim = 300 

    args = Args()
    set_seeds(43) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data and model weights...")
    _, _, test_df, ent2id, rel2id, _, h_cand, t_cand = read_files(args)

    model = DistMultModel(args.h_dim, len(ent2id), len(rel2id)).to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    model.eval()

    # Define different perturbation ratios to test
    ratios = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    results = []

    print("\nStarting Multi-Ratio Perturbation Experiment...")
    all_target_ids = list(t_cand)

    for ratio in ratios:
        print(f"\n>>> Testing Perturbation Ratio: {ratio*100:.0f}%")
        
        # 0.0 is the original baseline
        if ratio == 0.0:
            metrics = tester('DistMult', model, args, test_df.copy(), 
                            ent2id, rel2id, h_cand, t_cand)
        else:
            perturbed_df = test_df.copy()
            for i in range(len(perturbed_df)):
                if random.random() < ratio: 
                    perturbed_df.iloc[i, 2] = random.choice(all_target_ids)
            
            metrics = tester('DistMult', model, args, perturbed_df, 
                            ent2id, rel2id, h_cand, t_cand)
        
        # metrics[0] is Top-10, metrics[2] is Top-30, metrics[4] is Top-100
        results.append({
            'Ratio': ratio,
            'Top-10': metrics[0],
            'Top-30': metrics[2],
            'Top-100': metrics[4]
        })

    # Print summary table
    print("\n" + "="*60)
    print(f"{'Ratio (%)':<15} | {'Top-10':<12} | {'Top-30':<12} | {'Top-100':<12}")
    print("-" * 60)
    for res in results:
        print(f"{res['Ratio']*100:<15.0f} | {res['Top-10']:<12.4f} | {res['Top-30']:<12.4f} | {res['Top-100']:<12.4f}")
    print("="*60)
    print("Experiment complete. Use these values to plot your sensitivity curve.")

if __name__ == "__main__":
    run_multi_ratio_robustness()
