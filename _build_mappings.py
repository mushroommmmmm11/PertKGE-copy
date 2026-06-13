import os, sys
from easydict import EasyDict
import pandas as pd, numpy as np

sys.path.insert(0, r'e:\projects\大创1\src_npu')
from utils import read_files, set_seeds

args = EasyDict()
data_dir = r'e:\projects\大创1\src_npu\kg数据3'
args.cause_file = os.path.join(data_dir, 'cause.txt')
args.process_file = os.path.join(data_dir, 'process.txt')
args.effect_file = os.path.join(data_dir, 'effect.txt')
args.test_file = os.path.join(data_dir, 'test.txt')
args.pathway_extra_file = os.path.join(data_dir, 'human_gene_pathway_filtered.txt')
args.subtype_file = os.path.join(data_dir, 'subtype_epilepsy.txt')
args.subdisease_gene_file = os.path.join(data_dir, 'subdisease_gene.txt')
args.processed_data_file = r'e:\projects\大创1\processed_data\deepce\\'
args.load_processed_data = False
args.seed = 43

os.makedirs(args.processed_data_file, exist_ok=True)
set_seeds(43)

cause, pertkg_wo_cause, pertkg_wo_cause_global, test, ent2id, rel2id, pro2nc, h_cand, t_cand, subtype, subdisease_gene, edge_evidence_dict = read_files(args)
print('Success!')
print('ent2id: %d entities, rel2id: %d relations' % (len(ent2id), len(rel2id)))

new_ents = [k for k in ent2id if k.startswith('subtype:') or k == 'epilepsy']
print('New entities (subtype:, epilepsy): %d' % len(new_ents))
for e in sorted(new_ents)[:10]:
    print('  %s -> %d' % (e, ent2id[e]))

print('is_subtype_of id:', rel2id.get('is_subtype_of'))
print('subdisease_regulates id:', rel2id.get('subdisease_regulates'))
print('participates_in id:', rel2id.get('participates_in'))
