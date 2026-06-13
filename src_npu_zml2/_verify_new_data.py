"""Quick verification script for new integrated data."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

data_dir = r'e:\projects\大创1\processed_data\deepce'
ent2id = np.load(os.path.join(data_dir, 'ent2id.npy'), allow_pickle=True).item()
rel2id = np.load(os.path.join(data_dir, 'rel2id.npy'), allow_pickle=True).item()
pro2nc = np.load(os.path.join(data_dir, 'pro2nc.npy'), allow_pickle=True).item()

# Entity prefix counts
prefixes = {}
for k in ent2id:
    p = k.split(':')[0] if ':' in k else 'NO_PREFIX'
    prefixes[p] = prefixes.get(p, 0) + 1
print('=== ENTITY PREFIX COUNTS ===')
for p in sorted(prefixes, key=lambda x: -prefixes[x]):
    print(f'  {p}: {prefixes[p]}')
print(f'  Total: {len(ent2id)}')

# Relation summary
print('\n=== RELATIONS ===')
for k, v in sorted(rel2id.items(), key=lambda x: x[1]):
    print(f'  {v}: {k}')

# New data entities
print('\n=== NEW EPILEPSY ENTITIES ===')
subtypes = sorted([k for k in ent2id if k.startswith('subtype:')])
print(f'  subtype:xxx count: {len(subtypes)}')
print(f'  "epilepsy" in ent2id: {"epilepsy" in ent2id}')
print(f'  Sample subtypes: {subtypes[:5]}')

# Verify all subtype entities linked
print('\n=== NEW RELATION PRESENCE ===')
for r in ['is_subtype_of', 'subdisease_regulates', 'participates_in']:
    print(f'  {r}: id={rel2id.get(r, "MISSING!")}')
