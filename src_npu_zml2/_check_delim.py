import csv, os

base = r'e:/projects/大创1/src_npu/kg数据3'

files = ['human_gene_pathway_filtered.txt', 'subtype_epilepsy.txt', 'subdisease_gene.txt',
         'cause.txt', 'process.txt', 'effect.txt']

for fn in files:
    fp = os.path.join(base, fn)
    if not os.path.exists(fp):
        print(f'=== {fn}: NOT FOUND')
        continue
    with open(fp, encoding='utf-8', errors='replace') as f:
        lines = [f.readline() for _ in range(3)]
    print(f'=== {fn} ({os.path.getsize(fp)} bytes) ===')
    for i, l in enumerate(lines):
        print(f'  line{i}: {repr(l)}')
        parts = l.rstrip('\n').split('\t')
        if len(parts) > 1:
            print(f'    tab_split({len(parts)}): {parts}')
    print()
