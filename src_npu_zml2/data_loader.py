"""Data loading utilities — stratified sampling & evidence weighting (Sections 2.1–2.2)."""

import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


EVIDENCE_WEIGHTS = {
    "Definitive": 1.0,
    "Strong": 0.8,
    "Moderate": 0.5,
    "Supportive": 0.45,
    "Limited": 0.2,
}




class TrainDataset(Dataset):
    """PyTorch Dataset with pre-computed per-triple sampling & loss weights.

    Section 2.1 — stratified quota sampling: W_base = 1 / sqrt(N_r) per relation.
    Section 2.2 — evidence-level weighting for ``subdisease_regulates`` edges.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns ``["from", "rel", "to"]``.
    ent2id : dict
        Entity name → integer ID.
    rel2id : dict
        Relation name → integer ID.
    edge_evidence_dict : dict, optional
        ``(head_str, rel_str, tail_str) → classification_level``.
        Only consulted for ``subdisease_regulates`` triples.
    """

    def __init__(self, df, ent2id, rel2id, edge_evidence_dict=None):
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.n_ent = len(ent2id)

        # 2.1: per-relation counts -> W_base (VECTORIZED)
        rel_counts = df["rel"].value_counts().to_dict()
        self.relation_global_weight = {
            r: 1.0 / np.sqrt(cnt) for r, cnt in rel_counts.items()
        }

        # VECTORIZED: build ID lists and weights
        self.h = df["from"].map(ent2id).fillna(0).astype(int).values.tolist()
        self.r = df["rel"].map(rel2id).fillna(0).astype(int).values.tolist()
        self.t = df["to"].map(ent2id).fillna(0).astype(int).values.tolist()
        rel_w = df["rel"].map(self.relation_global_weight).fillna(1.0)
        self.sampling_weights = rel_w.values.tolist()
        self.loss_weights = rel_w.copy().values.tolist()
        if edge_evidence_dict is not None and len(edge_evidence_dict) > 0:
            mask = df["rel"] == "subdisease_regulates"
            if mask.any():
                for idx in mask[mask].index:
                    row = df.loc[idx]
                    key = (row["from"], "subdisease_regulates", row["to"])
                    classification = edge_evidence_dict.get(key, "Supportive")
                    self.loss_weights[idx] *= EVIDENCE_WEIGHTS.get(classification, 0.45)

    def get_sampling_weights(self):
        """Return the per-sample weights for WeightedRandomSampler."""
        return self.sampling_weights

    def __getitem__(self, idx):
        h = self.h[idx]
        r = self.r[idx]
        t = self.t[idx]

        # Simple random negative tail (placeholder — Section 4 replaces this)
        neg_t = t
        while neg_t == t:
            neg_t = np.random.randint(0, self.n_ent)

        return (
            torch.tensor(h, dtype=torch.long),
            torch.tensor(r, dtype=torch.long),
            torch.tensor(t, dtype=torch.long),
            torch.tensor(neg_t, dtype=torch.long),
            torch.tensor(self.loss_weights[idx], dtype=torch.float32),
        )

    def __len__(self):
        return len(self.h)


class DistributedWeightedRandomSampler(Sampler):
    """
    分布式加权采样器：先全局加权采样，再轮询分发给各个 Rank。
    保证全局概率分布绝对正确，且各卡数据量一致防止 barrier 死锁。
    """
    def __init__(self, weights, num_samples, world_size, rank, replacement=True):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.world_size = world_size
        self.rank = rank
        self.replacement = replacement
        self.epoch = 0

        # 向上取整保证每卡数据量一致，防止 dist.barrier() 死锁
        self.local_num_samples = math.ceil(self.num_samples / self.world_size)
        self.total_size = self.local_num_samples * self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # 1. 全局加权采样：所有 rank 用相同 seed 生成完全一致的采样序列
        g = torch.Generator()
        g.manual_seed(self.epoch)
        global_indices = torch.multinomial(
            self.weights, self.total_size, self.replacement, generator=g
        ).tolist()

        # 2. 轮询分发：Rank 0 拿 [0,8,16...], Rank 1 拿 [1,9,17...]
        local_indices = global_indices[self.rank : self.total_size : self.world_size]
        assert len(local_indices) == self.local_num_samples
        return iter(local_indices)

    def __len__(self):
        return self.local_num_samples
