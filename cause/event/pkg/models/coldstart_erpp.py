"""
Cold-Start for NEC: TTF + LoRA-TTF
====================================
方案 A: ColdStartTTF — 标准 ERPP 训练, 推理时对新类型 v 做梯度下降
方案 B: ColdStartSVD — 低秩分解 W·c, 推理时只优化 8 维 c
"""
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rnn import ExplainableRecurrentPointProcess
from ..utils.misc import AverageMeter
from ..utils.torch import generate_sequence_mask


# ════════════════════════════════════════════════════════════
# ColdStartTTF: 测试时微调 (Test-Time Finetuning)
# ════════════════════════════════════════════════════════════

class ColdStartTTF(ExplainableRecurrentPointProcess):
    """
    测试时微调冷启动。不改变训练流程。

    累积式微调:
      首次出现 → 随机 v → 5 步梯度下降
      后续出现 → 当前 v → 1 步梯度下降（累积信息）
    """

    def __init__(self, ttf_steps: int = 5, ttf_lr: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.ttf_steps = ttf_steps
        self.ttf_lr = ttf_lr
        self._seen: set[int] = set()
        self._ttf_done: set[int] = set()
        self._ttf_enabled: bool = False  # RCA 推理时设为 True

    def mark_seen(self, types: set[int]):
        self._seen |= types

    def forward(self, event_seqs, event_type='category',
                need_weights=True, target_type=-1, device=None):
        """每条序列独立 TTF: 不同序列 → 不同上下文 → 不同 v."""
        if self._ttf_enabled and event_type == 'category':
            cold_k = [k for k in event_seqs[:, :, 1].long().unique().tolist()
                      if k not in self._seen and k < self.current_n_types]
            if cold_k:
                dev = event_seqs.device
                self._ttf_done.clear()  # 新序列, 重新开始
                for k in cold_k:
                    self.refine_v(event_seqs, k, dev)
        return super().forward(event_seqs, event_type, need_weights, target_type, device)

    def refine_v(self, event_seqs: torch.Tensor, k: int, device: torch.device):
        """
        对新类型 k 做一步梯度下降，不断累积信息。
        event_seqs: [1, T, 2]
        """
        d = self.embedding_dim
        is_first = k not in self._ttf_done
        self._ttf_done.add(k)

        if is_first:
            v = nn.Parameter(torch.randn(d, device=device) * 0.02)
            n_steps = self.ttf_steps
        else:
            v = nn.Parameter(self.embed[str(k)].data.clone())
            n_steps = 1

        # 正则参考: 已知类型嵌入的均值, 防止 v 偏离太远
        known_vecs = torch.stack([self.embed[str(t)].data
                                   for t in range(self.current_n_types)
                                   if t in self._seen], dim=0)
        v_prior = known_vecs.mean(dim=0) if len(known_vecs) > 0 else torch.zeros(d, device=device)

        opt = torch.optim.Adam([v], lr=self.ttf_lr)

        self.eval()
        for _ in range(n_steps):
            orig = self.embed[str(k)].data.clone()
            self.embed[str(k)].data = v.data

            log_ints, _ = super().forward(event_seqs, need_weights=True, event_type='category')
            mask_k = (event_seqs[:, :, 1].long() == k)
            nll = -log_ints[:, :, k][mask_k].mean()
            reg = ((v - v_prior).pow(2).sum()) * 0.01   # 正则: 不偏离已知嵌入中心太远
            loss = nll + reg

            opt.zero_grad()
            loss.backward()
            opt.step()

            self.embed[str(k)].data = orig

        self.embed[str(k)].data = v.detach()
        return v.detach()


# ════════════════════════════════════════════════════════════
# ColdStartSVD: 推理时 SVD 分解 + TTF
# ════════════════════════════════════════════════════════════

class ColdStartSVD(ExplainableRecurrentPointProcess):
    """
    训练: 标准 ERPP (完全不变), 64 维嵌入
    推理: 对训练好的嵌入做 SVD → V ≈ W @ C
          已知类型用 W@c_k 重建, 新类型冻结 W 只优化 c_k (8 维)
    """

    def __init__(self, rank: int = 8, ttf_steps: int = 5, ttf_lr: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.ttf_steps = ttf_steps
        self.ttf_lr = ttf_lr
        self._seen: set[int] = set()
        self._ttf_done: set[int] = set()
        self._ttf_enabled: bool = False
        self._decomposed: bool = False

    def decompose_embeddings(self):
        """对训练好的嵌入做 SVD 分解, 构建 W 和 c_k."""
        if self._decomposed:
            return
        with torch.no_grad():
            vecs = torch.stack([self.embed[str(t)].data
                                for t in range(self.current_n_types)], dim=1)  # [d, N]
            U, S, Vt = torch.linalg.svd(vecs, full_matrices=False)
            r = min(self.rank, len(S))
            self.W = nn.Parameter(U[:, :r] @ torch.diag(S[:r]))  # [d, r]
            for t in range(self.current_n_types):
                self.embed[str(t)] = nn.Parameter(Vt[:r, t].clone())  # c_k = [r]
        self._decomposed = True

    def mark_seen(self, types: set[int]):
        self._seen |= types

    def forward(self, event_seqs, event_type='category',
                need_weights=True, target_type=-1, device=None):
        """每条序列独立 LoRA-TTF: 不同序列 → 不同上下文 → 不同 v."""
        if self._ttf_enabled and event_type == 'category':
            cold_k = [k for k in event_seqs[:, :, 1].long().unique().tolist()
                      if k not in self._seen and k < self.current_n_types]
            if cold_k:
                dev = event_seqs.device
                self._ttf_done.clear()
                for k in cold_k:
                    self.refine_c(event_seqs, k, dev)
        return super().forward(event_seqs, event_type, need_weights, target_type, device)

    def update_event_type(self, event_type, device):
        """c_k 是 rank 维, 不是 embedding_dim 维."""
        self.embed[str(event_type)] = nn.Parameter(
            torch.randn(self.rank, device=device) * 0.02
        )
        self.log_intensities_prior[str(event_type)] = nn.Parameter(
            torch.zeros(1, device=device), requires_grad=True
        )
        self.max_type_index = max(int(event_type), self.max_type_index)
        if self.optim is not None:
            self.optim.add_param_group({'params': self.log_intensities_prior[str(event_type)]})
            self.optim.add_param_group({'params': self.embed[str(event_type)]})

    def event_type2embedding(self, event_seqs, device=None):
        """原版格式: [B, T, d+1], 第 0 列为时间戳, 第 1: 列为 W @ c_k."""
        if device is None:
            device = self.get_model_device()
        embedding_seqs = []
        for seq in event_seqs:
            embedding_seqs.append(
                torch.stack([
                    torch.cat([
                        torch.FloatTensor([t]).to(device),
                        self.W @ self.setdefault_embed(int(event_type), device),
                    ]) if int(event_type) < self.current_n_types else
                    torch.cat([
                        torch.FloatTensor([t]).to(device),
                        torch.zeros(self.embedding_dim, device=device),
                    ])
                    for t, event_type in seq
                ], dim=0)
            )
        return torch.stack(embedding_seqs, dim=0)

    def return_all_parameters(self, dim=1):
        cs = torch.stack([self.embed[str(t)] for t in range(self.current_n_types)], dim=1)  # [r, n]
        result = self.W @ cs  # [d, n]
        if dim == 0:
            return result.T  # [n, d]
        return result  # [d, n]

    def refine_c(self, event_seqs: torch.Tensor, k: int, device: torch.device):
        """
        两阶段微调:
          Stage 1: 在 SVD 子空间优化 c_k (r 维), 收敛快
          Stage 2: 展开 v=W@c, 在全空间优化 v (d 维), 覆盖子空间外的模式
        """
        d = self.embedding_dim
        is_first = k not in self._ttf_done
        self._ttf_done.add(k)

        self.eval()
        if is_first:
            # Stage 1: 子空间优化 (r 维, 3 步)
            c = nn.Parameter(torch.randn(self.rank, device=device) * 0.02)
            opt_c = torch.optim.Adam([c], lr=self.ttf_lr)
            for _ in range(3):
                orig_c = self.embed[str(k)].data.clone()
                self.embed[str(k)].data = c.data
                log_ints, _ = super().forward(event_seqs, need_weights=True, event_type='category')
                mask_k = (event_seqs[:, :, 1].long() == k)
                loss = -log_ints[:, :, k][mask_k].mean()
                opt_c.zero_grad(); loss.backward(); opt_c.step()
                self.embed[str(k)].data = orig_c

            # Stage 2: 全空间优化 (d 维, 2 步), 从 W@c 初始化
            v = nn.Parameter((self.W @ c.detach()).clone())
            opt_v = torch.optim.Adam([v], lr=self.ttf_lr * 0.5)
            with torch.no_grad():
                self.embed[str(k)] = nn.Parameter(torch.randn(d, device=device) * 0.02)
            for _ in range(2):
                orig_v = self.embed[str(k)].data.clone()
                self.embed[str(k)].data = v.data
                log_ints, _ = super().forward(event_seqs, need_weights=True, event_type='category')
                mask_k = (event_seqs[:, :, 1].long() == k)
                loss = -log_ints[:, :, k][mask_k].mean()
                opt_v.zero_grad(); loss.backward(); opt_v.step()
                self.embed[str(k)].data = orig_v

            self.embed[str(k)].data = v.detach()
            return v.detach()
        else:
            # 后续出现: 全空间 1 步微调
            v = nn.Parameter(self.embed[str(k)].data.clone())
            opt_v = torch.optim.Adam([v], lr=self.ttf_lr * 0.5)
            orig_v = self.embed[str(k)].data.clone()
            self.embed[str(k)].data = v.data
            log_ints, _ = super().forward(event_seqs, need_weights=True, event_type='category')
            mask_k = (event_seqs[:, :, 1].long() == k)
            loss = -log_ints[:, :, k][mask_k].mean()
            opt_v.zero_grad(); loss.backward(); opt_v.step()
            self.embed[str(k)].data = orig_v
            self.embed[str(k)].data = v.detach()
            return v.detach()
