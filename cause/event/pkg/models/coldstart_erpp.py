"""
Cold-Start for NEC: TTF + LoRA-TTF
====================================
方案 A: ColdStartTTF — 标准 ERPP 训练, 推理时对新类型 v 做梯度下降
方案 B: ColdStartLoRA — 低秩分解 W·c, 推理时只优化 8 维 c
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
# ColdStartLoRA: 低秩分解 + TTF
# ════════════════════════════════════════════════════════════

class ColdStartLoRA(ExplainableRecurrentPointProcess):
    """
    低秩 TTF: 标准 ERPP 训练, 推理时对已学嵌入做 SVD 分解出低秩基,
    新类型在低秩空间做 TTF (16 维), 再映射回 64 维。
    训练与 ERPP 完全一致, 不引入额外参数。
    """

    def __init__(self, rank: int = 16, ttf_steps: int = 5, ttf_lr: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.ttf_steps = ttf_steps
        self.ttf_lr = ttf_lr
        self._seen: set[int] = set()
        self._ttf_done: set[int] = set()
        self._ttf_enabled: bool = False
        self.W: torch.Tensor | None = None     # 推理时从已知嵌入 SVD 得到
        self.c_buf: dict[str, torch.Tensor] = {}  # 已知类型的 c_k 缓存

    def mark_seen(self, types: set[int]):
        self._seen |= types

    def _build_lora_basis(self):
        """SVD 分解已知类型嵌入表 → W, c_k. 只在首次 TTF 时调用一次."""
        if self.W is not None:
            return
        device = self.get_model_device()
        known = [t for t in range(self.current_n_types) if t in self._seen]
        V = torch.stack([self.embed[str(t)].data for t in known], dim=0)  # [K, d]
        U, S, Vt = torch.linalg.svd(V.float(), full_matrices=False)
        r = min(self.rank, len(known), len(S))
        # V ≈ U[:,:r] @ diag(S[:r]) @ Vt[:r,:] → W = Vt[:r,:].T [d,r], c_k = diag(S) @ U[k,:r]^T [r]
        self.W = Vt[:r, :].T.to(device)  # [d, r]
        for i, t in enumerate(known):
            self.c_buf[str(t)] = (torch.diag(S[:r]) @ U[i, :r]).to(device)  # [r]

    def forward(self, event_seqs, event_type='category',
                need_weights=True, target_type=-1, device=None):
        """前向传播前对冷启动类型做低秩 TTF."""
        if self._ttf_enabled and event_type == 'category':
            self._build_lora_basis()
            cold_k = [k for k in event_seqs[:, :, 1].long().unique().tolist()
                      if k not in self._seen and k < self.current_n_types]
            if cold_k:
                dev = event_seqs.device
                self._ttf_done.clear()
                for k in cold_k:
                    self.refine_c(event_seqs, k, dev)
        return super().forward(event_seqs, event_type, need_weights, target_type, device)

    def refine_c(self, event_seqs: torch.Tensor, k: int, device: torch.device):
        """低秩 TTF: 优化 r 维 c, v = W @ c 映射回 d 维."""
        r = self.W.size(1)
        is_first = k not in self._ttf_done
        self._ttf_done.add(k)

        if is_first:
            c = nn.Parameter(torch.randn(r, device=device) * 0.02)
            n_steps = self.ttf_steps
        else:
            c = nn.Parameter(self.c_buf.get(str(k), torch.randn(r, device=device) * 0.02))
            n_steps = 1

        known_vecs = torch.stack([self.embed[str(t)].data
                                   for t in range(self.current_n_types)
                                   if t in self._seen], dim=0)
        v_prior = known_vecs.mean(dim=0) if len(known_vecs) > 0 else torch.zeros(self.embedding_dim, device=device)

        opt = torch.optim.Adam([c], lr=self.ttf_lr)
        self.eval()
        for _ in range(n_steps):
            v = self.W @ c  # [d]
            orig = self.embed[str(k)].data.clone()
            self.embed[str(k)].data = v

            log_ints, _ = super().forward(event_seqs, need_weights=True, event_type='category')
            mask_k = (event_seqs[:, :, 1].long() == k)
            nll = -log_ints[:, :, k][mask_k].mean()
            reg = ((v - v_prior).pow(2).sum()) * 0.01
            loss = nll + reg

            opt.zero_grad()
            loss.backward()
            opt.step()

            self.embed[str(k)].data = orig

        # 持久化最终的 v
        final_v = (self.W @ c).detach()
        self.embed[str(k)].data = final_v
        self.c_buf[str(k)] = c.detach()
        return final_v
