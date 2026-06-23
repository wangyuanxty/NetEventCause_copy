"""
Cold-Start for NEC: TTF + LoRA-TTF
====================================
方案 A: ColdStartTTF — 标准 ERPP 训练, 推理时对新类型 v 做梯度下降
方案 B: ColdStartLoRA — v_k = W·c_k 低秩嵌入, 推理时只优化 c_k
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
# ColdStartTTF: 测试时微调
# ════════════════════════════════════════════════════════════

class ColdStartTTF(ExplainableRecurrentPointProcess):
    """标准 ERPP 训练, 推理时对新类型 v 做累积式梯度下降."""

    def __init__(self, ttf_steps: int = 5, ttf_lr: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.ttf_steps = ttf_steps
        self.ttf_lr = ttf_lr
        self._seen: set[int] = set()
        self._ttf_done: set[int] = set()
        self._ttf_enabled: bool = False

    def mark_seen(self, types: set[int]):
        self._seen |= types

    def forward(self, event_seqs, event_type='category',
                need_weights=True, target_type=-1, device=None):
        if self._ttf_enabled and event_type == 'category':
            cold_k = [k for k in event_seqs[:, :, 1].long().unique().tolist()
                      if k not in self._seen and k < self.current_n_types]
            if cold_k:
                dev = event_seqs.device
                self._ttf_done.clear()
                for k in cold_k:
                    self.refine_v(event_seqs, k, dev)
        return super().forward(event_seqs, event_type, need_weights, target_type, device)

    def refine_v(self, event_seqs: torch.Tensor, k: int, device: torch.device):
        d = self.embedding_dim
        is_first = k not in self._ttf_done
        self._ttf_done.add(k)
        if is_first:
            v = nn.Parameter(torch.randn(d, device=device) * 0.02)
            n_steps = self.ttf_steps
        else:
            v = nn.Parameter(self.embed[str(k)].data.clone())
            n_steps = 1

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
            reg = ((v - v_prior).pow(2).sum()) * 0.01
            loss = nll + reg
            opt.zero_grad()
            loss.backward()
            opt.step()
            self.embed[str(k)].data = orig
        self.embed[str(k)].data = v.detach()
        return v.detach()


# ════════════════════════════════════════════════════════════
# ColdStartLoRA: 低秩嵌入 v_k = W @ c_k
# ════════════════════════════════════════════════════════════

class ColdStartLoRA(ExplainableRecurrentPointProcess):
    """
    推理时低秩 TTF。训练 = 标准 ERPP, 不学 W/c。
    推理时用梯度下降拟合低秩分解 W·c_k ≈ v_k, 然后冻结 W, TTF 新类型的 c。
    """

    def __init__(self, rank: int = 16, ttf_steps: int = 5, ttf_lr: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.ttf_steps = ttf_steps
        self.ttf_lr = ttf_lr
        self._seen: set[int] = set()
        self._ttf_done: set[int] = set()
        self._ttf_enabled: bool = False
        self.W: torch.Tensor | None = None

    def mark_seen(self, types: set[int]):
        self._seen |= types

    def _build_lora(self, lora_epochs: int = 200):
        """梯度下降拟合 W·c_k ≈ v_k for known types. 不修改 v_k."""
        if self.W is not None:
            return
        device = self.get_model_device()
        known = [t for t in range(self.current_n_types) if t in self._seen]
        V = torch.stack([self.embed[str(t)].data for t in known], dim=0)  # [K, d]
        r = min(self.rank, len(known))
        W = nn.Parameter(torch.randn(self.embedding_dim, r, device=device) * 0.02)
        C = nn.Parameter(torch.randn(r, len(known), device=device) * 0.02)
        opt = torch.optim.Adam([W, C], lr=0.01)
        for _ in range(lora_epochs):
            loss = ((W @ C).T - V).pow(2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.W = W.detach()
        self.c_map: dict[int, torch.Tensor] = {known[i]: C[:, i].detach() for i in range(len(known))}

    def forward(self, event_seqs, event_type='category',
                need_weights=True, target_type=-1, device=None):
        if self._ttf_enabled and event_type == 'category':
            self._build_lora()
            cold_k = [k for k in event_seqs[:, :, 1].long().unique().tolist()
                      if k not in self._seen and k < self.current_n_types]
            if cold_k:
                dev = event_seqs.device
                self._ttf_done.clear()
                for k in cold_k:
                    self.refine_c(event_seqs, k, dev)
        return super().forward(event_seqs, event_type, need_weights, target_type, device)

    def refine_c(self, event_seqs: torch.Tensor, k: int, device: torch.device):
        """冻结 W, 只优化 r 维 c_k."""
        r = self.W.size(1)
        is_first = k not in self._ttf_done
        self._ttf_done.add(k)
        if is_first:
            c = nn.Parameter(torch.randn(r, device=device) * 0.02)
            n_steps = self.ttf_steps
        else:
            c = nn.Parameter(self.c_map.get(k, torch.randn(r, device=device) * 0.02))
            n_steps = 1

        known_vecs = torch.stack([self.embed[str(t)].data
                                   for t in range(self.current_n_types)
                                   if t in self._seen], dim=0)
        v_prior = known_vecs.mean(dim=0) if len(known_vecs) > 0 else torch.zeros(self.embedding_dim, device=device)

        opt = torch.optim.Adam([c], lr=self.ttf_lr)
        self.eval()
        for _ in range(n_steps):
            v = self.W @ c
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
        final_v = (self.W @ c).detach()
        self.embed[str(k)].data = final_v
        self.c_map[k] = c.detach()
        return final_v
