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
        """前向传播前自动对冷启动类型做 TTF（仅 _ttf_enabled=True 且 category 模式时触发）"""
        if self._ttf_enabled and event_type == 'category':
            batch_k = event_seqs[:, :, 1].long().unique().tolist()
            cold_k = [k for k in batch_k if k not in self._seen and k < self.current_n_types]
            if cold_k:
                dev = event_seqs.device
                for k in cold_k:
                    self.refine_v(event_seqs, k, dev)
                self._seen |= set(cold_k)
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

        opt = torch.optim.Adam([v], lr=self.ttf_lr)

        self.eval()
        for _ in range(n_steps):
            orig = self.embed[str(k)].data.clone()
            self.embed[str(k)].data = v.data

            log_ints, _ = super().forward(event_seqs, need_weights=True, event_type='category')
            mask_k = (event_seqs[:, :, 1].long() == k)
            loss = -log_ints[:, :, k][mask_k].mean()

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
    低秩嵌入 + TTF.   v_k = W @ c_k,   W ∈ R^{d×r}, c_k ∈ R^r (r=8).

    训练: W + 所有已知 c_k 联合学习
    推理: 新类型 → W 冻结, c_k 做 TTF (只要 8 维, 收敛快)
    """

    def __init__(self, rank: int = 8, ttf_steps: int = 5, ttf_lr: float = 0.01, **kwargs):
        self.rank = rank  # 必须在 super() 之前, update_event_type 需要
        super().__init__(**kwargs)
        self.W = nn.Parameter(torch.randn(self.embedding_dim, rank) * 0.02)
        self.ttf_steps = ttf_steps
        self.ttf_lr = ttf_lr
        self._seen: set[int] = set()
        self._ttf_done: set[int] = set()
        self._ttf_enabled: bool = False

    def mark_seen(self, types: set[int]):
        self._seen |= types

    def forward(self, event_seqs, event_type='category',
                need_weights=True, target_type=-1, device=None):
        """前向传播前自动对冷启动类型做 LoRA-TTF（仅 _ttf_enabled=True 且 category 模式时触发）"""
        if self._ttf_enabled and event_type == 'category':
            batch_k = event_seqs[:, :, 1].long().unique().tolist()
            cold_k = [k for k in batch_k if k not in self._seen and k < self.current_n_types]
            if cold_k:
                dev = event_seqs.device
                for k in cold_k:
                    self.refine_c(event_seqs, k, dev)
                self._seen |= set(cold_k)
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
        """v_k = W @ c_k. 返回 [B, T, d+1], 第 0 列占位, 1: 为嵌入."""
        if device is None:
            device = self.get_model_device()
        B, T = event_seqs.size()[:2]
        e = torch.zeros(B, T, self.embedding_dim + 1,
                         device=device, dtype=torch.float)
        for k_str, c in self.embed.items():
            k = int(k_str)
            mask = (event_seqs[:, :, 1].long() == k)
            if mask.any():
                v = self.W @ c
                n = mask.sum().item()
                e[mask, 1:] = v.unsqueeze(0).expand(n, -1)
        return e

    def return_all_parameters(self, dim=1):
        device = self.W.device
        cs = torch.stack([self.embed[str(t)] for t in range(self.current_n_types)], dim=1)
        return self.W @ cs

    def refine_c(self, event_seqs: torch.Tensor, k: int, device: torch.device):
        """对新类型 k 做低秩 TTF: 冻结 W, 只优化 c_k (8 维)."""
        is_first = k not in self._ttf_done
        self._ttf_done.add(k)

        if is_first:
            c = nn.Parameter(torch.randn(self.rank, device=device) * 0.02)
            n_steps = self.ttf_steps
        else:
            c = nn.Parameter(self.embed[str(k)].data.clone())
            n_steps = 1

        opt = torch.optim.Adam([c], lr=self.ttf_lr)

        self.eval()
        for _ in range(n_steps):
            orig_c = self.embed[str(k)].data.clone()
            self.embed[str(k)].data = c.data

            log_ints, _ = super().forward(event_seqs, need_weights=True, event_type='category')
            mask_k = (event_seqs[:, :, 1].long() == k)
            loss = -log_ints[:, :, k][mask_k].mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            self.embed[str(k)].data = orig_c

        self.embed[str(k)].data = c.detach()
        return c.detach()
