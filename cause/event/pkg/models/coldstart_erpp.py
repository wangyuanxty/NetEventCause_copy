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
    低秩嵌入 + TTF.   v_k = W @ c_k.

    训练: 同标准 ERPP (无低秩约束).  推理前调用 decompose_from(erpp_checkpoint)
          对已训练嵌入做 SVD 分解得到 W 和 c_k, 之后 W 冻结, 冷启动只优化 c_k.
    """

    def __init__(self, rank: int = 16, ttf_steps: int = 5, ttf_lr: float = 0.01, **kwargs):
        self.rank = rank
        super().__init__(**kwargs)
        self.ttf_steps = ttf_steps
        self.ttf_lr = ttf_lr
        self._seen: set[int] = set()
        self._ttf_done: set[int] = set()
        self._ttf_enabled: bool = False
        self.W: nn.Parameter | None = None

        # 训练阶段使用标准 embedding (d 维), 不暴露 LoRA 结构
        self._decomposed = False

    def decompose_from(self, erpp_state_dict: dict):
        """从标准 ERPP checkpoint 分解嵌入表: V ∈ R^{d×n} → W @ C."""
        d = self.embedding_dim
        # 提取所有已知类型的嵌入
        n = len(erpp_state_dict) - sum('log_intensities_prior' in k for k in erpp_state_dict)
        # 收集嵌入矩阵
        vecs = []
        for key in sorted(erpp_state_dict.keys(),
                          key=lambda x: int(x) if x.isdigit() else -1):
            if key.isdigit():
                vecs.append(erpp_state_dict[key])
        if not vecs:
            return
        V = torch.stack(vecs, dim=1)  # [d, n]
        U, S, Vt = torch.linalg.svd(V.float(), full_matrices=False)
        r = min(self.rank, len(S))
        self.W = nn.Parameter(U[:, :r] @ torch.diag(S[:r]))   # [d, r]
        # 将分解后的 c_k 填入 self.embed
        for i, k in enumerate(sorted([k for k in erpp_state_dict if k.isdigit()],
                                      key=int)):
            self.embed[str(k)] = nn.Parameter(Vt[:r, i].clone())
        # 同时复制 log_intensities_prior
        for key in erpp_state_dict:
            if 'log_intensities_prior' in key:
                k = key.strip('log_intensities_prior')
                if k in self.log_intensities_prior:
                    self.log_intensities_prior[k].data.copy_(erpp_state_dict[key])
        self._decomposed = True

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
                        self.W @ self.embed[str(int(event_type))]
                        if self._decomposed and int(event_type) < self.current_n_types
                        else self.setdefault_embed(int(event_type), device),
                    ])
                    for t, event_type in seq
                ], dim=0)
            )
        return torch.stack(embedding_seqs, dim=0)

    def return_all_parameters(self, dim=1):
        if not self._decomposed:
            return super().return_all_parameters(dim)
        cs = torch.stack([self.embed[str(t)] for t in range(self.current_n_types)], dim=1)
        result = self.W @ cs
        if dim == 0:
            return result.T
        return result

    def refine_c(self, event_seqs: torch.Tensor, k: int, device: torch.device):
        """对新类型 k 做 LoRA-TTF: W 冻结, 只优化 c_k (r 维)."""
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

    def forward(self, event_seqs, event_type='category',
                need_weights=True, target_type=-1, device=None):
        """每条序列独立 LoRA-TTF."""
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
        """训练阶段创建标准 d 维嵌入; 推理阶段创建 r 维 c_k."""
        dim = self.rank if self._decomposed else self.embedding_dim
        self.embed[str(event_type)] = nn.Parameter(torch.randn(dim, device=device) * 0.02)
        self.log_intensities_prior[str(event_type)] = nn.Parameter(
            torch.zeros(1, device=device), requires_grad=True)
        self.max_type_index = max(int(event_type), self.max_type_index)
        if self.optim is not None:
            self.optim.add_param_group({'params': self.embed[str(event_type)]})
            self.optim.add_param_group({'params': self.log_intensities_prior[str(event_type)]})

    def mark_seen(self, types: set[int]):
        self._seen |= types
