"""
Cold-Start ERPP: NEC + ContextEmbedder
=======================================
在 ExplainableRecurrentPointProcess 的 forward 内部插入 ContextEmbedder。
两趟 forward 解循环依赖.

训练: 每 batch 随机 mask 已知类型 → ContextEmbedder(h(t-)) 生成嵌入
推理: cold_types 的类型 → ContextEmbedder(h(t-)) 生成嵌入
"""
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rnn import ExplainableRecurrentPointProcess
from ..utils.misc import AverageMeter
from ..utils.torch import generate_sequence_mask


class ContextEmbedder(nn.Module):
    """h(t-) [d] → type_embedding [d]"""

    def __init__(self, d_model: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class ColdStartERPP(ExplainableRecurrentPointProcess):
    """
    继承原 NEC, 在 forward 里加 ContextEmbedder。
    自动从训练数据中识别冷启动类型：训练时未见过的类型自动走 ContextEmbedder。
    """

    def __init__(self, embedder_hidden: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.embedder = ContextEmbedder(self.embedding_dim, hidden=embedder_hidden)
        n_types = kwargs.get('n_types', 1)
        self.register_buffer('_seen_types', torch.zeros(n_types, dtype=torch.bool))
        self._masked: set[int] = set()

    def set_masked(self, types: set[int]):
        self._masked = types

    def mark_seen(self, types: set[int]):
        for t in types:
            if t < len(self._seen_types):
                self._seen_types[t] = True

    def _types_to_replace(self) -> set[int]:
        if self.training:
            return self._masked
        return {t for t in range(self.current_n_types)
                if t >= len(self._seen_types) or not bool(self._seen_types[t].item())}

    def forward(
        self, event_seqs, event_type='category',
        need_weights=True, target_type=-1, device=None
    ):
        """
        两趟 forward:
          Pass 1 → history_emb
          Pass 2 → ContextEmbedder(history_emb) → 替换对应类型行 → log_intensities
        """
        if device is None:
            device = self.get_model_device()

        replace_types = self._types_to_replace()
        # 'feat' 模式: 嵌入已在 get_seq_contribution 中预处理, 跳过冷启动替换
        if not replace_types or event_type == 'feat':
            return super().forward(event_seqs, event_type, need_weights, target_type, device)

        # ── Pass 1: 用原嵌入表跑完整的 encoder ──
        batch_size, T = event_seqs.size()[:2]
        ts = F.pad(event_seqs[:, :, 0], (1, 0))
        dt = F.pad(ts[:, 1:] - ts[:, :-1], (1, 0))
        temp_feat = dt[:, :-1].unsqueeze(-1)

        if event_type == 'category':
            type_feat = self.event_type2embedding(event_seqs, device)[:, :-1, 1:]
        elif event_type == 'onehot':
            type_feat = event_seqs[:, :-1, 1:] @ self.return_all_parameters(dim=0)
        else:
            type_feat = event_seqs[:, :-1, 1:]

        type_feat = F.pad(type_feat, (0, 0, 1, 0))
        feat = torch.cat([temp_feat, type_feat], dim=-1)
        history_emb, *_ = self.seq_encoder(feat)
        history_emb = self.dropout(history_emb)

        # ── Pass 2: 为 replace_types 生成嵌入, 替换 decoder 中的对应部分 ──
        log_basis_feat = self.shallow_net(history_emb).view(
            batch_size, T, self.n_bases + 1, self.embedding_dim
        )

        # 原嵌入表 [n_types, d]
        all_embeds = self.return_all_parameters(dim=1)  # [d, n_types]

        # 对需要替换的类型, 从 history_emb 生成新嵌入
        # history_emb[:, i, :] 编码了事件 0..i-1, ≈ h(t_i-)
        # 事件 i 的嵌入用于 decoder 位置 i
        # 先算所有类型的 log_basis_weights
        log_basis_weights = log_basis_feat @ all_embeds  # [B, T, n_types, n_bases+1]

        # 冷启动类型: 逐位置生成嵌入, 替换对应位置的 decoder 输出
        for k in replace_types:
            if k >= self.current_n_types:
                continue
            k_mask = (event_seqs[:, :, 1].long() == k)  # [B, T]
            if not k_mask.any():
                continue
            gen_h = history_emb[k_mask]   # [n, hidden]
            gen_v = self.embedder(gen_h)  # [n, d], 每个位置独立生成
            b_idx, t_idx = k_mask.nonzero(as_tuple=True)
            # 替换: log_basis_weights[b, :, k, t] = log_basis_feat[b,t] · gen_v_i
            new_weight = (log_basis_feat[b_idx, t_idx] * gen_v.unsqueeze(1)).sum(dim=-1)
            log_basis_weights[b_idx, :, k, t_idx] = new_weight

        log_basis_weights = log_basis_weights.transpose(-2, -1).contiguous()

        # 强度: + basis_values → logsumexp
        basis_values = torch.cat(
            [basis.log_prob(dt[:, 1:, None]) for basis in self.bases], dim=2
        ).unsqueeze(-2)
        log_intensities = (log_basis_weights + basis_values).logsumexp(dim=-1)

        if need_weights:
            return log_intensities, log_basis_weights
        return log_intensities

    def train_epoch(
        self, train_dataloader, optim, valid_dataloader=None,
        device=None, **kwargs,
    ):
        """重写父类 train_epoch: 每 batch 随机 mask 类型训练 ContextEmbedder"""
        self.train()
        self.optim = optim
        mask_ratio = kwargs.get('mask_ratio', 0.15)
        train_types = list(range(self.current_n_types))

        train_metrics = defaultdict(AverageMeter)

        for batch in train_dataloader:
            if device:
                batch = batch.to(device)

            # 记录本 batch 出现的类型
            batch_types = set(batch[:, :, 1].long().flatten().tolist())
            self.mark_seen(batch_types)

            # 随机 mask (只 mask 训练集中出现过的已知类型)
            n_mask = max(1, int(len(train_types) * mask_ratio))
            masked = set(np.random.choice(train_types, n_mask, replace=False))
            self.set_masked(masked)

            seq_length = (batch.abs().sum(-1) > 0).sum(-1)
            mask = generate_sequence_mask(seq_length)

            log_intensities, log_basis_weights = self.forward(
                batch, need_weights=True, event_type='category'
            )

            nll = self._eval_nll(batch, log_intensities, log_basis_weights, mask)

            T_total = batch[range(batch.shape[0]), seq_length - 1, 0].sum()
            prior_log_intensities_loss = (
                -self.return_all_log_prior().expand(*(batch.size()[:2]), -1)
                .gather(dim=2, index=batch[:, :, 1:].long())
                .squeeze(-1).masked_select(mask).sum()
            ) + self.return_all_log_prior().exp().sum() * T_total

            loss = nll + prior_log_intensities_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_metrics['loss'].update(loss.item(), batch.size(0))
            train_metrics['nll'].update(nll.item(), batch.size(0))
            train_metrics['acc'].update(
                self._eval_acc(batch, log_intensities, mask), seq_length.sum()
            )

        if valid_dataloader is not None:
            valid_metrics = self._eval_epoch(valid_dataloader, device)
        else:
            valid_metrics = {}

        return train_metrics, valid_metrics

    @torch.no_grad()
    def _eval_epoch(self, dataloader, device):
        self.eval()
        metrics = defaultdict(AverageMeter)
        for batch in dataloader:
            if device:
                batch = batch.to(device)
            seq_length = (batch.abs().sum(-1) > 0).sum(-1)
            mask = generate_sequence_mask(seq_length)
            log_intensities, log_basis_weights = self.forward(
                batch, need_weights=True, event_type='category'
            )
            nll = self._eval_nll(batch, log_intensities, log_basis_weights, mask)
            metrics['nll'].update(nll.item(), batch.size(0))
            metrics['acc'].update(
                self._eval_acc(batch, log_intensities, mask), seq_length.sum()
            )
        return metrics

    def get_seq_contribution(self, batch, device=None, steps=50, **kwargs):
        from functools import partial
        from ..explain.integrated_gradient import batch_integrated_gradient
        from ..utils.torch import set_eval_mode, generate_sequence_mask

        replace_types = self._types_to_replace()
        if not replace_types:
            return super().get_seq_contribution(batch, device, steps, **kwargs)

        set_eval_mode(self)
        for param in self.parameters():
            param.requires_grad_(False)
        if device:
            batch = batch.to(device)
        B, T = batch.size()[:2]

        # ── 计算 history_emb ──
        ts = F.pad(batch[:, :, 0], (1, 0))
        dt = F.pad(ts[:, 1:] - ts[:, :-1], (1, 0))
        temp_feat = dt[:, :-1].unsqueeze(-1)
        type_feat = self.event_type2embedding(batch)[:, :-1, 1:]
        type_feat = F.pad(type_feat, (0, 0, 1, 0))
        feat = torch.cat([temp_feat, type_feat], dim=-1)
        history_emb, *_ = self.seq_encoder(feat)
        history_emb = self.dropout(history_emb)

        # ── 逐位置生成冷启动嵌入, 构建 inputs ──
        inputs = self.event_type2embedding(batch)  # [B, T, 1+d]
        type_col = batch[:, :, 1].long()
        for k in replace_types:
            k_mask = (type_col == k)
            if not k_mask.any():
                continue
            gen_v = self.embedder(history_emb[k_mask])  # [n, d], 每位置独立
            b_idx, t_idx = k_mask.nonzero(as_tuple=True)
            inputs[b_idx, t_idx, 1:] = gen_v

        baselines = F.pad(inputs[:, :, :1], (0, self.embedding_dim))
        seq_lengths = (batch.abs().sum(-1) > 0).sum(-1)
        event_scores = torch.zeros(B, T, T - 1, device=device)

        def _func(X, p, b):
            li = self.forward(X, event_type='feat', need_weights=False)
            return torch.gather(li[:, p], dim=-1, index=b[:, p, 1:].repeat(X.size(0), 1).long())

        for pos in range(T):
            mask_p = (seq_lengths > pos)
            ig = batch_integrated_gradient(
                partial(_func, p=pos, b=batch), inputs, baselines=baselines,
                mask=mask_p, steps=steps,
            )
            event_scores[:, pos] = ig[:, :-1].sum(-1)

        log_ints = self.forward(inputs, event_type='feat', need_weights=False)
        log_ints_events = log_ints.gather(dim=2, index=batch[:, :, 1:].long()).squeeze(-1)
        base_ints_events = self.forward(baselines, event_type='feat', need_weights=False)
        base_ints_events = base_ints_events.gather(dim=2, index=batch[:, :, 1:].long()).squeeze(-1)
        prior_ints_events = self.event_prior_intensities(batch)

        return (event_scores.detach().cpu(), log_ints_events.detach().cpu(),
                base_ints_events.detach().cpu(), prior_ints_events.detach().cpu())
