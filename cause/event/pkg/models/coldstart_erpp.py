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
        n_types = kwargs.get('n_types', 1)
        super().__init__(**kwargs)
        self.embedder = ContextEmbedder(self.embedding_dim, hidden=embedder_hidden)
        self.register_buffer('_seen_types', torch.zeros(n_types, dtype=torch.bool))
        self._masked: set[int] = set()
        self._cold_gen_v = None  # 注入 per-position embedding: {k: (b_idx, t_idx, gen_v)}

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
        """先跑父类 forward, 再对冷启动类型逐位置替换 decoder 权重."""
        if device is None:
            device = self.get_model_device()

        replace_types = self._types_to_replace()
        cold_data = self._cold_gen_v  # 由 get_seq_contribution 注入

        # 无冷启动类型需要替换 → 直接走父类
        if not replace_types and cold_data is None:
            return super().forward(event_seqs, event_type, need_weights, target_type, device)
        if cold_data is None and event_type == 'feat':
            return super().forward(event_seqs, event_type, need_weights, target_type, device)

        # ── 跑父类 encoder 拿 history_emb ──
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

        log_basis_feat = self.shallow_net(history_emb).view(
            batch_size, T, self.n_bases + 1, self.embedding_dim
        )
        all_embeds = self.return_all_parameters(dim=1)  # [d, n_types]
        log_basis_weights = log_basis_feat @ all_embeds  # [B, T, n_bases+1, n_types]

        # ── 替换冷启动类型的 decoder 权重 (逐位置 embedding) ──
        # event_type='feat' 时 event_seqs[:,:,1:] 是嵌入值不是 type ID,
        # 此时用 cold_data 里预先存在的 per-position embedding
        if cold_data is not None and event_type == 'feat':
            for k, (b_idx, t_idx, gen_v) in cold_data.items():
                new_w = (log_basis_feat[b_idx, t_idx] * gen_v.unsqueeze(1)).sum(dim=-1)
                for j in range(len(b_idx)):
                    log_basis_weights[b_idx[j], t_idx[j], :, k] = new_w[j]
        else:
            # event_type='category' 时可以直接用 type ID 定位
            for k in replace_types:
                if k >= self.current_n_types:
                    continue
                k_mask = (event_seqs[:, :, 1].long() == k)
                if not k_mask.any():
                    continue
                gen_h = history_emb[k_mask]
                gen_v = self.embedder(gen_h)
                b_idx, t_idx = k_mask.nonzero(as_tuple=True)
                new_w = (log_basis_feat[b_idx, t_idx] * gen_v.unsqueeze(1)).sum(dim=-1)
                for j in range(len(b_idx)):
                    log_basis_weights[b_idx[j], t_idx[j], :, k] = new_w[j]

        log_basis_weights = log_basis_weights.transpose(-2, -1).contiguous()
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
            batch_types = set(batch[:, :, 1].long().flatten().tolist())
            self.mark_seen(batch_types)
            n_mask = max(1, int(len(train_types) * mask_ratio))
            masked = set(np.random.choice(train_types, n_mask, replace=False))
            self.set_masked(masked)

            seq_length = (batch.abs().sum(-1) > 0).sum(-1)
            mask_t = generate_sequence_mask(seq_length)
            log_intensities, log_basis_weights = self.forward(
                batch, need_weights=True, event_type='category'
            )
            nll = self._eval_nll(batch, log_intensities, log_basis_weights, mask_t)
            T_total = batch[range(batch.shape[0]), seq_length - 1, 0].sum()
            prior_log_intensities_loss = (
                -self.return_all_log_prior().expand(*(batch.size()[:2]), -1)
                .gather(dim=2, index=batch[:, :, 1:].long())
                .squeeze(-1).masked_select(mask_t).sum()
            ) + self.return_all_log_prior().exp().sum() * T_total
            loss = nll + prior_log_intensities_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_metrics['loss'].update(loss.item(), batch.size(0))
            train_metrics['nll'].update(nll.item(), batch.size(0))
            train_metrics['acc'].update(
                self._eval_acc(batch, log_intensities, mask_t), seq_length.sum()
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
        """
        覆写父类: 冷启动类型逐位置生成嵌入, 注入 forward 供 IG 循环和强度计算使用.
        """
        from functools import partial
        from ..explain.integrated_gradient import batch_integrated_gradient
        from ..utils.torch import set_eval_mode

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
        dt_fwd = F.pad(ts[:, 1:] - ts[:, :-1], (1, 0))
        temp_feat = dt_fwd[:, :-1].unsqueeze(-1)
        type_feat = self.event_type2embedding(batch)[:, :-1, 1:]
        type_feat = F.pad(type_feat, (0, 0, 1, 0))
        feat = torch.cat([temp_feat, type_feat], dim=-1)
        history_emb, *_ = self.seq_encoder(feat)
        history_emb = self.dropout(history_emb)

        # ── 逐位置生成冷启动嵌入 (与训练一致) ──
        type_col = batch[:, :, 1].long()
        cold_data = {}
        for k in replace_types:
            k_mask = (type_col == k)
            if not k_mask.any():
                continue
            b_idx, t_idx = k_mask.nonzero(as_tuple=True)
            gen_v = self.embedder(history_emb[k_mask])  # [n, d], 每位置独立
            cold_data[k] = (b_idx, t_idx, gen_v)

        # ── 注入 cold_data 到 forward, 跑 IG ──
        self._cold_gen_v = cold_data
        inputs = self.event_type2embedding(batch)
        for k, (b_idx, t_idx, gen_v) in cold_data.items():
            inputs[b_idx, t_idx, 1:] = gen_v

        baselines = F.pad(inputs[:, :, :1], (0, self.embedding_dim))
        seq_lengths = (batch.abs().sum(-1) > 0).sum(-1)
        event_scores = torch.zeros(B, T, T - 1, device=device)

        def _func(X, p, b):
            li, _ = self.forward(X, event_type='feat', need_weights=True)
            return torch.gather(li[:, p], dim=-1, index=b[:, p, 1:].repeat(X.size(0), 1).long())

        for pos in range(T):
            mask_p = (seq_lengths > pos)
            ig = batch_integrated_gradient(
                partial(_func, p=pos, b=batch), inputs, baselines=baselines,
                mask=mask_p, steps=steps,
            )
            event_scores[:, pos] = ig[:, :-1].sum(-1)

        # ── 计算最终强度 ──
        log_ints, _ = self.forward(inputs, event_type='feat', need_weights=True)
        log_ints_events = log_ints.gather(dim=2, index=batch[:, :, 1:].long()).squeeze(-1)
        base_ints, _ = self.forward(baselines, event_type='feat', need_weights=True)
        base_ints_events = base_ints.gather(dim=2, index=batch[:, :, 1:].long()).squeeze(-1)
        prior_ints_events = self.event_prior_intensities(batch)

        self._cold_gen_v = None
        return (event_scores.detach().cpu(), log_ints_events.detach().cpu(),
                base_ints_events.detach().cpu(), prior_ints_events.detach().cpu())
