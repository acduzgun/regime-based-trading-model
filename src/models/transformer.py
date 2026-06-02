from __future__ import annotations

from typing import List, Optional, Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing import WindowSpec, build_yearly_schedule, prepare_arrays


# ---------------------------------------------------------------------------
# Typed attention transformer
#
# No lookback — uses only the current time step from each window.
# Each x-feature and each regime variable becomes its own token.
# Joint attention with relation-type bias (f→f, f→r, r→f, r→r per head).
# Regime information flows into feature tokens through attention.
# Readout: flatten final feature tokens → MLP head.
# ---------------------------------------------------------------------------

def _validate_attention_shape(d_model: int, n_heads: int) -> None:
    if d_model % n_heads != 0:
        raise ValueError(f'd_model={d_model} must be divisible by n_heads={n_heads}')


def _ffn(d_model: int, ffn_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, ffn_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(ffn_dim, d_model),
    )


class TypedAttentionBlock(nn.Module):
    """One pre-norm typed-attention + typed-FFN block."""

    def __init__(
        self,
        n_feat: int,
        n_cond: int,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
    ):
        super().__init__()
        _validate_attention_shape(d_model, n_heads)
        self.n_feat = n_feat
        self.n_cond = n_cond
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.scale = self.head_dim ** -0.5

        # Attention pre-norms (separate per stream)
        self.norm_f_attn = nn.LayerNorm(d_model)
        self.norm_r_attn = nn.LayerNorm(d_model)

        # Separate Q/K/V projections per stream
        self.Q_f = nn.Linear(d_model, d_model)
        self.K_f = nn.Linear(d_model, d_model)
        self.V_f = nn.Linear(d_model, d_model)
        self.Q_r = nn.Linear(d_model, d_model)
        self.K_r = nn.Linear(d_model, d_model)
        self.V_r = nn.Linear(d_model, d_model)

        # Relation-type bias: (n_heads, 2, 2); type 0=feature, 1=regime
        self.relation_bias = nn.Parameter(torch.zeros(n_heads, 2, 2))

        # Separate output projections
        self.O_f = nn.Linear(d_model, d_model)
        self.O_r = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # FFN pre-norms (separate per stream)
        self.norm_f_ffn = nn.LayerNorm(d_model)
        self.norm_r_ffn = nn.LayerNorm(d_model)

        # Separate FFNs
        self.ffn_f = _ffn(d_model, ffn_dim, dropout)
        self.ffn_r = _ffn(d_model, ffn_dim, dropout)

    def forward(self, H_f: torch.Tensor, H_r: torch.Tensor):
        # H_f: (B, n_feat, d_model),  H_r: (B, n_cond, d_model)
        B = H_f.size(0)
        N = self.n_feat + self.n_cond

        # Pre-norm → Q/K/V per stream → concat
        H_fn, H_rn = self.norm_f_attn(H_f), self.norm_r_attn(H_r)
        Q = torch.cat([self.Q_f(H_fn), self.Q_r(H_rn)], dim=1)   # (B, N, d)
        K = torch.cat([self.K_f(H_fn), self.K_r(H_rn)], dim=1)
        V = torch.cat([self.V_f(H_fn), self.V_r(H_rn)], dim=1)

        # Split heads: (B, N, d) → (B, n_heads, N, head_dim)
        def split_heads(x):
            return x.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, n_heads, N, N)

        # Relation-type bias: (n_heads, 2, 2) → (1, n_heads, N, N)
        types = torch.zeros(N, dtype=torch.long, device=H_f.device)
        types[self.n_feat:] = 1
        bias = self.relation_bias[:, types][:, :, types]            # (n_heads, N, N)
        scores = scores + bias.unsqueeze(0)

        out = torch.matmul(torch.softmax(scores, dim=-1), V)        # (B, n_heads, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)

        # Split → separate output projections → residual
        H_f = H_f + self.dropout(self.O_f(out[:, :self.n_feat]))
        H_r = H_r + self.dropout(self.O_r(out[:, self.n_feat:]))

        # FFN pre-norm → separate FFNs → residual
        H_f = H_f + self.dropout(self.ffn_f(self.norm_f_ffn(H_f)))
        H_r = H_r + self.dropout(self.ffn_r(self.norm_r_ffn(H_r)))

        return H_f, H_r


class TypedAttentionTransformer(nn.Module):
    def __init__(
        self,
        n_x_features: int,
        n_cond_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.n_x_features = n_x_features
        self.n_cond_features = n_cond_features

        # Each x-feature / regime variable → its own Linear(1 → d_model)
        self.feature_embeds = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_x_features)])
        self.regime_embeds  = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_cond_features)])

        self.blocks = nn.ModuleList([
            TypedAttentionBlock(n_x_features, n_cond_features, d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

        flat_dim = n_x_features * d_model          # 8 × d_model
        self.head = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, 1),
        )

    def forward(self, X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Accept (B, L, n) from windowed dataloader — use only current time step
        if X.dim() == 3:
            X = X[:, -1, :]   # (B, n_x)
            C = C[:, -1, :]   # (B, n_cond)

        B = X.size(0)

        # Embed each feature/regime scalar separately
        H_f = torch.stack([self.feature_embeds[j](X[:, j:j+1]) for j in range(self.n_x_features)], dim=1)
        H_r = torch.stack([self.regime_embeds[k](C[:, k:k+1]) for k in range(self.n_cond_features)], dim=1)

        for block in self.blocks:
            H_f, H_r = block(H_f, H_r)

        z = H_f.reshape(B, -1)              # (B, n_x * d_model)
        return self.head(z).squeeze(-1)     # (B,)


# ---------------------------------------------------------------------------
# Directed regime attention transformer
#
# Features and regimes have separate self-attention streams.
# Regime tokens influence feature tokens through a one-way regime->feature
# cross-attention block. Regime tokens are never read out directly.
# ---------------------------------------------------------------------------


class DirectedRegimeAttentionBlock(nn.Module):
    """Separate self-attention streams plus regime-to-feature cross-attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
    ):
        super().__init__()
        _validate_attention_shape(d_model, n_heads)

        self.norm_f_self = nn.LayerNorm(d_model)
        self.norm_r_self = nn.LayerNorm(d_model)
        self.feature_self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.regime_self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.norm_f_cross = nn.LayerNorm(d_model)
        self.norm_r_cross = nn.LayerNorm(d_model)
        self.regime_to_feature_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.norm_f_ffn = nn.LayerNorm(d_model)
        self.norm_r_ffn = nn.LayerNorm(d_model)
        self.ffn_f = _ffn(d_model, ffn_dim, dropout)
        self.ffn_r = _ffn(d_model, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H_f: torch.Tensor, H_r: torch.Tensor):
        H_fn = self.norm_f_self(H_f)
        f_self, _ = self.feature_self_attn(H_fn, H_fn, H_fn, need_weights=False)
        H_f = H_f + self.dropout(f_self)

        H_rn = self.norm_r_self(H_r)
        r_self, _ = self.regime_self_attn(H_rn, H_rn, H_rn, need_weights=False)
        H_r = H_r + self.dropout(r_self)

        H_fq = self.norm_f_cross(H_f)
        H_rkv = self.norm_r_cross(H_r)
        f_cross, _ = self.regime_to_feature_attn(H_fq, H_rkv, H_rkv, need_weights=False)
        H_f = H_f + self.dropout(f_cross)

        H_f = H_f + self.dropout(self.ffn_f(self.norm_f_ffn(H_f)))
        H_r = H_r + self.dropout(self.ffn_r(self.norm_r_ffn(H_r)))
        return H_f, H_r


class DirectedRegimeAttentionTransformer(nn.Module):
    def __init__(
        self,
        n_x_features: int,
        n_cond_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.n_x_features = n_x_features
        self.n_cond_features = n_cond_features

        self.feature_embeds = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_x_features)])
        self.regime_embeds = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_cond_features)])

        self.blocks = nn.ModuleList([
            DirectedRegimeAttentionBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

        flat_dim = n_x_features * d_model
        self.head = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, 1),
        )

    def forward(self, X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        if X.dim() == 3:
            X = X[:, -1, :]
            C = C[:, -1, :]

        B = X.size(0)
        H_f = torch.stack([self.feature_embeds[j](X[:, j:j+1]) for j in range(self.n_x_features)], dim=1)
        H_r = torch.stack([self.regime_embeds[k](C[:, k:k+1]) for k in range(self.n_cond_features)], dim=1)

        for block in self.blocks:
            H_f, H_r = block(H_f, H_r)

        z = H_f.reshape(B, -1)
        return self.head(z).squeeze(-1)


# ---------------------------------------------------------------------------
# Regime-bias attention transformer
#
# Feature tokens attend only to feature tokens. A regime summary creates
# attention-logit biases and feature gates, so regime affects how features
# interact without becoming an attended/readout token itself.
# ---------------------------------------------------------------------------


class RegimeBiasedFeatureBlock(nn.Module):
    """Feature self-attention whose attention logits and gates are regime-conditioned."""

    def __init__(
        self,
        n_feat: int,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
    ):
        super().__init__()
        _validate_attention_shape(d_model, n_heads)
        self.n_feat = n_feat
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.norm_attn = nn.LayerNorm(d_model)
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.O = nn.Linear(d_model, d_model)

        self.bias_head = nn.Linear(d_model, n_heads * n_feat * n_feat)
        self.gate_head = nn.Linear(d_model, n_feat * d_model)
        nn.init.zeros_(self.bias_head.weight)
        nn.init.zeros_(self.bias_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = _ffn(d_model, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H_f: torch.Tensor, regime_summary: torch.Tensor) -> torch.Tensor:
        B = H_f.size(0)
        Hn = self.norm_attn(H_f)
        Q = self.Q(Hn).view(B, self.n_feat, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.K(Hn).view(B, self.n_feat, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.V(Hn).view(B, self.n_feat, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        regime_bias = self.bias_head(regime_summary).view(B, self.n_heads, self.n_feat, self.n_feat)
        scores = scores + 0.25 * torch.tanh(regime_bias)

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(self.dropout(attn), V)
        out = out.transpose(1, 2).contiguous().view(B, self.n_feat, self.d_model)
        H_f = H_f + self.dropout(self.O(out))

        gate = self.gate_head(regime_summary).view(B, self.n_feat, self.d_model)
        H_f = H_f * (1.0 + 0.5 * torch.tanh(gate))
        H_f = H_f + self.dropout(self.ffn(self.norm_ffn(H_f)))
        return H_f


class RegimeBiasAttentionTransformer(nn.Module):
    def __init__(
        self,
        n_x_features: int,
        n_cond_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.n_x_features = n_x_features
        self.n_cond_features = n_cond_features

        self.feature_embeds = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_x_features)])
        self.regime_encoder = nn.Sequential(
            nn.LayerNorm(n_cond_features),
            nn.Linear(n_cond_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.blocks = nn.ModuleList([
            RegimeBiasedFeatureBlock(n_x_features, d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

        flat_dim = n_x_features * d_model
        self.head = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, 1),
        )

    def forward(self, X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        if X.dim() == 3:
            X = X[:, -1, :]
            C = C[:, -1, :]

        B = X.size(0)
        H_f = torch.stack([self.feature_embeds[j](X[:, j:j+1]) for j in range(self.n_x_features)], dim=1)
        regime_summary = self.regime_encoder(C)

        for block in self.blocks:
            H_f = block(H_f, regime_summary)

        z = H_f.reshape(B, -1)
        return self.head(z).squeeze(-1)


# ---------------------------------------------------------------------------
# Training loop (generic — works for any model shape)
# ---------------------------------------------------------------------------

def _daily_sharpe(
    pred: np.ndarray,
    returns: np.ndarray,
    trade_dates: np.ndarray,
    annualization: int = 252,
) -> float:
    """Backtest-style daily Sharpe used for epoch selection."""
    if pred is None or returns is None or trade_dates is None:
        return float('nan')

    data = pd.DataFrame({
        'trade_date': pd.to_datetime(trade_dates).date,
        'pred': np.asarray(pred, dtype=float),
        'ret': np.asarray(returns, dtype=float),
    }).dropna()
    if data.empty:
        return float('nan')

    data['pnl'] = data['pred'].clip(-3, 3) * data['ret']
    daily_pnl = data.groupby('trade_date', sort=True)['pnl'].sum()
    if len(daily_pnl) < 2:
        return float('nan')

    std = daily_pnl.std()
    if std == 0 or not np.isfinite(std):
        return float('nan')
    return float(np.sqrt(annualization) * daily_pnl.mean() / std)

def fit_transformer(
    model: nn.Module,
    X_tr: np.ndarray,
    C_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    C_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    epochs: int = 100,
    batch_size: int = 1024,
    patience: int = 10,
    lr_scheduler_patience: int = 3,
    warmup_fraction: float = 0.01,
    grad_clip: float = 1.0,
    train_returns: Optional[np.ndarray] = None,
    val_returns: Optional[np.ndarray] = None,
    train_dates: Optional[np.ndarray] = None,
    val_dates: Optional[np.ndarray] = None,
):
    """AdamW + ReduceLROnPlateau + early stopping on inner validation Sharpe.

    Data stays on CPU; batches are moved to device one at a time to avoid OOM
    with large windowed datasets (e.g. 2M windows × 32 steps).

    Returns (model, train_losses, val_losses, grad_norms, train_sharpes,
    val_sharpes) — last-fold histories for diagnostics. MSE is still
    tracked for both train and inner-val slices, but the best checkpoint is
    selected by inner-val Sharpe.
    """
    def to_cpu(a: np.ndarray) -> torch.Tensor:
        return torch.tensor(a, dtype=torch.float32)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_tr_cpu = to_cpu(X_tr)
    C_tr_cpu = to_cpu(C_tr)
    y_tr_cpu = to_cpu(y_tr)
    X_val_cpu = to_cpu(X_val)
    C_val_cpu = to_cpu(C_val)
    y_val_cpu = to_cpu(y_val)

    pin = device.type == 'cuda'
    loader = DataLoader(
        TensorDataset(X_tr_cpu, C_tr_cpu, y_tr_cpu),
        batch_size=min(batch_size, len(X_tr)),
        shuffle=True,
        pin_memory=pin,
    )

    total_steps = epochs * len(loader)
    warmup_steps = max(1, int(float(warmup_fraction) * total_steps))
    warmup_steps = min(warmup_steps, len(loader))
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0 / warmup_steps, end_factor=1.0, total_iters=warmup_steps,
    )
    lr_scheduler_patience = int(lr_scheduler_patience)
    if lr_scheduler_patience >= patience:
        lr_scheduler_patience = max(0, int(patience) - 2)
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=lr_scheduler_patience, factor=0.5, min_lr=1e-6,
    )

    print(f'  lr={lr}  weight_decay={weight_decay}  grad_clip={grad_clip}  '
          f'epochs={epochs}  early_stop_patience={patience}  '
          f'lr_scheduler_patience={lr_scheduler_patience}  batch_size={batch_size}  '
          f'warmup_steps={warmup_steps}/{total_steps}  warmup_fraction={warmup_fraction}  '
          f'stop_metric=val_sharpe')

    def predict_cpu(X_cpu: torch.Tensor, C_cpu: torch.Tensor) -> torch.Tensor:
        parts = []
        with torch.no_grad():
            for i in range(0, len(X_cpu), batch_size):
                parts.append(model(
                    X_cpu[i:i+batch_size].to(device),
                    C_cpu[i:i+batch_size].to(device),
                ).cpu())
        return torch.cat(parts)

    best_val_sharpe, best_state, no_improve = -float('inf'), None, 0
    train_losses: list = []
    val_losses: list = []
    train_sharpes: list = []
    val_sharpes: list = []
    grad_norms: list = []
    global_step = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        batch_norms: list = []
        for Xb, Cb, yb in loader:
            Xb = Xb.to(device, non_blocking=pin)
            Cb = Cb.to(device, non_blocking=pin)
            yb = yb.to(device, non_blocking=pin)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(Xb, Cb), yb)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if global_step < warmup_steps:
                warmup_sched.step()
            global_step += 1
            epoch_loss += loss.item()
            batch_norms.append(norm.item())
            n_batches += 1
        bn = np.array(batch_norms)
        grad_norms.append({
            'mean': float(bn.mean()),
            'p95': float(np.percentile(bn, 95)),
            'max': float(bn.max()),
        })

        model.eval()
        train_pred = predict_cpu(X_tr_cpu, C_tr_cpu)
        val_pred = predict_cpu(X_val_cpu, C_val_cpu)
        train_loss = nn.functional.mse_loss(train_pred, y_tr_cpu).item()
        val_loss = nn.functional.mse_loss(val_pred, y_val_cpu).item()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_pred_np = train_pred.numpy()
        val_pred_np = val_pred.numpy()
        pred_scale = train_pred_np.std() + 1e-12
        train_sharpe = _daily_sharpe(
            train_pred_np / pred_scale, train_returns, train_dates
        )
        val_sharpe = _daily_sharpe(
            val_pred_np / pred_scale, val_returns, val_dates
        )
        train_sharpes.append(train_sharpe)
        val_sharpes.append(val_sharpe)

        stop_metric = val_sharpe if np.isfinite(val_sharpe) else -float('inf')
        if global_step >= warmup_steps:
            plateau_sched.step(stop_metric)
        improved = best_state is None or stop_metric > best_val_sharpe + 1e-12
        print(
            f'  epoch {epoch+1:3d}  '
            f'train_mse={train_loss:.6f}  train_sharpe={train_sharpe:.4f}  '
            f'val_mse={val_loss:.6f}  val_sharpe={val_sharpe:.4f}  '
            f'lr={optimizer.param_groups[0]["lr"]:.2e}'
            + ('  *' if improved else '')
        )

        if improved:
            best_val_sharpe = stop_metric
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses, grad_norms, train_sharpes, val_sharpes


# ---------------------------------------------------------------------------
# Windowing helpers
# ---------------------------------------------------------------------------

def _sliding_windows(arr: np.ndarray, lookback: int) -> np.ndarray:
    """Return (N-L+1, L, F) view via stride tricks — no Python loop, no copy."""
    from numpy.lib.stride_tricks import sliding_window_view
    # sliding_window_view over axis=0 → (N-L+1, F, L), then swap to (N-L+1, L, F)
    return np.ascontiguousarray(
        sliding_window_view(arr, window_shape=lookback, axis=0).swapaxes(1, 2)
    )


def _make_train_windows(
    X: np.ndarray, C: np.ndarray, y: np.ndarray, years: np.ndarray, lookback: int
):
    """Sliding windows over training data. First lookback-1 samples are dropped."""
    X_win = _sliding_windows(X, lookback)   # (N-L+1, L, F)
    C_win = _sliding_windows(C, lookback)
    idx = np.arange(lookback - 1, len(X))
    return X_win, C_win, y[idx], years[idx]


def _make_test_windows(
    X_train: np.ndarray, C_train: np.ndarray,
    X_test: np.ndarray, C_test: np.ndarray,
    lookback: int,
):
    """Test windows using last lookback-1 train rows as left context."""
    ctx = lookback - 1
    X_ctx = np.vstack([X_train[-ctx:] if ctx > 0 else X_train[:0], X_test])
    C_ctx = np.vstack([C_train[-ctx:] if ctx > 0 else C_train[:0], C_test])
    X_win = _sliding_windows(X_ctx, lookback)
    C_win = _sliding_windows(C_ctx, lookback)
    return X_win, C_win


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def train_predict_fixed_split(
    df: pd.DataFrame,
    x_cols: List[str],
    cond_cols: List[str],
    model_class: Type[nn.Module],
    model_kwargs: dict,
    train_end_year: int = 2021,
    train_end_date: Optional[str] = None,
    val_years: Optional[List[int]] = None,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
    target_col: str = 'ret_fopen',
    date_col: str = 'msgStamp',
    trade_date_col: str = 'trade_date',
    x_clip: float = 3.0,
    y_clip_quantile: Optional[float] = 0.01,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    epochs: int = 100,
    batch_size: int = 1024,
    patience: int = 10,
    lr_scheduler_patience: int = 3,
    warmup_fraction: float = 0.01,
    grad_clip: float = 1.0,
    inner_val_mode: str = 'tail_fraction',
    inner_val_fraction: float = 0.15,
    device: Optional[torch.device] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train once on historical data; predict a held-out validation window.

    Intended for fast HP search. Inner val is used for early stopping only.
    Training can end by whole year (train_end_year) or by explicit inclusive
    date (train_end_date).
    Validation can be selected either by whole years (val_years) or by an
    explicit inclusive date range (val_start/val_end).
    Default inner_val_mode='tail_fraction' keeps the historical behavior used
    in the initial experiments; inner_val_mode='last_year' withholds the full
    final training year for a cleaner temporal diagnostic.
    Predictions normalised by train-window prediction std.
    Returns held-out predictions plus .attrs diagnostics, including
    .attrs['inner_val_preds'] for the temporal validation slice inside
    the training period.
    """
    if val_years is None:
        val_years = [2022, 2023]
    lookback = model_kwargs.get('lookback', 32)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).reset_index(drop=True)
    data['_year'] = data[date_col].dt.year
    if trade_date_col not in data.columns:
        data[trade_date_col] = data[date_col].dt.date

    data_dates = pd.to_datetime(data[date_col])
    if data_dates.dt.tz is not None:
        data_dates = data_dates.dt.tz_convert(None)
    if train_end_date is not None:
        train_cutoff = pd.Timestamp(train_end_date)
        train_mask_all = data_dates < train_cutoff + pd.Timedelta(days=1)
        train_end_label = train_cutoff.date().isoformat()
    else:
        train_mask_all = data['_year'] <= train_end_year
        train_end_label = str(train_end_year)

    train_df = data[train_mask_all].dropna(
        subset=x_cols + cond_cols + [target_col]
    ).copy()
    if val_start is not None or val_end is not None:
        val_mask = pd.Series(True, index=data.index)
        if val_start is not None:
            val_mask &= data_dates >= pd.Timestamp(val_start)
        if val_end is not None:
            val_mask &= data_dates < pd.Timestamp(val_end) + pd.Timedelta(days=1)
    else:
        val_mask = data['_year'].isin(val_years)
    val_df = data[val_mask].dropna(subset=x_cols + cond_cols).copy()

    if len(train_df) < lookback or len(val_df) == 0:
        return pd.DataFrame(columns=[date_col, trade_date_col, 'pred'])

    X_train, X_val, C_train, C_val, y_train = prepare_arrays(
        train_df, val_df, x_cols, cond_cols, target_col, x_clip, y_clip_quantile
    )

    train_years_arr = train_df['_year'].values
    train_dates_arr = pd.to_datetime(train_df[date_col]).to_numpy()
    X_tr_win, C_tr_win, y_tr_win, yr_win = _make_train_windows(
        X_train, C_train, y_train, train_years_arr, lookback
    )
    date_win = train_dates_arr[lookback - 1:]
    X_val_win, C_val_win = _make_test_windows(X_train, C_train, X_val, C_val, lookback)
    train_window_rows = train_df.iloc[lookback - 1:].reset_index(drop=True)
    train_ret_win = train_window_rows[target_col].to_numpy(dtype=float)
    train_dates_win = train_window_rows[trade_date_col].to_numpy()

    if inner_val_mode == 'last_year':
        if train_end_date is not None:
            cutoff = pd.Timestamp(train_end_date)
            cutoff_start = pd.Timestamp(cutoff.year, 1, 1)
            cutoff_next = cutoff + pd.Timedelta(days=1)
            iv_mask = (date_win >= np.datetime64(cutoff_start)) & (date_win < np.datetime64(cutoff_next))
        else:
            iv_mask = (yr_win == train_end_year)
        train_mask = ~iv_mask
    elif inner_val_mode == 'tail_fraction':
        n_iv = max(1, int(inner_val_fraction * len(X_tr_win)))
        train_mask = np.arange(len(X_tr_win)) < len(X_tr_win) - n_iv
        iv_mask = ~train_mask
    else:
        raise ValueError(
            f"Unknown inner_val_mode={inner_val_mode!r}. "
            "Choose 'tail_fraction' or 'last_year'."
        )

    if not iv_mask.any() or not train_mask.any():
        n_iv = max(1, int(inner_val_fraction * len(X_tr_win)))
        train_mask = np.arange(len(X_tr_win)) < len(X_tr_win) - n_iv
        iv_mask = ~train_mask

    X_tr, C_tr, y_tr = X_tr_win[train_mask], C_tr_win[train_mask], y_tr_win[train_mask]
    X_iv, C_iv, y_iv = X_tr_win[iv_mask], C_tr_win[iv_mask], y_tr_win[iv_mask]
    train_ret, inner_val_ret = train_ret_win[train_mask], train_ret_win[iv_mask]
    train_dates, inner_val_dates = train_dates_win[train_mask], train_dates_win[iv_mask]
    if len(X_tr) == 0:
        X_tr, C_tr, y_tr = X_tr_win, C_tr_win, y_tr_win
        X_iv, C_iv, y_iv = X_tr_win, C_tr_win, y_tr_win
        train_ret = inner_val_ret = train_ret_win
        train_dates = inner_val_dates = train_dates_win

    model = model_class(
        n_x_features=len(x_cols),
        n_cond_features=len(cond_cols),
        **model_kwargs,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  n_params={n_params:,}')

    model, train_losses, val_losses, grad_norms, train_sharpes, val_sharpes = fit_transformer(
        model, X_tr, C_tr, y_tr, X_iv, C_iv, y_iv,
        device=device, lr=lr, weight_decay=weight_decay,
        epochs=epochs, batch_size=batch_size, patience=patience,
        lr_scheduler_patience=lr_scheduler_patience,
        warmup_fraction=warmup_fraction, grad_clip=grad_clip,
        train_returns=train_ret, val_returns=inner_val_ret,
        train_dates=train_dates, val_dates=inner_val_dates,
    )

    model.eval()

    def predict(X: np.ndarray, C: np.ndarray, chunk: int = 1024) -> np.ndarray:
        parts = []
        with torch.no_grad():
            for i in range(0, len(X), chunk):
                parts.append(model(
                    torch.tensor(X[i:i + chunk], dtype=torch.float32).to(device),
                    torch.tensor(C[i:i + chunk], dtype=torch.float32).to(device),
                ).cpu().numpy())
        return np.concatenate(parts)

    train_std = predict(X_tr_win, C_tr_win).std() + 1e-12
    pred_inner_val = predict(X_iv, C_iv) / train_std
    pred_val = predict(X_val_win, C_val_win) / train_std

    inner_val_rows = train_window_rows.loc[iv_mask].copy()
    inner_val_out = inner_val_rows[[date_col, trade_date_col, target_col]].copy()
    inner_val_out['pred'] = pred_inner_val
    inner_val_out['train_end_year'] = train_end_year
    inner_val_out['train_end_date'] = train_end_label
    inner_val_out = inner_val_out.sort_values(date_col).reset_index(drop=True)

    out_cols = [date_col, trade_date_col]
    if target_col in val_df.columns:
        out_cols.append(target_col)
    out = val_df[out_cols].copy()
    out['pred'] = pred_val
    out['train_end_year'] = train_end_year
    out['train_end_date'] = train_end_label

    result = out.sort_values(date_col).reset_index(drop=True)
    result.attrs['train_losses'] = train_losses
    result.attrs['val_losses'] = val_losses
    result.attrs['train_sharpes'] = train_sharpes
    result.attrs['val_sharpes'] = val_sharpes
    result.attrs['grad_norms'] = grad_norms
    result.attrs['n_params'] = n_params
    val_sharpes_arr = np.asarray(val_sharpes, dtype=float)
    if val_sharpes_arr.size and np.isfinite(val_sharpes_arr).any():
        result.attrs['best_epoch_by_val_sharpe'] = int(np.nanargmax(val_sharpes_arr)) + 1
        result.attrs['best_val_sharpe'] = float(np.nanmax(val_sharpes_arr))
    result.attrs['inner_val_preds'] = inner_val_out
    result.attrs['inner_val_start'] = inner_val_out[date_col].min()
    result.attrs['inner_val_end'] = inner_val_out[date_col].max()
    result.attrs['inner_val_years'] = sorted(inner_val_out[date_col].dt.year.unique().tolist())
    result.attrs['inner_val_mode'] = inner_val_mode
    result.attrs['inner_val_fraction'] = inner_val_fraction
    result.attrs['n_inner_val'] = len(inner_val_out)
    result.attrs['train_end_year'] = train_end_year
    result.attrs['train_end_date'] = train_end_label
    result.attrs['val_start'] = result[date_col].min() if len(result) else None
    result.attrs['val_end'] = result[date_col].max() if len(result) else None
    result.attrs['val_years'] = sorted(result[date_col].dt.year.unique().tolist()) if len(result) else []
    print(f'  train_end={train_end_label}  n_train_win={len(X_tr_win):,}  '
          f'n_inner_val={len(inner_val_out):,}  n_val={len(X_val_win):,}  '
          f'epochs={len(train_losses)}')
    return result


def walk_forward_transformer(
    df: pd.DataFrame,
    x_cols: List[str],
    cond_cols: List[str],
    model_class: Type[nn.Module],
    model_kwargs: dict,
    target_col: str = 'ret_fopen',
    date_col: str = 'msgStamp',
    trade_date_col: str = 'trade_date',
    window_spec: Optional[WindowSpec] = None,
    x_clip: float = 3.0,
    y_clip_quantile: Optional[float] = 0.01,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    epochs: int = 100,
    batch_size: int = 1024,
    patience: int = 10,
    lr_scheduler_patience: int = 3,
    warmup_fraction: float = 0.01,
    grad_clip: float = 1.0,
    device: Optional[torch.device] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Walk-forward causal temporal transformer.

    lookback is read from model_kwargs (default 32). Each fold creates sliding
    windows of length lookback. The last lookback-1 training rows serve as left
    context for the first test windows (no leakage — they are training data).
    Inner val (last training year) is used only for early stopping.
    Predictions are normalised by train-window prediction std.
    """
    lookback = model_kwargs.get('lookback', 32)

    if window_spec is None:
        window_spec = WindowSpec(mode='expanding')
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).reset_index(drop=True)
    data['year'] = data[date_col].dt.to_period('Y')
    if trade_date_col not in data.columns:
        data[trade_date_col] = data[date_col].dt.date

    schedule = build_yearly_schedule(data, date_col, window_spec)
    outputs = []
    last_train_losses: list = []
    last_val_losses: list = []
    last_train_sharpes: list = []
    last_val_sharpes: list = []
    last_grad_norms: list = []

    for step in schedule:
        train_years, pred_year = step['train_years'], step['pred_year']

        train_df = data.loc[data['year'].isin(train_years)].dropna(
            subset=x_cols + cond_cols + [target_col]
        ).copy()
        test_df = data.loc[data['year'] == pred_year].dropna(
            subset=x_cols + cond_cols
        ).copy()

        if len(train_df) < lookback or len(test_df) == 0:
            continue

        X_train, X_test, C_train, C_test, y_train = prepare_arrays(
            train_df, test_df, x_cols, cond_cols, target_col, x_clip, y_clip_quantile
        )

        train_years_arr = train_df['year'].values
        X_tr_win, C_tr_win, y_tr_win, yr_win = _make_train_windows(
            X_train, C_train, y_train, train_years_arr, lookback
        )
        X_test_win, C_test_win = _make_test_windows(
            X_train, C_train, X_test, C_test, lookback
        )
        train_window_rows = train_df.iloc[lookback - 1:].reset_index(drop=True)
        train_ret_win = train_window_rows[target_col].to_numpy(dtype=float)
        train_dates_win = train_window_rows[trade_date_col].to_numpy()

        # Inner val: last training year (by prediction-target year)
        iv_mask = (yr_win == train_years[-1])
        it_mask = ~iv_mask
        X_tr, C_tr, y_tr = X_tr_win[it_mask], C_tr_win[it_mask], y_tr_win[it_mask]
        X_iv, C_iv, y_iv = X_tr_win[iv_mask], C_tr_win[iv_mask], y_tr_win[iv_mask]
        train_ret, inner_val_ret = train_ret_win[it_mask], train_ret_win[iv_mask]
        train_dates, inner_val_dates = train_dates_win[it_mask], train_dates_win[iv_mask]
        if len(X_tr) == 0:
            X_tr, C_tr, y_tr = X_tr_win, C_tr_win, y_tr_win
            X_iv, C_iv, y_iv = X_tr_win, C_tr_win, y_tr_win
            train_ret = inner_val_ret = train_ret_win
            train_dates = inner_val_dates = train_dates_win

        model = model_class(
            n_x_features=len(x_cols),
            n_cond_features=len(cond_cols),
            **model_kwargs,
        )
        model, train_losses, val_losses, grad_norms, train_sharpes, val_sharpes = fit_transformer(
            model, X_tr, C_tr, y_tr, X_iv, C_iv, y_iv,
            device=device, lr=lr, weight_decay=weight_decay,
            epochs=epochs, batch_size=batch_size, patience=patience,
            lr_scheduler_patience=lr_scheduler_patience,
            warmup_fraction=warmup_fraction,
            grad_clip=grad_clip,
            train_returns=train_ret, val_returns=inner_val_ret,
            train_dates=train_dates, val_dates=inner_val_dates,
        )
        last_train_losses = train_losses
        last_val_losses = val_losses
        last_train_sharpes = train_sharpes
        last_val_sharpes = val_sharpes
        last_grad_norms = grad_norms

        model.eval()

        def predict(X: np.ndarray, C: np.ndarray, chunk: int = 1024) -> np.ndarray:
            parts = []
            with torch.no_grad():
                for i in range(0, len(X), chunk):
                    parts.append(model(
                        torch.tensor(X[i:i + chunk], dtype=torch.float32).to(device),
                        torch.tensor(C[i:i + chunk], dtype=torch.float32).to(device),
                    ).cpu().numpy())
            return np.concatenate(parts)

        train_std = predict(X_tr_win, C_tr_win).std() + 1e-12
        pred_test = predict(X_test_win, C_test_win) / train_std

        out_cols = [date_col, trade_date_col]
        if target_col in test_df.columns:
            out_cols.append(target_col)
        out = test_df[out_cols].copy()
        out['pred'] = pred_test
        out['pred_year'] = str(pred_year)
        out['train_start_year'] = str(train_years[0])
        out['train_end_year'] = str(train_years[-1])
        outputs.append(out)
        print(f'  pred_year={pred_year}  n_train_win={len(X_tr_win):,}  n_test={len(X_test):,}')

    if not outputs:
        return pd.DataFrame(columns=[date_col, trade_date_col, 'pred'])

    result = pd.concat(outputs).sort_values(date_col).reset_index(drop=True)
    result.attrs['train_losses'] = last_train_losses
    result.attrs['val_losses'] = last_val_losses
    result.attrs['train_sharpes'] = last_train_sharpes
    result.attrs['val_sharpes'] = last_val_sharpes
    result.attrs['grad_norms'] = last_grad_norms
    last_val_sharpes_arr = np.asarray(last_val_sharpes, dtype=float)
    if last_val_sharpes_arr.size and np.isfinite(last_val_sharpes_arr).any():
        result.attrs['best_epoch_by_val_sharpe'] = int(np.nanargmax(last_val_sharpes_arr)) + 1
        result.attrs['best_val_sharpe'] = float(np.nanmax(last_val_sharpes_arr))
    return result
