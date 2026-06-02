"""Modal training functions for XGBoost and Transformer models.

Usage from a notebook or script:
    import modal, io, pickle, pandas as pd
    train_xgboost = modal.Function.lookup('regime-trading-model', 'train_xgboost')
    raw = train_xgboost.remote(xgb_params={...})
    result = pickle.loads(raw)
    preds = pd.read_parquet(io.BytesIO(result['preds']))
    for k, v in result.get('attrs', {}).items():
        preds.attrs[k] = v

Upload data once:
    python modal_train.py upload  # uploads data/model_data.parquet to volume
"""
from __future__ import annotations

import datetime as _datetime
import io
import pickle
import sys

import modal

# ---------------------------------------------------------------------------
# Image: includes torch + all deps; src/ is mounted at /src
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install(
        'pandas>=2.0',
        'numpy',
        'scikit-learn',
        'xgboost==1.7.6',
        'torch',
        'pyarrow',
        'matplotlib',
    )
    .add_local_dir('src', '/src')
)

volume = modal.Volume.from_name('regime-trading-model-data', create_if_missing=True)
app = modal.App('regime-trading-model')

DATA_PATH = '/data/model_data.parquet'


def _pack_preds(preds) -> bytes:
    """Serialize predictions while keeping rich attrs out of Parquet metadata."""
    def pack_attr(value):
        if hasattr(value, 'to_parquet'):
            buf = io.BytesIO()
            attr_df = value.copy(deep=False)
            attr_df.attrs.clear()
            attr_df.to_parquet(buf, index=False)
            return {
                '__attr_type__': 'dataframe_parquet',
                'data': buf.getvalue(),
            }
        if isinstance(value, dict):
            return {k: pack_attr(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [pack_attr(v) for v in value]
        if isinstance(value, _datetime.datetime):
            return value.isoformat()
        if isinstance(value, _datetime.date):
            return value.isoformat()
        if hasattr(value, 'isoformat') and value.__class__.__module__.startswith('pandas'):
            return value.isoformat()
        if hasattr(value, 'item') and value.__class__.__module__.startswith('numpy'):
            return value.item()
        if hasattr(value, 'tolist') and value.__class__.__module__.startswith('numpy'):
            return value.tolist()
        return value

    attrs = {k: pack_attr(v) for k, v in preds.attrs.items()}
    parquet_preds = preds.copy(deep=False)
    parquet_preds.attrs.clear()

    buf = io.BytesIO()
    parquet_preds.to_parquet(buf, index=False)
    return pickle.dumps({'preds': buf.getvalue(), 'attrs': attrs})


def _load_data():
    sys.path.insert(0, '/src')
    import pandas as pd
    df = pd.read_parquet(DATA_PATH)
    df['cond2'] = df['cond2'].ffill().bfill()
    df['cond3'] = df['cond3'].ffill().bfill()
    x_cols = [c for c in df.columns if c.startswith('x')]
    cond_cols = [c for c in df.columns if c.startswith('con')]
    return df, x_cols, cond_cols


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={'/data': volume},
    timeout=3600,
    memory=4096,
)
def train_xgboost(
    xgb_params: dict | None = None,
    x_clip: float = 3.0,
    y_clip_quantile: float = 0.01,
    early_stopping_rounds: int = 20,
    random_state: int = 42,
) -> bytes:
    """Train walk-forward XGBoost and return predictions as parquet bytes."""
    sys.path.insert(0, '/src')
    sys.path.insert(0, '/')
    from models.xgboost_model import walk_forward_xgboost
    from preprocessing import WindowSpec

    df, x_cols, cond_cols = _load_data()
    preds = walk_forward_xgboost(
        df=df,
        x_cols=x_cols,
        cond_cols=cond_cols,
        window_spec=WindowSpec(mode='expanding'),
        xgb_params=xgb_params,
        x_clip=x_clip,
        y_clip_quantile=y_clip_quantile,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
    )
    return _pack_preds(preds)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={'/data': volume},
    gpu='T4',
    timeout=3600,
    memory=8192,
)
def train_transformer(
    model_type: str = 'typed_attention',
    model_kwargs: dict | None = None,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    epochs: int = 100,
    batch_size: int = 1024,
    patience: int = 10,
    lr_scheduler_patience: int = 3,
    warmup_fraction: float = 0.01,
    x_clip: float = 3.0,
    y_clip_quantile: float = 0.01,
    random_state: int = 42,
) -> bytes:
    """Train walk-forward causal temporal transformer and return predictions as parquet bytes."""
    import torch
    sys.path.insert(0, '/src')
    sys.path.insert(0, '/')
    from models.transformer import (
        DirectedRegimeAttentionTransformer,
        RegimeBiasAttentionTransformer,
        TypedAttentionTransformer,
        walk_forward_transformer,
    )
    from preprocessing import WindowSpec

    model_map = {
        'typed_attention': TypedAttentionTransformer,
        'typed_joint_attention': TypedAttentionTransformer,
        'directed_regime_attention': DirectedRegimeAttentionTransformer,
        'regime_bias_attention': RegimeBiasAttentionTransformer,
    }
    if model_type not in model_map:
        raise ValueError(f'Unknown model_type: {model_type}. Choose from {list(model_map)}')

    if model_kwargs is None:
        model_kwargs = {
            'd_model': 32, 'n_heads': 4, 'n_layers': 2,
            'ffn_dim': 128, 'dropout': 0.3,
        }

    df, x_cols, cond_cols = _load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    preds = walk_forward_transformer(
        df=df,
        x_cols=x_cols,
        cond_cols=cond_cols,
        model_class=model_map[model_type],
        model_kwargs=model_kwargs,
        window_spec=WindowSpec(mode='expanding'),
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        lr_scheduler_patience=lr_scheduler_patience,
        warmup_fraction=warmup_fraction,
        x_clip=x_clip,
        y_clip_quantile=y_clip_quantile,
        device=device,
        random_state=random_state,
    )
    return _pack_preds(preds)


# ---------------------------------------------------------------------------
# Fixed-split (fast HP search): train once ≤ train_end_year, predict val_years
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={'/data': volume},
    gpu='T4',
    timeout=3600,
    memory=8192,
)
def train_transformer_fixed(
    model_type: str = 'typed_attention',
    model_kwargs: dict | None = None,
    train_end_year: int = 2021,
    train_end_date: str | None = None,
    val_years: list | None = None,
    val_start: str | None = None,
    val_end: str | None = None,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    epochs: int = 100,
    batch_size: int = 1024,
    patience: int = 10,
    lr_scheduler_patience: int = 3,
    warmup_fraction: float = 0.01,
    inner_val_mode: str = 'tail_fraction',
    inner_val_fraction: float = 0.15,
    x_clip: float = 3.0,
    y_clip_quantile: float = 0.01,
    random_state: int = 42,
) -> bytes:
    """Train once on historical data and predict the held-out validation window."""
    import torch
    sys.path.insert(0, '/src')
    sys.path.insert(0, '/')
    from models.transformer import (
        DirectedRegimeAttentionTransformer,
        RegimeBiasAttentionTransformer,
        TypedAttentionTransformer,
        train_predict_fixed_split,
    )

    model_map = {
        'typed_attention': TypedAttentionTransformer,
        'typed_joint_attention': TypedAttentionTransformer,
        'directed_regime_attention': DirectedRegimeAttentionTransformer,
        'regime_bias_attention': RegimeBiasAttentionTransformer,
    }
    if model_type not in model_map:
        raise ValueError(f'Unknown model_type: {model_type}. Choose from {list(model_map)}')

    if model_kwargs is None:
        model_kwargs = {
            'd_model': 32, 'n_heads': 4, 'n_layers': 2,
            'ffn_dim': 128, 'dropout': 0.3,
        }
    if val_years is None:
        val_years = [2022, 2023]

    df, x_cols, cond_cols = _load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    preds = train_predict_fixed_split(
        df=df, x_cols=x_cols, cond_cols=cond_cols,
        model_class=model_map[model_type], model_kwargs=model_kwargs,
        train_end_year=train_end_year, train_end_date=train_end_date,
        val_years=val_years,
        val_start=val_start, val_end=val_end,
        lr=lr, weight_decay=weight_decay, grad_clip=grad_clip,
        epochs=epochs, batch_size=batch_size, patience=patience,
        lr_scheduler_patience=lr_scheduler_patience,
        warmup_fraction=warmup_fraction,
        inner_val_mode=inner_val_mode, inner_val_fraction=inner_val_fraction,
        x_clip=x_clip, y_clip_quantile=y_clip_quantile,
        device=device, random_state=random_state,
    )
    return _pack_preds(preds)


@app.function(
    image=image,
    volumes={'/data': volume},
    timeout=3600,
    memory=4096,
)
def train_xgboost_fixed(
    xgb_params: dict | None = None,
    train_end_year: int = 2021,
    train_end_date: str | None = None,
    val_years: list | None = None,
    val_start: str | None = None,
    val_end: str | None = None,
    val_end_date: str | None = None,
    x_clip: float = 3.0,
    y_clip_quantile: float = 0.01,
    early_stopping_rounds: int = 25,
    random_state: int = 42,
    verbose: bool = False,
) -> bytes:
    """Train once on historical data and predict a held-out window. Returns parquet bytes."""
    sys.path.insert(0, '/src')
    sys.path.insert(0, '/')
    from models.xgboost_model import train_predict_fixed_split_xgboost

    if val_years is None:
        val_years = [2022, 2023]

    df, x_cols, cond_cols = _load_data()
    preds = train_predict_fixed_split_xgboost(
        df=df, x_cols=x_cols, cond_cols=cond_cols,
        train_end_year=train_end_year,
        train_end_date=train_end_date,
        val_years=val_years,
        val_start=val_start,
        val_end=val_end,
        val_end_date=val_end_date,
        xgb_params=xgb_params, x_clip=x_clip, y_clip_quantile=y_clip_quantile,
        early_stopping_rounds=early_stopping_rounds, random_state=random_state,
        verbose=verbose,
    )
    return _pack_preds(preds)


if __name__ == '__main__':
    import pathlib
    import sys as _sys

    if len(_sys.argv) > 1 and _sys.argv[1] == 'upload':
        for candidate in [
            pathlib.Path('data/model_data.parquet'),
            pathlib.Path('notebooks/model_data.parquet'),
            pathlib.Path('model_data.parquet'),
            pathlib.Path('../data/model_data.parquet'),
        ]:
            if candidate.exists():
                local_path = candidate
                break
        else:
            print('Error: model_data.parquet not found.')
            _sys.exit(1)
        print(f'Uploading {local_path} ({local_path.stat().st_size / 1e6:.1f} MB) ...')
        with volume.batch_upload() as batch:
            batch.put_file(str(local_path), 'model_data.parquet')
        print(f'Done. File available at {DATA_PATH} inside Modal containers.')
    else:
        print('Usage: python modal_train.py upload')
        print('       modal deploy modal_train.py')
