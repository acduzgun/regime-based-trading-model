from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class WindowSpec:
    mode: str           # 'rolling' or 'expanding'
    train_years: int = 4
    min_train_years: int = 4


def build_yearly_schedule(
    df: pd.DataFrame,
    date_col: str,
    window_spec: WindowSpec,
) -> List[Dict]:
    tmp = df[[date_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    years = np.array(sorted(tmp[date_col].dt.to_period('Y').unique()))

    schedule = []
    for i in range(len(years) - 1):
        pred_year = years[i + 1]

        if window_spec.mode == 'rolling':
            start_idx = i - window_spec.train_years + 1
            if start_idx < 0:
                continue
            train_years = years[start_idx:i + 1]
        elif window_spec.mode == 'expanding':
            if i + 1 < window_spec.min_train_years:
                continue
            train_years = years[:i + 1]
        else:
            raise ValueError(f"Unknown window mode: {window_spec.mode}")

        schedule.append({'train_years': train_years, 'pred_year': pred_year})

    return schedule


def fit_clip_and_scale_cond(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cond_cols: List[str],
    q_low: float = 0.01,
    q_high: float = 0.99,
    final_clip: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    C_train = train_df[cond_cols].to_numpy(dtype=float)
    C_test = test_df[cond_cols].to_numpy(dtype=float)

    lower = np.quantile(C_train, q_low, axis=0)
    upper = np.quantile(C_train, q_high, axis=0)
    C_train = np.clip(C_train, lower, upper)
    C_test = np.clip(C_test, lower, upper)

    mean = C_train.mean(axis=0)
    std = C_train.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)

    C_train = np.clip((C_train - mean) / std, -final_clip, final_clip)
    C_test = np.clip((C_test - mean) / std, -final_clip, final_clip)

    return C_train, C_test


def prepare_arrays(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    x_cols: List[str],
    cond_cols: List[str],
    target_col: str,
    x_clip: float = 3.0,
    y_clip_quantile: Optional[float] = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X_train, X_test, C_train, C_test, y_train — all clipped/scaled."""
    X_train = np.clip(train_df[x_cols].to_numpy(dtype=float), -x_clip, x_clip)
    X_test = np.clip(test_df[x_cols].to_numpy(dtype=float), -x_clip, x_clip)
    y_train = train_df[target_col].to_numpy(dtype=float)

    if y_clip_quantile is not None:
        lo, hi = np.quantile(y_train, [y_clip_quantile, 1.0 - y_clip_quantile])
        y_train = np.clip(y_train, lo, hi)

    C_train, C_test = fit_clip_and_scale_cond(train_df, test_df, cond_cols)

    return X_train, X_test, C_train, C_test, y_train
