from __future__ import annotations

import numpy as np
import pandas as pd


def backtest(
    df: pd.DataFrame,
    feature_col: str,
    ret_col: str,
    date_col: str = 'trade_date',
    annualization: int = 252,
) -> dict:
    data = df[[date_col, feature_col, ret_col]].dropna().copy()
    data[date_col] = pd.to_datetime(data[date_col]).dt.date

    data['alpha'] = data[feature_col].clip(-3, 3)
    data['pnl'] = data['alpha'] * data[ret_col]
    data['gross'] = data['alpha'].abs()
    stats = data.groupby(date_col, sort=True).agg(
        daily_pnl=('pnl', 'sum'),
        daily_gross=('gross', 'sum'),
    ).dropna()
    daily_pnl = stats['daily_pnl']
    daily_gross = stats['daily_gross']

    return {
        'cumulative_returns': daily_pnl.cumsum(),
        'sharpe_ratio': np.sqrt(annualization) * daily_pnl.mean() / daily_pnl.std(),
        'average_return': daily_pnl.sum() / daily_gross.sum() if daily_gross.sum() > 0 else 0.0,
    }
