import numpy as np
import pandas as pd
from WindPy import w

def calculate_volatility(prices: pd.Series) -> float:
    """内部函数，根据价格序列计算年化波动率"""
    daily_returns = prices.pct_change().dropna()
    # 假设一年有 252 个交易日
    return daily_returns.std() * np.sqrt(252)

w.start()

wind_data = w.wsd("000300.SH", "close", "20250812", "20250911", Period="D")
volatility = calculate_volatility( pd.Series(data=wind_data.Data[0], index=wind_data.Times))
print(wind_data)
print(volatility)