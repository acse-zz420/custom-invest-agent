# tools/financial_calculator.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from WindPy import w
from llama_index.core.tools import FunctionTool

# --- 连接 Wind API ---
def _get_wind_connection():
    if not w.isconnected():
        w.start()
    return w

# --- 1: 计算波动率 ---
def _calculate_volatility(prices: pd.Series) -> float:
    """内部函数，根据价格序列计算年化波动率"""
    daily_returns = prices.pct_change().dropna()
    # 假设一年有 252 个交易日
    return daily_returns.std() * np.sqrt(252)

# --- 2: 计算夏普比率 ---
def _calculate_sharpe_ratio(prices: pd.Series, risk_free_rate: float = 0.02) -> float:
    """内部函数，根据价格序列计算年化夏普比率"""
    daily_returns = prices.pct_change().dropna()
    excess_returns = daily_returns - (risk_free_rate / 252)
    # 年化夏普比率
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

# --- 3: 计算最大回撤 ---
def _calculate_max_drawdown(prices: pd.Series) -> float:
    """内部函数，计算最大回撤"""
    cumulative_max = prices.cummax()
    drawdown = (prices - cumulative_max) / cumulative_max
    return drawdown.min()


def calculate_financial_metric(
        security_code: str,
        metric: str,
        period_in_days: int = 90
) -> str:
    """
    一个强大的金融指标计算器。当需要计算特定证券（股票或指数）在过去一段时间内的
    量化指标时，请使用此工具。

    支持的 'metric' (指标) 包括:
    - 'volatility': 年化波动率，衡量资产价格的波动剧烈程度。
    - 'sharpe_ratio': 夏普比率，衡量每单位风险所能带来的超额回报。
    - 'max_drawdown': 最大回撤，衡量资产在历史上的最大跌幅。
    - 'close': 获取最近一个交易日的收盘价。
    """
    print(f"--- 金融计算器接收到任务: code={security_code}, metric={metric}, days={period_in_days} ---")

    # 1. 定义一个字典，将指标名称映射到具体的计算函数
    METRIC_CALCULATORS = {
        'volatility': _calculate_volatility,
        'sharpe_ratio': _calculate_sharpe_ratio,
        'max_drawdown': _calculate_max_drawdown,
    }

    try:
        wind_conn = _get_wind_connection()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_in_days)

        # 2. 获取所需的基础数据
        print(f"正在从 Wind 获取 {security_code} 的历史收盘价...")
        api_result = wind_conn.wsd(
            security_code, "close",
            start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"),
            "Period=D;PriceAdj=F"
        )
        if api_result.ErrorCode != 0: return f"Wind 数据获取失败: {api_result.Data[0][0]}"

        prices = pd.Series(api_result.Data[0], index=api_result.Times)

        # 3. 根据 'metric' 参数，调用不同的计算逻辑
        if metric in METRIC_CALCULATORS:
            calculator_func = METRIC_CALCULATORS[metric]
            result_value = calculator_func(prices)
            # 格式化输出
            if metric in ['volatility', 'max_drawdown']:
                formatted_result = f"{result_value:.2%}"
            else:
                formatted_result = f"{result_value:.2f}"

            return (f"根据过去 {period_in_days} 天的数据计算，{security_code} 的 {metric} 指标值为: "
                    f"{formatted_result}")

        elif metric == 'close':
            latest_close = prices.iloc[-1]
            latest_date = prices.index[-1].strftime('%Y-%m-%d')
            return f"{security_code} 在 {latest_date} 的收盘价为: {latest_close:.2f}"

        else:
            supported_metrics = list(METRIC_CALCULATORS.keys()) + ['close']
            return f"错误: 不支持的指标 '{metric}'。可用指标包括: {supported_metrics}"

    except Exception as e:
        return f"执行计算时发生错误: {e}"


financial_calculator_tool = FunctionTool.from_defaults(
    fn=calculate_financial_metric,
    name="financial_metric_calculator",
    description=(
        "一个多功能的金融指标计算器，用于获取或计算股票/指数的历史量化指标。"
        "使用它来计算波动率(volatility), 夏普比率(sharpe_ratio), 最大回撤(max_drawdown), "
        "或获取最新的收盘价(close)。"
    )
)