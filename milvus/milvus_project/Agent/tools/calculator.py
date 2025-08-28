# tools/financial_calculator.py
import numpy as np
from llama_index.core.tools import FunctionTool


def calculate_return_rate(start_price: float, end_price: float) -> str:
    """
    根据一个起始价格和一个结束价格计算简单收益率。
    这个工具只接受两个独立的数字作为输入。
    """
    if start_price == 0:
        return "错误：起始价格不能为零。"
    simple_return = (end_price - start_price) / start_price
    return f"计算出的简单收益率是: {simple_return:.4%}"


def calculate_volatility(daily_prices_str: str) -> str:
    """
    根据一个以逗号分隔的日度价格字符串，计算年化波动率。
    例如，输入 "100.0, 102.5, 101.0"。
    """
    try:
        # 在 Python 代码中处理字符串到列表的转换，而不是让 LLM 处理
        prices = [float(p.strip()) for p in daily_prices_str.split(',')]
        if len(prices) < 2:
            return "错误：价格数据至少需要两个点。"

        log_returns = np.log(np.array(prices[1:]) / np.array(prices[:-1]))
        daily_volatility = np.std(log_returns)
        annualized_volatility = daily_volatility * np.sqrt(252)
        return f"计算出的年化波动率是: {annualized_volatility:.4%}"
    except Exception as e:
        return f"处理价格字符串时出错: {e}。请确保输入是以逗号分隔的数字字符串。"

# 将函数包装成 LlamaIndex Tools
return_rate_tool = FunctionTool.from_defaults(fn=calculate_return_rate)
volatility_tool = FunctionTool.from_defaults(fn=calculate_volatility)


financial_tools = [return_rate_tool, volatility_tool]