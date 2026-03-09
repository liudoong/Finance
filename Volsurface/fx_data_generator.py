"""
FX期权波动率数据下载参数生成器

用法：
    import pandas as pd
    from fx_data_generator import generate_fx_vol_parameters

    # 生成EURUSD的所有126条时间序列参数
    df = generate_fx_vol_parameters("EURUSD")

    # df 是一个DataFrame，包含两列：
    # - 'label': 标签字符串
    # - 'expression': 表达式字符串
"""

import pandas as pd


def generate_fx_vol_parameters(currency_pair: str) -> pd.DataFrame:
    """
    生成指定货币对的完整波动率曲面下载参数

    参数:
        currency_pair (str): 货币对字符串，如 "EURUSD", "GBPJPY" 等

    返回:
        pd.DataFrame: 包含126行2列的DataFrame
            - 'label': 数据标签
            - 'expression': 下载表达式
    """

    # 定义期限及其代码映射
    tenors = [
        ("1 Week", "V1W"),
        ("1 Month", "V1M"),
        ("2 Month", "V2M"),
        ("3 Month", "V3M"),
        ("4 Month", "V4M"),
        ("5 Month", "V5M"),
        ("6 Month", "V6M"),
        ("9 Month", "V9M"),
        ("1 Year", "V1Y"),
        ("18 Month", "V18M"),
        ("2 Year", "V2Y"),
        ("3 Year", "V3Y"),
        ("4 Year", "V4Y"),
        ("5 Year", "V5Y"),
    ]

    # 定义期权类型配置（按波动率微笑顺序：从左到右）
    # 格式: (label_suffix, expression_code)
    option_configs = [
        ("25d put vol", "25P"),        # 左翼 OTM Put
        ("10d put vol", "10P"),
        ("Implied Vol", ""),           # ATM 中心
        ("10d call vol", "10C"),
        ("25d call vol", "25C"),       # 右翼 OTM Call
        ("25d risk reversal", "25RR"), # 微笑偏斜
        ("25d butterfly", "25BF"),     # 微笑凸度
        ("10d risk reversal", "10RR"),
        ("10d butterfly", "10BF"),
    ]

    labels = []
    expressions = []

    # 遍历所有期限
    for tenor_name, tenor_code in tenors:
        # 遍历所有期权类型
        for label_suffix, option_code in option_configs:
            # 构建label
            label = f"{currency_pair} {label_suffix} {tenor_name}"

            # 构建expression
            expression = f"FX[{currency_pair}{option_code}{tenor_code}]"

            labels.append(label)
            expressions.append(expression)

    # 创建DataFrame
    df = pd.DataFrame({
        'label': labels,
        'expression': expressions
    })

    return df
