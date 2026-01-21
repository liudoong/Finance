"""
FX期权波动率数据下载参数生成器

功能：根据货币对（如 "EURUSD"）生成完整波动率曲面所需的所有时间序列下载参数

用法：
    from fx_data_generator import generate_fx_vol_parameters

    # 生成EURUSD的所有126条时间序列参数
    params = generate_fx_vol_parameters("EURUSD")

    # params 是一个包含126个字典的列表，每个字典有三个键：
    # - 'label': 标签字符串
    # - 'expression': 表达式字符串
    # - 'description': 描述信息（可选，用于调试）
"""

def generate_fx_vol_parameters(currency_pair: str) -> list:
    """
    生成指定货币对的完整波动率曲面下载参数

    参数:
        currency_pair (str): 货币对字符串，如 "EURUSD", "GBPJPY" 等

    返回:
        list: 包含126个参数字典的列表，每个字典包含：
            - label (str): 数据标签
            - expression (str): 下载表达式
            - description (str): 描述信息
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

    # 定义期权类型配置
    # 格式: (label_suffix, expression_code, description)
    option_configs = [
        # ATM (没有delta)
        ("Implied Vol", "", "ATM"),

        # 25 delta
        ("25d call vol", "25C", "25 Delta Call"),
        ("25d put vol", "25P", "25 Delta Put"),
        ("25d risk reversal", "25RR", "25 Delta Risk Reversal"),
        ("25d butterfly", "25BF", "25 Delta Butterfly"),

        # 10 delta
        ("10d call vol", "10C", "10 Delta Call"),
        ("10d put vol", "10P", "10 Delta Put"),
        ("10d risk reversal", "10RR", "10 Delta Risk Reversal"),
        ("10d butterfly", "10BF", "10 Delta Butterfly"),
    ]

    parameters = []

    # 遍历所有期限
    for tenor_name, tenor_code in tenors:
        # 遍历所有期权类型
        for label_suffix, option_code, description in option_configs:
            # 构建label
            label = f"{currency_pair} {label_suffix} {tenor_name}"

            # 构建expression
            expression = f"FX[{currency_pair}{option_code}{tenor_code}]"

            # 添加到参数列表
            parameters.append({
                'label': label,
                'expression': expression,
                'description': f"{currency_pair} - {tenor_name} - {description}"
            })

    return parameters


def print_parameters(currency_pair: str, max_display: int = 20):
    """
    打印生成的参数（用于调试和验证）

    参数:
        currency_pair (str): 货币对字符串
        max_display (int): 最多显示多少条（默认20条，设为None显示全部）
    """
    params = generate_fx_vol_parameters(currency_pair)

    print(f"\n{'='*80}")
    print(f"货币对: {currency_pair}")
    print(f"总共生成: {len(params)} 条时间序列参数")
    print(f"{'='*80}\n")

    display_count = len(params) if max_display is None else min(max_display, len(params))

    for i, param in enumerate(params[:display_count], 1):
        print(f"{i:3d}. Label:       {param['label']}")
        print(f"     Expression:  {param['expression']}")
        print(f"     Description: {param['description']}")
        print()

    if max_display and len(params) > max_display:
        print(f"... 还有 {len(params) - max_display} 条参数未显示 ...\n")

    print(f"{'='*80}")
    print(f"验证: 14个期限 × 9个期权类型 = {14 * 9} 条时间序列")
    print(f"实际生成: {len(params)} 条")
    print(f"{'='*80}\n")


def get_parameters_by_tenor(currency_pair: str, tenor: str) -> list:
    """
    获取特定期限的所有参数（9条）

    参数:
        currency_pair (str): 货币对
        tenor (str): 期限，如 "1 Month", "3 Month", "1 Year" 等

    返回:
        list: 该期限的9条参数
    """
    all_params = generate_fx_vol_parameters(currency_pair)
    return [p for p in all_params if tenor in p['label']]


def get_parameters_by_type(currency_pair: str, option_type: str) -> list:
    """
    获取特定期权类型的所有参数（14条 - 对应14个期限）

    参数:
        currency_pair (str): 货币对
        option_type (str): 期权类型，如 "ATM", "25d call", "10d put" 等

    返回:
        list: 该期权类型的14条参数
    """
    all_params = generate_fx_vol_parameters(currency_pair)

    # 构建搜索关键词
    if option_type.upper() == "ATM":
        search_key = "Implied Vol"
    else:
        search_key = option_type.lower()

    return [p for p in all_params if search_key in p['label'].lower()]


# 示例用法
if __name__ == "__main__":
    # 示例1: 生成EURUSD的所有参数
    print("\n【示例1】生成 EURUSD 的前20条参数:")
    print_parameters("EURUSD", max_display=20)

    # 示例2: 生成GBPJPY的所有参数
    print("\n【示例2】生成 GBPJPY 的前10条参数:")
    print_parameters("GBPJPY", max_display=10)

    # 示例3: 获取特定期限的参数
    print("\n【示例3】获取 EURUSD 4 Month 的所有参数:")
    params_4m = get_parameters_by_tenor("EURUSD", "4 Month")
    for i, p in enumerate(params_4m, 1):
        print(f"{i}. {p['label']} -> {p['expression']}")

    # 示例4: 获取特定类型的参数
    print("\n【示例4】获取 EURUSD 所有 25d call 的参数:")
    params_25c = get_parameters_by_type("EURUSD", "25d call")
    for i, p in enumerate(params_25c, 1):
        print(f"{i}. {p['label']} -> {p['expression']}")

    # 示例5: 在实际下载函数中使用
    print("\n【示例5】模拟下载流程:")
    print("-" * 80)

    def mock_download_function(label, expression, start_date):
        """模拟的下载函数"""
        return f"下载成功: {label} | {expression} | 起始日期: {start_date}"

    # 获取EURUSD的前3条参数
    eurusd_params = generate_fx_vol_parameters("EURUSD")[:3]
    start_date = "2020-01-01"

    for param in eurusd_params:
        result = mock_download_function(
            label=param['label'],
            expression=param['expression'],
            start_date=start_date
        )
        print(result)

    print("-" * 80)
