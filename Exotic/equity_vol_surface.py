"""
Equity Volatility Surface Calibration Module
============================================

专门用于从市场期权数据校准股票波动率曲面的独立模块。

支持的模型:
- Heston: 随机波动率模型
- SABR: 市场标准的波动率微笑模型

输入:
- Excel文件: 包含期权隐含波动率数据
- 模型选择: 'heston' 或 'sabr'

输出:
- QuantLib格式的波动率曲面对象
- 校准后的模型参数

使用示例:
    from equity_vol_surface import VolatilitySurfaceCalibrator

    # 方式1: 手动指定spot price
    calibrator = VolatilitySurfaceCalibrator(
        option_data_file="spx_infvol_20260109.xlsx",
        spot_price=5900.0,
        risk_free_rate=0.045,
        model_type='heston'  # 或 'sabr'
    )

    # 方式2: 自动从Yahoo Finance下载spot price (推荐)
    calibrator = VolatilitySurfaceCalibrator(
        option_data_file="spx_infvol_20260109.xlsx",
        spot_price='auto',  # 自动提取ticker并下载历史价格
        model_type='heston'
    )

    vol_surface = calibrator.calibrate()
    params = calibrator.get_parameters()
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Literal, Union
from dataclasses import dataclass
import re
import os

try:
    import QuantLib as ql
    HAS_QUANTLIB = True
except ImportError:
    HAS_QUANTLIB = False
    print("Warning: QuantLib not available. Some features will be limited.")


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class VolatilityPoint:
    """单个波动率数据点"""
    strike: float
    expiry_date: datetime
    implied_vol: float
    tenor_days: int
    moneyness: float
    option_type: str = 'P'  # P for Put, C for Call


@dataclass
class HestonParameters:
    """Heston模型参数"""
    v0: float      # 初始方差
    kappa: float   # 均值回归速度
    theta: float   # 长期方差
    sigma: float   # 波动率的波动率
    rho: float     # 相关系数

    def __str__(self):
        return f"""Heston Parameters:
  v0    (初始方差):     {self.v0:.6f}  (vol={np.sqrt(self.v0):.2%})
  theta (长期方差):     {self.theta:.6f}  (vol={np.sqrt(self.theta):.2%})
  kappa (均值回归):     {self.kappa:.4f}
  sigma (vol of vol):  {self.sigma:.4f}
  rho   (相关系数):     {self.rho:.4f}"""


@dataclass
class SABRParameters:
    """SABR模型参数 (按期限存储)"""
    alpha: Dict[str, float]  # 波动率水平 (按tenor)
    beta: float              # 弹性参数 (通常固定)
    rho: Dict[str, float]    # 相关系数 (按tenor)
    nu: Dict[str, float]     # 波动率的波动率 (按tenor)

    def __str__(self):
        result = "SABR Parameters:\n"
        result += f"  beta (固定): {self.beta:.4f}\n"
        result += "\n  按期限的参数:\n"
        for tenor in sorted(self.alpha.keys()):
            result += f"    {tenor:>6s}: α={self.alpha[tenor]:.4f}, ρ={self.rho[tenor]:.4f}, ν={self.nu[tenor]:.4f}\n"
        return result


# =============================================================================
# 主校准类
# =============================================================================

class VolatilitySurfaceCalibrator:
    """
    波动率曲面校准器

    从Excel期权数据校准Heston或SABR波动率曲面
    """

    def __init__(
        self,
        option_data_file: str,
        spot_price: Union[float, str] = 'auto',
        risk_free_rate: float = 0.045,
        dividend_yield: float = 0.0,
        model_type: Literal['heston', 'sabr'] = 'heston',
        valuation_date: Optional[datetime] = None
    ):
        """
        初始化校准器

        Args:
            option_data_file: Excel文件路径 (包含Strike, Implied Volatility等列)
            spot_price: 现货价格，可以是:
                - float: 手动指定的现货价格
                - 'auto': 从Yahoo Finance自动下载 (根据合约名提取ticker)
            risk_free_rate: 无风险利率
            dividend_yield: 股息率 (默认0)
            model_type: 'heston' 或 'sabr'
            valuation_date: 估值日期 (默认今天)
        """
        self.option_file = option_data_file
        self.spot_price_input = spot_price
        self.r = risk_free_rate
        self.q = dividend_yield
        self.model_type = model_type.lower()
        self.valuation_date = valuation_date or datetime.now()

        # 提取ticker和data_date (如果需要自动下载spot price)
        self.ticker = None
        self.data_date = None

        # 如果spot_price是'auto'，则需要从文件中提取ticker和日期
        if spot_price == 'auto':
            self._extract_ticker_and_date()
            self.spot = self._fetch_spot_from_yahoo()
        else:
            self.spot = float(spot_price)

        # 数据容器
        self.vol_points: List[VolatilityPoint] = []
        self.tenors: List[str] = []
        self.heston_params: Optional[HestonParameters] = None
        self.sabr_params: Optional[SABRParameters] = None

        # QuantLib对象
        self.ql_vol_surface = None

        print(f"初始化波动率曲面校准器:")
        print(f"  模型类型: {self.model_type.upper()}")
        if spot_price == 'auto':
            print(f"  Ticker: {self.ticker}")
            print(f"  数据日期: {self.data_date.strftime('%Y-%m-%d')}")
        print(f"  现货价格: {self.spot:,.2f}")
        print(f"  无风险利率: {self.r:.2%}")
        print(f"  估值日期: {self.valuation_date.strftime('%Y-%m-%d')}")

    def _extract_ticker_and_date(self) -> None:
        """
        从Excel文件中提取ticker和数据日期

        方法:
        1. 从文件名提取日期 (如 spx_infvol_20260109.xlsx -> 2026-01-09)
        2. 从合约名提取ticker (如 SPXW260112P02800000 -> SPX)
        """
        print("\n自动提取Ticker和数据日期...")

        # 1. 从文件名提取日期
        filename = os.path.basename(self.option_file)
        date_pattern = r'(\d{8})'  # 匹配8位数字日期
        date_match = re.search(date_pattern, filename)

        if date_match:
            date_str = date_match.group(1)
            try:
                self.data_date = datetime.strptime(date_str, '%Y%m%d')
                print(f"  从文件名提取数据日期: {self.data_date.strftime('%Y-%m-%d')}")
            except ValueError:
                self.data_date = self.valuation_date
                print(f"  无法解析日期，使用估值日期: {self.data_date.strftime('%Y-%m-%d')}")
        else:
            self.data_date = self.valuation_date
            print(f"  文件名无日期信息，使用估值日期: {self.data_date.strftime('%Y-%m-%d')}")

        # 2. 从合约名提取ticker
        try:
            df = pd.read_excel(self.option_file)

            if 'Contract Name' in df.columns:
                # 取第一个合约名
                sample_contract = df['Contract Name'].iloc[0]

                # 从合约名提取ticker
                # 格式: SPXW260112P02800000 -> ticker可能是SPX或SPXW
                # 通常是前3-4个字母
                ticker_match = re.match(r'^([A-Z]+)', sample_contract)

                if ticker_match:
                    ticker_raw = ticker_match.group(1)

                    # 处理常见的期权ticker格式
                    # SPXW -> ^SPX (SPX index)
                    # SPY -> SPY (SPY ETF)
                    # AAPL -> AAPL (stock)

                    if ticker_raw in ['SPX', 'SPXW']:
                        self.ticker = '^SPX'  # Yahoo Finance格式
                    elif ticker_raw in ['VIX', 'VIXW']:
                        self.ticker = '^VIX'
                    elif ticker_raw in ['NDX', 'NDXW']:
                        self.ticker = '^NDX'
                    else:
                        # 对于其他ticker，去掉W后缀
                        self.ticker = ticker_raw.rstrip('W')

                    print(f"  从合约名提取Ticker: {ticker_raw} -> {self.ticker}")
                else:
                    raise ValueError("无法从合约名提取ticker")
            else:
                raise ValueError("Excel文件缺少 'Contract Name' 列")

        except Exception as e:
            print(f"  警告: 提取ticker失败 ({e})")
            # 默认使用SPX
            self.ticker = '^SPX'
            print(f"  使用默认ticker: {self.ticker}")

    def _fetch_spot_from_yahoo(self) -> float:
        """
        从Yahoo Finance下载历史价格，并返回对应日期的spot price

        Returns:
            对应日期的现货价格
        """
        print(f"\n从Yahoo Finance下载 {self.ticker} 历史价格...")

        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "需要安装yfinance来自动下载spot price: pip install yfinance"
            )

        try:
            # 下载一段时间的历史数据 (前后各7天)
            start_date = self.data_date - timedelta(days=7)
            end_date = self.data_date + timedelta(days=7)

            print(f"  下载时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")

            ticker_obj = yf.Ticker(self.ticker)
            hist = ticker_obj.history(start=start_date, end=end_date)

            if hist.empty:
                raise ValueError(f"无法下载 {self.ticker} 的历史数据")

            # 找到最接近data_date的日期
            hist.index = pd.to_datetime(hist.index).tz_localize(None)

            # 尝试精确匹配
            if self.data_date.date() in hist.index.date:
                spot_price = hist.loc[hist.index.date == self.data_date.date(), 'Close'].iloc[0]
                print(f"  找到精确日期 {self.data_date.strftime('%Y-%m-%d')} 的收盘价: {spot_price:,.2f}")
            else:
                # 找最近的日期
                closest_idx = (hist.index - self.data_date).abs().argmin()
                closest_date = hist.index[closest_idx]
                spot_price = hist['Close'].iloc[closest_idx]
                print(f"  未找到精确日期，使用最近日期 {closest_date.strftime('%Y-%m-%d')} 的收盘价: {spot_price:,.2f}")

            return float(spot_price)

        except Exception as e:
            print(f"  错误: 无法从Yahoo Finance获取数据 - {e}")
            print(f"  请手动指定spot_price参数")
            raise

    def load_data(self) -> None:
        """从Excel加载期权数据"""
        print(f"\n加载数据: {self.option_file}")

        df = pd.read_excel(self.option_file)

        # 验证必需列
        required_cols = ['Strike', 'Implied Volatility']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必需列: {col}")

        # 清理数据
        df = df.dropna(subset=['Implied Volatility'])

        # 转换IV格式 (百分比 -> 小数)
        if df['Implied Volatility'].mean() > 1.0:
            df['Implied Volatility'] = df['Implied Volatility'] / 100.0

        # 提取到期日
        tenors_found = set()

        for _, row in df.iterrows():
            strike = float(row['Strike'])
            iv = float(row['Implied Volatility'])

            # 从合约名提取到期日
            if 'Contract Name' in df.columns:
                contract = row['Contract Name']
                expiry, tenor, opt_type = self._parse_contract_name(contract)
            else:
                # 默认1个月
                expiry = self.valuation_date + timedelta(days=30)
                tenor = "30D"
                opt_type = 'P'

            # 计算moneyness
            moneyness = strike / self.spot

            # 有效性检查
            if 0.01 <= iv <= 3.0 and 0.3 <= moneyness <= 2.0:
                tenor_days = (expiry - self.valuation_date).days

                vol_point = VolatilityPoint(
                    strike=strike,
                    expiry_date=expiry,
                    implied_vol=iv,
                    tenor_days=tenor_days,
                    moneyness=moneyness,
                    option_type=opt_type
                )

                self.vol_points.append(vol_point)
                tenors_found.add(tenor)

        self.tenors = sorted(list(tenors_found))

        print(f"  加载 {len(self.vol_points)} 个有效数据点")
        print(f"  发现 {len(self.tenors)} 个不同期限: {self.tenors}")

        # 按期限分组统计
        tenor_counts = {}
        for point in self.vol_points:
            tenor = f"{point.tenor_days}D"
            tenor_counts[tenor] = tenor_counts.get(tenor, 0) + 1

        print(f"\n  按期限分布:")
        for tenor, count in sorted(tenor_counts.items()):
            print(f"    {tenor:>6s}: {count:>3d} 个点")

    def _parse_contract_name(self, contract_name: str) -> Tuple[datetime, str, str]:
        """
        解析合约名称提取到期日

        格式: SPXW260112P02800000
              SPXW YYMMDD P KKKKKKKK
        """
        try:
            if len(contract_name) >= 10:
                date_str = contract_name[4:10]
                expiry = datetime.strptime('20' + date_str, '%Y%m%d')
                days = (expiry - self.valuation_date).days
                tenor = f"{days}D"

                # 提取期权类型
                opt_type = 'P' if 'P' in contract_name[10:12] else 'C'

                return expiry, tenor, opt_type
        except:
            pass

        # 默认值
        expiry = self.valuation_date + timedelta(days=30)
        return expiry, "30D", 'P'

    def calibrate(self) -> object:
        """
        执行校准并返回QuantLib波动率曲面

        Returns:
            QuantLib BlackVarianceSurface 或 HestonModel对象
        """
        print(f"\n{'='*80}")
        print(f"开始校准 {self.model_type.upper()} 波动率曲面")
        print(f"{'='*80}")

        # 加载数据
        self.load_data()

        # 根据模型类型校准
        if self.model_type == 'heston':
            self._calibrate_heston()
        elif self.model_type == 'sabr':
            self._calibrate_sabr()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        # 构建QuantLib曲面
        self._build_quantlib_surface()

        print(f"\n{'='*80}")
        print("校准完成 ✓")
        print(f"{'='*80}\n")

        return self.ql_vol_surface

    def _calibrate_heston(self) -> None:
        """校准Heston模型参数"""
        print("\n校准Heston模型参数...")

        # 计算ATM波动率
        atm_vols = [p.implied_vol for p in self.vol_points
                    if 0.95 <= p.moneyness <= 1.05]
        atm_vol = np.median(atm_vols) if atm_vols else np.median([p.implied_vol for p in self.vol_points])

        print(f"  ATM波动率: {atm_vol:.2%}")

        # 1. v0: 初始方差 (从ATM波动率)
        v0 = atm_vol ** 2

        # 2. theta: 长期方差 (简化：等于v0)
        theta = v0

        # 3. rho: 从波动率偏度估计
        rho = self._estimate_rho_from_skew()

        # 4. sigma: 从波动率曲率估计
        sigma = self._estimate_vol_of_vol()

        # 5. kappa: 均值回归速度 (典型股票值)
        kappa = 2.0

        self.heston_params = HestonParameters(
            v0=v0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            rho=rho
        )

        print(f"\n{self.heston_params}")

        # 验证Feller条件
        feller = 2 * kappa * theta
        feller_check = feller > sigma ** 2
        print(f"\nFeller条件: 2κθ={feller:.4f} {'>' if feller_check else '<'} σ²={sigma**2:.4f} {'✓' if feller_check else '✗'}")

    def _calibrate_sabr(self) -> None:
        """校准SABR模型参数 (按期限)"""
        print("\n校准SABR模型参数...")

        # SABR参数容器
        alpha_dict = {}
        rho_dict = {}
        nu_dict = {}
        beta = 0.5  # 典型股票值 (0.5 for lognormal, 1 for normal)

        # 按期限分组校准
        tenors_set = set(f"{p.tenor_days}D" for p in self.vol_points)

        for tenor in sorted(tenors_set):
            print(f"\n  校准期限: {tenor}")

            # 该期限的数据点
            tenor_points = [p for p in self.vol_points if f"{p.tenor_days}D" == tenor]

            if len(tenor_points) < 3:
                print(f"    数据点不足，跳过")
                continue

            # 提取strikes和vols
            strikes = np.array([p.strike for p in tenor_points])
            vols = np.array([p.implied_vol for p in tenor_points])

            # 计算ATM vol (作为alpha初值)
            atm_idx = np.argmin(np.abs(strikes - self.spot))
            atm_vol = vols[atm_idx]

            # 估计SABR参数
            alpha, rho, nu = self._estimate_sabr_params(
                strikes, vols, self.spot, atm_vol, beta, tenor_points[0].tenor_days
            )

            alpha_dict[tenor] = alpha
            rho_dict[tenor] = rho
            nu_dict[tenor] = nu

            print(f"    α={alpha:.4f}, ρ={rho:.4f}, ν={nu:.4f}")

        self.sabr_params = SABRParameters(
            alpha=alpha_dict,
            beta=beta,
            rho=rho_dict,
            nu=nu_dict
        )

        print(f"\n{self.sabr_params}")

    def _estimate_rho_from_skew(self) -> float:
        """从波动率偏度估计rho"""
        # OTM put vs OTM call波动率
        otm_puts = [p for p in self.vol_points if p.moneyness < 0.95]
        otm_calls = [p for p in self.vol_points if p.moneyness > 1.05]

        if otm_puts and otm_calls:
            put_vol_avg = np.mean([p.implied_vol for p in otm_puts])
            call_vol_avg = np.mean([p.implied_vol for p in otm_calls])
            skew = put_vol_avg - call_vol_avg

            # 映射: 更强的skew -> 更负的rho
            rho = np.clip(-0.5 - skew * 3.0, -0.9, -0.3)
            print(f"  从偏度估计 ρ: {rho:.4f} (skew={skew:.2%})")
            return float(rho)
        else:
            print(f"  使用默认 ρ: -0.7")
            return -0.7

    def _estimate_vol_of_vol(self) -> float:
        """从波动率方差估计sigma"""
        vols = np.array([p.implied_vol for p in self.vol_points])
        vol_variance = np.var(vols)
        sigma = np.clip(np.sqrt(vol_variance) * 3.0, 0.1, 1.0)
        print(f"  从曲率估计 σ: {sigma:.4f}")
        return float(sigma)

    def _estimate_sabr_params(
        self,
        strikes: np.ndarray,
        market_vols: np.ndarray,
        forward: float,
        atm_vol: float,
        beta: float,
        tenor_days: int
    ) -> Tuple[float, float, float]:
        """
        简化的SABR参数估计

        使用解析近似公式估计alpha, rho, nu
        """
        T = tenor_days / 365.0

        # 1. Alpha: 从ATM波动率
        # α ≈ σ_ATM * F^(1-β)
        alpha = atm_vol * (forward ** (1 - beta))

        # 2. Rho: 从波动率斜率
        # 计算25-delta put和call的vol差
        if len(strikes) >= 5:
            sorted_idx = np.argsort(strikes)
            strikes_sorted = strikes[sorted_idx]
            vols_sorted = market_vols[sorted_idx]

            # 低strike (put side) vs 高strike (call side)
            n = len(strikes_sorted)
            low_vol = np.mean(vols_sorted[:n//3])
            high_vol = np.mean(vols_sorted[-n//3:])

            # rho从skew估计
            skew = (low_vol - high_vol) / atm_vol
            rho = np.clip(-skew * 2.0, -0.9, 0.9)
        else:
            rho = -0.3  # 默认

        # 3. Nu: 从波动率曲率
        vol_var = np.var(market_vols)
        nu = np.clip(np.sqrt(vol_var) * 2.0, 0.1, 1.0)

        return alpha, rho, nu

    def _build_quantlib_surface(self) -> None:
        """构建QuantLib波动率曲面对象"""
        if not HAS_QUANTLIB:
            print("QuantLib不可用，跳过曲面构建")
            return

        print("\n构建QuantLib波动率曲面...")

        # 设置QuantLib日期
        ql.Settings.instance().evaluationDate = ql.Date(
            self.valuation_date.day,
            self.valuation_date.month,
            self.valuation_date.year
        )

        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        day_count = ql.Actual365Fixed()

        if self.model_type == 'heston':
            # Heston模型 -> HestonModel对象
            self.ql_vol_surface = self._build_heston_model(calendar, day_count)
        else:
            # SABR模型 -> BlackVarianceSurface
            self.ql_vol_surface = self._build_sabr_surface(calendar, day_count)

        print("  QuantLib曲面构建完成 ✓")

    def _build_heston_model(self, calendar, day_count) -> ql.HestonModel:
        """构建QuantLib Heston模型"""
        calculation_date = ql.Settings.instance().evaluationDate

        # 创建yield curve handles
        risk_free_curve = ql.YieldTermStructureHandle(
            ql.FlatForward(calculation_date, self.r, day_count)
        )
        dividend_curve = ql.YieldTermStructureHandle(
            ql.FlatForward(calculation_date, self.q, day_count)
        )

        # 创建Heston过程
        heston_process = ql.HestonProcess(
            risk_free_curve,
            dividend_curve,
            ql.QuoteHandle(ql.SimpleQuote(self.spot)),
            self.heston_params.v0,
            self.heston_params.kappa,
            self.heston_params.theta,
            self.heston_params.sigma,
            self.heston_params.rho
        )

        # 创建Heston模型
        heston_model = ql.HestonModel(heston_process)

        print(f"    Heston模型已创建")
        return heston_model

    def _build_sabr_surface(self, calendar, day_count) -> ql.BlackVarianceSurface:
        """构建QuantLib SABR波动率曲面"""
        calculation_date = ql.Settings.instance().evaluationDate

        # 准备数据: 按tenor和strike组织
        expiry_dates = []
        strikes_list = []

        # 按tenor分组
        tenor_groups = {}
        for point in self.vol_points:
            tenor = f"{point.tenor_days}D"
            if tenor not in tenor_groups:
                tenor_groups[tenor] = []
            tenor_groups[tenor].append(point)

        # 为每个tenor创建数据
        for tenor in sorted(tenor_groups.keys()):
            points = tenor_groups[tenor]
            if not points:
                continue

            # 使用第一个点的到期日
            expiry = points[0].expiry_date
            ql_date = ql.Date(expiry.day, expiry.month, expiry.year)

            expiry_dates.append(ql_date)
            strikes_list.append([p.strike for p in points])

        # 创建volatility matrix
        # 简化：使用market vols (未经SABR调整)
        vol_matrix = []
        for tenor in sorted(tenor_groups.keys()):
            points = sorted(tenor_groups[tenor], key=lambda x: x.strike)
            vol_row = [p.implied_vol for p in points]
            vol_matrix.append(vol_row)

        # 找出所有独特的strikes (用于surface)
        all_strikes = sorted(set(p.strike for p in self.vol_points))

        # 创建BlackVarianceSurface
        # 注意：这需要矩形网格，所以我们需要插值
        matrix = ql.Matrix(len(expiry_dates), len(all_strikes))

        for i, tenor in enumerate(sorted(tenor_groups.keys())):
            points = {p.strike: p.implied_vol for p in tenor_groups[tenor]}
            for j, strike in enumerate(all_strikes):
                # 如果该strike存在，使用实际vol，否则插值
                if strike in points:
                    matrix[i][j] = points[strike]
                else:
                    # 简单线性插值
                    matrix[i][j] = self._interpolate_vol(strike, points)

        vol_surface = ql.BlackVarianceSurface(
            calculation_date,
            calendar,
            expiry_dates,
            all_strikes,
            matrix,
            day_count
        )

        print(f"    SABR曲面已创建: {len(expiry_dates)} 期限 x {len(all_strikes)} 行权价")
        return vol_surface

    def _interpolate_vol(self, target_strike: float, strike_vol_dict: Dict[float, float]) -> float:
        """简单的线性插值"""
        strikes = sorted(strike_vol_dict.keys())

        if target_strike <= strikes[0]:
            return strike_vol_dict[strikes[0]]
        if target_strike >= strikes[-1]:
            return strike_vol_dict[strikes[-1]]

        # 找到相邻点
        for i in range(len(strikes) - 1):
            if strikes[i] <= target_strike <= strikes[i+1]:
                k1, k2 = strikes[i], strikes[i+1]
                v1, v2 = strike_vol_dict[k1], strike_vol_dict[k2]
                # 线性插值
                weight = (target_strike - k1) / (k2 - k1)
                return v1 + weight * (v2 - v1)

        return np.mean(list(strike_vol_dict.values()))

    def get_parameters(self) -> Dict:
        """
        获取校准后的参数

        Returns:
            包含模型参数的字典
        """
        if self.model_type == 'heston':
            return {
                'model_type': 'heston',
                'parameters': self.heston_params,
                'v0': self.heston_params.v0,
                'kappa': self.heston_params.kappa,
                'theta': self.heston_params.theta,
                'sigma': self.heston_params.sigma,
                'rho': self.heston_params.rho
            }
        else:
            return {
                'model_type': 'sabr',
                'parameters': self.sabr_params,
                'alpha': self.sabr_params.alpha,
                'beta': self.sabr_params.beta,
                'rho': self.sabr_params.rho,
                'nu': self.sabr_params.nu
            }

    def export_surface_data(self, output_file: str = None) -> pd.DataFrame:
        """
        导出波动率曲面数据为DataFrame

        Args:
            output_file: 可选，保存为CSV的文件名

        Returns:
            包含曲面数据的DataFrame
        """
        data = []
        for point in self.vol_points:
            data.append({
                'Strike': point.strike,
                'Expiry': point.expiry_date.strftime('%Y-%m-%d'),
                'Tenor_Days': point.tenor_days,
                'Moneyness': point.moneyness,
                'Implied_Vol': point.implied_vol,
                'Option_Type': point.option_type
            })

        df = pd.DataFrame(data)

        if output_file:
            df.to_csv(output_file, index=False)
            print(f"\n波动率曲面数据已导出: {output_file}")

        return df

    def plot_volatility_surface(self, output_file: str = None, show_plot: bool = True) -> None:
        """
        绘制波动率曲面图

        生成两个图:
        1. 波动率微笑曲线 (按期限分组)
        2. 3D波动率曲面图

        Args:
            output_file: 可选，保存图片的文件名 (如 'vol_surface.png')
            show_plot: 是否显示图片 (默认True)
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            raise ImportError("需要安装matplotlib来绘图: pip install matplotlib")

        if len(self.vol_points) == 0:
            print("警告: 没有数据点可绘制")
            return

        # 创建图形
        fig = plt.figure(figsize=(16, 6))

        # ====================================================================
        # 图1: 波动率微笑曲线 (按期限)
        # ====================================================================
        ax1 = fig.add_subplot(1, 2, 1)

        # 按期限分组
        tenor_groups = {}
        for point in self.vol_points:
            tenor = f"{point.tenor_days}D"
            if tenor not in tenor_groups:
                tenor_groups[tenor] = {'moneyness': [], 'vol': []}
            tenor_groups[tenor]['moneyness'].append(point.moneyness)
            tenor_groups[tenor]['vol'].append(point.implied_vol)

        # 绘制每个期限的微笑曲线
        colors = plt.cm.viridis(np.linspace(0, 1, len(tenor_groups)))
        for i, (tenor, data) in enumerate(sorted(tenor_groups.items())):
            # 按moneyness排序
            sorted_pairs = sorted(zip(data['moneyness'], data['vol']))
            moneyness_sorted = [m for m, v in sorted_pairs]
            vol_sorted = [v for m, v in sorted_pairs]

            ax1.plot(moneyness_sorted, vol_sorted, 'o-',
                    color=colors[i], label=tenor, linewidth=2, markersize=6)

        # 标记ATM
        ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='ATM')

        ax1.set_xlabel('Moneyness (Strike/Spot)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Implied Volatility', fontsize=12, fontweight='bold')
        ax1.set_title('波动率微笑曲线 (Volatility Smile)', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0.5, right=1.5)

        # 格式化y轴为百分比
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

        # ====================================================================
        # 图2: 3D波动率曲面
        # ====================================================================
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

        # 准备3D数据
        strikes = np.array([p.strike for p in self.vol_points])
        tenor_days = np.array([p.tenor_days for p in self.vol_points])
        vols = np.array([p.implied_vol for p in self.vol_points])

        # 创建散点图
        scatter = ax2.scatter(strikes, tenor_days, vols,
                            c=vols, cmap='viridis', s=50, alpha=0.6)

        # 如果数据点足够且有多个期限，创建网格曲面
        unique_tenors = len(np.unique(tenor_days))

        if len(self.vol_points) > 10 and unique_tenors > 1:
            try:
                from scipy.interpolate import griddata

                # 创建网格
                strike_range = np.linspace(strikes.min(), strikes.max(), 30)
                tenor_range = np.linspace(tenor_days.min(), tenor_days.max(), 20)
                strike_grid, tenor_grid = np.meshgrid(strike_range, tenor_range)

                # 插值
                vol_grid = griddata(
                    (strikes, tenor_days), vols,
                    (strike_grid, tenor_grid),
                    method='cubic'
                )

                # 绘制曲面
                ax2.plot_surface(strike_grid, tenor_grid, vol_grid,
                               cmap='viridis', alpha=0.3, edgecolor='none')
            except ImportError:
                print("提示: 安装scipy可以显示平滑的3D曲面: pip install scipy")
            except Exception as e:
                print(f"提示: 无法生成平滑3D曲面 ({e})")
        elif unique_tenors == 1:
            print("提示: 数据只有单一期限，3D曲面将仅显示散点图")

        ax2.set_xlabel('Strike', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Days to Expiry', fontsize=10, fontweight='bold')
        ax2.set_zlabel('Implied Vol', fontsize=10, fontweight='bold')
        ax2.set_title('3D波动率曲面 (3D Volatility Surface)', fontsize=12, fontweight='bold')

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax2, pad=0.1, shrink=0.8)
        cbar.set_label('Implied Volatility', fontsize=10)

        # 调整布局
        plt.tight_layout()

        # 保存图片
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"\n波动率曲面图已保存: {output_file}")

        # 显示图片
        if show_plot:
            plt.show()
        else:
            plt.close()


# =============================================================================
# 便利函数
# =============================================================================

def calibrate_vol_surface(
    option_file: str,
    spot_price: float,
    model_type: Literal['heston', 'sabr'] = 'heston',
    risk_free_rate: float = 0.045,
    **kwargs
) -> Tuple[object, Dict]:
    """
    快速校准接口

    Args:
        option_file: Excel文件路径
        spot_price: 现货价格
        model_type: 'heston' 或 'sabr'
        risk_free_rate: 无风险利率
        **kwargs: 其他参数传递给VolatilitySurfaceCalibrator

    Returns:
        (QuantLib波动率曲面对象, 参数字典)

    Example:
        vol_surface, params = calibrate_vol_surface(
            "spx_infvol_20260109.xlsx",
            spot_price=5900.0,
            model_type='heston'
        )
    """
    calibrator = VolatilitySurfaceCalibrator(
        option_data_file=option_file,
        spot_price=spot_price,
        risk_free_rate=risk_free_rate,
        model_type=model_type,
        **kwargs
    )

    vol_surface = calibrator.calibrate()
    params = calibrator.get_parameters()

    return vol_surface, params


# =============================================================================
# 测试/示例代码
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("波动率曲面校准模块 - 示例")
    print("="*80)

    # 示例1: Heston模型
    print("\n示例1: 校准Heston模型")
    print("-"*80)

    heston_calibrator = VolatilitySurfaceCalibrator(
        option_data_file="spx_infvol_20260109.xlsx",
        spot_price=5900.0,
        risk_free_rate=0.045,
        model_type='heston'
    )

    heston_surface = heston_calibrator.calibrate()
    heston_params = heston_calibrator.get_parameters()

    # 导出数据
    df = heston_calibrator.export_surface_data("vol_surface_data.csv")
    print(f"\n导出了 {len(df)} 个数据点")

    # 示例2: 自动获取spot price (推荐方式)
    print("\n" + "="*80)
    print("示例2: 使用自动spot price下载")
    print("-"*80)

    try:
        auto_calibrator = VolatilitySurfaceCalibrator(
            option_data_file="spx_infvol_20260109.xlsx",
            spot_price='auto',  # 自动从Yahoo Finance下载
            risk_free_rate=0.045,
            model_type='heston'
        )

        auto_surface = auto_calibrator.calibrate()
        auto_params = auto_calibrator.get_parameters()

        print("\n自动获取spot price成功！")
    except ImportError:
        print("\n需要安装yfinance: pip install yfinance")
    except Exception as e:
        print(f"\n自动获取spot price失败: {e}")
        print("请使用手动指定spot_price的方式")

    # 示例3: SABR模型
    print("\n" + "="*80)
    print("示例3: 校准SABR模型")
    print("-"*80)

    sabr_calibrator = VolatilitySurfaceCalibrator(
        option_data_file="spx_infvol_20260109.xlsx",
        spot_price=5900.0,
        risk_free_rate=0.045,
        model_type='sabr'
    )

    sabr_surface = sabr_calibrator.calibrate()
    sabr_params = sabr_calibrator.get_parameters()

    print("\n" + "="*80)
    print("示例完成！")
    print("="*80)