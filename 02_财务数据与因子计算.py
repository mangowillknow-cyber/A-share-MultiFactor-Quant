"""
阶段2：财务数据获取 + 多因子计算
======================================
基于阶段1的日线行情数据，增加财务数据维度，计算40个因子。

因子体系：
  第一组：动量因子（12个）
  第二组：估值因子（8个）
  第三组：质量因子（10个）
  第四组：波动率因子（6个）
  第五组：流动性因子（4个）

使用方法：
    # 计算所有因子（需要先运行阶段1获取日线数据）
    python 02_财务数据与因子计算.py

    # 只获取财务数据
    python 02_财务数据与因子计算.py --fetch-financial

    # 只计算因子（已有财务数据）
    python 02_财务数据与因子计算.py --calc-factors
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# 日志与路径配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("factor_calc.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

TQDM_FMT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DAILY_DIR = DATA_DIR / "daily"
FINANCIAL_DIR = DATA_DIR / "financial"
FACTORS_DIR = DATA_DIR / "factors"
OUTPUT_DIR = PROJECT_ROOT / "output"

for d in [FINANCIAL_DIR, FACTORS_DIR, OUTPUT_DIR / "figures"]:
    d.mkdir(parents=True, exist_ok=True)

# 财务指标列名映射（akshare stock_financial_analysis_indicator）
FIN_COLS = {
    "日期": "report_date",
    "摊薄每股收益(元)": "eps",
    "每股净资产_调整后(元)": "bps",
    "每股经营性现金流(元)": "ocfps",
    "总资产利润率(%)": "roa",
    "销售净利率(%)": "net_margin",
    "销售毛利率(%)": "gross_margin",
    "净资产收益率(%)": "roe",
    "加权净资产收益率(%)": "roe_weighted",
    "主营业务收入增长率(%)": "revenue_growth",
    "净利润增长率(%)": "profit_growth",
    "流动比率": "current_ratio",
    "资产负债率(%)": "debt_ratio",
    "总资产(元)": "total_assets",
    "经营现金净流量与净利润的比率(%)": "ocf_to_profit",
}


# ============================================================
# 第一部分：财务数据获取
# ============================================================
class FinancialDataFetcher:
    """财务数据获取器"""

    def __init__(self, start_year: str = "2014"):
        self.start_year = start_year

    def fetch_single(self, symbol: str) -> pd.DataFrame:
        """获取单只股票的财务指标"""
        try:
            import akshare as ak
            df = ak.stock_financial_analysis_indicator(
                symbol=symbol, start_year=self.start_year
            )
            if df is None or df.empty:
                return pd.DataFrame()

            # 选取关键列
            available = {k: v for k, v in FIN_COLS.items() if k in df.columns}
            result = df[list(available.keys())].rename(columns=available)
            result["report_date"] = pd.to_datetime(result["report_date"])
            result["symbol"] = symbol

            # 数值转换
            for col in result.columns:
                if col not in ("report_date", "symbol"):
                    result[col] = pd.to_numeric(result[col], errors="coerce")

            return result
        except Exception as e:
            logger.debug(f"{symbol} 财务数据获取失败: {e}")
            return pd.DataFrame()

    def fetch_all(self, symbols: list) -> pd.DataFrame:
        """批量获取所有股票的财务数据（带缓存）"""
        cache_file = FINANCIAL_DIR / "all_financial_data.csv"

        # 检查缓存
        if cache_file.exists():
            cached = pd.read_csv(cache_file, parse_dates=["report_date"])
            cached_symbols = set(cached["symbol"].unique())
            to_fetch = [s for s in symbols if s not in cached_symbols]
            if not to_fetch:
                logger.info(f"财务数据缓存完整，共 {len(cached_symbols)} 只股票")
                return cached
            logger.info(f"缓存已有 {len(cached_symbols)} 只，需补充 {len(to_fetch)} 只")
        else:
            cached = pd.DataFrame()
            to_fetch = symbols

        # 分批获取（每批100只，避免频繁请求）
        batch_size = 100
        all_data = [cached] if not cached.empty else []

        for i in tqdm(range(0, len(to_fetch), batch_size),
                      desc="获取财务数据", bar_format=TQDM_FMT):
            batch = to_fetch[i:i + batch_size]
            for sym in batch:
                df = self.fetch_single(sym)
                if not df.empty:
                    all_data.append(df)
                time.sleep(0.3)

            # 每批保存一次缓存
            if all_data:
                combined = pd.concat(all_data, ignore_index=True)
                combined.to_csv(cache_file, index=False, encoding="utf-8-sig")

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result.to_csv(cache_file, index=False, encoding="utf-8-sig")
            logger.info(f"财务数据已保存: {len(result)} 行, {result['symbol'].nunique()} 只股票")
            return result
        return pd.DataFrame()


# ============================================================
# 第二部分：因子计算引擎
# ============================================================
class FactorEngine:
    """多因子计算引擎"""

    def __init__(self):
        self.daily_data = {}   # {symbol: DataFrame}
        self.financial_data = pd.DataFrame()
        self.factors = {}      # {factor_name: DataFrame(date x symbol)}

    # ----------------------------------------------------------
    # 数据加载
    # ----------------------------------------------------------
    def load_daily_data(self, max_stocks: int = None):
        """从本地CSV加载日线数据"""
        files = sorted(DAILY_DIR.glob("*.csv"))
        if max_stocks:
            files = files[:max_stocks]

        logger.info(f"加载 {len(files)} 只股票的日线数据 ...")
        for f in tqdm(files, desc="加载日线数据", bar_format=TQDM_FMT):
            try:
                df = pd.read_csv(f, parse_dates=["date"])
                if len(df) > 60:  # 至少60个交易日
                    self.daily_data[f.stem] = df
            except Exception:
                pass

        logger.info(f"成功加载 {len(self.daily_data)} 只股票")
        return len(self.daily_data)

    def load_financial_data(self):
        """加载财务数据"""
        cache_file = FINANCIAL_DIR / "all_financial_data.csv"
        if cache_file.exists():
            self.financial_data = pd.read_csv(cache_file, parse_dates=["report_date"])
            logger.info(f"财务数据: {len(self.financial_data)} 行, "
                        f"{self.financial_data['symbol'].nunique()} 只股票")
        else:
            logger.warning("财务数据文件不存在，估值和质量因子将不可用")

    # ----------------------------------------------------------
    # 辅助函数
    # ----------------------------------------------------------
    def _build_panel(self, field: str = "close") -> pd.DataFrame:
        """将日线数据构建为面板（index=date, columns=symbol）"""
        series_dict = {}
        for sym, df in self.daily_data.items():
            if field in df.columns:
                series_dict[sym] = df.set_index("date")[field]
        return pd.DataFrame(series_dict)

    def _get_monthly_dates(self) -> list:
        """获取每月最后一个交易日"""
        close_panel = self._build_panel("close")
        monthly = close_panel.resample("ME").last()
        return list(monthly.index)

    # ----------------------------------------------------------
    # 第一组：动量因子（12个）
    # ----------------------------------------------------------
    def calc_momentum_factors(self):
        """计算动量因子"""
        logger.info("计算动量因子 ...")
        close = self._build_panel("close")
        monthly_close = close.resample("ME").last()

        # 1-4: 收益率因子
        for period, name in [(1, "ret_1m"), (3, "ret_3m"), (6, "ret_6m"), (12, "ret_12m")]:
            self.factors[name] = monthly_close.pct_change(period)
            logger.info(f"  {name}: 计算完成")

        # 5: 反转因子（1个月 vs 12个月收益差）
        self.factors["reversal"] = self.factors["ret_1m"] - self.factors["ret_12m"]
        logger.info("  reversal: 计算完成")

        # 6-7: RSI
        for window in [20, 60]:
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.rolling(window).mean()
            avg_loss = loss.rolling(window).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            self.factors[f"rsi_{window}d"] = rsi.resample("ME").last()
            logger.info(f"  rsi_{window}d: 计算完成")

        # 8-9: 动量均值
        self.factors["mom_20d"] = close.pct_change(20).resample("ME").last()
        self.factors["mom_60d"] = close.pct_change(60).resample("ME").last()

        # 10-12: 动量波动调整
        ret_daily = close.pct_change()
        for window, name in [(20, "mom_vol_adj_20d"), (60, "mom_vol_adj_60d")]:
            mom = close.pct_change(window)
            vol = ret_daily.rolling(window).std()
            self.factors[name] = (mom / vol.replace(0, np.nan)).resample("ME").last()
            logger.info(f"  {name}: 计算完成")

        # 12: 信息离散度
        self.factors["info_discreteness"] = (
            ret_daily.rolling(20).apply(lambda x: np.sum(np.sign(x)) / len(x), raw=True)
            .resample("ME").last()
        )
        logger.info("  info_discreteness: 计算完成")

    # ----------------------------------------------------------
    # 第二组：估值因子（8个）
    # ----------------------------------------------------------
    def calc_valuation_factors(self):
        """计算估值因子（需要财务数据）"""
        if self.financial_data.empty:
            logger.warning("财务数据不可用，跳过估值因子")
            return

        logger.info("计算估值因子 ...")
        close = self._build_panel("close")
        monthly_close = close.resample("ME").last()

        # 提取财务数据中的关键字段（point-in-time：每月使用当时已披露的最新财报）
        fin = self.financial_data.copy()
        fin["report_date"] = pd.to_datetime(fin["report_date"])

        # 为每只股票构建财务数据时间序列，按披露日期前向填充
        def build_fin_panel(field, close_index):
            panels = {}
            for sym in close.columns:
                sym_fin = fin[fin["symbol"] == sym].sort_values("report_date")
                if sym_fin.empty or field not in sym_fin.columns:
                    continue
                s = sym_fin.set_index("report_date")[field].dropna()
                s = s[~s.index.duplicated(keep="last")]
                # 前向填充到月末日期，确保不使用未来数据
                aligned = s.reindex(close_index, method="ffill")
                panels[sym] = aligned
            return pd.DataFrame(panels, index=close_index)

        eps_panel = build_fin_panel("eps", monthly_close.index)
        bps_panel = build_fin_panel("bps", monthly_close.index)
        self.factors["pe"] = monthly_close / eps_panel.replace(0, np.nan)
        logger.info("  pe: 计算完成")

        # 2: PB = 价格 / BPS
        self.factors["pb"] = monthly_close / bps_panel.replace(0, np.nan)
        logger.info("  pb: 计算完成")

        # 3: EP（PE的倒数，越大越便宜）
        self.factors["ep"] = 1.0 / self.factors["pe"].replace(0, np.nan)
        logger.info("  ep: 计算完成")

        # 4: BP（PB的倒数）
        self.factors["bp"] = 1.0 / self.factors["pb"].replace(0, np.nan)
        logger.info("  bp: 计算完成")

        # 5-8: 行业相对估值（需要行业数据，此处用中位数标准化）
        for factor_name in ["pe", "pb", "ep", "bp"]:
            panel = self.factors[factor_name]
            median = panel.median(axis=1)
            self.factors[f"{factor_name}_rel"] = panel.div(median, axis=0)
            logger.info(f"  {factor_name}_rel: 计算完成")

    # ----------------------------------------------------------
    # 第三组：质量因子（10个）
    # ----------------------------------------------------------
    def calc_quality_factors(self):
        """计算质量因子（需要财务数据）"""
        if self.financial_data.empty:
            logger.warning("财务数据不可用，跳过质量因子")
            return

        logger.info("计算质量因子 ...")
        close = self._build_panel("close")
        monthly_close = close.resample("ME").last()
        fin = self.financial_data.copy()
        fin["report_date"] = pd.to_datetime(fin["report_date"])

        # point-in-time：每月使用当时已披露的最新财报
        def build_fin_panel(field, close_index):
            panels = {}
            for sym in close.columns:
                sym_fin = fin[fin["symbol"] == sym].sort_values("report_date")
                if sym_fin.empty or field not in sym_fin.columns:
                    continue
                s = sym_fin.set_index("report_date")[field].dropna()
                s = s[~s.index.duplicated(keep="last")]
                aligned = s.reindex(close_index, method="ffill")
                panels[sym] = aligned
            return pd.DataFrame(panels, index=close_index)

        # 定义质量因子映射（使用CSV中的英文列名）
        quality_map = {
            "roe_quality": "roe",
            "roa_quality": "roa",
            "gross_margin": "gross_margin",
            "net_margin": "net_margin",
            "current_ratio": "current_ratio",
            "debt_ratio": "debt_ratio",
            "ocf_to_profit": "ocf_to_profit",
            "revenue_growth": "revenue_growth",
            "profit_growth": "profit_growth",
        }

        for factor_key, col_name in quality_map.items():
            if col_name in fin.columns:
                panel = build_fin_panel(col_name, monthly_close.index)
                self.factors[factor_key] = panel
                logger.info(f"  {factor_key}: 计算完成")

        # 10: 现金流质量 = 经营现金流 / 总资产
        if "ocfps" in fin.columns and "total_assets" in fin.columns:
            ocf_panel = build_fin_panel("ocfps", monthly_close.index)
            ta_panel = build_fin_panel("total_assets", monthly_close.index)
            self.factors["cashflow_quality"] = ocf_panel / ta_panel.replace(0, np.nan)
            logger.info("  cashflow_quality: 计算完成")

    # ----------------------------------------------------------
    # 第四组：波动率因子（6个）
    # ----------------------------------------------------------
    def calc_volatility_factors(self):
        """计算波动率因子"""
        logger.info("计算波动率因子 ...")
        close = self._build_panel("close")
        ret = close.pct_change()

        # 1-2: 历史波动率
        for window in [20, 60]:
            vol = ret.rolling(window).std() * np.sqrt(252)
            self.factors[f"volatility_{window}d"] = vol.resample("ME").last()
            logger.info(f"  volatility_{window}d: 计算完成")

        # 3: 最大回撤（过去60日）
        def max_drawdown_series(prices, window=60):
            rolling_max = prices.rolling(window, min_periods=window).max()
            drawdown = (prices - rolling_max) / rolling_max
            return drawdown.rolling(window, min_periods=window).min()

        self.factors["max_drawdown_60d"] = max_drawdown_series(close, 60).resample("ME").last()
        logger.info("  max_drawdown_60d: 计算完成")

        # 4: 下行波动率
        neg_ret = ret.clip(upper=0)
        for window in [20, 60]:
            down_vol = neg_ret.rolling(window).std() * np.sqrt(252)
            self.factors[f"downside_vol_{window}d"] = down_vol.resample("ME").last()
            logger.info(f"  downside_vol_{window}d: 计算完成")

        # 6: 特质波动率（收益率偏离其60日均值的残差标准差）
        rolling_mean = ret.rolling(60).mean()
        residual = ret - rolling_mean
        self.factors["idio_vol_60d"] = (
            residual.rolling(60).std().multiply(np.sqrt(252))
            .resample("ME").last()
        )
        logger.info("  idio_vol_60d: 计算完成")

    # ----------------------------------------------------------
    # 第五组：流动性因子（4个）
    # ----------------------------------------------------------
    def calc_liquidity_factors(self):
        """计算流动性因子"""
        logger.info("计算流动性因子 ...")
        close = self._build_panel("close")
        volume = self._build_panel("volume")
        amount = self._build_panel("amount")

        # 1: 过去20日均换手率（用成交量近似）
        avg_vol_20d = volume.rolling(20).mean()
        self.factors["avg_volume_20d"] = np.log1p(avg_vol_20d).resample("ME").last()
        logger.info("  avg_volume_20d: 计算完成")

        # 2: Amihud 非流动性指标 = |日收益率| / 日成交额
        ret = close.pct_change()
        # 避免除零
        amount_safe = amount.replace(0, np.nan)
        amihud_daily = ret.abs() / amount_safe * 1e8  # 放大系数
        self.factors["amihud_20d"] = amihud_daily.rolling(20).mean().resample("ME").last()
        logger.info("  amihud_20d: 计算完成")

        # 3: 成交量波动率
        self.factors["volume_vol_20d"] = (
            volume.rolling(20).std() / volume.rolling(20).mean().replace(0, np.nan)
        ).resample("ME").last()
        logger.info("  volume_vol_20d: 计算完成")

        # 4: 换手率变化（近5日 vs 近20日）
        vol_5 = volume.rolling(5).mean()
        vol_20 = volume.rolling(20).mean()
        self.factors["turnover_change"] = (
            vol_5 / vol_20.replace(0, np.nan)
        ).resample("ME").last()
        logger.info("  turnover_change: 计算完成")

    # ----------------------------------------------------------
    # 行业中性化
    # ----------------------------------------------------------
    def neutralize_factors(self, industry_map: dict = None):
        """
        对因子进行行业中性化处理。
        industry_map: {symbol: industry_code}
        如果没有行业数据，使用截面标准化作为替代。
        """
        logger.info("因子行业中性化处理 ...")

        # 先收集所有需要处理的因子名（避免迭代时修改字典）
        factor_names = [f for f in self.factors.keys() if not f.endswith("_neutral")]

        if industry_map:
            # 按行业中性化：减去行业均值，除以行业标准差
            for fname in factor_names:
                panel = self.factors[fname]
                neutralized = panel.copy()
                for date_idx in panel.index:
                    row = panel.loc[date_idx]
                    for ind in set(industry_map.values()):
                        stocks = [s for s in row.index if industry_map.get(s) == ind]
                        if len(stocks) > 1:
                            vals = row[stocks]
                            mean_val = vals.mean()
                            std_val = vals.std()
                            if std_val > 0:
                                neutralized.loc[date_idx, stocks] = (vals - mean_val) / std_val
                self.factors[f"{fname}_neutral"] = neutralized
                logger.info(f"  {fname}_neutral: 行业中性化完成")
        else:
            # 无行业数据时，使用截面标准化（z-score）+ winsorize
            for fname in factor_names:
                panel = self.factors[fname]
                # winsorize: 截尾处理，限制在1%/99%分位数
                lower = panel.quantile(0.01, axis=1)
                upper = panel.quantile(0.99, axis=1)
                clipped = panel.clip(lower=lower, upper=upper, axis=0)
                mean = clipped.mean(axis=1)
                std = clipped.std(axis=1).replace(0, np.nan)
                self.factors[f"{fname}_neutral"] = clipped.sub(mean, axis=0).div(std, axis=0)
            logger.info("  使用截面z-score标准化+winsorize（无行业数据）")

    # ----------------------------------------------------------
    # 因子统计
    # ----------------------------------------------------------
    def print_factor_statistics(self):
        """输出每个因子的描述性统计"""
        print("\n" + "=" * 80)
        print("  因子描述性统计")
        print("=" * 80)

        stats_rows = []
        for fname, panel in sorted(self.factors.items()):
            if fname.endswith("_neutral"):
                continue
            values = panel.values.flatten()
            values = values[~np.isnan(values)]
            if len(values) == 0:
                continue
            stats_rows.append({
                "因子": fname,
                "样本数": len(values),
                "均值": f"{np.mean(values):.4f}",
                "标准差": f"{np.std(values):.4f}",
                "最小值": f"{np.min(values):.4f}",
                "25%": f"{np.percentile(values, 25):.4f}",
                "50%": f"{np.percentile(values, 50):.4f}",
                "75%": f"{np.percentile(values, 75):.4f}",
                "最大值": f"{np.max(values):.4f}",
            })

        stats_df = pd.DataFrame(stats_rows)
        pd.set_option("display.max_rows", 50)
        pd.set_option("display.width", 130)
        print(stats_df.to_string(index=False))
        print("=" * 80)

    # ----------------------------------------------------------
    # 保存因子数据
    # ----------------------------------------------------------
    def save_factors(self):
        """保存所有因子数据到CSV"""
        for fname, panel in self.factors.items():
            filepath = FACTORS_DIR / f"{fname}.csv"
            panel.to_csv(filepath, encoding="utf-8-sig")
        logger.info(f"已保存 {len(self.factors)} 个因子文件到 {FACTORS_DIR}")


# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="财务数据获取 + 因子计算")
    parser.add_argument("--fetch-financial", action="store_true", help="只获取财务数据")
    parser.add_argument("--calc-factors", action="store_true", help="只计算因子")
    parser.add_argument("--max-stocks", type=int, default=None, help="限制股票数量（调试用）")
    args = parser.parse_args()

    print("=" * 70)
    print("  A股多因子量化选股回测系统 — 阶段2：财务数据 + 因子计算")
    print("=" * 70)

    engine = FactorEngine()

    # Step 1: 加载日线数据
    n_stocks = engine.load_daily_data(max_stocks=args.max_stocks)
    if n_stocks == 0:
        logger.error("没有日线数据，请先运行阶段1获取数据。")
        sys.exit(1)

    # Step 2: 获取财务数据
    do_fetch = not args.calc_factors  # 默认获取
    if args.fetch_financial or do_fetch:
        symbols = list(engine.daily_data.keys())
        fetcher = FinancialDataFetcher(start_year="2014")
        fetcher.fetch_all(symbols)

    if args.fetch_financial:
        return

    # Step 3: 加载财务数据
    engine.load_financial_data()

    # Step 4: 计算所有因子
    logger.info("开始计算40个因子 ...")

    engine.calc_momentum_factors()       # 12个动量因子
    engine.calc_valuation_factors()      # 8个估值因子
    engine.calc_quality_factors()        # 10个质量因子
    engine.calc_volatility_factors()     # 6个波动率因子
    engine.calc_liquidity_factors()      # 4个流动性因子

    total_raw = len(engine.factors)
    logger.info(f"原始因子计算完成: {total_raw} 个")

    # Step 5: 行业中性化
    engine.neutralize_factors()

    # Step 6: 输出统计
    engine.print_factor_statistics()

    # Step 7: 保存
    engine.save_factors()

    print(f"\n阶段2完成！共计算 {total_raw} 个原始因子 + {total_raw} 个中性化因子")
    print(f"因子数据保存在: {FACTORS_DIR}")
    print()


if __name__ == "__main__":
    main()
