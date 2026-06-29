"""
阶段3：回测框架搭建 + 双均线策略示例
======================================
从零搭建 qltrader 风格的回测框架，支持日频/月频调仓。

核心类：
  DataLoader   - 从本地CSV加载数据，提供 current()/history() 接口
  Position     - 单只股票持仓管理
  Portfolio    - 组合管理（总资产、现金、持仓、净值）
  BacktestEngine - 回测引擎（佣金、滑点、定时任务）

示例策略：
  双均线策略（5日线金叉20日线买入，死叉卖出）

使用方法：
    python 03_回测引擎搭建.py
"""

import sys
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DAILY_DIR = DATA_DIR / "daily"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 中文字体配置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# DataLoader - 数据加载器
# ============================================================
class DataLoader:
    """从本地CSV加载日线数据，提供 current()/history() 接口"""

    def __init__(self, data_dir: Path = DAILY_DIR):
        self.data_dir = data_dir
        self._cache: Dict[str, pd.DataFrame] = {}
        self._all_dates: list = []
        self._current_idx: int = 0
        self._current_date: Optional[pd.Timestamp] = None

    def load(self, symbols: list = None, start_date: str = "2015-01-01",
             end_date: str = "2025-06-20"):
        """加载数据到内存"""
        files = sorted(self.data_dir.glob("*.csv"))
        if symbols:
            files = [f for f in files if f.stem in symbols]

        logger.info(f"加载 {len(files)} 只股票数据 ...")
        for f in files:
            try:
                df = pd.read_csv(f, parse_dates=["date"])
                df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
                if not df.empty:
                    df = df.sort_values("date").set_index("date")
                    self._cache[f.stem] = df
            except Exception:
                pass

        # 构建交易日历
        all_dates = set()
        for df in self._cache.values():
            all_dates.update(df.index)
        self._all_dates = sorted(all_dates)
        logger.info(f"加载完成: {len(self._cache)} 只股票, {len(self._all_dates)} 个交易日")
        return self

    @property
    def current_date(self) -> pd.Timestamp:
        return self._current_date

    @property
    def trading_dates(self) -> list:
        return self._all_dates

    def set_date(self, date: pd.Timestamp):
        """设置当前日期"""
        self._current_date = date

    def current(self, symbol: str, field: str = "close") -> float:
        """获取当前日期的指定字段值"""
        if symbol not in self._cache:
            return np.nan
        df = self._cache[symbol]
        if self._current_date in df.index:
            return df.loc[self._current_date, field]
        return np.nan

    def history(self, symbol: str, field: str = "close",
                window: int = 20) -> pd.Series:
        """获取截至当前日期的历史窗口数据"""
        if symbol not in self._cache:
            return pd.Series(dtype=float)
        df = self._cache[symbol]
        mask = df.index <= self._current_date
        return df.loc[mask, field].tail(window)

    def get_symbols(self) -> list:
        return list(self._cache.keys())

    def has_data(self, symbol: str) -> bool:
        if symbol not in self._cache:
            return False
        return self._current_date in self._cache[symbol].index


# ============================================================
# Position - 持仓
# ============================================================
@dataclass
class Position:
    """单只股票持仓"""
    symbol: str
    shares: int
    cost_price: float
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def cost_value(self) -> float:
        return self.shares * self.cost_price

    @property
    def pnl(self) -> float:
        return self.market_value - self.cost_value

    @property
    def pnl_pct(self) -> float:
        if self.cost_value == 0:
            return 0.0
        return self.pnl / self.cost_value


# ============================================================
# Portfolio - 组合管理
# ============================================================
class Portfolio:
    """组合管理（总资产、现金、持仓列表）"""

    def __init__(self, initial_cash: float = 1_000_000.0,
                 commission_rate: float = 0.0015,
                 slippage_rate: float = 0.001):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.commission_rate = commission_rate  # 双边佣金
        self.slippage_rate = slippage_rate      # 滑点

        # 记录
        self.nav_history: list = []   # [(date, nav)]
        self.trade_history: list = []  # [(date, symbol, side, price, shares, amount)]

    @property
    def total_market_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def total_value(self) -> float:
        return self.cash + self.total_market_value

    @property
    def nav(self) -> float:
        return self.total_value / self.initial_cash

    def update_prices(self, data: DataLoader):
        """更新所有持仓的当前价格"""
        for sym, pos in self.positions.items():
            price = data.current(sym, "close")
            if not np.isnan(price):
                pos.current_price = price

    def order_shares(self, symbol: str, shares: int, price: float,
                     date: pd.Timestamp = None):
        """按股数下单（正数买入，负数卖出）"""
        if shares == 0 or np.isnan(price) or price <= 0:
            return

        if shares > 0:
            # 买入
            actual_price = price * (1 + self.slippage_rate)
            cost = shares * actual_price
            commission = cost * self.commission_rate
            total_cost = cost + commission

            if total_cost > self.cash:
                # 资金不足，调整买入数量
                shares = int(self.cash / (actual_price * (1 + self.commission_rate)) / 100) * 100
                if shares <= 0:
                    return
                cost = shares * actual_price
                commission = cost * self.commission_rate
                total_cost = cost + commission

            self.cash -= total_cost

            if symbol in self.positions:
                pos = self.positions[symbol]
                total_shares = pos.shares + shares
                pos.cost_price = (pos.cost_price * pos.shares + actual_price * shares) / total_shares
                pos.shares = total_shares
            else:
                self.positions[symbol] = Position(symbol, shares, actual_price, actual_price)

            self.trade_history.append((date, symbol, "BUY", actual_price, shares, total_cost))

        else:
            # 卖出
            sell_shares = abs(shares)
            if symbol not in self.positions or self.positions[symbol].shares < sell_shares:
                if symbol in self.positions:
                    sell_shares = self.positions[symbol].shares
                else:
                    return

            actual_price = price * (1 - self.slippage_rate)
            proceeds = sell_shares * actual_price
            commission = proceeds * self.commission_rate
            net_proceeds = proceeds - commission

            self.cash += net_proceeds

            pos = self.positions[symbol]
            pos.shares -= sell_shares
            if pos.shares <= 0:
                del self.positions[symbol]

            self.trade_history.append((date, symbol, "SELL", actual_price, sell_shares, net_proceeds))

    def order_target_percent(self, symbol: str, target_pct: float,
                             price: float, date: pd.Timestamp = None):
        """按目标持仓比例下单"""
        if np.isnan(price) or price <= 0:
            return

        target_value = self.total_value * target_pct
        current_value = 0
        if symbol in self.positions:
            current_value = self.positions[symbol].market_value

        diff_value = target_value - current_value
        shares = int(np.floor(diff_value / price / 100)) * 100  # 统一向下取整，避免卖出不足

        if shares != 0:
            self.order_shares(symbol, shares, price, date)

    def record_nav(self, date: pd.Timestamp):
        """记录当日净值"""
        self.nav_history.append((date, self.nav))

    def get_nav_df(self) -> pd.DataFrame:
        """获取净值序列"""
        df = pd.DataFrame(self.nav_history, columns=["date", "nav"])
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")


# ============================================================
# BacktestEngine - 回测引擎
# ============================================================
class BacktestEngine:
    """日频/月频调仓回测引擎"""

    def __init__(self, data: DataLoader, portfolio: Portfolio):
        self.data = data
        self.portfolio = portfolio
        self._schedules: list = []  # [(frequency, func)]

    def schedule(self, func: Callable, frequency: str = "monthly"):
        """
        注册定时任务。
        frequency: 'daily', 'weekly', 'monthly'
        """
        self._schedules.append((frequency, func))

    def _is_rebalance_day(self, date: pd.Timestamp, frequency: str) -> bool:
        """判断是否为调仓日"""
        if frequency == "daily":
            return True
        elif frequency == "weekly":
            return date.weekday() == 4  # 周五
        elif frequency == "monthly":
            # 每月最后一个交易日
            next_dates = [d for d in self.data.trading_dates if d > date]
            if next_dates:
                return date.month != next_dates[0].month
            return True
        return False

    def run(self, start_date: str = "2015-01-05", end_date: str = "2025-06-20"):
        """运行回测"""
        logger.info(f"回测区间: {start_date} ~ {end_date}")
        logger.info(f"初始资金: {self.portfolio.initial_cash:,.0f}")

        dates = [d for d in self.data.trading_dates
                 if pd.Timestamp(start_date) <= d <= pd.Timestamp(end_date)]

        for date in dates:
            self.data.set_date(date)
            self.portfolio.update_prices(self.data)

            # 执行定时任务
            for freq, func in self._schedules:
                if self._is_rebalance_day(date, freq):
                    func(self.data, self.portfolio, date)

            self.portfolio.update_prices(self.data)
            self.portfolio.record_nav(date)

        logger.info(f"回测完成: {len(dates)} 个交易日, {len(self.portfolio.trade_history)} 笔交易")
        return self.portfolio.get_nav_df()


# ============================================================
# 示例策略：双均线策略
# ============================================================
class DualMAStrategy:
    """双均线策略：5日线金叉20日线买入，死叉卖出"""

    def __init__(self, symbol: str, short_window: int = 5, long_window: int = 20):
        self.symbol = symbol
        self.short_window = short_window
        self.long_window = long_window
        self.holding = False

    def __call__(self, data: DataLoader, portfolio: Portfolio, date: pd.Timestamp):
        # 注：本策略使用当日收盘价计算均线并同时交易，存在 look-ahead bias。
        # 实盘中应使用前一日信号决定今日开盘交易。
        short_ma = data.history(self.symbol, "close", self.short_window).mean()
        long_ma = data.history(self.symbol, "close", self.long_window).mean()
        price = data.current(self.symbol, "close")

        if np.isnan(short_ma) or np.isnan(long_ma) or np.isnan(price):
            return

        if short_ma > long_ma and not self.holding:
            # 金叉买入
            portfolio.order_target_percent(self.symbol, 0.95, price, date)
            if self.symbol in portfolio.positions:
                self.holding = True
                logger.debug(f"{date.date()} BUY {self.symbol} @ {price:.2f}")

        elif short_ma < long_ma and self.holding:
            # 死叉卖出
            if self.symbol in portfolio.positions:
                shares = portfolio.positions[self.symbol].shares
                portfolio.order_shares(self.symbol, -shares, price, date)
                self.holding = False
                logger.debug(f"{date.date()} SELL {self.symbol} @ {price:.2f}")


# ============================================================
# 绩效分析
# ============================================================
class PerformanceAnalyzer:
    """回测绩效分析器"""

    def __init__(self, nav_df: pd.DataFrame, benchmark_df: pd.DataFrame = None):
        self.nav = nav_df
        self.benchmark = benchmark_df

    def calc_metrics(self) -> dict:
        """计算核心绩效指标"""
        nav = self.nav["nav"]
        returns = nav.pct_change().dropna()

        # 总收益率
        total_return = (nav.iloc[-1] / nav.iloc[0]) - 1

        # 年化收益率
        years = (nav.index[-1] - nav.index[0]).days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1

        # 年化波动率
        annual_vol = returns.std() * np.sqrt(252)

        # 夏普比率（假设无风险利率2.5%）
        rf = 0.025
        sharpe = (annual_return - rf) / annual_vol if annual_vol > 0 else 0

        # 最大回撤
        cummax = nav.cummax()
        drawdown = (nav - cummax) / cummax
        max_drawdown = drawdown.min()

        # 最大回撤持续天数
        dd_start = drawdown.idxmin()
        dd_peak = cummax[:dd_start].idxmax() if dd_start in cummax.index else dd_start

        # 月度胜率
        monthly_returns = returns.resample("ME").sum()
        win_rate = (monthly_returns > 0).sum() / len(monthly_returns) if len(monthly_returns) > 0 else 0

        # 卡尔玛比率
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 索提诺比率
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (annual_return - rf) / downside_vol if downside_vol > 0 else 0

        return {
            "总收益率": f"{total_return:.2%}",
            "年化收益率": f"{annual_return:.2%}",
            "年化波动率": f"{annual_vol:.2%}",
            "夏普比率": f"{sharpe:.3f}",
            "最大回撤": f"{max_drawdown:.2%}",
            "卡尔玛比率": f"{calmar:.3f}",
            "索提诺比率": f"{sortino:.3f}",
            "月度胜率": f"{win_rate:.2%}",
            "交易天数": len(nav),
            "回测年数": f"{years:.1f}",
        }

    def plot_nav_curve(self, save_path: Path = None):
        """净值曲线图（策略 vs 基准）"""
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.nav.index, self.nav["nav"], label="策略净值", linewidth=1.5, color="#e74c3c")

        if self.benchmark is not None:
            bench_nav = self.benchmark["nav"] / self.benchmark["nav"].iloc[0]
            ax.plot(bench_nav.index, bench_nav, label="沪深300基准", linewidth=1, color="#3498db", alpha=0.7)

        ax.set_title("策略净值曲线", fontsize=14)
        ax.set_xlabel("日期")
        ax.set_ylabel("净值")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"净值曲线已保存: {save_path}")
        plt.close()

    def plot_drawdown(self, save_path: Path = None):
        """回撤曲线图"""
        nav = self.nav["nav"]
        cummax = nav.cummax()
        drawdown = (nav - cummax) / cummax

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.fill_between(drawdown.index, drawdown.values, 0, color="#e74c3c", alpha=0.4)
        ax.plot(drawdown.index, drawdown.values, color="#c0392b", linewidth=0.8)
        ax.set_title("策略回撤曲线", fontsize=14)
        ax.set_xlabel("日期")
        ax.set_ylabel("回撤幅度")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"回撤曲线已保存: {save_path}")
        plt.close()

    def plot_monthly_heatmap(self, save_path: Path = None):
        """月度收益热力图"""
        returns = self.nav["nav"].pct_change().dropna()
        monthly = returns.resample("ME").sum()

        # 构建年-月矩阵
        monthly_df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        })
        pivot = monthly_df.pivot_table(values="return", index="year", columns="month", aggfunc="sum")
        pivot.columns = [f"{m}月" for m in pivot.columns]

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=-0.15, vmax=0.15)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        # 添加数值标注
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=8)

        plt.colorbar(im, ax=ax, label="月度收益率")
        ax.set_title("月度收益热力图", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"月度热力图已保存: {save_path}")
        plt.close()


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 70)
    print("  A股多因子量化选股回测系统 — 阶段3：回测引擎 + 双均线策略")
    print("=" * 70)

    # Step 1: 加载数据
    data = DataLoader(DAILY_DIR)
    data.load(start_date="2015-01-01", end_date="2025-06-20")

    if not data.get_symbols():
        logger.error("没有可用数据，请先运行阶段1。")
        sys.exit(1)

    # 选择一只股票做示例（000001 平安银行）
    test_symbol = "000001"
    if test_symbol not in data.get_symbols():
        test_symbol = data.get_symbols()[0]

    logger.info(f"示例股票: {test_symbol}")

    # Step 2: 创建组合和引擎
    portfolio = Portfolio(
        initial_cash=1_000_000,
        commission_rate=0.0015,
        slippage_rate=0.001,
    )

    engine = BacktestEngine(data, portfolio)

    # Step 3: 注册双均线策略
    strategy = DualMAStrategy(test_symbol, short_window=5, long_window=20)
    engine.schedule(strategy, frequency="daily")

    # Step 4: 运行回测
    nav_df = engine.run(start_date="2015-01-05", end_date="2025-06-20")

    # Step 5: 绩效分析
    analyzer = PerformanceAnalyzer(nav_df)
    metrics = analyzer.calc_metrics()

    print("\n" + "=" * 70)
    print(f"  双均线策略绩效报告（{test_symbol}）")
    print("=" * 70)
    for k, v in metrics.items():
        print(f"  {k:<12}: {v}")
    print("=" * 70)

    # Step 6: 交易记录
    print(f"\n交易记录（共 {len(portfolio.trade_history)} 笔）:")
    print("-" * 75)
    print(f"  {'日期':<12} {'操作':<6} {'股票':<8} {'价格':<10} {'股数':<10} {'金额':<15}")
    print("-" * 75)
    for trade in portfolio.trade_history[:30]:
        date, sym, side, price, shares, amount = trade
        print(f"  {str(date.date()):<12} {side:<6} {sym:<8} {price:<10.2f} {shares:<10} {amount:<15,.0f}")
    if len(portfolio.trade_history) > 30:
        print(f"  ... 共 {len(portfolio.trade_history)} 笔交易，仅显示前30笔")
    print("-" * 75)

    # Step 7: 可视化
    print("\n生成图表 ...")
    analyzer.plot_nav_curve(FIGURES_DIR / "fig1_nav_curve.png")
    analyzer.plot_drawdown(FIGURES_DIR / "fig2_drawdown.png")
    analyzer.plot_monthly_heatmap(FIGURES_DIR / "fig3_monthly_heatmap.png")

    # Step 8: 保存每日净值
    nav_df.to_csv(OUTPUT_DIR / "daily_nav.csv", encoding="utf-8-sig")
    logger.info(f"每日净值已保存: {OUTPUT_DIR / 'daily_nav.csv'}")

    print(f"\n图表已保存到: {FIGURES_DIR}")
    print()


if __name__ == "__main__":
    main()
