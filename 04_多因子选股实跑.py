"""
阶段4：多因子选股策略实跑
======================================
基于阶段2计算的因子，执行完整的多因子选股策略。

功能：
1. 因子有效性检验（IC分析）
2. 分层回测（五分位组合）
3. 多因子合成（IC_IR加权）
4. 绩效报告

使用方法：
    python 04_多因子选股实跑.py
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

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
FACTORS_DIR = DATA_DIR / "factors"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 数据加载
# ============================================================
def load_monthly_returns() -> pd.DataFrame:
    """计算每月月末收盘价和下月收益率"""
    logger.info("加载日线数据并计算月度收益率 ...")
    files = sorted(DAILY_DIR.glob("*.csv"))

    close_dict = {}
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=["date"])
            if len(df) > 60:
                close_dict[f.stem] = df.set_index("date")["close"]
        except Exception:
            pass

    close_panel = pd.DataFrame(close_dict)
    monthly_close = close_panel.resample("ME").last()

    # 下月收益率
    monthly_returns = monthly_close.pct_change().shift(-1)
    logger.info(f"月度数据: {len(monthly_close)} 个月, {monthly_close.shape[1]} 只股票")
    return monthly_returns


def load_factors() -> Dict[str, pd.DataFrame]:
    """加载所有原始因子（不含中性化）"""
    factors = {}
    for f in sorted(FACTORS_DIR.glob("*.csv")):
        if f.stem.endswith("_neutral"):
            continue
        try:
            df = pd.read_csv(f, parse_dates=[0], index_col=0)
            if df.shape[0] > 0 and df.shape[1] > 0:
                factors[f.stem] = df
        except Exception:
            pass
    logger.info(f"加载 {len(factors)} 个因子")
    return factors


# ============================================================
# IC分析器
# ============================================================
class ICAnalyzer:
    """因子有效性检验（Rank IC 分析）"""

    def __init__(self, factors: Dict[str, pd.DataFrame], monthly_returns: pd.DataFrame):
        self.factors = factors
        self.monthly_returns = monthly_returns
        self.ic_series: Dict[str, pd.Series] = {}

    def calc_ic(self) -> pd.DataFrame:
        """计算每个因子每月的 Rank IC"""
        logger.info("计算 Rank IC ...")
        ic_results = {}

        for fname, factor_panel in self.factors.items():
            ic_list = []
            dates = []

            for date in factor_panel.index:
                if date not in self.monthly_returns.index:
                    continue

                factor_values = factor_panel.loc[date].dropna()
                return_values = self.monthly_returns.loc[date].dropna()

                # 取交集
                common = factor_values.index.intersection(return_values.index)
                if len(common) < 5:
                    continue

                f_vals = factor_values[common]
                r_vals = return_values[common]

                # Spearman Rank IC
                ic, _ = stats.spearmanr(f_vals, r_vals)
                ic_list.append(ic)
                dates.append(date)

            if ic_list:
                ic_series = pd.Series(ic_list, index=pd.DatetimeIndex(dates), name=fname)
                ic_results[fname] = ic_series
                self.ic_series[fname] = ic_series

        self.ic_df = pd.DataFrame(ic_results)
        logger.info(f"IC 计算完成: {len(ic_results)} 个因子")
        return self.ic_df

    def summary(self) -> pd.DataFrame:
        """输出 IC 统计摘要"""
        rows = []
        for fname, ic in self.ic_series.items():
            ic_clean = ic.dropna()
            if len(ic_clean) < 6:
                continue
            rows.append({
                "因子": fname,
                "IC均值": f"{ic_clean.mean():.4f}",
                "IC标准差": f"{ic_clean.std():.4f}",
                "IR": f"{ic_clean.mean() / ic_clean.std():.3f}" if ic_clean.std() > 0 else "N/A",
                "IC>0占比": f"{(ic_clean > 0).mean():.2%}",
                "IC绝对值均值": f"{ic_clean.abs().mean():.4f}",
                "t统计量": f"{ic_clean.mean() / ic_clean.std() * np.sqrt(len(ic_clean)):.2f}" if ic_clean.std() > 0 else "N/A",
            })
        return pd.DataFrame(rows)

    def yearly_summary(self) -> pd.DataFrame:
        """按年度拆解的 IC 表"""
        yearly_data = {}
        for fname, ic in self.ic_series.items():
            for year, group in ic.groupby(ic.index.year):
                if year not in yearly_data:
                    yearly_data[year] = {}
                yearly_data[year][fname] = group.mean()

        df = pd.DataFrame(yearly_data).T
        df.index.name = "年份"
        return df


# ============================================================
# 分层回测
# ============================================================
class LayeredBacktest:
    """分层回测（五分位组合分析）"""

    def __init__(self, factor_panel: pd.DataFrame, monthly_returns: pd.DataFrame,
                 n_groups: int = 5):
        self.factor_panel = factor_panel
        self.monthly_returns = monthly_returns
        self.n_groups = n_groups
        self.group_returns: pd.DataFrame = None

    def run(self) -> pd.DataFrame:
        """运行分层回测"""
        group_returns_dict = {f"G{i+1}": [] for i in range(self.n_groups)}
        group_returns_dict["long_short"] = []
        dates = []

        for date in self.factor_panel.index:
            if date not in self.monthly_returns.index:
                continue

            factor_values = self.factor_panel.loc[date].dropna()
            return_values = self.monthly_returns.loc[date].dropna()

            common = factor_values.index.intersection(return_values.index)
            if len(common) < self.n_groups * 2:
                continue

            f_vals = factor_values[common]
            r_vals = return_values[common].clip(-0.99, 2.0)  # 裁剪极端收益，上限200%

            # 按因子值分组
            try:
                groups = pd.qcut(f_vals, self.n_groups, labels=False, duplicates="drop")
            except ValueError:
                continue

            dates.append(date)
            for g in range(self.n_groups):
                stocks = groups[groups == g].index
                group_returns_dict[f"G{g+1}"].append(r_vals[stocks].mean())

            # 多空组合
            group_returns_dict["long_short"].append(
                group_returns_dict[f"G{self.n_groups}"][-1] - group_returns_dict["G1"][-1]
            )

        self.group_returns = pd.DataFrame(group_returns_dict, index=pd.DatetimeIndex(dates))
        return self.group_returns

    def performance_metrics(self) -> pd.DataFrame:
        """计算各组绩效指标"""
        if self.group_returns is None:
            return pd.DataFrame()

        rows = []
        for col in self.group_returns.columns:
            ret = self.group_returns[col]
            cum_ret = (1 + ret).cumprod()

            total_ret = cum_ret.iloc[-1] - 1
            years = len(ret) / 12
            annual_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
            annual_vol = ret.std() * np.sqrt(12)
            sharpe = (annual_ret - 0.025) / annual_vol if annual_vol > 0 else 0

            cummax = cum_ret.cummax()
            max_dd = ((cum_ret - cummax) / cummax).min()

            win_rate = (ret > 0).mean()

            rows.append({
                "组合": col,
                "累计收益": f"{total_ret:.2%}",
                "年化收益": f"{annual_ret:.2%}",
                "年化波动": f"{annual_vol:.2%}",
                "夏普比率": f"{sharpe:.3f}",
                "最大回撤": f"{max_dd:.2%}",
                "月度胜率": f"{win_rate:.2%}",
            })

        return pd.DataFrame(rows)


# ============================================================
# 多因子合成
# ============================================================
class MultiFactorComposite:
    """多因子合成（IC_IR 加权）"""

    def __init__(self, ic_analyzer: ICAnalyzer):
        self.ic_analyzer = ic_analyzer

    def select_factors(self, min_ir: float = 0.1) -> list:
        """筛选 IR > 阈值的因子"""
        summary = self.ic_analyzer.summary()
        selected = []

        for _, row in summary.iterrows():
            try:
                ir = float(row["IR"]) if row["IR"] != "N/A" else 0
                ic_gt0 = float(row["IC>0占比"].strip("%")) / 100

                if ir > min_ir and ic_gt0 > 0.5:
                    selected.append(row["因子"])
            except (ValueError, TypeError):
                continue

        # 如果没有符合条件的因子，取 IR 最高的几个
        if not selected and not summary.empty:
            ir_values = []
            for _, row in summary.iterrows():
                try:
                    ir = float(row["IR"]) if row["IR"] != "N/A" else 0
                    ir_values.append((row["因子"], ir))
                except (ValueError, TypeError):
                    continue
            ir_values.sort(key=lambda x: abs(x[1]), reverse=True)
            selected = [name for name, ir in ir_values[:5] if abs(ir) > 0]

        logger.info(f"筛选出 {len(selected)} 个因子: {selected}")
        return selected

    def composite(self, selected_factors: list) -> pd.DataFrame:
        """IC_IR 加权合成复合因子"""
        if not selected_factors:
            logger.warning("没有有效因子，使用所有因子的等权合成")
            selected_factors = list(self.ic_analyzer.ic_series.keys())

        # 计算每个因子的 IC_IR 权重
        weights = {}
        for fname in selected_factors:
            ic = self.ic_analyzer.ic_series.get(fname)
            if ic is not None:
                ir = ic.mean() / ic.std() if ic.std() > 0 else 0
                weights[fname] = ir

        # 归一化权重
        total = sum(abs(v) for v in weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        logger.info("因子权重:")
        for fname, w in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
            logger.info(f"  {fname}: {w:.4f}")

        # 合成（需要先标准化每个因子）
        composite_panel = None
        for fname, w in weights.items():
            factor_file = FACTORS_DIR / f"{fname}.csv"
            if not factor_file.exists():
                continue

            panel = pd.read_csv(factor_file, parse_dates=[0], index_col=0)

            # 截面标准化
            mean = panel.mean(axis=1)
            std = panel.std(axis=1).replace(0, np.nan)
            normalized = panel.sub(mean, axis=0).div(std, axis=0)

            if composite_panel is None:
                composite_panel = normalized * w
            else:
                # 对齐日期和股票
                common_dates = composite_panel.index.intersection(normalized.index)
                common_stocks = composite_panel.columns.intersection(normalized.columns)
                composite_panel = composite_panel.loc[common_dates, common_stocks]
                normalized_aligned = normalized.loc[common_dates, common_stocks]
                composite_panel = composite_panel + normalized_aligned * w

        return composite_panel


# ============================================================
# 可视化
# ============================================================
class Visualizer:
    """结果可视化"""

    @staticmethod
    def plot_ic_heatmap(ic_df: pd.DataFrame, save_path: Path = None):
        """IC 时间序列热力图"""
        if ic_df.empty:
            logger.warning("IC 数据为空，跳过热力图")
            return

        # 按年度汇总
        yearly_ic = ic_df.groupby(ic_df.index.year).mean()
        if yearly_ic.empty:
            return

        fig, ax = plt.subplots(figsize=(16, 8))
        im = ax.imshow(yearly_ic.values.T, cmap="RdYlGn_r", aspect="auto", vmin=-0.1, vmax=0.1)

        ax.set_xticks(range(len(yearly_ic.index)))
        ax.set_xticklabels(yearly_ic.index, rotation=45)
        ax.set_yticks(range(len(yearly_ic.columns)))
        ax.set_yticklabels(yearly_ic.columns, fontsize=8)

        for i in range(len(yearly_ic.columns)):
            for j in range(len(yearly_ic.index)):
                val = yearly_ic.values[j, i]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)

        plt.colorbar(im, ax=ax, label="Rank IC")
        ax.set_title("因子 IC 年度热力图", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_layered_nav(group_returns: pd.DataFrame, factor_name: str,
                         save_path: Path = None):
        """分层净值曲线"""
        fig, ax = plt.subplots(figsize=(14, 6))
        # 填充缺失月份为0收益，避免净值曲线断裂
        cum_ret = (1 + group_returns.fillna(0)).cumprod()

        # 高区分度配色：G1~G5 从冷色到暖色，long_short 用黑色
        palette = {
            "G1": "#1f77b4",  # 蓝
            "G2": "#2ca02c",  # 绿
            "G3": "#ff7f0e",  # 橙
            "G4": "#d62728",  # 红
            "G5": "#9467bd",  # 紫
            "long_short": "#333333",  # 深灰
        }

        for col in cum_ret.columns:
            c = palette.get(col, "#888888")
            lw = 2.0 if col == "long_short" else 1.2
            ls = "--" if col == "long_short" else "-"
            ax.plot(cum_ret.index, cum_ret[col], label=col,
                    linewidth=lw, linestyle=ls, color=c)

        ax.set_title(f"{factor_name} 分层净值曲线", fontsize=14)
        ax.set_xlabel("日期")
        ax.set_ylabel("净值")
        ax.legend(ncol=3, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color="gray", linewidth=0.5, linestyle=":")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_ic_bar_chart(summary_df: pd.DataFrame, save_path: Path = None):
        """IC 均值柱状图"""
        if summary_df.empty:
            logger.warning("IC 摘要为空，跳过柱状图")
            return
        fig, ax = plt.subplots(figsize=(14, 6))

        names = summary_df["因子"].values
        ic_means = [float(v) for v in summary_df["IC均值"].values]
        colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in ic_means]

        ax.barh(range(len(names)), ic_means, color=colors, alpha=0.7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("IC 均值")
        ax.set_title("因子 IC 均值", fontsize=14)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 70)
    print("  A股多因子量化选股回测系统 — 阶段4：多因子选股实跑")
    print("=" * 70)

    # Step 1: 加载数据
    monthly_returns = load_monthly_returns()
    factors = load_factors()

    if not factors:
        logger.error("没有因子数据，请先运行阶段2。")
        sys.exit(1)

    # Step 2: IC 分析
    ic_analyzer = ICAnalyzer(factors, monthly_returns)
    ic_df = ic_analyzer.calc_ic()

    # IC 摘要
    summary = ic_analyzer.summary()
    print("\n" + "=" * 100)
    print("  因子 IC 统计摘要")
    print("=" * 100)
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.width", 120)
    print(summary.to_string(index=False))
    print("=" * 100)

    # 年度 IC
    yearly_ic = ic_analyzer.yearly_summary()
    print("\n年度 IC 均值:")
    print(yearly_ic.to_string(float_format=lambda x: f"{x:.4f}"))

    # Step 3: 分层回测（对前5个动量因子）
    print("\n" + "=" * 100)
    print("  分层回测结果")
    print("=" * 100)

    key_factors = ["ret_1m", "ret_6m", "reversal", "volatility_20d", "amihud_20d"]
    all_layered_results = {}

    for fname in key_factors:
        factor_file = FACTORS_DIR / f"{fname}.csv"
        if not factor_file.exists():
            continue

        factor_panel = pd.read_csv(factor_file, parse_dates=[0], index_col=0)
        layered = LayeredBacktest(factor_panel, monthly_returns, n_groups=5)
        group_returns = layered.run()

        if group_returns is not None and len(group_returns) > 0:
            metrics = layered.performance_metrics()
            all_layered_results[fname] = {"metrics": metrics, "returns": group_returns}

            print(f"\n--- {fname} ---")
            print(metrics.to_string(index=False))

            # 保存图表
            Visualizer.plot_layered_nav(
                group_returns, fname, FIGURES_DIR / f"fig4_layered_{fname}.png"
            )

    # Step 4: 多因子合成
    print("\n" + "=" * 100)
    print("  多因子合成")
    print("=" * 100)

    composite = MultiFactorComposite(ic_analyzer)
    selected = composite.select_factors(min_ir=0.1)

    if selected:
        composite_panel = composite.composite(selected)
        if composite_panel is not None and not composite_panel.empty:
            layered = LayeredBacktest(composite_panel, monthly_returns, n_groups=5)
            group_returns = layered.run()

            if group_returns is not None and len(group_returns) > 0:
                metrics = layered.performance_metrics()
                print("\n复合因子分层回测:")
                print(metrics.to_string(index=False))

                Visualizer.plot_layered_nav(
                    group_returns, "复合因子",
                    FIGURES_DIR / "fig5_composite_layered.png"
                )

    # Step 5: 可视化
    print("\n生成图表 ...")
    Visualizer.plot_ic_heatmap(ic_df, FIGURES_DIR / "fig6_ic_heatmap.png")
    Visualizer.plot_ic_bar_chart(summary, FIGURES_DIR / "fig7_ic_bar_chart.png")

    # Step 6: 保存结果
    summary.to_csv(OUTPUT_DIR / "ic_summary.csv", index=False, encoding="utf-8-sig")
    ic_df.to_csv(OUTPUT_DIR / "ic_series.csv", encoding="utf-8-sig")
    logger.info(f"结果已保存到 {OUTPUT_DIR}")

    print(f"\n图表已保存到: {FIGURES_DIR}")
    print("阶段4完成！")
    print()


if __name__ == "__main__":
    main()
