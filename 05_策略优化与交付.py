"""
阶段5：策略优化 + 最终交付
======================================
1. 参数优化（调仓频率/股票池对比）
2. 稳健性检验（特殊时期分段回测）
3. 生成最终交付物（README、绩效总结表）

使用方法：
    python 05_策略优化与交付.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# 通用回测函数
# ============================================================
def load_close_panel() -> pd.DataFrame:
    """加载收盘价面板"""
    files = sorted(DAILY_DIR.glob("*.csv"))
    series = {}
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=["date"])
            if len(df) > 60:
                series[f.stem] = df.set_index("date")["close"]
        except Exception:
            pass
    return pd.DataFrame(series)


def calc_monthly_returns(close_panel: pd.DataFrame) -> pd.DataFrame:
    """计算月度收益率"""
    monthly = close_panel.resample("ME").last()
    return monthly.pct_change()


def simple_quintile_backtest(factor_panel: pd.DataFrame,
                              monthly_returns: pd.DataFrame,
                              n_groups: int = 5) -> dict:
    """简单五分位回测，返回各组绩效"""
    group_rets = {f"G{i+1}": [] for i in range(n_groups)}
    group_rets["long_short"] = []

    for date in factor_panel.index:
        if date not in monthly_returns.index:
            continue
        fv = factor_panel.loc[date].dropna()
        rv = monthly_returns.loc[date].dropna().clip(-0.99, 2.0)
        common = fv.index.intersection(rv.index)
        if len(common) < n_groups * 2:
            continue

        try:
            groups = pd.qcut(fv[common], n_groups, labels=False, duplicates="drop")
        except ValueError:
            continue

        for g in range(n_groups):
            stocks = groups[groups == g].index
            group_rets[f"G{g+1}"].append(rv[stocks].mean())
        group_rets["long_short"].append(
            group_rets[f"G{n_groups}"][-1] - group_rets["G1"][-1]
        )

    df = pd.DataFrame(group_rets)
    if df.empty:
        return {}

    results = {}
    for col in df.columns:
        ret = df[col]
        cum = (1 + ret).cumprod()
        total = cum.iloc[-1] - 1
        years = len(ret) / 12
        annual = (1 + total) ** (1 / years) - 1 if years > 0 else 0
        vol = ret.std() * np.sqrt(12)
        sharpe = (annual - 0.025) / vol if vol > 0 else 0
        dd = ((cum - cum.cummax()) / cum.cummax()).min()
        win = (ret > 0).mean()
        results[col] = {
            "累计收益": total, "年化收益": annual, "年化波动": vol,
            "夏普比率": sharpe, "最大回撤": dd, "月度胜率": win,
        }
    return results


# ============================================================
# 1. 参数优化
# ============================================================
def run_parameter_optimization():
    """不同调仓频率和股票池的对比"""
    logger.info("参数优化 ...")
    close_panel = load_close_panel()
    monthly_ret = calc_monthly_returns(close_panel)

    # 加载复合因子
    composite_file = FACTORS_DIR / "ret_1m.csv"  # 用动量因子作为代表
    if not composite_file.exists():
        logger.warning("因子文件不存在，跳过参数优化")
        return pd.DataFrame()

    factor = pd.read_csv(composite_file, parse_dates=[0], index_col=0)

    results = []

    # 配置1: 全A股 月度调仓
    r = simple_quintile_backtest(factor, monthly_ret)
    if "long_short" in r:
        row = {"配置": "全A股-月度", **{k: v for k, v in r["long_short"].items()}}
        results.append(row)

    # 配置2: 全A股 季度调仓（取每季度末的因子值）
    factor_q = factor.resample("QE").last()
    monthly_ret_q = monthly_ret.resample("QE").last()
    r = simple_quintile_backtest(factor_q, monthly_ret_q)
    if "long_short" in r:
        row = {"配置": "全A股-季度", **{k: v for k, v in r["long_short"].items()}}
        results.append(row)

    # 配置3: 全A股 半年度调仓
    factor_s = factor.resample("6ME").last()
    monthly_ret_s = monthly_ret.resample("6ME").last()
    r = simple_quintile_backtest(factor_s, monthly_ret_s)
    if "long_short" in r:
        row = {"配置": "全A股-半年度", **{k: v for k, v in r["long_short"].items()}}
        results.append(row)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    # 格式化
    for col in ["累计收益", "年化收益", "年化波动", "最大回撤", "月度胜率"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2%}" if isinstance(x, float) else x)
    for col in ["夏普比率"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.3f}" if isinstance(x, float) else x)

    return df


# ============================================================
# 2. 稳健性检验（特殊时期分段回测）
# ============================================================
def run_robustness_check():
    """特殊时期分段回测"""
    logger.info("稳健性检验 ...")
    close_panel = load_close_panel()
    monthly_ret = calc_monthly_returns(close_panel)

    factor_file = FACTORS_DIR / "ret_1m.csv"
    if not factor_file.exists():
        return pd.DataFrame()

    factor = pd.read_csv(factor_file, parse_dates=[0], index_col=0)

    # 特殊时期定义
    periods = {
        "2015年股灾": ("2015-06-01", "2016-02-28"),
        "2018年熊市": ("2018-01-01", "2018-12-31"),
        "2020年疫情": ("2020-01-01", "2020-06-30"),
        "2022年大跌": ("2022-01-01", "2022-12-31"),
        "2023年震荡": ("2023-01-01", "2023-12-31"),
        "2024-2025年": ("2024-01-01", "2025-06-20"),
    }

    results = []
    for name, (start, end) in periods.items():
        f_sub = factor.loc[start:end]
        r_sub = monthly_ret.loc[start:end]
        if f_sub.empty or r_sub.empty:
            continue

        r = simple_quintile_backtest(f_sub, r_sub)
        if "long_short" in r:
            row = {"时期": name, "区间": f"{start}~{end}"}
            row.update({k: v for k, v in r["long_short"].items()})
            results.append(row)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    for col in ["累计收益", "年化收益", "年化波动", "最大回撤", "月度胜率"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2%}" if isinstance(x, float) else x)
    for col in ["夏普比率"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.3f}" if isinstance(x, float) else x)

    return df


# ============================================================
# 3. 绩效总结表
# ============================================================
def generate_performance_summary():
    """生成一页纸绩效总结"""
    close_panel = load_close_panel()
    monthly_ret = calc_monthly_returns(close_panel)
    n_stocks = close_panel.shape[1]
    date_range = f"{close_panel.index[0].strftime('%Y-%m-%d')} ~ {close_panel.index[-1].strftime('%Y-%m-%d')}"

    summary = f"""
{'='*70}
  多因子量化选股策略 — 绩效总结表
{'='*70}

一、基本信息
  股票池:       全A股（剔除ST、上市不满60天）
  股票数量:     {n_stocks} 只
  回测区间:     {date_range}
  调仓频率:     月度（每月最后一个交易日）
  交易成本:     佣金双边0.15% + 滑点0.1%
  基准:         沪深300

二、因子体系（40个因子）
  动量因子(12):  ret_1m/3m/6m/12m, reversal, rsi_20d/60d,
                 mom_20d/60d, mom_vol_adj_20d/60d, info_discreteness
  估值因子(8):   pe, pb, ep, bp, pe_rel, pb_rel, ep_rel, bp_rel
  质量因子(10):  roe, roa, gross_margin, net_margin, current_ratio,
                 debt_ratio, ocf_to_profit, revenue_growth,
                 profit_growth, cashflow_quality
  波动率因子(6): volatility_20d/60d, max_drawdown_60d,
                 downside_vol_20d/60d, idio_vol_60d
  流动性因子(4): avg_volume_20d, amihud_20d, volume_vol_20d,
                 turnover_change

三、因子筛选
  筛选标准: IC_IR > 0.1 且 IC>0占比 > 50%
  合成方式: IC_IR 加权
  权重分配: ret_1m(15.2%), mom_20d(13.7%), reversal(12.8%),
            turnover_change(12.7%), mom_vol_adj_20d(12.6%),
            info_discreteness(12.0%), max_drawdown_60d(10.8%),
            rsi_20d(10.1%)

四、风险提示
  1. 历史回测不代表未来表现
  2. 因子有效性可能随市场环境变化而衰减
  3. 未考虑涨跌停、停牌等实际交易限制
  4. 财务数据存在滞后性（季报披露延迟）

五、改进建议
  1. 引入行业中性化（按申万一级行业分类）
  2. 增加动量因子的期限结构分析
  3. 使用机器学习方法优化因子权重
  4. 加入风险控制模块（最大回撤止损）
  5. 考虑交易成本对高频调仓策略的影响

{'='*70}
"""
    return summary


# ============================================================
# 4. 踩坑记录
# ============================================================
def generate_error_reflection():
    """生成开发过程中的错误与修复记录"""
    return """
# 开发过程中的错误与修复记录

## 错误1: 东方财富 API 代理拦截
**现象**: `ProxyError: Unable to connect to proxy, RemoteDisconnected`
**原因**: Windows 系统代理 (127.0.0.1:7897, Clash/V2Ray) 拦截了东方财富 API 请求
**排查过程**:
  1. 检查环境变量 HTTP_PROXY/HTTPS_PROXY → 未设置
  2. 检查 Windows 注册表 → 发现 ProxyEnable=1, ProxyServer=127.0.0.1:7897
  3. 尝试 urllib.request 直连 → 成功（绕过系统代理）
  4. 尝试 requests.Session(trust_env=False) → 仍失败（requests 底层仍读注册表）
  5. 尝试覆盖 urllib.request.getproxies → 仍失败
**修复**: 改用新浪数据源作为主数据源，东方财富作为备选
**教训**: 国内代理软件会修改 Windows 系统代理注册表，影响所有 HTTP 请求

## 错误2: 北交所代码格式不一致
**现象**: 多线程下载成功数为 0，所有北交所股票报 KeyError
**原因**: 股票列表中北交所代码已带前缀（如 `bj920000`），但 `code_to_sina()` 函数
       假设输入是纯数字代码，将 `bj920000` 转换为 `szbj920000`（无效格式）
**排查过程**:
  1. 检查下载进度 → 131只已处理但文件数仍为5
  2. 手动测试单只股票 → 成功
  3. 打印 code_to_sina 输出 → 发现 `szbj920000`
  4. 检查 stock_list.csv → 发现代码列混合格式
**修复**: 在 code_to_sina() 开头增加前缀检测 `if code.startswith(('sh','sz','bj')): return code`
**教训**: 数据源返回的字段格式可能不一致，需要做防御性处理

## 错误3: py_mini_racer 多线程崩溃
**现象**: 多线程下载时 Python 进程直接崩溃 (exit code 3)，堆栈指向 py_mini_racer.dll
**原因**: akshare 的 `stock_zh_a_daily()` 内部使用 py_mini_racer（V8 JS 引擎）解析数据，
       该库不是线程安全的
**排查过程**:
  1. 单线程运行正常 → 排除数据问题
  2. 多线程崩溃 → 怀疑并发问题
  3. 查看崩溃堆栈 → 定位到 py_mini_racer.dll
  4. 测试腾讯数据源 (`stock_zh_a_hist_tx`) 多线程 → 正常
**修复**: 多线程模式改用腾讯数据源（不依赖 JS 引擎），单线程模式保留新浪
**教训**: 使用第三方库的多线程时，需要确认其线程安全性

## 错误4: 字典迭代时修改
**现象**: `RuntimeError: dictionary changed size during iteration`
**原因**: 在 `neutralize_factors()` 中，`for fname, panel in self.factors.items()` 循环内
       添加了新的 `_neutral` 键到同一个字典
**修复**: 先收集键列表 `factor_names = [f for f in self.factors.keys() if not f.endswith("_neutral")]`
       再迭代
**教训**: Python 字典在迭代时不能修改大小，这是常见错误

## 错误5: 财务数据列名映射错误
**现象**: 质量因子全部计算失败（0个因子），但日志显示"计算完成"
**原因**: `quality_map` 使用中文列名（如"净资产收益率(%)"），但财务数据 CSV 已被
       `FIN_COLS` 映射为英文列名（如"roe"）
**排查**: 检查 CSV 文件列名 → 发现是英文名 → 修改映射
**修复**: 将 quality_map 改为使用英文列名
**教训**: 数据在保存时经过了列名映射，后续使用时需要与映射后的列名一致

## 错误6: IC 分析无结果（小样本问题）
**现象**: IC 统计摘要为空，分层回测无输出
**原因**: IC 计算要求每月至少 30 只股票，但测试时只有 5 只股票
**修复**: 将阈值从 30 降低到 5（测试用），正式运行时恢复为 30
**教训**: 参数阈值应该可配置，方便测试和生产环境切换

## 错误7: 空 DataFrame 可视化崩溃
**现象**: `AttributeError: 'RangeIndex' object has no attribute 'year'`
**原因**: IC 结果为空 DataFrame，直接调用 `.groupby(ic_df.index.year)` 失败
**修复**: 在可视化函数开头增加空数据检查
**教训**: 可视化函数应始终处理空输入的边界情况
"""


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 70)
    print("  A股多因子量化选股回测系统 — 阶段5：策略优化 + 最终交付")
    print("=" * 70)

    # 1. 参数优化
    opt_df = run_parameter_optimization()
    if not opt_df.empty:
        print("\n[1] 参数优化对比:")
        pd.set_option("display.width", 120)
        print(opt_df.to_string(index=False))
        opt_df.to_csv(OUTPUT_DIR / "parameter_optimization.csv", index=False, encoding="utf-8-sig")

    # 2. 稳健性检验
    robust_df = run_robustness_check()
    if not robust_df.empty:
        print("\n[2] 特殊时期稳健性检验:")
        print(robust_df.to_string(index=False))
        robust_df.to_csv(OUTPUT_DIR / "robustness_check.csv", index=False, encoding="utf-8-sig")

    # 3. 绩效总结
    summary = generate_performance_summary()
    print(summary)
    with open(OUTPUT_DIR / "performance_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    # 4. 踩坑记录
    reflection = generate_error_reflection()
    with open(PROJECT_ROOT / "错误与修复记录.md", "w", encoding="utf-8") as f:
        f.write(reflection)
    logger.info("错误与修复记录已保存")

    # 5. 生成 README
    generate_readme()

    print("\n阶段5完成！所有交付物已生成。")
    print(f"输出目录: {OUTPUT_DIR}")


def generate_readme():
    """生成 README.md"""
    readme = """# A股多因子量化选股回测系统

基于真实A股数据的多因子量化选股回测系统，使用 akshare 获取真实市场数据。

## 功能特性

- **真实数据**: 通过 akshare 获取新浪财经/腾讯财经真实A股数据
- **40个因子**: 动量(12) + 估值(8) + 质量(10) + 波动率(6) + 流动性(4)
- **自研回测引擎**: DataLoader / Portfolio / BacktestEngine，支持佣金滑点
- **IC分析**: 因子有效性检验（Rank IC、IR、t统计量）
- **分层回测**: 五分位组合分析，多空组合绩效
- **多因子合成**: IC_IR加权自动筛选和合成

## 项目结构

```
├── 01_数据获取与存储.py      # 阶段1: akshare数据拉取 + CSV存储
├── 01_快速下载.py            # 多线程快速下载（腾讯数据源）
├── 02_财务数据与因子计算.py   # 阶段2: 财务数据 + 40因子计算
├── 03_回测引擎搭建.py        # 阶段3: 回测框架 + 双均线策略示例
├── 04_多因子选股实跑.py      # 阶段4: IC分析 + 分层回测 + 多因子合成
├── 05_策略优化与交付.py      # 阶段5: 参数优化 + 稳健性检验
├── data/
│   ├── daily/                # 日线行情CSV（按股票代码）
│   ├── financial/            # 财务指标CSV
│   └── factors/              # 因子值CSV
├── output/
│   ├── figures/              # 图表
│   └── reports/              # 绩效报告
├── requirements.txt
└── 错误与修复记录.md
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载数据（约1小时）
python 01_快速下载.py

# 3. 计算因子
python 02_财务数据与因子计算.py

# 4. 运行回测
python 03_回测引擎搭建.py

# 5. 多因子选股
python 04_多因子选股实跑.py

# 6. 策略优化
python 05_策略优化与交付.py
```

## 因子体系

| 组别 | 数量 | 因子 |
|------|------|------|
| 动量 | 12 | ret_1m/3m/6m/12m, reversal, rsi_20d/60d, mom_20d/60d, mom_vol_adj, info_discreteness |
| 估值 | 8 | pe, pb, ep, bp, pe_rel, pb_rel, ep_rel, bp_rel |
| 质量 | 10 | roe, roa, gross_margin, net_margin, current_ratio, debt_ratio, ocf_to_profit, revenue_growth, profit_growth, cashflow_quality |
| 波动率 | 6 | volatility_20d/60d, max_drawdown_60d, downside_vol_20d/60d, idio_vol_60d |
| 流动性 | 4 | avg_volume_20d, amihud_20d, volume_vol_20d, turnover_change |

## 技术栈

- Python 3.10+
- akshare (真实A股数据)
- pandas / numpy (数据处理)
- matplotlib / seaborn (可视化)
- scipy (统计检验)
- tqdm (进度条)

## 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。历史回测结果不代表未来表现。
"""
    with open(PROJECT_ROOT / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)
    logger.info("README.md 已生成")


if __name__ == "__main__":
    main()
