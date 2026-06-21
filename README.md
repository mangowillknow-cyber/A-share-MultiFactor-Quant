# A股多因子量化选股回测系统

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
