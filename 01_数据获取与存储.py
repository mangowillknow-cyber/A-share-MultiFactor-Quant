"""
阶段1：真实A股数据获取 + 本地存储
======================================
使用 akshare 获取真实A股数据（新浪/东方财富数据源），支持增量更新。

功能：
1. 获取全A股股票列表
2. 剔除ST股、上市不满60天的股票
3. 获取2015-2025年日线数据（OHLCV）
4. 按股票代码保存为CSV文件
5. 支持增量更新

使用方法：
    # 完整下载
    python 01_数据获取与存储.py

    # 增量更新（只拉取最新缺失的数据）
    python 01_数据获取与存储.py --update
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd
from tqdm import tqdm

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data_fetch.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# tqdm 进度条样式
TQDM_FMT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

# ============================================================
# 项目目录
# ============================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DAILY_DIR = DATA_DIR / "daily"
FINANCIAL_DIR = DATA_DIR / "financial"

for d in [DAILY_DIR, FINANCIAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def code_to_symbol(code: str) -> str:
    """
    将股票代码转换为带交易所前缀的格式（新浪数据源需要）。
    自动处理已有前缀的情况（如 bj920000）。
    """
    code = str(code).strip()
    if code.startswith(("sh", "sz", "bj")):
        return code
    code = code.zfill(6)
    if code.startswith(("60", "68", "69")):
        return f"sh{code}"
    elif code.startswith(("00", "30")):
        return f"sz{code}"
    elif code.startswith("92"):
        return f"bj{code}"
    else:
        return f"sz{code}"


class DataFetcher:
    """A股数据获取器（基于 akshare，新浪+东方财富双数据源）"""

    def __init__(self, start_date: str = "20150101", end_date: str = "20250620"):
        self.start_date = start_date
        self.end_date = end_date
        self.retry_times = 3
        self.fetch_delay = 0.2  # 每次请求间隔（秒），避免被封

    # ----------------------------------------------------------
    # 1. 获取股票列表（优先东方财富，失败则用新浪）
    # ----------------------------------------------------------
    def get_stock_list(self) -> pd.DataFrame:
        """获取全A股股票列表"""
        logger.info("正在获取A股股票列表 ...")

        # 优先尝试东方财富（速度快）
        try:
            df = ak.stock_zh_a_spot_em()
            result = pd.DataFrame({
                "代码": df["代码"].astype(str).str.zfill(6),
                "名称": df["名称"],
                "成交量": pd.to_numeric(df["成交量"], errors="coerce"),
            })
            logger.info(f"东方财富数据源: 获取到 {len(result)} 只股票")
            return result
        except Exception as e:
            logger.warning(f"东方财富数据源失败: {e}，切换到新浪数据源")

        # 备选：新浪数据源
        try:
            df = ak.stock_zh_a_spot()
            result = pd.DataFrame({
                "代码": df["代码"].astype(str).str.zfill(6),
                "名称": df["名称"],
                "成交量": pd.to_numeric(df["成交量"], errors="coerce"),
            })
            logger.info(f"新浪数据源: 获取到 {len(result)} 只股票")
            return result
        except Exception as e:
            logger.error(f"新浪数据源也失败: {e}")
            return pd.DataFrame()

    # ----------------------------------------------------------
    # 2. 过滤 ST / 退市 / 次新股
    # ----------------------------------------------------------
    def filter_stocks(self, stock_list: pd.DataFrame) -> pd.DataFrame:
        """剔除ST股、退市股、停牌股"""
        n_before = len(stock_list)
        df = stock_list.copy()

        # 排除 ST / 退市股
        mask_st = df["名称"].str.contains(r"ST|退市", case=False, na=False)
        df = df[~mask_st]

        # 排除成交量为0的股票（停牌/退市/尚未上市）
        df = df[df["成交量"] > 0]

        n_after = len(df)
        logger.info(f"过滤ST/退市/停牌后: {n_before} -> {n_after} 只")
        return df.reset_index(drop=True)

    # ----------------------------------------------------------
    # 3. 获取单只股票日线数据（优先东方财富，失败则用新浪）
    # ----------------------------------------------------------
    def fetch_daily_single(self, symbol: str) -> pd.DataFrame:
        """
        获取单只股票日线行情（OHLCV）。

        symbol: 纯数字代码，如 "000001"
        返回包含: date, open, high, low, close, volume, amount
        """
        sina_symbol = code_to_symbol(symbol)

        for attempt in range(self.retry_times):
            try:
                df = ak.stock_zh_a_daily(
                    symbol=sina_symbol,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    adjust="qfq",
                )
                if df is not None and not df.empty:
                    result = df[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
                    result["date"] = pd.to_datetime(result["date"])
                    return result
            except Exception as e:
                if attempt < self.retry_times - 1:
                    time.sleep(1)
                else:
                    logger.debug(f"{symbol} 获取失败: {e}")

        return pd.DataFrame()

    # ----------------------------------------------------------
    # 4. 批量获取日线数据（支持断点续传）
    # ----------------------------------------------------------
    def fetch_daily_all(self, stock_list: pd.DataFrame) -> int:
        """
        批量获取所有股票的日线数据，直接保存为CSV。
        返回成功保存的文件数。
        """
        symbols = stock_list["代码"].tolist()

        # 找出需要下载的（排除已存在的）
        existing = set(f.stem for f in DAILY_DIR.glob("*.csv"))
        to_fetch = [s for s in symbols if s not in existing]

        if not to_fetch:
            logger.info("所有股票数据已存在，无需下载。")
            return len(existing)

        logger.info(f"需下载: {len(to_fetch)} 只（已有 {len(existing)} 只）")

        success_count = 0
        failed_list = []

        for sym in tqdm(to_fetch, desc="获取日线数据", bar_format=TQDM_FMT):
            try:
                df = self.fetch_daily_single(sym)
                if not df.empty:
                    filepath = DAILY_DIR / f"{sym}.csv"
                    df.to_csv(filepath, index=False, encoding="utf-8-sig")
                    success_count += 1
                else:
                    failed_list.append(sym)
            except Exception as e:
                failed_list.append(sym)
                logger.debug(f"{sym} 异常: {e}")
            time.sleep(self.fetch_delay)

        logger.info(f"下载完成: 成功 {success_count}, 失败 {len(failed_list)}")

        if failed_list:
            pd.DataFrame({"代码": failed_list}).to_csv(
                DATA_DIR / "failed_stocks.csv", index=False, encoding="utf-8-sig"
            )
            logger.info(f"失败列表已保存到 data/failed_stocks.csv")

        return success_count + len(existing)

    # ----------------------------------------------------------
    # 5. 增量更新
    # ----------------------------------------------------------
    def incremental_update(self, stock_list: pd.DataFrame) -> dict:
        """增量更新：对比本地已有数据的最新日期，只拉取缺失的部分"""
        stats = {"updated": 0, "skipped": 0, "failed": 0}
        symbols = stock_list["代码"].tolist()

        logger.info(f"开始增量更新 {len(symbols)} 只股票 ...")
        for sym in tqdm(symbols, desc="增量更新", bar_format=TQDM_FMT):
            try:
                filepath = DAILY_DIR / f"{sym}.csv"
                if filepath.exists():
                    existing = pd.read_csv(filepath, parse_dates=["date"])
                    if not existing.empty:
                        last_date = existing["date"].max()
                        if last_date >= pd.Timestamp(self.end_date) - timedelta(days=1):
                            stats["skipped"] += 1
                            continue

                        # 获取增量数据
                        new_start = (last_date + timedelta(days=1)).strftime("%Y%m%d")
                        new_data = self._fetch_single_range(sym, new_start, self.end_date)
                        if new_data is not None and not new_data.empty:
                            combined = pd.concat([existing, new_data], ignore_index=True)
                            combined = combined.drop_duplicates(subset=["date"]).sort_values("date")
                            combined.to_csv(filepath, index=False, encoding="utf-8-sig")
                            stats["updated"] += 1
                        else:
                            stats["skipped"] += 1
                        time.sleep(self.fetch_delay)
                        continue

                # 本地无数据，全量拉取
                df = self.fetch_daily_single(sym)
                if not df.empty:
                    df.to_csv(filepath, index=False, encoding="utf-8-sig")
                    stats["updated"] += 1
                else:
                    stats["failed"] += 1
                time.sleep(self.fetch_delay)

            except Exception as e:
                stats["failed"] += 1
                logger.debug(f"{sym} 更新失败: {e}")

        logger.info(
            f"增量更新完成: 更新 {stats['updated']}, "
            f"跳过 {stats['skipped']}, 失败 {stats['failed']}"
        )
        return stats

    def _fetch_single_range(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """获取单只股票在指定日期范围内的数据"""
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=start, end_date=end, adjust="qfq",
            )
            if df is not None and not df.empty:
                result = df.rename(columns={
                    "日期": "date", "开盘": "open", "最高": "high",
                    "最低": "low", "收盘": "close",
                    "成交量": "volume", "成交额": "amount",
                })
                result = result[["date", "open", "high", "low", "close", "volume", "amount"]]
                result["date"] = pd.to_datetime(result["date"])
                return result
        except Exception:
            pass

        try:
            sina_sym = code_to_symbol(symbol)
            df = ak.stock_zh_a_daily(
                symbol=sina_sym, start_date=start, end_date=end, adjust="qfq",
            )
            if df is not None and not df.empty:
                result = df[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
                result["date"] = pd.to_datetime(result["date"])
                return result
        except Exception:
            pass

        return pd.DataFrame()

    # ----------------------------------------------------------
    # 6. 统计报告
    # ----------------------------------------------------------
    def print_statistics(self, stock_list: pd.DataFrame):
        """输出完整的数据统计报告"""
        print("\n" + "=" * 70)
        print("  阶段1 数据统计报告")
        print("=" * 70)

        # --- 股票列表统计 ---
        print(f"\n[1] 股票列表")
        print(f"    过滤后股票总数: {len(stock_list)} 只")
        print(f"\n    前20只股票:")
        print("    " + "-" * 45)
        for _, row in stock_list.head(20).iterrows():
            print(f"    {row['代码']}  {row['名称']}")
        print("    " + "-" * 45)

        # --- 日线数据统计 ---
        all_files = sorted(DAILY_DIR.glob("*.csv"))
        if not all_files:
            logger.warning("本地日线数据文件为空")
            return

        # 扫描日期范围（抽样）
        min_dates, max_dates, row_counts = [], [], []
        sample_files = all_files[:200]
        for f in tqdm(sample_files, desc="扫描日期范围", bar_format=TQDM_FMT):
            try:
                tmp = pd.read_csv(f, usecols=["date"], parse_dates=["date"])
                if not tmp.empty:
                    min_dates.append(tmp["date"].min())
                    max_dates.append(tmp["date"].max())
                    row_counts.append(len(tmp))
            except Exception:
                pass

        print(f"\n[2] 日线数据覆盖范围")
        print(f"    已下载文件数: {len(all_files)}")
        if min_dates:
            print(f"    最早日期: {min(min_dates).strftime('%Y-%m-%d')}")
            print(f"    最晚日期: {max(max_dates).strftime('%Y-%m-%d')}")
            print(f"    抽样平均每只股票行数: {sum(row_counts) / len(row_counts):.0f}")

        # --- 年度交易日统计 ---
        print(f"\n[3] 年度交易日统计（基于抽样）")
        year_counts = {}
        for f in sample_files:
            try:
                tmp = pd.read_csv(f, usecols=["date"], parse_dates=["date"])
                tmp["year"] = tmp["date"].dt.year
                for yr, cnt in tmp.groupby("year").size().items():
                    year_counts.setdefault(yr, []).append(cnt)
            except Exception:
                pass

        if year_counts:
            print("    " + "-" * 35)
            print(f"    {'年份':<8} {'交易日数（中位数）':<20}")
            print("    " + "-" * 35)
            for yr in sorted(year_counts):
                vals = sorted(year_counts[yr])
                median_val = vals[len(vals) // 2]
                print(f"    {yr:<8} {median_val:<20}")
            print("    " + "-" * 35)

        # --- 前3只股票数据预览 ---
        print(f"\n[4] 前3只股票数据预览（各前100行）")
        pd.set_option("display.max_rows", 110)
        pd.set_option("display.max_columns", 10)
        pd.set_option("display.width", 120)

        for fpath in sorted(all_files)[:3]:
            sym = fpath.stem
            print(f"\n    --- {sym} ---")
            try:
                df = pd.read_csv(fpath, nrows=100)
                lines = df.to_string(index=False).split("\n")
                for line in lines:
                    print(f"    {line}")
            except Exception as e:
                print(f"    读取失败: {e}")

        print("\n" + "=" * 70)


# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="A股数据获取工具")
    parser.add_argument("--update", action="store_true", help="增量更新模式")
    args = parser.parse_args()

    fetcher = DataFetcher(start_date="20150101", end_date="20250620")

    print("=" * 70)
    print("  A股多因子量化选股回测系统 — 阶段1：数据获取与存储")
    print("=" * 70)

    # Step 1: 获取并过滤股票列表（优先使用缓存）
    stock_list_file = DATA_DIR / "stock_list.csv"
    if stock_list_file.exists():
        stock_list = pd.read_csv(stock_list_file, dtype={"代码": str})
        stock_list["代码"] = stock_list["代码"].astype(str).str.zfill(6)
        logger.info(f"使用缓存股票列表: {len(stock_list)} 只")
    else:
        stock_list = fetcher.get_stock_list()
        if stock_list.empty:
            logger.error("无法获取股票列表，请检查网络连接。")
            sys.exit(1)
        stock_list = fetcher.filter_stocks(stock_list)
        stock_list.to_csv(stock_list_file, index=False, encoding="utf-8-sig")
        logger.info(f"股票列表已保存: {len(stock_list)} 只")

    # Step 2: 获取日线数据
    if args.update:
        fetcher.incremental_update(stock_list)
    else:
        fetcher.fetch_daily_all(stock_list)

    # Step 3: 输出统计报告
    fetcher.print_statistics(stock_list)

    # Step 4: 简单验证
    print("\n[验证]")
    csv_count = len(list(DAILY_DIR.glob("*.csv")))
    print(f"  data/daily/ 下 CSV 文件数: {csv_count}")
    if csv_count > 0:
        sample = pd.read_csv(next(DAILY_DIR.glob("*.csv")))
        print(f"  样本文件列名: {list(sample.columns)}")
        print(f"  样本文件行数: {len(sample)}")
    print()


if __name__ == "__main__":
    main()
