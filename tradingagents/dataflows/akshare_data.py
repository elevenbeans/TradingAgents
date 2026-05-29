from typing import Annotated, Optional
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import akshare as ak
import os
import re
from .stockstats_utils import _clean_dataframe
from .config import get_config
from .utils import safe_ticker_component


def _normalize_hk_symbol(symbol: str) -> str:
    """Normalize HK stock symbol to akshare format (5-digit zero-padded).

    Handles formats: '0700.HK', '0700', '700', '00700'. Returns the
    original symbol unchanged if it doesn't look like a HK stock.
    """
    s = symbol.upper().strip()
    if s.endswith('.HK'):
        s = s[:-3]
    s = s.strip()
    if s.isdigit():
        return s.zfill(5)
    return symbol


def _is_hk_ticker(symbol: str) -> bool:
    """Check if the ticker is a Hong Kong stock."""
    s = symbol.upper().strip()
    if s.endswith('.HK'):
        return True
    s = s.replace('.HK', '')
    return bool(re.match(r'^\d{1,5}$', s.strip()))


def _normalize_a_share_symbol(symbol: str) -> str:
    """Normalize A-share symbol to akshare format (6 digits, no suffix).

    Handles formats: '600519.SS', '600519', '000001.SZ', '000001'.
    """
    s = symbol.upper().strip()
    s = re.sub(r'\.(SS|SZ)$', '', s)
    return s


def _is_a_share_ticker(symbol: str) -> bool:
    """Check if the ticker is a China A-share stock."""
    s = symbol.upper().strip()
    if s.endswith('.SS') or s.endswith('.SZ'):
        return True
    # 6-digit codes with no suffix are commonly A-share
    s_no_suffix = re.sub(r'\.(SS|SZ)$', '', s)
    if re.match(r'^\d{6}$', s_no_suffix):
        return True
    return False


def get_stock_data_akshare(
    symbol: Annotated[str, "ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Retrieve stock price data (OHLCV) via akshare.

    Supports HK stocks (.HK) via East Money API.
    Accessible from mainland China without proxy.
    """
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    start_clean = start_date.replace("-", "")
    end_clean = end_date.replace("-", "")

    try:
        if _is_hk_ticker(symbol):
            sym = _normalize_hk_symbol(symbol)
            data = ak.stock_hk_hist(
                symbol=sym,
                period="daily",
                start_date=start_clean,
                end_date=end_clean,
                adjust="qfq",
            )
        elif _is_a_share_ticker(symbol):
            sym = _normalize_a_share_symbol(symbol)
            data = ak.stock_zh_a_hist(
                symbol=sym,
                period="daily",
                start_date=start_clean,
                end_date=end_clean,
                adjust="qfq",
            )
        else:
            return (
                f"Akshare data source does not support symbol '{symbol}'. "
                "Currently supports HK stock tickers (e.g. 0700.HK) "
                "and A-share tickers (e.g. 600519.SS, 000001.SZ)."
            )

        if data.empty:
            return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"

        expected_cols = {"日期", "开盘", "收盘", "最高", "最低", "成交量"}
        if not expected_cols.issubset(data.columns):
            return (
                f"Unexpected data format for '{symbol}'. "
                f"Got columns: {list(data.columns)}"
            )

        result = pd.DataFrame()
        result["Date"] = data["日期"]
        result["Open"] = pd.to_numeric(data["开盘"], errors="coerce").round(2)
        result["High"] = pd.to_numeric(data["最高"], errors="coerce").round(2)
        result["Low"] = pd.to_numeric(data["最低"], errors="coerce").round(2)
        result["Close"] = pd.to_numeric(data["收盘"], errors="coerce").round(2)
        result["Volume"] = pd.to_numeric(data["成交量"], errors="coerce")
        if "Adj Close" in data.columns:
            result["Adj Close"] = pd.to_numeric(data["Adj Close"], errors="coerce").round(2)
        else:
            result["Adj Close"] = result["Close"]

        result = result.sort_values("Date")

        csv_string = result.to_csv(index=False)
        header = f"# Stock data for {symbol} from {start_date} to {end_date}\n"
        header += f"# Total records: {len(result)}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + csv_string

    except Exception as e:
        return f"Error retrieving stock data for {symbol}: {str(e)}"


def load_ohlcv_akshare(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch OHLCV data via akshare with caching, filtered to prevent look-ahead bias.

    Mirrors the interface of stockstats_utils.load_ohlcv but uses akshare
    instead of yfinance. Accessible from mainland China without proxy.
    """
    safe_symbol = safe_ticker_component(symbol)
    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date)
    today_date = pd.Timestamp.today()
    start_date = today_date - pd.DateOffset(years=5)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = today_date.strftime("%Y-%m-%d")

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{safe_symbol}-akshare-data-{start_str}-{end_str}.csv",
    )

    if os.path.exists(data_file):
        data = pd.read_csv(data_file, on_bad_lines="skip", encoding="utf-8")
    else:
        if _is_hk_ticker(symbol):
            sym = _normalize_hk_symbol(symbol)
            raw = ak.stock_hk_hist(
                symbol=sym,
                period="daily",
                start_date=start_str.replace("-", ""),
                end_date=end_str.replace("-", ""),
                adjust="qfq",
            )
            data = pd.DataFrame()
            data["Date"] = pd.to_datetime(raw["日期"])
            data["Open"] = pd.to_numeric(raw["开盘"], errors="coerce")
            data["High"] = pd.to_numeric(raw["最高"], errors="coerce")
            data["Low"] = pd.to_numeric(raw["最低"], errors="coerce")
            data["Close"] = pd.to_numeric(raw["收盘"], errors="coerce")
            data["Volume"] = pd.to_numeric(raw["成交量"], errors="coerce")
        elif _is_a_share_ticker(symbol):
            sym = _normalize_a_share_symbol(symbol)
            raw = ak.stock_zh_a_hist(
                symbol=sym,
                period="daily",
                start_date=start_str.replace("-", ""),
                end_date=end_str.replace("-", ""),
                adjust="qfq",
            )
            data = pd.DataFrame()
            data["Date"] = pd.to_datetime(raw["日期"])
            data["Open"] = pd.to_numeric(raw["开盘"], errors="coerce")
            data["High"] = pd.to_numeric(raw["最高"], errors="coerce")
            data["Low"] = pd.to_numeric(raw["最低"], errors="coerce")
            data["Close"] = pd.to_numeric(raw["收盘"], errors="coerce")
            data["Volume"] = pd.to_numeric(raw["成交量"], errors="coerce")
        else:
            raise ValueError(f"Akshare data source does not support symbol '{symbol}'")

        data.to_csv(data_file, index=False, encoding="utf-8")

    data = _clean_dataframe(data)
    data = data[data["Date"] <= curr_date_dt]

    return data


def get_stock_stats_indicators_window_akshare(
    symbol: Annotated[str, "ticker symbol"],
    indicator: Annotated[str, "technical indicator name"],
    curr_date: Annotated[str, "The current trading date, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """Calculate technical indicators using akshare OHLCV data + stockstats."""
    from stockstats import wrap

    best_ind_params = {
        "close_50_sma": (
            "50 SMA: A medium-term trend indicator. "
            "Usage: Identify trend direction and serve as dynamic support/resistance."
        ),
        "close_200_sma": (
            "200 SMA: A long-term trend benchmark. "
            "Usage: Confirm overall market trend and identify golden/death cross setups."
        ),
        "close_10_ema": (
            "10 EMA: A responsive short-term average. "
            "Usage: Capture quick shifts in momentum and potential entry points."
        ),
        "macd": (
            "MACD: Computes momentum via differences of EMAs. "
            "Usage: Look for crossovers and divergence as signals of trend changes."
        ),
        "macds": (
            "MACD Signal: An EMA smoothing of the MACD line. "
            "Usage: Use crossovers with the MACD line to trigger trades."
        ),
        "macdh": (
            "MACD Histogram: Shows the gap between the MACD line and its signal. "
            "Usage: Visualize momentum strength and spot divergence early."
        ),
        "rsi": (
            "RSI: Measures momentum to flag overbought/oversold conditions. "
            "Usage: Apply 70/30 thresholds and watch for divergence."
        ),
        "boll": (
            "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
            "Usage: Acts as a dynamic benchmark for price movement."
        ),
        "boll_ub": (
            "Bollinger Upper Band: Typically 2 standard deviations above the middle line."
        ),
        "boll_lb": (
            "Bollinger Lower Band: Typically 2 standard deviations below the middle line."
        ),
        "atr": (
            "ATR: Averages true range to measure volatility. "
            "Usage: Set stop-loss levels and adjust position sizes."
        ),
        "vwma": (
            "VWMA: A moving average weighted by volume. "
            "Usage: Confirm trends by integrating price action with volume data."
        ),
        "mfi": (
            "MFI: Money Flow Index, uses both price and volume to measure buying/selling pressure."
        ),
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} is not supported. Choose from: {list(best_ind_params.keys())}"
        )

    end_date = curr_date
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    try:
        data = load_ohlcv_akshare(symbol, curr_date)
        df = wrap(data)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

        df[indicator]

        result_dict = {}
        for _, row in df.iterrows():
            date_str = row["Date"]
            val = row[indicator]
            result_dict[date_str] = str(val) if not pd.isna(val) else "N/A"

        current_dt = curr_date_dt
        ind_string = ""
        while current_dt >= before:
            date_str = current_dt.strftime("%Y-%m-%d")
            value = result_dict.get(date_str, "N/A: Not a trading day (weekend or holiday)")
            ind_string += f"{date_str}: {value}\n"
            current_dt = current_dt - relativedelta(days=1)

    except Exception as e:
        return f"Error calculating {indicator} for {symbol}: {str(e)}"

    result_str = (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "No description available.")
    )

    return result_str


def get_fundamentals_akshare(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date, yyyy-mm-dd"] = None,
) -> str:
    """Get company fundamentals overview via akshare.

    Combines company profile with key financial indicators.
    """
    try:
        header = f"# Company Fundamentals for {ticker}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        lines = []

        if _is_hk_ticker(ticker):
            sym = _normalize_hk_symbol(ticker)

            profile = ak.stock_hk_company_profile_em(symbol=sym)
            if not profile.empty:
                for _, row in profile.iterrows():
                    for col in profile.columns:
                        val = row[col]
                        if pd.notna(val) and str(val).strip():
                            label = col.replace("_", " ").strip()
                            lines.append(f"{label}: {val}")

            indicators = ak.stock_hk_financial_indicator_em(symbol=sym)
            if not indicators.empty:
                lines.append("\n--- Key Financial Indicators ---")
                indicator_labels = {
                    "基本每股收益(元)": "EPS (Basic)",
                    "每股净资产(元)": "Book Value per Share",
                    "每股经营现金流(元)": "Operating CF per Share",
                    "股息率TTM(%)": "Dividend Yield (TTM)",
                    "总市值(港元)": "Market Cap (HKD)",
                    "营业总收入": "Total Revenue",
                    "净利润": "Net Income",
                    "销售净利率(%)": "Net Profit Margin",
                    "股东权益回报率(%)": "ROE",
                    "总资产回报率(%)": "ROA",
                    "市盈率": "PE Ratio",
                    "市净率": "PB Ratio",
                }
                for _, row in indicators.iterrows():
                    for col, label in indicator_labels.items():
                        if col in row and pd.notna(row[col]):
                            lines.append(f"{label}: {row[col]}")

        elif _is_a_share_ticker(ticker):
            sym = _normalize_a_share_symbol(ticker)

            info = ak.stock_individual_info_em(symbol=sym)
            if not info.empty:
                for _, row in info.iterrows():
                    item = row.get("item", "")
                    value = row.get("value", "")
                    if pd.notna(item) and pd.notna(value):
                        lines.append(f"{item}: {value}")

            fin = ak.stock_financial_abstract(symbol=sym, indicator="按年度")
            if not fin.empty:
                lines.append("\n--- Key Financial Indicators ---")
                fin_str = fin.to_string(index=False)
                lines.append(fin_str)

            profit = ak.stock_profit_sheet_by_report_em(symbol=sym)
            if not profit.empty:
                lines.append("\n--- Profit Forecast ---")
                profit_str = profit.to_string(index=False)
                lines.append(profit_str)

        else:
            return f"No fundamentals data available for '{ticker}' via akshare"

        if lines:
            return header + "\n".join(lines)
        return f"No fundamentals data found for '{ticker}'"

    except Exception as e:
        return f"Error retrieving fundamentals for {ticker}: {str(e)}"


def _fetch_financial_report(ticker: str, report_type: str, curr_date: str = None) -> pd.DataFrame:
    """Fetch stock financial report from East Money, pivoted to wide format.

    East Money returns data in long format (one row per item per period).
    This function pivots to wide format as columns=dates, rows=items,
    which matches the format expected by downstream tools and agents.

    Args:
        ticker: HK stock ticker (e.g. 0700.HK) or A-share ticker (e.g. 600519.SS)
        report_type: "资产负债表", "利润表", or "现金流量表"
        curr_date: if set, filters to periods on or before this date
    """
    if _is_hk_ticker(ticker):
        sym = _normalize_hk_symbol(ticker)
        indicator = "年度"
        raw = ak.stock_financial_hk_report_em(stock=sym, symbol=report_type, indicator=indicator)
    elif _is_a_share_ticker(ticker):
        sym = _normalize_a_share_symbol(ticker)
        raw = ak.stock_financial_report_sina(stock=sym, symbol=report_type)
    else:
        raise ValueError(f"Unsupported ticker: {ticker}")

    if raw.empty:
        return raw

    # Check format: long format (REPORT_DATE + STD_ITEM_NAME + AMOUNT) or wide
    if "STD_ITEM_NAME" in raw.columns and "REPORT_DATE" in raw.columns:
        raw["REPORT_DATE"] = pd.to_datetime(raw["REPORT_DATE"])

        data = raw.pivot_table(
            index="STD_ITEM_NAME",
            columns="REPORT_DATE",
            values="AMOUNT",
            aggfunc="first",
        )

        data.columns.name = None
        data.index.name = None

        if curr_date:
            cutoff = pd.Timestamp(curr_date)
            data = data[[c for c in data.columns if c <= cutoff]]

        data.columns = [c.strftime("%Y-%m-%d") for c in data.columns]
        data = data.reset_index()
        data.columns = ["Item"] + list(data.columns[1:])

        return data

    # Already in wide format or other format, return as-is
    return raw


def get_balance_sheet_akshare(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date, YYYY-MM-DD"] = None,
) -> str:
    """Get balance sheet data via akshare financial reports."""
    try:
        if _is_hk_ticker(ticker) or _is_a_share_ticker(ticker):
            data = _fetch_financial_report(ticker, "资产负债表", curr_date)

            if data.empty:
                return f"No balance sheet data found for '{ticker}'"

            csv_string = data.to_csv(index=False)
            header = f"# Balance Sheet data for {ticker} ({freq})\n"
            header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            return header + csv_string

        return f"No balance sheet data for '{ticker}' via akshare"

    except Exception as e:
        return f"Error retrieving balance sheet for {ticker}: {str(e)}"


def get_cashflow_akshare(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date, YYYY-MM-DD"] = None,
) -> str:
    """Get cash flow data via akshare financial reports."""
    try:
        if _is_hk_ticker(ticker) or _is_a_share_ticker(ticker):
            data = _fetch_financial_report(ticker, "现金流量表", curr_date)

            if data.empty:
                return f"No cash flow data found for '{ticker}'"

            csv_string = data.to_csv(index=False)
            header = f"# Cash Flow data for {ticker} ({freq})\n"
            header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            return header + csv_string

        return f"No cash flow data for '{ticker}' via akshare"

    except Exception as e:
        return f"Error retrieving cash flow for {ticker}: {str(e)}"


def get_income_statement_akshare(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date, YYYY-MM-DD"] = None,
) -> str:
    """Get income statement data via akshare financial reports."""
    try:
        if _is_hk_ticker(ticker) or _is_a_share_ticker(ticker):
            data = _fetch_financial_report(ticker, "利润表", curr_date)

            if data.empty:
                return f"No income statement data found for '{ticker}'"

            csv_string = data.to_csv(index=False)
            header = f"# Income Statement data for {ticker} ({freq})\n"
            header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            return header + csv_string

        return f"No income statement data for '{ticker}' via akshare"

    except Exception as e:
        return f"Error retrieving income statement for {ticker}: {str(e)}"


def get_insider_transactions_akshare(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """Get insider transaction information.

    Supported: A-share holders info (top shareholders).
    Not available: HK stock insider transactions.
    """
    try:
        if _is_a_share_ticker(ticker):
            sym = _normalize_a_share_symbol(ticker)
            holders = ak.stock_holdernumber(symbol=sym)
            if not holders.empty:
                header = f"# Shareholder Info for {ticker}\n"
                header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                return header + holders.to_csv(index=False, encoding="utf-8")
            return f"No shareholder data for '{ticker}'"
        return (
            f"Insider transaction data is not available for '{ticker}' via akshare. "
            "Currently supports A-share tickers."
        )
    except Exception as e:
        return f"Error retrieving insider data for {ticker}: {str(e)}"


def get_news_akshare(
    ticker: Annotated[str, "ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Get news for a ticker via akshare.

    Akshare's HK news coverage is limited. Returns what's available
    from East Money's HK stock news.
    """
    try:
        header = f"# News for {ticker} from {start_date} to {end_date}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        if _is_hk_ticker(ticker):
            try:
                news_detail = ak.stock_hk_famous_spot_em()
                articles = []
                if not news_detail.empty:
                    for _, row in news_detail.iterrows():
                        articles.append(str(dict(row)))
                if articles:
                    return header + "\n".join(articles[:20])
                return header + "No specific news articles found for this period."
            except Exception:
                return header + "News data temporarily unavailable."

        elif _is_a_share_ticker(ticker):
            sym = _normalize_a_share_symbol(ticker)
            try:
                news = ak.stock_info_js(symbol=sym)
                if not news.empty:
                    return header + news.to_csv(index=False, encoding="utf-8")
                return header + "No news articles found."
            except Exception:
                return header + "News data temporarily unavailable."

        return f"News data for '{ticker}' is not available via akshare."

    except Exception as e:
        return f"Error retrieving news for {ticker}: {str(e)}"


def get_global_news_akshare(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[Optional[int], "Days to look back"] = None,
    limit: Annotated[Optional[int], "Max articles"] = None,
) -> str:
    """Get global/macro news via akshare.

    Falls back to Chinese financial news sources since akshare
    does not have direct global news feeds.
    """
    config = get_config()
    if limit is None:
        limit = config.get("global_news_article_limit", 10)
    if look_back_days is None:
        look_back_days = config.get("global_news_lookback_days", 7)

    header = f"# Global Macro News\n"
    header += f"# Retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    header += f"# Lookback: {look_back_days} days\n\n"

    try:
        news_items = []

        try:
            cn_news = ak.stock_info_global_em()
            if not cn_news.empty:
                news_items.append(cn_news.head(limit).to_string(index=False))
        except Exception:
            pass

        try:
            hk_info = ak.macro_china_hk_market_info()
            if not hk_info.empty:
                news_items.append(hk_info.to_string(index=False))
        except Exception:
            pass

        if news_items:
            return header + "\n\n".join(news_items)
        return header + "No global news data available."
    except Exception:
        return header + "Global news data temporarily unavailable."


def get_dragon_tiger_akshare(
    ticker: Annotated[str, "ticker symbol"],
    trade_date: Annotated[str, "Current date in yyyy-mm-dd format"],
) -> str:
    """Get Dragon-Tiger Board data (top trading seats) for an A-share stock.

    The Dragon-Tiger Board discloses the top 5 brokerage trading desks
    by buy/sell volume for stocks with significant price moves or turnover.
    Heavy institutional buying is bullish; heavy retail buying often signals
    distribution (出货).
    """
    header = f"# Dragon-Tiger Board (龙虎榜) for {ticker}\n"
    header += f"# Retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    if not _is_a_share_ticker(ticker):
        return header + "Dragon-Tiger Board data is only available for A-share stocks."

    try:
        sym = _normalize_a_share_symbol(ticker)
        data = ak.stock_lhb_detail_em(date=trade_date.replace("-", ""))
        if data.empty:
            return header + "No Dragon-Tiger Board data found for this date."

        filtered = data[data["代码"] == sym]
        if filtered.empty:
            return header + f"Stock {ticker} did not appear on the Dragon-Tiger Board on {trade_date}."

        cols = [c for c in ["代码", "名称", "上榜原因", "营业部净买入额", "买入金额", "卖出金额", "成交额", "总成交额占比", "涨跌幅", "换手率"]
                if c in filtered.columns]
        result = filtered[cols].reset_index(drop=True)

        lines = []
        for _, row in result.iterrows():
            for col in cols:
                lines.append(f"{col}: {row[col]}")
            lines.append("---")

        return header + "\n".join(lines)
    except Exception as e:
        return header + f"Error retrieving Dragon-Tiger data: {str(e)}"


def get_lockup_expiry_akshare(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """Get upcoming restricted share lockup expiration (解禁) data.

    Restricted shares (限售股) become freely tradable after a lockup period.
    Major unlock events create selling pressure — this tool surfaces upcoming
    and recent unlock data for risk assessment.
    """
    header = f"# Lockup Expiration (限售股解禁) for {ticker}\n"
    header += f"# Retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    if not _is_a_share_ticker(ticker):
        return header + "Lockup expiration data is only available for A-share stocks."

    try:
        sym = _normalize_a_share_symbol(ticker)

        try:
            data = ak.stock_restricted_release_queue_em(symbol=sym)
        except Exception:
            try:
                data = ak.share_restricted_list_em(symbol=sym)
            except Exception:
                return header + f"No lockup expiration data found for {ticker} (API may have changed)."

        if data.empty:
            return header + f"No upcoming lockup expirations found for {ticker}."

        csv_str = data.to_csv(index=False, encoding="utf-8")
        return header + csv_str
    except Exception as e:
        return header + f"Error retrieving lockup expiration data: {str(e)}"


def get_northbound_flow_akshare(
    trade_date: Annotated[str, "Current date in yyyy-mm-dd format"],
) -> str:
    """Get Northbound capital flow (北向资金) data via Stock Connect.

    Northbound flows from Hong Kong-based foreign investors through
    Shanghai/Shenzhen-HK Stock Connect are a key daily sentiment barometer
    for A-shares. Net inflows signal foreign confidence; persistent outflows
    suggest bearish sentiment.
    """
    header = f"# Northbound Capital Flow (北向资金)\n"
    header += f"# Retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    try:
        data = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
        if data.empty:
            return header + "Northbound flow data is temporarily unavailable."

        date_str = trade_date.replace("-", "")
        if "日期" in data.columns:
            data["日期_str"] = data["日期"].astype(str).str[:10]
            filtered = data[data["日期_str"] == date_str]
        else:
            filtered = data

        if filtered.empty:
            recent = data.tail(5) if len(data) > 5 else data
            csv_str = recent.to_csv(index=False, encoding="utf-8")
            return header + f"No data for exact date {trade_date}. Showing most recent records:\n\n" + csv_str

        csv_str = filtered.to_csv(index=False, encoding="utf-8")
        return header + csv_str
    except Exception as e:
        return header + f"Error retrieving northbound flow data: {str(e)}"
