from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_dragon_tiger(
    ticker: Annotated[str, "Ticker symbol"],
    trade_date: Annotated[str, "Current date in yyyy-mm-dd format"],
) -> str:
    """Retrieve Dragon-Tiger Board (龙虎榜) data for an A-share stock.

    The Dragon-Tiger Board discloses the top 5 brokerage trading desks (席位)
    ranked by buy/sell volume. It triggers when a stock has significant price
    movement or abnormal turnover. Heavy institutional buying is bullish;
    heavy retail buying often signals distribution (出货).

    Args:
        ticker: A-share ticker with exchange suffix (e.g. 600519.SS)
        trade_date: Analysis date in YYYY-MM-DD format

    Returns:
        str: Dragon-Tiger Board data or message if stock didn't appear on the board
    """
    return route_to_vendor("get_dragon_tiger", ticker, trade_date)


@tool
def get_lockup_expiry(
    ticker: Annotated[str, "Ticker symbol"],
) -> str:
    """Retrieve upcoming restricted share lockup expiration (限售股解禁) data.

    Restricted shares held by insiders, institutions, or pre-IPO investors
    become freely tradable after a lockup period (typically 6-36 months).
    Major unlock events create downward selling pressure on the stock price.

    Args:
        ticker: A-share ticker with exchange suffix (e.g. 600519.SS)

    Returns:
        str: Upcoming lockup expirations with share counts and dates
    """
    return route_to_vendor("get_lockup_expiry", ticker)


@tool
def get_northbound_flow(
    trade_date: Annotated[str, "Current date in yyyy-mm-dd format"],
) -> str:
    """Retrieve Northbound capital flow (北向资金) data from Stock Connect.

    Northbound flows represent Hong Kong-based foreign institutional money
    flowing into A-shares via Shanghai/Shenzhen-HK Stock Connect (沪深港通).
    Consistent net inflows signal foreign confidence; persistent net outflows
    suggest bearish sentiment or capital flight.

    Args:
        trade_date: Analysis date in YYYY-MM-DD format

    Returns:
        str: Northbound net flow data (daily net buy/sell amounts by market)
    """
    return route_to_vendor("get_northbound_flow", trade_date)
