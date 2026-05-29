import re

from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)
from tradingagents.agents.utils.cn_market_tools import (
    get_dragon_tiger,
    get_hk_ipo,
    get_hk_short_selling,
    get_lockup_expiry,
    get_northbound_flow,
    get_southbound_flow,
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Applied to every agent whose output reaches the saved report —
    analysts, researchers, debaters, research manager, trader, and
    portfolio manager — so a non-English run produces a fully localized
    report rather than a mix of languages.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


A_SHARE_MARKET_RULES = """
## A-Share Market Rules (China)

You are analyzing a Chinese A-share stock. The following rules are critical for your analysis:

### Trading Mechanics
- **T+1 Settlement**: Shares bought today cannot be sold until the next trading day. Intraday trading is not possible — this affects liquidity and position-sizing decisions.
- **Price Limits (涨跌幅限制)**:
  - Main board (主板): ±10% daily price limit
  - ChiNext (创业板, 300xxx): ±20% daily price limit
  - STAR Market (科创板, 688xxx): ±20% daily price limit
  - ST (Special Treatment) stocks: ±5% daily price limit
- **No Intraday Short Selling**: Retail investors cannot short-sell; only qualified institutional investors can participate in margin trading & securities lending (融资融券).
- **Lot Size**: Minimum trading unit is 100 shares (一手), in multiples of 100.
- **Stamp Duty (印花税)**: 0.05% on sell only (collected by the government).

### Special Treatment (ST / *ST)
- Stocks labeled **ST** or ***ST** are under Special Treatment due to financial distress, negative net assets, or audit issues.
- *ST indicates risk of delisting. These stocks have tighter price limits (±5%) and higher risk.
- Always check whether the stock carries ST status and factor this into your risk assessment.

### Dragon-Tiger Board (龙虎榜)
- A daily disclosure of the top 5 brokerage trading desks (席位) by buy/sell volume for stocks with significant price moves or turnover.
- Heavy institutional buying on the Dragon-Tiger Board is a bullish signal; heavy retail/散户户 buying often indicates distribution.
- If Dragon-Tiger data is available in the news, analyze it as a sentiment signal.

### Lockup Expiration (解禁)
- Restricted shares (限售股) held by insiders, institutions, or pre-IPO investors become freely tradable after a lockup period (typically 6–36 months).
- Major lockup expirations create selling pressure — always check for upcoming 解禁 events.
- Use the insider transactions / shareholder data to identify concentrated unlocking risks.

### Financial Report Disclosure (财报披露)
- **Annual Report (年报)**: Must be disclosed by April 30 of the following year.
- **Q1 Report (一季报)**: Must be disclosed by April 30.
- **Semi-Annual Report (中报/半年报)**: Must be disclosed by August 31.
- **Q3 Report (三季报)**: Must be disclosed by October 31.
- Key windows to watch: pre-disclosure blackout periods, earnings surprises, and window-dressing (粉饰报表) by management.

### Other Key Considerations
- **Policy Sensitivity**: A-shares are highly sensitive to government policy, regulatory changes, and macro-prudential signals from the CSRC (证监会) and central bank (人民银行).
- **Sector Rotation**: Hot money (游资) frequently rotates between sectors based on policy themes — momentum can be sudden and sharp.
- **Northbound Capital (北向资金)**: Flows through Stock Connect (沪深港通) from Hong Kong-based foreign investors are a daily sentiment barometer.
"""

HK_MARKET_RULES = """
## Hong Kong Stock Market Rules (港股)

You are analyzing a Hong Kong-listed stock. The following rules are critical for your analysis:

### Trading Mechanics
- **T+2 Settlement**: Shares are delivered 2 business days after the trade date. No intraday restriction on selling — you can buy and sell on the same day.
- **No Daily Price Limits**: HK stocks have no price bands — they can gap dramatically on news/earnings. Position sizing and stop-losses are essential.
- **Lot Size (每手股数)**: Each stock has a fixed board lot size set by the HKEX (香港交易所). Common lot sizes are 100, 200, 500, 1000, or 2000 shares depending on the stock price tier. Always check the lot size before sizing positions.
- **Short Selling**: Allowed only for **designated securities** (HKEX maintains an eligible list). Short selling data is publicly disclosed daily — use `get_hk_short_selling` to check short interest trends.
- **Trading Hours**: Morning session 9:30–12:00, Afternoon session 13:00–16:00. No lunch break closure. Pre-opening session 9:00–9:30.
- **Stamp Duty (印花税)**: 0.1% on transaction value (collected on both buy and sell). Reduced from 0.13% effective November 2023.
- **No Capital Gains Tax**: Hong Kong does not tax capital gains or dividends for individual investors. However, H-share dividends are subject to 10% PRC withholding tax.

### Stock Types
- **Blue Chips (蓝筹股)**: Large-cap stocks, typically Hang Seng Index constituents (e.g. 0700.HK Tencent, 0005.HK HSBC).
- **H-Shares (H股)**: Mainland Chinese companies incorporated in the PRC and listed in HK. Subject to 10% dividend withholding tax.
- **Red Chips (红筹股)**: Mainland Chinese state-owned enterprises incorporated outside the PRC and listed in HK.
- **Growth Enterprise Market (GEM/创业板)**: Smaller, higher-risk stocks with the prefix "8" (e.g. 8xxx.HK).

### Stock Connect (沪深港通)
- **Southbound Capital (南向资金)**: Mainland Chinese investors buying HK stocks via Shanghai/Shenzhen-HK Stock Connect. Significant southbound inflows are bullish for HK stocks, especially H-shares.
- **Northbound Capital (北向资金)**: Foreign investors buying A-shares through the same mechanism — impacts A-share sentiment but also indicates overall cross-border appetite.
- **Quota System**: Daily quotas apply to both directions; exhaustion of southbound quota can cap upside momentum.

### Financial Reporting (财务披露)
- **Annual Report**: Must be published within 4 months of fiscal year-end (for Dec year-end: by April 30).
- **Interim Report (半年报)**: Must be published within 3 months of the period end.
- HK-listed companies follow HKFRS (Hong Kong Financial Reporting Standards), aligned with IFRS.

### Key Indices
- **Hang Seng Index (恒生指数, ^HSI)**: The primary benchmark for HK stocks. HSI inclusion/exclusion drives significant passive fund flows.
- **Hang Seng China Enterprises Index (恒生中国企业指数, ^HSCEI)**: Tracks H-shares — relevant for mainland-linked HK stocks.
- **Hang Seng Tech Index (恒生科技指数)**: Tracks major HK-listed tech companies like Tencent, Alibaba, Meituan, etc.

### Other Key Considerations
- **Typhoon/Rain Signal Closures**: The HKEX may close or shorten trading during typhoon signal No. 8 or black rainstorm warnings. Check for unexpected trading halts on extreme weather days.
- **IPO / Block Trade Activity**: Large block trades and new listings can create supply overhang. Monitor share registrations for placement activity.
- **Currency Exposure**: HK stocks trade in HKD (pegged to USD at ~7.75–7.85). Mainland businesses report in RMB — currency translation affects reported financials.
"""


def _is_a_share_ticker(ticker: str) -> bool:
    """Detect A-share stocks by exchange suffix or 6-digit numeric code."""
    t = ticker.strip().upper()
    if t.endswith(".SS") or t.endswith(".SZ"):
        return True
    if t.isdigit() and len(t) == 6:
        return True
    return False


def _is_hk_ticker(ticker: str) -> bool:
    """Detect HK stocks by .HK suffix or 1-5 digit numeric code."""
    t = ticker.strip().upper()
    if t.endswith(".HK"):
        return True
    t_clean = t.replace(".HK", "")
    return bool(re.match(r"^\d{1,5}$", t_clean))


def build_instrument_context(ticker: str, asset_type: str = "stock") -> str:
    """Describe the exact instrument and inject market-specific rules."""
    instrument_label = "asset" if asset_type == "crypto" else "instrument"
    extra_hint = (
        " Treat it as a crypto asset rather than a company, and do not assume company fundamentals are available."
        if asset_type == "crypto"
        else ""
    )

    context = (
        f"The {instrument_label} to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
        + extra_hint
    )

    if asset_type == "stock":
        if _is_a_share_ticker(ticker):
            context += A_SHARE_MARKET_RULES
        elif _is_hk_ticker(ticker):
            context += HK_MARKET_RULES

    return context

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
