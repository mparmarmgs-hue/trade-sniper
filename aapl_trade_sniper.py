"""
The Trade Sniper
A rule-based trading strategy application for any stock ticker
Prevents "catching a falling knife" and alerts only when reversal criteria are met
Originally designed for AAPL analysis, now supports all tickers
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="The Trade Sniper - Multi-Ticker Strategy Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Clean minimal theme
st.markdown("""
<style>
    /* Traffic Light Status Boxes - Main Feature */
    .red-stop {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: 800;
        margin: 25px 0;
        box-shadow: 0 8px 20px rgba(220, 38, 38, 0.3);
        border: 3px solid #ef4444;
        letter-spacing: 1px;
    }

    .green-go {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: 800;
        margin: 25px 0;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.3);
        border: 3px solid #34d399;
        letter-spacing: 1px;
    }

    /* Alert Boxes - Status Messages */
    .yellow-watch {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: #1f2937;
        padding: 18px;
        border-radius: 10px;
        font-weight: 600;
        margin: 15px 0;
        border-left: 5px solid #f97316;
    }

    .warning-box {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 18px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #f87171;
        font-weight: 500;
    }

    .info-box {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 18px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #60a5fa;
        font-weight: 500;
    }

    .success-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 18px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #34d399;
        font-weight: 500;
    }

    /* Headers - Enhanced visibility */
    h1 {
        color: #10b981 !important;
        font-weight: 700 !important;
        padding-bottom: 10px !important;
        border-bottom: 3px solid #10b981 !important;
    }

    h2 {
        color: #60a5fa !important;
        font-weight: 600 !important;
        margin-top: 30px !important;
    }

    h3 {
        color: #a78bfa !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def safe_float(value, default=0.0):
    """Safely convert to float"""
    try:
        if pd.isna(value):
            return default
        if hasattr(value, 'item'):
            return float(value.item())
        return float(value)
    except:
        return default

def calculate_sma(data, window):
    """Calculate Simple Moving Average"""
    return data['Close'].rolling(window=window, min_periods=1).mean()

def calculate_rsi(data, period=14):
    """Calculate RSI"""
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(data, period=14):
    """Calculate Average True Range"""
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def get_seasonality_data():
    """Hardcoded monthly average returns for AAPL (historical patterns)"""
    return {
        'January': 2.5,
        'February': -1.2,
        'March': -0.8,
        'April': -0.5,
        'May': 3.1,
        'June': 1.8,
        'July': 2.9,
        'August': 0.3,
        'September': -2.1,
        'October': 1.4,
        'November': 4.2,
        'December': 3.8
    }

# Sidebar - Ticker Selection
st.sidebar.title("🎯 Trade Sniper Settings")
st.sidebar.markdown("### 📈 Stock Selection")
ticker_symbol = st.sidebar.text_input(
    "Ticker Symbol",
    value="AAPL",
    help="Enter any stock ticker symbol (e.g., AAPL, TSLA, MSFT, GOOGL)"
).upper().strip()

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Risk Settings")
total_capital = st.sidebar.number_input(
    "Total Capital ($)",
    min_value=1000,
    max_value=10000000,
    value=10000,
    step=1000
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Position Sizing")
tier1_percent = 20
tier2_percent = 30
tier1_capital = total_capital * (tier1_percent / 100)
tier2_capital = total_capital * (tier2_percent / 100)

st.sidebar.metric("Tier 1 Entry (20%)", f"${tier1_capital:,.2f}")
st.sidebar.metric("Tier 2 Entry (30%)", f"${tier2_capital:,.2f}")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Advanced Settings")
lookback_days = st.sidebar.slider("Lookback Period (Days)", 90, 730, 365)
volatility_threshold = st.sidebar.slider("Volatility Threshold (%)", 1.0, 5.0, 2.5, 0.1)

# Main title
st.title(f"🎯 The Trade Sniper — {ticker_symbol}")
st.markdown("**Rule-Based Trading Strategy | Prevent Catching Falling Knives**")

# Fetch data
@st.cache_data(ttl=300)
def fetch_stock_data(ticker_symbol, days=365):
    """Fetch stock data with caching"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get historical data
        hist = ticker.history(start=start_date, end=end_date)

        # Flatten multi-level columns if present
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        # Get info
        info = ticker.info

        return hist, info
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None, None

# Fetch long-term data for historical context
@st.cache_data(ttl=3600)
def fetch_long_term_data(ticker_symbol):
    """Fetch 20-year monthly data"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*20)
        hist = ticker.history(start=start_date, end=end_date, interval="1mo")
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        return hist
    except Exception as e:
        st.error(f"Error fetching long-term data: {str(e)}")
        return None

# Load data
with st.spinner(f"Loading {ticker_symbol} data..."):
    data, info = fetch_stock_data(ticker_symbol, lookback_days)
    long_term_data = fetch_long_term_data(ticker_symbol)

if data is None or data.empty:
    st.error(f"Unable to fetch {ticker_symbol} data. Please check the ticker symbol and your internet connection.")
    st.stop()

# Calculate indicators
data['SMA5'] = calculate_sma(data, 5)
data['SMA21'] = calculate_sma(data, 21)
data['SMA50'] = calculate_sma(data, 50)
data['SMA100'] = calculate_sma(data, 100)
data['RSI14'] = calculate_rsi(data, 14)
data['ATR14'] = calculate_atr(data, 14)

# Get current values
current_price = safe_float(data['Close'].iloc[-1])
sma5_current = safe_float(data['SMA5'].iloc[-1])
sma21_current = safe_float(data['SMA21'].iloc[-1])
sma50_current = safe_float(data['SMA50'].iloc[-1])
rsi_current = safe_float(data['RSI14'].iloc[-1])
atr_current = safe_float(data['ATR14'].iloc[-1])
volume_current = safe_float(data['Volume'].iloc[-1])
volume_avg = safe_float(data['Volume'].rolling(10).mean().iloc[-1])

# Get fundamentals
peg_ratio = info.get('pegRatio', None)
pe_ratio = info.get('trailingPE', None)
week52_high = info.get('fiftyTwoWeekHigh', None)
week52_low = info.get('fiftyTwoWeekLow', None)

# Calculate daily volatility
daily_returns = data['Close'].pct_change()
daily_volatility = safe_float(daily_returns.iloc[-1] * 100)

# Check for RSI crossover
rsi_previous = safe_float(data['RSI14'].iloc[-2])
rsi_crossover = rsi_previous < 30 and rsi_current >= 30

# === MODULE 1: TRAFFIC LIGHT STATUS ===
st.markdown("---")
st.header("🚦 Traffic Light Status (Primary Filter)")

if current_price < sma21_current:
    st.markdown(f"""
    <div class="red-stop">
        🛑 DO NOT ENTER - TREND IS BEARISH 🛑<br>
        Price ${current_price:.2f} is BELOW 21-Day SMA ${sma21_current:.2f}
    </div>
    """, unsafe_allow_html=True)
    traffic_light = "RED"
else:
    st.markdown(f"""
    <div class="green-go">
        ✅ TREND SUPPORTED - SAFE TO SCALE IN ✅<br>
        Price ${current_price:.2f} is ABOVE 21-Day SMA ${sma21_current:.2f}
    </div>
    """, unsafe_allow_html=True)
    traffic_light = "GREEN"

# === MODULE 2: ENTRY TRIGGER CALCULATOR ===
st.markdown("---")
st.header("🎯 Entry Trigger Calculator (The 20/30 Rule)")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Current Price", f"${current_price:.2f}")

with col2:
    gap_to_5day = sma5_current - current_price
    if gap_to_5day > 0:
        st.metric("Gap to 5-Day SMA", f"${gap_to_5day:.2f}", delta="Must rise", delta_color="inverse")
    else:
        st.metric("Above 5-Day SMA", f"${abs(gap_to_5day):.2f}", delta="Tier 1 Active", delta_color="normal")

with col3:
    gap_to_21day = sma21_current - current_price
    if gap_to_21day > 0:
        st.metric("Gap to 21-Day SMA", f"${gap_to_21day:.2f}", delta="Must rise", delta_color="inverse")
    else:
        st.metric("Above 21-Day SMA", f"${abs(gap_to_21day):.2f}", delta="Tier 2 Active", delta_color="normal")

# Entry signals
st.subheader("Entry Signals")
if current_price > sma5_current:
    st.markdown(f"""
    <div class="success-box">
        ✅ TIER 1 TRIGGER: Price crossed above 5-Day SMA<br>
        → Deploy ${tier1_capital:,.2f} (20% of capital)<br>
        → Entry Price: ${current_price:.2f}
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="info-box">
        ⏳ Waiting for Tier 1 Trigger<br>
        Price must rise ${gap_to_5day:.2f} to cross 5-Day SMA (${sma5_current:.2f})<br>
        Then deploy ${tier1_capital:,.2f} (20% capital)
    </div>
    """, unsafe_allow_html=True)

if current_price > sma21_current:
    st.markdown(f"""
    <div class="success-box">
        ✅ TIER 2 TRIGGER: Price crossed above 21-Day SMA<br>
        → Deploy ${tier2_capital:,.2f} (30% of capital)<br>
        → Entry Price: ${current_price:.2f}
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="info-box">
        ⏳ Waiting for Tier 2 Trigger<br>
        Price must rise ${gap_to_21day:.2f} to cross 21-Day SMA (${sma21_current:.2f})<br>
        Then add ${tier2_capital:,.2f} (30% capital)
    </div>
    """, unsafe_allow_html=True)

# === MODULE 3: FUNDAMENTAL & RISK RED FLAGS ===
st.markdown("---")
st.header("⚠️ Fundamental & Risk Red Flags")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if peg_ratio:
        st.metric("PEG Ratio", f"{peg_ratio:.2f}", delta="Target < 1.0")
        if peg_ratio > 1.0:
            st.markdown(f"""
            <div class="warning-box">
                ⚠️ Valuation Drag: PEG is {peg_ratio:.2f} (Target < 1.0)<br>
                Growth is expensive
            </div>
            """, unsafe_allow_html=True)
    else:
        st.metric("PEG Ratio", "N/A")

with col2:
    if pe_ratio:
        st.metric("P/E Ratio", f"{pe_ratio:.2f}")
    else:
        st.metric("P/E Ratio", "N/A")

with col3:
    st.metric("Daily Volatility", f"{abs(daily_volatility):.2f}%")
    if abs(daily_volatility) > volatility_threshold:
        st.markdown(f"""
        <div class="warning-box">
            ⚠️ High Volatility Warning<br>
            Use Wider Stops
        </div>
        """, unsafe_allow_html=True)

with col4:
    atr_percent = (atr_current / current_price) * 100
    st.metric("14-Day ATR", f"{atr_percent:.2f}%")

# === MODULE 4: FALLING KNIFE PREVENTER ===
st.markdown("---")
st.header("🔪 Falling Knife Preventer (RSI + Volume)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("RSI Analysis")
    st.metric("14-Day RSI", f"{rsi_current:.2f}")

    if rsi_current < 30:
        st.markdown("""
        <div class="yellow-watch">
            ⚠️ OVERSOLD - WATCHLIST ONLY<br>
            Do not buy yet. Wait for momentum reversal.
        </div>
        """, unsafe_allow_html=True)
    elif rsi_crossover:
        st.markdown("""
        <div class="success-box">
            ✅ MOMENTUM REVERSAL DETECTED<br>
            RSI crossed above 30 from oversold territory
        </div>
        """, unsafe_allow_html=True)
    elif rsi_current > 70:
        st.markdown("""
        <div class="warning-box">
            ⚠️ OVERBOUGHT<br>
            Risk of pullback. Consider taking profits.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            📊 RSI in neutral zone (30-70)
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.subheader("Volume Analysis")
    volume_ratio = volume_current / volume_avg if volume_avg > 0 else 1.0
    st.metric("Volume vs 10-Day Avg", f"{volume_ratio:.2f}x")

    price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
    price_down = price_change < 0

    if volume_ratio > 1.2 and price_down:
        st.markdown("""
        <div class="warning-box">
            🔻 SELLING PRESSURE INTENSIFYING<br>
            High volume + Down day (No Yellow Bar yet)
        </div>
        """, unsafe_allow_html=True)
    elif volume_ratio > 1.2 and not price_down:
        st.markdown("""
        <div class="success-box">
            📈 BUYING PRESSURE DETECTED<br>
            High volume + Up day
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            📊 Volume normal
        </div>
        """, unsafe_allow_html=True)

# === MODULE 5: MAIN CHART ===
st.markdown("---")
st.header("📈 Technical Analysis Chart")

fig = go.Figure()

# Candlestick
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name=ticker_symbol,
    increasing_line_color='#00cc66',
    decreasing_line_color='#ff4444'
))

# SMAs
fig.add_trace(go.Scatter(x=data.index, y=data['SMA5'], name='SMA 5', line=dict(color='#ffaa00', width=1)))
fig.add_trace(go.Scatter(x=data.index, y=data['SMA21'], name='SMA 21', line=dict(color='#4a90e2', width=2)))
fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name='SMA 50', line=dict(color='#9b59b6', width=1)))

fig.update_layout(
    title=f'{ticker_symbol} Price Action with Moving Averages',
    yaxis_title='Price ($)',
    xaxis_title='Date',
    template='plotly_dark',
    height=600,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, width='stretch')

# === MODULE 6: SEASONALITY CHECKER ===
st.markdown("---")
st.header("📅 Seasonality Checker")

current_month = datetime.now().strftime('%B')
seasonality_data = get_seasonality_data()
current_month_avg = seasonality_data.get(current_month, 0)

col1, col2 = st.columns([2, 1])

with col1:
    # Create seasonality bar chart
    months = list(seasonality_data.keys())
    returns = list(seasonality_data.values())
    colors = ['#ff4444' if r < 0 else '#00cc66' for r in returns]

    fig_season = go.Figure(data=[
        go.Bar(x=months, y=returns, marker_color=colors)
    ])

    fig_season.update_layout(
        title='AAPL Historical Monthly Average Returns (10-Year Pattern)',
        yaxis_title='Average Return (%)',
        template='plotly_dark',
        height=400
    )

    st.plotly_chart(fig_season, width='stretch')

with col2:
    st.metric("Current Month", current_month)
    st.metric("Historical Avg Return", f"{current_month_avg:.1f}%")

    if current_month_avg < 0:
        st.markdown(f"""
        <div class="warning-box">
            ⚠️ SEASONAL CAUTION<br>
            {current_month} is historically weak<br>
            (Avg: {current_month_avg:.1f}%)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="success-box">
            ✅ SEASONAL TAILWIND<br>
            {current_month} is historically strong<br>
            (Avg: {current_month_avg:.1f}%)
        </div>
        """, unsafe_allow_html=True)

# === MODULE 7: HISTORICAL CONTEXT (THE CISCO TEST) ===
st.markdown("---")
st.header("📊 Historical Context (The 'Cisco Test')")

st.markdown("""
**Question:** Is the current chart parabolic like Cisco 2000 or PayPal?
**Reference:** Strategy warns about 20-year structural corrections.
""")

if long_term_data is not None and not long_term_data.empty:
    fig_hist = go.Figure()

    fig_hist.add_trace(go.Scatter(
        x=long_term_data.index,
        y=long_term_data['Close'],
        name=f'{ticker_symbol} Monthly',
        line=dict(color='#4a90e2', width=2),
        fill='tozeroy',
        fillcolor='rgba(74, 144, 226, 0.1)'
    ))

    fig_hist.update_layout(
        title=f'{ticker_symbol} 20-Year Monthly Chart - Structural Analysis',
        yaxis_title='Price ($)',
        xaxis_title='Year',
        template='plotly_dark',
        height=500,
        yaxis_type='log'  # Log scale to better see long-term trends
    )

    st.plotly_chart(fig_hist, width='stretch')

    st.info(f"""
    📌 **Structural Analysis:**
    Look for parabolic moves followed by multi-year corrections.
    The strategy references Cisco (2000) and PayPal as examples of 20-year structural declines.
    Current {ticker_symbol} position: Assess if we're in a similar corrective phase.
    """)
else:
    st.warning("Unable to load 20-year historical data.")

# === FINAL SUMMARY DASHBOARD ===
st.markdown("---")
st.header("🎯 Trade Decision Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Status Check")
    status_score = 0
    if traffic_light == "GREEN":
        st.success("✅ Traffic Light: GREEN")
        status_score += 1
    else:
        st.error("🛑 Traffic Light: RED")

    if current_price > sma5_current:
        st.success("✅ Above 5-Day SMA")
        status_score += 1
    else:
        st.warning("⏳ Below 5-Day SMA")

    if rsi_current >= 30:
        st.success("✅ RSI > 30")
        status_score += 1
    else:
        st.warning("⚠️ RSI < 30 (Oversold)")

with col2:
    st.subheader("Risk Factors")
    risk_score = 0
    if peg_ratio and peg_ratio > 1.0:
        st.warning(f"⚠️ High PEG ({peg_ratio:.2f})")
        risk_score += 1
    else:
        st.success("✅ PEG Acceptable")

    if abs(daily_volatility) > volatility_threshold:
        st.warning(f"⚠️ High Volatility ({abs(daily_volatility):.2f}%)")
        risk_score += 1
    else:
        st.success("✅ Normal Volatility")

    if current_month_avg < 0:
        st.warning(f"⚠️ Weak Seasonal ({current_month})")
        risk_score += 1
    else:
        st.success("✅ Strong Seasonal")

with col3:
    st.subheader("Action Plan")

    if traffic_light == "RED":
        st.error("🛑 DO NOT ENTER - WAIT")
        st.markdown("Price below 21-day SMA. Stay in cash.")
    elif status_score >= 2 and risk_score <= 1:
        st.success("✅ CONDITIONS FAVORABLE")
        if current_price > sma5_current:
            st.markdown(f"**Deploy Tier 1:** ${tier1_capital:,.2f}")
        if current_price > sma21_current:
            st.markdown(f"**Deploy Tier 2:** ${tier2_capital:,.2f}")
    else:
        st.warning("⏳ WATCHLIST - NOT YET")
        st.markdown("Wait for better setup")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>🎯 The Trade Sniper | Multi-Ticker Rule-Based Strategy | Data from yfinance</p>
    <p><small>This is not financial advice. Trade at your own risk.</small></p>
</div>
""", unsafe_allow_html=True)
