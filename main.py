"""
FX Spot Monitor & Analytics Dashboard — single-file build.
Run: streamlit run main.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="FX Monitor", layout="wide", initial_sidebar_state="expanded")

# ─── Config ───────────────────────────────────────────────────────────────
PAIRS = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X",
    "USD/CHF": "CHF=X",    "AUD/USD": "AUDUSD=X", "USD/CAD": "CAD=X",
    "NZD/USD": "NZDUSD=X", "EUR/GBP": "EURGBP=X", "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
}
USD_PAIRS = {  # quote per 1 USD
    "EUR": "EUR=X", "GBP": "GBP=X", "JPY": "JPY=X", "CHF": "CHF=X",
    "CAD": "CAD=X", "AUD": "AUD=X", "NZD": "NZD=X",
}
PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]

# ─── Data layer ───────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

@st.cache_data(ttl=300, show_spinner=False)
def get_pair_closes(period: str) -> pd.DataFrame:
    out = {}
    for name, tkr in PAIRS.items():
        d = fetch_history(tkr, period)
        if not d.empty:
            out[name] = d["Close"]
    return pd.DataFrame(out).dropna(how="all")

@st.cache_data(ttl=60, show_spinner=False)
def get_latest_rates() -> pd.DataFrame:
    rows = []
    for name, tkr in PAIRS.items():
        d = fetch_history(tkr, "5d")
        if len(d) >= 2:
            last, prev = float(d["Close"].iloc[-1]), float(d["Close"].iloc[-2])
            rows.append({"Pair": name, "Rate": last, "Change": last - prev,
                         "Change %": (last / prev - 1) * 100})
    return pd.DataFrame(rows)

@st.cache_data(ttl=300, show_spinner=False)
def get_usd_rates() -> dict:
    rates = {"USD": 1.0}
    for ccy, tkr in USD_PAIRS.items():
        d = fetch_history(tkr, "5d")
        if not d.empty:
            rates[ccy] = float(d["Close"].iloc[-1])
    return rates

# ─── Analytics ────────────────────────────────────────────────────────────
def build_cross_rate_matrix(usd_rates: dict) -> pd.DataFrame:
    ccys = list(usd_rates.keys())
    m = pd.DataFrame(index=ccys, columns=ccys, dtype=float)
    for b in ccys:
        for q in ccys:
            # 1 b = (usd_rates[q] / usd_rates[b]) q, since usd_rates = quote per USD
            m.loc[b, q] = usd_rates[q] / usd_rates[b]
    return m

def rolling_vol(closes: pd.Series, window: int) -> pd.Series:
    return closes.pct_change().rolling(window).std() * np.sqrt(252) * 100

def correlation_matrix(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.pct_change().tail(window).corr()

def bollinger(closes: pd.Series, w: int, k: float):
    ma = closes.rolling(w).mean()
    sd = closes.rolling(w).std()
    return ma, ma + k * sd, ma - k * sd

def rsi(closes: pd.Series, w: int = 14) -> pd.Series:
    delta = closes.diff()
    up = delta.clip(lower=0).rolling(w).mean()
    dn = (-delta.clip(upper=0)).rolling(w).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

# ─── Charts ───────────────────────────────────────────────────────────────
def price_chart(df, pair, ma_s, ma_l, bb_w, bb_k):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name=pair))
    c = df["Close"]
    fig.add_trace(go.Scatter(x=df.index, y=c.rolling(ma_s).mean(), name=f"MA{ma_s}", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=c.rolling(ma_l).mean(), name=f"MA{ma_l}", line=dict(width=1)))
    ma, up, lo = bollinger(c, bb_w, bb_k)
    fig.add_trace(go.Scatter(x=df.index, y=up, name="BB Up", line=dict(dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=lo, name="BB Low", line=dict(dash="dot", width=1), fill="tonexty", fillcolor="rgba(100,100,250,0.08)"))
    fig.update_layout(title=f"{pair} Price", xaxis_rangeslider_visible=False, height=600, template="plotly_dark")
    return fig

def volatility_chart(closes, pair, w):
    v = rolling_vol(closes, w)
    r = rsi(closes)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v.index, y=v, name=f"Vol {w}d (ann %)", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=r.index, y=r, name="RSI 14", yaxis="y2", line=dict(color="cyan")))
    fig.update_layout(title=f"{pair} Volatility & RSI", height=500, template="plotly_dark",
                      yaxis=dict(title="Vol %"), yaxis2=dict(title="RSI", overlaying="y", side="right", range=[0, 100]))
    return fig

def correlation_heatmap(corr):
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(title="Pair Return Correlations", height=600, template="plotly_dark")
    return fig

# ─── Sidebar ──────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.header("Controls")
    pair = st.sidebar.selectbox("Pair", list(PAIRS.keys()))
    period = st.sidebar.select_slider("History", PERIODS, value="6mo")

    st.sidebar.subheader("Technicals")
    ma_short = st.sidebar.slider("MA short", 5, 50, 20)
    ma_long = st.sidebar.slider("MA long", 20, 200, 50)
    bb_window = st.sidebar.slider("Bollinger window", 10, 60, 20)
    bb_std = st.sidebar.slider("Bollinger std", 1.0, 4.0, 2.0, 0.1)

    st.sidebar.subheader("Stats")
    vol_window = st.sidebar.slider("Vol window (d)", 5, 60, 20)
    corr_window = st.sidebar.slider("Corr lookback (d)", 20, 252, 60)

    st.sidebar.subheader("P&L Simulator")
    direction = st.sidebar.radio("Direction", ["Long", "Short"], horizontal=True)
    entry_rate = st.sidebar.number_input("Entry rate", value=1.0, step=0.0001, format="%.4f")
    notional = st.sidebar.number_input("Notional (base ccy)", value=100000, step=10000)

    return dict(pair=pair, ticker=PAIRS[pair], period=period, ma_short=ma_short, ma_long=ma_long,
                bb_window=bb_window, bb_std=bb_std, vol_window=vol_window, corr_window=corr_window,
                direction=direction, entry_rate=entry_rate, notional=notional)

# ─── Tables / widgets ─────────────────────────────────────────────────────
def render_summary_metrics(closes, pair, w):
    last = float(closes.iloc[-1]); prev = float(closes.iloc[-2])
    chg = last - prev; pct = (last / prev - 1) * 100
    v = rolling_vol(closes, w).iloc[-1]
    hi, lo = float(closes.max()), float(closes.min())
    c = st.columns(5)
    c[0].metric(pair, f"{last:.4f}", f"{chg:+.4f} ({pct:+.2f}%)")
    c[1].metric("Period High", f"{hi:.4f}")
    c[2].metric("Period Low", f"{lo:.4f}")
    c[3].metric(f"Vol {w}d", f"{v:.2f}%")
    c[4].metric("Bars", f"{len(closes)}")

def render_spot_table(df):
    if df.empty: st.warning("No rates."); return
    st.dataframe(df.style.format({"Rate": "{:.4f}", "Change": "{:+.4f}", "Change %": "{:+.2f}%"})
                 .background_gradient(subset=["Change %"], cmap="RdYlGn"), use_container_width=True)

def render_cross_matrix(m):
    st.dataframe(m.style.format("{:.4f}").background_gradient(cmap="Blues"), use_container_width=True)

def render_pnl(pair, direction, entry, current, notional):
    pips_mult = 100 if "JPY" in pair else 10000
    diff = (current - entry) if direction == "Long" else (entry - current)
    pips = diff * pips_mult
    quote_pnl = diff * notional
    c = st.columns(4)
    c[0].metric("Entry", f"{entry:.4f}")
    c[1].metric("Current", f"{current:.4f}")
    c[2].metric("Pips", f"{pips:+.1f}")
    c[3].metric("P&L (quote)", f"{quote_pnl:+,.2f}")
    # Sensitivity curve
    rng = np.linspace(entry * 0.95, entry * 1.05, 100)
    pnl = (rng - entry if direction == "Long" else entry - rng) * notional
    fig = go.Figure(go.Scatter(x=rng, y=pnl, mode="lines", line=dict(color="lime")))
    fig.add_vline(x=current, line_dash="dash", line_color="white")
    fig.add_hline(y=0, line_color="gray")
    fig.update_layout(title="P&L vs Rate", xaxis_title="Rate", yaxis_title="P&L", height=400, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    st.title("FX Spot Monitor")
    st.caption("Live rates, technicals, volatility and cross-rate analytics")
    p = render_sidebar()

    with st.spinner("Loading..."):
        df = fetch_history(p["ticker"], p["period"])
    if df.empty:
        st.error(f"No data for {p['pair']}."); return
    closes = df["Close"]

    render_summary_metrics(closes, p["pair"], p["vol_window"])
    st.markdown("---")

    t1, t2, t3, t4, t5, t6 = st.tabs(["Chart", "Volatility", "Correlation", "Cross Rates", "P&L", "Spot Rates"])

    with t1:
        st.plotly_chart(price_chart(df, p["pair"], p["ma_short"], p["ma_long"], p["bb_window"], p["bb_std"]), use_container_width=True)
    with t2:
        st.plotly_chart(volatility_chart(closes, p["pair"], p["vol_window"]), use_container_width=True)
    with t3:
        with st.spinner("Computing..."):
            pc = get_pair_closes(p["period"])
        if pc.empty: st.warning("Insufficient data.")
        else: st.plotly_chart(correlation_heatmap(correlation_matrix(pc, p["corr_window"])), use_container_width=True)
    with t4:
        with st.spinner("Building..."):
            usd = get_usd_rates()
        if len(usd) < 2: st.warning("Insufficient data.")
        else:
            st.subheader("Cross-Rate Matrix (1 Base = X Quote)")
            render_cross_matrix(build_cross_rate_matrix(usd))
    with t5:
        st.subheader(f"P&L Simulator — {p['pair']}")
        render_pnl(p["pair"], p["direction"], p["entry_rate"], float(closes.iloc[-1]), p["notional"])
    with t6:
        st.subheader("Current Spot Rates")
        with st.spinner("Fetching..."):
            rates = get_latest_rates()
        render_spot_table(rates)

if __name__ == "__main__":
    main()
