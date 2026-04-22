"""FX Spot Monitor & Analytics Dashboard — single-file build."""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="FX Monitor", layout="wide", initial_sidebar_state="expanded")

PAIRS = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X",
    "USD/CHF": "CHF=X",    "AUD/USD": "AUDUSD=X", "USD/CAD": "CAD=X",
    "NZD/USD": "NZDUSD=X", "EUR/GBP": "EURGBP=X", "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
}
USD_PAIRS = {"EUR": "EUR=X", "GBP": "GBP=X", "JPY": "JPY=X", "CHF": "CHF=X",
             "CAD": "CAD=X", "AUD": "AUD=X", "NZD": "NZD=X"}
PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]

# ─── Data ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(ticker: str, period: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True, threads=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
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
        if d.empty or len(d) < 2:
            continue
        last, prev = float(d["Close"].iloc[-1]), float(d["Close"].iloc[-2])
        rows.append({"Pair": name, "Rate": last,
                     "Change": last - prev, "Change %": (last / prev - 1) * 100})
    return pd.DataFrame(rows)

@st.cache_data(ttl=300, show_spinner=False)
def get_usd_rates() -> dict:
    out = {"USD": 1.0}
    for ccy, tkr in USD_PAIRS.items():
        d = fetch_history(tkr, "5d")
        if not d.empty:
            out[ccy] = float(d["Close"].iloc[-1])
    return out

# ─── Analytics ───────────────────────────────────────────────────────────
def bollinger(s, w, k):
    m = s.rolling(w).mean()
    sd = s.rolling(w).std()
    return m, m + k * sd, m - k * sd

def rolling_vol(s, w):
    return s.pct_change().rolling(w).std() * np.sqrt(252) * 100

def correlation_matrix(df, w):
    return df.pct_change().tail(w).corr()

def build_cross_rate_matrix(usd_rates: dict) -> pd.DataFrame:
    ccy = list(usd_rates.keys())
    M = pd.DataFrame(index=ccy, columns=ccy, dtype=float)
    for b in ccy:
        for q in ccy:
            # 1 base = X quote  =>  (USD per base inverse) * (quote per USD)
            # usd_rates stores "quote per 1 USD" (except USD=1)
            M.loc[b, q] = usd_rates[q] / usd_rates[b]
    return M

# ─── UI ──────────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.header("Controls")
    pair = st.sidebar.selectbox("Pair", list(PAIRS.keys()))
    period = st.sidebar.select_slider("History", PERIODS, value="6mo")
    st.sidebar.subheader("Technicals")
    ma_s = st.sidebar.slider("MA short", 5, 100, 20)
    ma_l = st.sidebar.slider("MA long", 20, 300, 50)
    bb_w = st.sidebar.slider("Bollinger window", 10, 100, 20)
    bb_k = st.sidebar.slider("Bollinger std", 1.0, 4.0, 2.0, 0.1)
    st.sidebar.subheader("Volatility / Correlation")
    vol_w = st.sidebar.slider("Vol window (days)", 5, 120, 30)
    corr_w = st.sidebar.slider("Corr window (days)", 20, 500, 90)
    st.sidebar.subheader("P&L Simulator")
    direction = st.sidebar.radio("Direction", ["Long", "Short"], horizontal=True)
    entry = st.sidebar.number_input("Entry rate", value=1.0, step=0.0001, format="%.4f")
    notional = st.sidebar.number_input("Notional (base ccy)", value=100000, step=1000)
    return dict(pair=pair, ticker=PAIRS[pair], period=period,
                ma_short=ma_s, ma_long=ma_l, bb_window=bb_w, bb_std=bb_k,
                vol_window=vol_w, corr_window=corr_w,
                direction=direction, entry_rate=entry, notional=notional)

def price_chart(df, pair, ma_s, ma_l, bb_w, bb_k):
    c = df["Close"]
    mid, up, lo = bollinger(c, bb_w, bb_k)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name=pair))
    fig.add_trace(go.Scatter(x=c.index, y=c.rolling(ma_s).mean(), name=f"MA{ma_s}"))
    fig.add_trace(go.Scatter(x=c.index, y=c.rolling(ma_l).mean(), name=f"MA{ma_l}"))
    fig.add_trace(go.Scatter(x=up.index, y=up, name="BB up", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=lo.index, y=lo, name="BB lo", line=dict(dash="dot"),
                             fill="tonexty", fillcolor="rgba(100,100,250,0.1)"))
    fig.update_layout(height=550, xaxis_rangeslider_visible=False, template="plotly_dark")
    return fig

def volatility_chart(closes, pair, w):
    v = rolling_vol(closes, w).dropna()
    fig = px.area(v, title=f"{pair} — Annualized Rolling Volatility ({w}d), %")
    fig.update_layout(template="plotly_dark", showlegend=False, height=450)
    return fig

def correlation_heatmap(corr):
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(template="plotly_dark", height=550)
    return fig

def render_summary_metrics(closes, pair, w):
    last = float(closes.iloc[-1])
    prev = float(closes.iloc[-2]) if len(closes) > 1 else last
    chg = (last / prev - 1) * 100
    hi, lo = float(closes.max()), float(closes.min())
    vol = rolling_vol(closes, w).dropna()
    v = float(vol.iloc[-1]) if not vol.empty else 0.0
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(pair, f"{last:.4f}", f"{chg:+.2f}%")
    c2.metric("Period High", f"{hi:.4f}")
    c3.metric("Period Low", f"{lo:.4f}")
    c4.metric(f"Vol {w}d", f"{v:.2f}%")
    c5.metric("Observations", len(closes))

def render_spot_table(df):
    if df.empty:
        st.warning("No rates available."); return
    st.dataframe(df.style.format({"Rate": "{:.4f}", "Change": "{:+.4f}",
                                  "Change %": "{:+.2f}%"})
                 .background_gradient(subset=["Change %"], cmap="RdYlGn"),
                 use_container_width=True)

def render_cross_matrix(m):
    st.dataframe(m.style.format("{:.4f}").background_gradient(cmap="Blues"),
                 use_container_width=True)

def render_pnl(pair, direction, entry, current, notional):
    sign = 1 if direction == "Long" else -1
    pnl_quote = sign * (current - entry) * notional
    pnl_pct = sign * (current / entry - 1) * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Current rate", f"{current:.4f}")
    c2.metric("P&L (quote ccy)", f"{pnl_quote:,.2f}")
    c3.metric("P&L %", f"{pnl_pct:+.2f}%")

# ─── Main ────────────────────────────────────────────────────────────────
def main():
    st.title("FX Spot Monitor")
    st.caption("Live rates, technicals, volatility and cross-rate analytics")
    p = render_sidebar()

    with st.spinner("Loading..."):
        df = fetch_history(p["ticker"], p["period"])
    if df.empty:
        st.error(f"No data for {p['pair']}. Yahoo Finance may be rate-limiting. Retry in ~1 min.")
        return
    closes = df["Close"]

    render_summary_metrics(closes, p["pair"], p["vol_window"])
    st.markdown("---")

    t1, t2, t3, t4, t5, t6 = st.tabs(
        ["Chart", "Volatility", "Correlation", "Cross Rates", "P&L", "Spot Rates"])

    with t1:
        st.plotly_chart(price_chart(df, p["pair"], p["ma_short"], p["ma_long"],
                                    p["bb_window"], p["bb_std"]), use_container_width=True)
    with t2:
        st.plotly_chart(volatility_chart(closes, p["pair"], p["vol_window"]),
                        use_container_width=True)
    with t3:
        with st.spinner("Computing..."):
            pc = get_pair_closes(p["period"])
        if pc.empty or pc.shape[1] < 2:
            st.warning("Insufficient data.")
        else:
            st.plotly_chart(correlation_heatmap(correlation_matrix(pc, p["corr_window"])),
                            use_container_width=True)
    with t4:
        with st.spinner("Building..."):
            usd = get_usd_rates()
        if len(usd) < 2:
            st.warning("Insufficient data.")
        else:
            st.subheader("Cross-Rate Matrix (1 Base = X Quote)")
            render_cross_matrix(build_cross_rate_matrix(usd))
    with t5:
        st.subheader(f"P&L Simulator — {p['pair']}")
        render_pnl(p["pair"], p["direction"], p["entry_rate"],
                   float(closes.iloc[-1]), p["notional"])
    with t6:
        st.subheader("Current Spot Rates")
        with st.spinner("Fetching..."):
            rates = get_latest_rates()
        render_spot_table(rates)

if __name__ == "__main__":
    main()
