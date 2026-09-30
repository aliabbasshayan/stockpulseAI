import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from streamlit_lottie import st_lottie
except Exception:
    st_lottie = None

st.set_page_config(
    page_title="StockPulse AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Palette & Config ----------
BG, PANEL, LINE = "#080b12", "#10151f", "#1e2636"
TEXT, MUTED = "#e6ebf5", "#7d8aa5"
CYAN, UP, DOWN, AMBER, VIOLET = "#22d3ee", "#34d399", "#fb7185", "#fbbf24", "#a78bfa"
WATCHLIST = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "NFLX"]
FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA_5",
    "SMA_20",
    "SMA_50",
    "RSI",
    "Daily_Return",
    "MACD",
    "Volatility_10",
]

# ---------- CSS ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"], .stApp { font-family:'Space Grotesk',sans-serif; }
.stApp { background:
  radial-gradient(900px 500px at 85% -10%, rgba(34,211,238,.10), transparent 60%),
  radial-gradient(700px 500px at -10% 10%, rgba(167,139,250,.10), transparent 60%), #080b12; }
header[data-testid="stHeader"], #MainMenu, footer { visibility:hidden; height:0; }
.block-container { padding-top:1rem; max-width:1500px; }
section[data-testid="stSidebar"] { background:#0b0f18; border-right:1px solid #1e2636; }

.tape { overflow:hidden; border-bottom:1px solid #1e2636; background:#0b0f18;
  margin:0 -4rem 1.4rem -4rem; padding:10px 0; white-space:nowrap; }
.tape-inner { display:inline-flex; gap:42px; animation:scroll 40s linear infinite; }
.tape:hover .tape-inner { animation-play-state:paused; }
.tape span { font-family:'JetBrains Mono',monospace; font-size:.82rem; color:#7d8aa5; }
.tape b { color:#e6ebf5; margin-right:8px; font-weight:600; }
@keyframes scroll { from{transform:translateX(0)} to{transform:translateX(-50%)} }

.hero { display:flex; align-items:flex-end; justify-content:space-between; gap:24px;
  animation:rise .7s ease-out both; }
@keyframes rise { from{opacity:0; transform:translateY(18px)} to{opacity:1; transform:none} }
.hero .sym { font-size:.95rem; color:#7d8aa5; letter-spacing:.02em; display:flex; align-items:center; gap:10px; }
.dot { width:9px; height:9px; border-radius:50%; background:#34d399; animation:pulse 1.8s infinite; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(52,211,153,.6)} 100%{box-shadow:0 0 0 12px rgba(52,211,153,0)} }
.hero .price { font-size:4.6rem; font-weight:700; line-height:1; letter-spacing:-.04em; margin:6px 0 0 0; color:#e6ebf5; }
.hero .chg { font-family:'JetBrains Mono',monospace; font-size:1.1rem; margin-top:10px; }
.hero .name { font-size:1.5rem; font-weight:500; color:#e6ebf5; }

.kpi { background:linear-gradient(160deg,#121826,#0d121c); border:1px solid #1e2636; border-radius:10px;
  padding:14px 16px; height:100%; transition:border-color .25s, transform .25s; }
.kpi:hover { border-color:#22d3ee; transform:translateY(-3px); }
.kpi .l { color:#7d8aa5; font-size:.78rem; }
.kpi .v { font-family:'JetBrains Mono',monospace; font-size:1.35rem; font-weight:600; margin-top:4px; color:#e6ebf5; }
.kpi .s { font-size:.76rem; color:#7d8aa5; margin-top:2px; }

.forecast { border:1px solid #22d3ee55; border-radius:14px; padding:26px 28px;
  background:linear-gradient(135deg,rgba(34,211,238,.10),rgba(167,139,250,.08)); }
.forecast .l { color:#7d8aa5; font-size:.85rem; }
.forecast .big { font-size:3.4rem; font-weight:700; letter-spacing:-.03em; line-height:1.1; }
.verdict { display:inline-block; padding:6px 14px; border-radius:999px; font-weight:700; font-size:.9rem; }

.stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:1px solid #1e2636; }
.stTabs [data-baseweb="tab"] { background:transparent; color:#7d8aa5; padding:10px 18px; border-radius:8px 8px 0 0; }
.stTabs [aria-selected="true"] { color:#22d3ee !important; background:rgba(34,211,238,.08) !important; }
.stTabs [data-baseweb="tab-highlight"] { background:#22d3ee; }
h3, h4 { letter-spacing:-.01em; }
.note { color:#7d8aa5; font-size:.8rem; }
@media (prefers-reduced-motion: reduce) { * { animation:none !important; transition:none !important; } }
</style>
""",
    unsafe_allow_html=True,
)


# ---------- Helpers ----------
def human(n):
    for unit, d in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= d:
            return f"{n/d:.2f}{unit}"
    return f"{n:.0f}"


def tone(x):
    return UP if x >= 0 else DOWN


def kpi(label, value, sub="", color=TEXT):
    return (
        f'<div class="kpi"><div class="l">{label}</div>'
        f'<div class="v" style="color:{color}">{value}</div><div class="s">{sub}</div></div>'
    )


def layout(fig, h=520, **kw):
    fig.update_layout(
        template="plotly_dark",
        height=h,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(16,21,31,.6)",
        hovermode="x unified",
        font=dict(family="Space Grotesk", color=TEXT),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.04, x=1, xanchor="right"),
        **kw,
    )
    fig.update_xaxes(
        gridcolor=LINE, showspikes=True, spikecolor=MUTED, spikethickness=1
    )
    fig.update_yaxes(gridcolor=LINE)
    return fig


@st.cache_data(ttl=3600)
def load_lottie(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=900)
def load_raw(ticker, period):
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[cols].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


@st.cache_data(ttl=3600)
def company_name(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker


@st.cache_data(ttl=300)
def load_tape():
    try:
        px = yf.download(WATCHLIST, period="5d", progress=False)["Close"].dropna()
        if isinstance(px.columns, pd.MultiIndex):
            px.columns = px.columns.get_level_values(0)
        last, prev = px.iloc[-1], px.iloc[-2]
        return [
            (t, float(last[t]), float((last[t] / prev[t] - 1) * 100))
            for t in WATCHLIST
            if t in px.columns
        ]
    except Exception:
        return []


@st.cache_data(ttl=900)
def load_many(tickers, period):
    px = yf.download(
        list(tickers), period=period, auto_adjust=True, progress=False
    )["Close"]
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(0)
    return px.dropna(how="all")


def add_features(df):
    d = df.copy()
    c = d["Close"]
    d["SMA_5"], d["SMA_20"], d["SMA_50"] = (
        c.rolling(w).mean() for w in (5, 20, 50)
    )
    d["Daily_Return"] = c.pct_change()
    d["Volatility_10"] = d["Daily_Return"].rolling(10).std()

    delta = c.diff()
    up, dn = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.ewm(alpha=1 / 14, adjust=False).mean() / (
        dn.ewm(alpha=1 / 14, adjust=False).mean() + 1e-9
    )
    d["RSI"] = 100 - (100 / (1 + rs))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_Hist"] = d["MACD"] - d["MACD_Signal"]

    std = c.rolling(20).std()
    d["BB_Mid"], d["BB_Up"], d["BB_Low"] = (
        d["SMA_20"],
        d["SMA_20"] + 2 * std,
        d["SMA_20"] - 2 * std,
    )
    return d


def make_model(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def recursive_forecast(raw, model, steps):
    hist = raw.copy()
    rows = []
    for _ in range(steps):
        feats = add_features(hist)[FEATURES].iloc[[-1]]
        p = float(model.predict(feats)[0])
        nxt = hist.index[-1] + pd.offsets.BDay(1)
        hist.loc[nxt] = [p, p, p, p, hist["Volume"].tail(10).mean()]
        rows.append((nxt, p))
    return pd.DataFrame(rows, columns=["Date", "Predicted Close"]).set_index(
        "Date"
    )


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### Controls")
    pick = st.selectbox("Ticker", WATCHLIST)
    custom = st.text_input(
        "Or type any ticker", placeholder="e.g. AMD, BTC-USD, RELIANCE.NS"
    )
    ticker = custom.strip().upper() or pick
    period = st.select_slider(
        "History loaded", ["1y", "2y", "5y", "10y"], value="5y"
    )
    view = st.radio(
        "Chart window", ["1M", "3M", "6M", "1Y", "All"], index=3, horizontal=True
    )
    style = st.radio("Price style", ["Candles", "Line"], horizontal=True)
    overlays = st.multiselect(
        "Overlays",
        ["SMA 5", "SMA 20", "SMA 50", "Bollinger"],
        default=["SMA 20", "Bollinger"],
    )
    st.markdown("---")
    st.markdown("### Model")
    alpha = st.slider("Ridge alpha", 0.01, 50.0, 1.0)
    steps = st.slider("Forecast horizon (trading days)", 1, 10, 5)
    st.markdown("---")
    anim = load_lottie(
        "https://assets5.lottiefiles.com/packages/lf20_vnikbew6.json"
    )
    if anim and st_lottie:
        st_lottie(anim, height=120, key="anim")

# ---------- Ticker Tape ----------
tape = load_tape()
if tape:
    items = "".join(
        f'<span><b>{t}</b>${p:,.2f} <span style="color:{tone(c)}">{c:+.2f}%</span></span>'
        for t, p, c in tape
    )
    st.markdown(
        f'<div class="tape"><div class="tape-inner">{items}{items}</div></div>',
        unsafe_allow_html=True,
    )

# ---------- Load Data ----------
raw = load_raw(ticker, period)
if raw.empty or len(raw) < 120:
    st.error(
        f"Not enough data for '{ticker}'. Check the symbol and try a longer history."
    )
    st.stop()

df = add_features(raw)
last, prev = df.iloc[-1], df.iloc[-2]
chg = float(last["Close"] - prev["Close"])
pct = float((chg / prev["Close"]) * 100)

# ---------- Model ----------
full = df.copy()
full["Target"] = full["Close"].shift(-1)
train_df = full.dropna(subset=FEATURES + ["Target"])
split = int(len(train_df) * 0.8)
Xtr, Xte = train_df[FEATURES].iloc[:split], train_df[FEATURES].iloc[split:]
ytr, yte = train_df["Target"].iloc[:split], train_df["Target"].iloc[split:]

eval_model = make_model(alpha).fit(Xtr, ytr)
pred_te = eval_model.predict(Xte)
mae = float(mean_absolute_error(yte, pred_te))
rmse = float(np.sqrt(mean_squared_error(yte, pred_te)))
r2 = float(r2_score(yte, pred_te))
naive_mae = float(mean_absolute_error(yte, Xte["Close"]))
dir_acc = float(
    np.mean(np.sign(pred_te - Xte["Close"]) == np.sign(yte - Xte["Close"]))
    * 100
)

final_model = make_model(alpha).fit(train_df[FEATURES], train_df["Target"])
fc = recursive_forecast(raw, final_model, steps)
next_price = float(fc["Predicted Close"].iloc[0])
exp_diff = float(next_price - last["Close"])
exp_pct = float((exp_diff / last["Close"]) * 100)

# ---------- Hero Header ----------
name = company_name(ticker)
st.markdown(
    f"""
<div class="hero">
  <div>
    <div class="sym"><span class="dot"></span>{ticker} · last close {df.index[-1]:%d %b %Y}</div>
    <div class="name">{name}</div>
    <div class="price">${last['Close']:,.2f}</div>
    <div class="chg" style="color:{tone(chg)}">{chg:+.2f} ({pct:+.2f}%) today</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

hi52, lo52 = (
    float(df["High"].tail(252).max()),
    float(df["Low"].tail(252).min()),
)
k = st.columns(6)
k[0].markdown(kpi("Open", f"${last['Open']:.2f}"), unsafe_allow_html=True)
k[1].markdown(
    kpi("Day high", f"${last['High']:.2f}", color=UP), unsafe_allow_html=True
)
k[2].markdown(
    kpi("Day low", f"${last['Low']:.2f}", color=DOWN), unsafe_allow_html=True
)
k[3].markdown(
    kpi(
        "Volume",
        human(last["Volume"]),
        f"avg 20d {human(df['Volume'].tail(20).mean())}",
    ),
    unsafe_allow_html=True,
)
k[4].markdown(
    kpi(
        "52w high",
        f"${hi52:.2f}",
        f"{(last['Close']/hi52-1)*100:+.1f}% away",
    ),
    unsafe_allow_html=True,
)
k[5].markdown(
    kpi(
        "52w low", f"${lo52:.2f}", f"{(last['Close']/lo52-1)*100:+.1f}% above"
    ),
    unsafe_allow_html=True,
)
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

days = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "All": len(df)}[view]
v = df.tail(days)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Market", "Indicators", "AI Forecast", "Compare"]
)

# ---------- Tab 1: Market ----------
with tab1:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
    )
    if "Bollinger" in overlays:
        fig.add_trace(
            go.Scatter(
                x=v.index,
                y=v["BB_Up"],
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=v.index,
                y=v["BB_Low"],
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(167,139,250,.10)",
                name="Bollinger (20, 2σ)",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
    if style == "Candles":
        fig.add_trace(
            go.Candlestick(
                x=v.index,
                open=v["Open"],
                high=v["High"],
                low=v["Low"],
                close=v["Close"],
                name="Price",
                increasing_line_color=UP,
                decreasing_line_color=DOWN,
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=v.index,
                y=v["Close"],
                name="Close",
                line=dict(color=CYAN, width=2.4),
                fill="tozeroy",
                fillcolor="rgba(34,211,238,.06)",
            ),
            row=1,
            col=1,
        )
        fig.update_yaxes(
            range=[v["Close"].min() * 0.97, v["Close"].max() * 1.02],
            row=1,
            col=1,
        )
    for label, col_name, color in (
        ("SMA 5", "SMA_5", AMBER),
        ("SMA 20", "SMA_20", VIOLET),
        ("SMA 50", "SMA_50", "#f97316"),
    ):
        if label in overlays:
            fig.add_trace(
                go.Scatter(
                    x=v.index,
                    y=v[col_name],
                    name=label,
                    line=dict(color=color, width=1.5),
                ),
                row=1,
                col=1,
            )
    vol_colors = np.where(v["Close"] >= v["Open"], UP, DOWN)
    fig.add_trace(
        go.Bar(
            x=v.index,
            y=v["Volume"],
            marker_color=vol_colors,
            opacity=0.55,
            name="Volume",
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    st.plotly_chart(layout(fig, 600), use_container_width=True)

# ---------- Tab 2: Indicators ----------
with tab2:
    a, b = st.columns([2, 1])
    with a:
        f2 = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.5, 0.5],
            subplot_titles=("RSI (14)", "MACD (12, 26, 9)"),
        )
        f2.add_trace(
            go.Scatter(
                x=v.index,
                y=v["RSI"],
                name="RSI",
                line=dict(color=CYAN, width=2),
            ),
            row=1,
            col=1,
        )
        f2.add_hrect(
            y0=70,
            y1=100,
            fillcolor=DOWN,
            opacity=0.08,
            line_width=0,
            row=1,
            col=1,
        )
        f2.add_hrect(
            y0=0, y1=30, fillcolor=UP, opacity=0.08, line_width=0, row=1, col=1
        )
        f2.add_trace(
            go.Bar(
                x=v.index,
                y=v["MACD_Hist"],
                name="Histogram",
                marker_color=np.where(v["MACD_Hist"] >= 0, UP, DOWN),
                opacity=0.6,
            ),
            row=2,
            col=1,
        )
        f2.add_trace(
            go.Scatter(
                x=v.index,
                y=v["MACD"],
                name="MACD",
                line=dict(color=AMBER, width=1.6),
            ),
            row=2,
            col=1,
        )
        f2.add_trace(
            go.Scatter(
                x=v.index,
                y=v["MACD_Signal"],
                name="Signal",
                line=dict(color=VIOLET, width=1.6),
            ),
            row=2,
            col=1,
        )
        f2.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        st.plotly_chart(layout(f2, 560), use_container_width=True)
    with b:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(last["RSI"]),
                number=dict(font=dict(size=46, family="JetBrains Mono")),
                title=dict(text="RSI now"),
                gauge=dict(
                    axis=dict(range=[0, 100]),
                    bar=dict(color=CYAN, thickness=0.25),
                    bgcolor=PANEL,
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 30], color="rgba(52,211,153,.25)"),
                        dict(range=[30, 70], color="rgba(125,138,165,.10)"),
                        dict(range=[70, 100], color="rgba(251,113,133,.25)"),
                    ],
                ),
            )
        )
        gauge.update_layout(
            height=260,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT),
            margin=dict(l=20, r=20, t=50, b=0),
        )
        st.plotly_chart(gauge, use_container_width=True)

        signals = [
            ("RSI", 1 if last["RSI"] < 30 else -1 if last["RSI"] > 70 else 0),
            ("MACD vs signal", 1 if last["MACD"] > last["MACD_Signal"] else -1),
            ("Price vs SMA 20", 1 if last["Close"] > last["SMA_20"] else -1),
            ("SMA 5 vs SMA 20", 1 if last["SMA_5"] > last["SMA_20"] else -1),
            ("ML forecast", 1 if exp_diff > 0 else -1),
        ]
        score = sum(s for _, s in signals)
        verdict, vcol = (
            ("Bullish", UP)
            if score >= 2
            else ("Bearish", DOWN)
            if score <= -2
            else ("Neutral", AMBER)
        )
        st.markdown(
            f'<div class="verdict" style="background:{vcol}22;color:{vcol};border:1px solid {vcol}66">'
            f"{verdict} · score {score:+d}/5</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        for n, s in signals:
            c = UP if s > 0 else DOWN if s < 0 else MUTED
            t = "Bullish" if s > 0 else "Bearish" if s < 0 else "Neutral"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:5px 0;"
                f"border-bottom:1px solid {LINE}'><span style='color:{MUTED}'>{n}</span>"
                f"<span style='color:{c};font-family:JetBrains Mono'>{t}</span></div>",
                unsafe_allow_html=True,
            )

# ---------- Tab 3: Forecast ----------
with tab3:
    col = UP if exp_diff > 0 else DOWN
    arrow = "▲" if exp_diff > 0 else "▼"
    l, r = st.columns([1, 1.6])
    with l:
        st.markdown(
            f"""
        <div class="forecast">
          <div class="l">Predicted next close · {fc.index[0]:%a %d %b}</div>
          <div class="big" style="color:{CYAN}">${next_price:,.2f}</div>
          <div style="color:{col};font-family:'JetBrains Mono';font-size:1.15rem;margin-top:6px">
            {arrow} {exp_diff:+.2f} ({exp_pct:+.2f}%) vs ${last['Close']:,.2f}</div>
          <div class="note" style="margin-top:14px">Typical error on held-out data: ±${mae:.2f}</div>
        </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        m = st.columns(2)
        m[0].markdown(
            kpi("MAE", f"${mae:.2f}", f"naive baseline ${naive_mae:.2f}"),
            unsafe_allow_html=True,
        )
        m[1].markdown(kpi("RMSE", f"${rmse:.2f}"), unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        m = st.columns(2)
        m[0].markdown(kpi("R²", f"{r2:.4f}"), unsafe_allow_html=True)
        m[1].markdown(
            kpi(
                "Direction hit-rate",
                f"{dir_acc:.1f}%",
                "up/down called right",
            ),
            unsafe_allow_html=True,
        )
    with r:
        ctx = df["Close"].tail(60)
        f3 = go.Figure()
        f3.add_trace(
            go.Scatter(
                x=ctx.index, y=ctx, name="Close", line=dict(color=CYAN, width=2.4)
            )
        )
        xs = [ctx.index[-1]] + list(fc.index)
        ys = [float(ctx.iloc[-1])] + list(fc["Predicted Close"])
        band = rmse * np.sqrt(np.arange(0, len(ys)) + 0.0)
        f3.add_trace(
            go.Scatter(
                x=xs,
                y=np.array(ys) + band,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        f3.add_trace(
            go.Scatter(
                x=xs,
                y=np.array(ys) - band,
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(251,191,36,.14)",
                name="±RMSE band",
                hoverinfo="skip",
            )
        )
        f3.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                name="Forecast",
                mode="lines+markers",
                line=dict(color=AMBER, width=2.4, dash="dot"),
                marker=dict(size=7),
            )
        )
        f3.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        st.plotly_chart(layout(f3, 440), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Backtest: predicted vs actual (test set)")
        f4 = go.Figure()
        f4.add_trace(
            go.Scatter(
                x=yte.index,
                y=yte,
                name="Actual",
                line=dict(color=CYAN, width=2),
            )
        )
        f4.add_trace(
            go.Scatter(
                x=yte.index,
                y=pred_te,
                name="Predicted",
                line=dict(color=AMBER, width=1.6),
            )
        )
        st.plotly_chart(layout(f4, 340), use_container_width=True)
    with c2:
        st.markdown("##### What drives the model")
        coef = pd.Series(
            final_model.named_steps["ridge"].coef_, index=FEATURES
        ).sort_values()
        f5 = go.Figure(
            go.Bar(
                x=coef.values,
                y=coef.index,
                orientation="h",
                marker_color=np.where(coef.values >= 0, UP, DOWN),
            )
        )
        st.plotly_chart(layout(f5, 340), use_container_width=True)

    st.dataframe(fc.style.format("${:,.2f}"), use_container_width=True)
    st.download_button(
        "Download forecast CSV",
        fc.to_csv().encode(),
        f"{ticker}_forecast.csv",
        "text/csv",
    )
    st.warning(
        "Experimental model for learning only. Forecasts 2+ days ahead feed on the model's own "
        "predictions and get less reliable each step. This is not financial advice."
    )

# ---------- Tab 4: Compare ----------
with tab4:
    chosen = st.multiselect(
        "Tickers to compare",
        WATCHLIST,
        default=["AAPL", "MSFT", "NVDA", "TSLA"],
    )
    if len(chosen) >= 2:
        px = load_many(tuple(chosen), period).dropna().tail(days)
        norm = px / px.iloc[0] * 100
        f6 = go.Figure()
        for t in norm.columns:
            f6.add_trace(
                go.Scatter(
                    x=norm.index, y=norm[t], name=str(t), line=dict(width=2.2)
                )
            )
        f6.update_yaxes(title="Indexed to 100")
        f6.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        st.plotly_chart(layout(f6, 440), use_container_width=True)

        corr = px.pct_change().dropna().corr()
        f7 = go.Figure(
            go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                zmin=-1,
                zmax=1,
                colorscale=[[0, DOWN], [0.5, PANEL], [1, CYAN]],
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
            )
        )
        st.markdown("##### Daily-return correlation")
        st.plotly_chart(layout(f7, 380), use_container_width=True)
    else:
        st.info("Pick at least two tickers to compare.")