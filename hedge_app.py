"""
Cross-Asset Hedge — Liquidity-Adjusted Black-Scholes
Streamlit application
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")  # non-interactive backend required for Streamlit
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cross-Asset Hedge Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
CYAN    = "#00b4d8"
PURPLE  = "#7c3aed"
GREEN   = "#3fb950"
RED     = "#f85149"
GOLD    = "#d29922"
NAVY    = "#0d1117"
CARD    = "#161b22"
BORDER  = "#30363d"
TEXT    = "#e6edf3"
MUTED   = "#8b949e"

CLASS_COLORS = {
    "Equity_SmallCap": "#00b4d8",
    "Real_Estate":     "#f85149",
    "EM_Debt":         "#d29922",
    "Commodities":     "#3fb950",
    "HY_Credit":       "#a855f7",
    "Hedge":           "#8b949e",
}

TICKER_NAMES = {
    # Equity Small-Cap
    "IWM":  "iShares Russell 2000 ETF",
    "VIOV": "Vanguard S&P Small-Cap 600 Value ETF",
    "SLYV": "SPDR S&P 600 Small Cap Value ETF",
    # Real Estate
    "VNQ":  "Vanguard Real Estate ETF",
    "IYR":  "iShares US Real Estate ETF",
    "REM":  "iShares Mortgage Real Estate ETF",
    # EM Debt
    "EMB":  "iShares JP Morgan USD EM Bond ETF",
    "EMLC": "VanEck Local Currency EM Bond ETF",
    "VWOB": "Vanguard Emerging Markets Govt Bond ETF",
    # Commodities
    "SLV":  "iShares Silver Trust",
    "PDBC": "Invesco Optimum Yield Diversified Commodity ETF",
    "DJP":  "iPath Bloomberg Commodity Index ETN",
    # High Yield Credit
    "HYG":  "iShares iBoxx $ High Yield Corporate Bond ETF",
    "JNK":  "SPDR Bloomberg High Yield Bond ETF",
    "USHY": "iShares Broad USD High Yield Corporate Bond ETF",
    # Hedge instruments
    "SPY":  "SPDR S&P 500 ETF Trust",
    "QQQ":  "Invesco Nasdaq-100 ETF",
    "TLT":  "iShares 20+ Year Treasury Bond ETF",
    "GLD":  "SPDR Gold Shares ETF",
    "UUP":  "Invesco DB US Dollar Index Bullish Fund",
    "LQD":  "iShares iBoxx $ Investment Grade Corporate Bond ETF",
}

# ── Plotly base template ──────────────────────────────────────────────────────
def chart_layout(**kwargs):
    """Return a consistent dark Plotly layout dict."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,27,34,0.6)",
        font=dict(color=MUTED, family="Inter, 'Segoe UI', sans-serif", size=12),
        title_font=dict(color=TEXT, size=14, family="Inter, 'Segoe UI', sans-serif"),
        legend=dict(
            bgcolor="rgba(22,27,34,0.8)",
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(color=TEXT, size=11),
        ),
        xaxis=dict(
            gridcolor="rgba(48,54,61,0.6)",
            linecolor=BORDER,
            tickcolor=MUTED,
            tickfont=dict(color=MUTED),
            title_font=dict(color=MUTED),
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(48,54,61,0.6)",
            linecolor=BORDER,
            tickcolor=MUTED,
            tickfont=dict(color=MUTED),
            title_font=dict(color=MUTED),
            zeroline=False,
        ),
        margin=dict(l=48, r=24, t=52, b=48),
        hoverlabel=dict(
            bgcolor=CARD,
            bordercolor=BORDER,
            font=dict(color=TEXT, size=12),
        ),
    )
    base.update(kwargs)
    return base


# ── Global CSS injection ──────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Import font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {{
    font-family: 'Inter', 'Segoe UI', sans-serif;
}}

/* ── Hide default header/footer ── */
#MainMenu, footer, header {{ visibility: hidden; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {NAVY}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {MUTED}; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: #0d1117 !important;
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] .stMarkdown h3 {{
    color: {MUTED};
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 20px 0 6px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid {BORDER};
}}
[data-testid="stSidebar"] label {{
    color: {TEXT} !important;
    font-size: 0.82rem !important;
}}

/* ── Tabs ── */
[data-baseweb="tab-list"] {{
    gap: 2px;
    background: {CARD};
    padding: 4px;
    border-radius: 10px;
    border: 1px solid {BORDER};
}}
[data-baseweb="tab"] {{
    border-radius: 8px !important;
    color: {MUTED} !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
    transition: all 0.2s;
}}
[aria-selected="true"][data-baseweb="tab"] {{
    background: linear-gradient(135deg, rgba(0,180,216,0.18) 0%, rgba(124,58,237,0.18) 100%) !important;
    color: {CYAN} !important;
    border-bottom: 2px solid {CYAN} !important;
}}

/* ── KPI card ── */
.kpi-card {{
    background: linear-gradient(135deg, rgba(0,180,216,0.06) 0%, rgba(124,58,237,0.06) 100%);
    border: 1px solid rgba(0,180,216,0.2);
    border-radius: 12px;
    padding: 18px 22px;
    position: relative;
    overflow: hidden;
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {CYAN}, {PURPLE});
}}
.kpi-label {{
    color: {MUTED};
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}}
.kpi-value {{
    color: {TEXT};
    font-size: 1.55rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    line-height: 1.1;
}}
.kpi-delta {{
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 4px;
}}
.kpi-delta.pos {{ color: {GREEN}; }}
.kpi-delta.neg {{ color: {RED}; }}

/* ── Section header ── */
.section-header {{
    color: {TEXT};
    font-size: 1rem;
    font-weight: 600;
    padding: 0 0 10px 0;
    margin-bottom: 16px;
    border-bottom: 1px solid {BORDER};
    display: flex;
    align-items: center;
    gap: 8px;
}}

/* ── Formula box ── */
.formula-box {{
    background: rgba(22,27,34,0.9);
    border: 1px solid {BORDER};
    border-left: 3px solid {CYAN};
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.82rem;
    color: {CYAN};
    line-height: 1.7;
    margin: 8px 0;
}}

/* ── Feature card (landing) ── */
.feature-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px;
    height: 100%;
    transition: border-color 0.2s;
}}
.feature-card:hover {{ border-color: {CYAN}; }}
.feature-icon {{
    font-size: 1.6rem;
    margin-bottom: 10px;
}}
.feature-title {{
    color: {TEXT};
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 6px;
}}
.feature-desc {{
    color: {MUTED};
    font-size: 0.78rem;
    line-height: 1.5;
}}

/* ── Status badge ── */
.badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}}
.badge-cyan  {{ background: rgba(0,180,216,0.15);  color: {CYAN};  border: 1px solid rgba(0,180,216,0.3); }}
.badge-green {{ background: rgba(63,185,80,0.15);  color: {GREEN}; border: 1px solid rgba(63,185,80,0.3); }}
.badge-red   {{ background: rgba(248,81,73,0.15);  color: {RED};   border: 1px solid rgba(248,81,73,0.3); }}
.badge-gold  {{ background: rgba(210,153,34,0.15); color: {GOLD};  border: 1px solid rgba(210,153,34,0.3); }}

/* ── Dataframe overrides ── */
[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}

/* ── Divider ── */
.divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, {BORDER}, transparent);
    margin: 24px 0;
}}

/* ── Hero gradient text ── */
.hero-title {{
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e6edf3 0%, {CYAN} 50%, {PURPLE} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 6px;
}}
.hero-sub {{
    color: {MUTED};
    font-size: 0.88rem;
    font-weight: 400;
    letter-spacing: 0.03em;
}}

/* ── Streamlit default metric override ── */
[data-testid="metric-container"] {{
    background: linear-gradient(135deg, rgba(0,180,216,0.06) 0%, rgba(124,58,237,0.06) 100%);
    border: 1px solid rgba(0,180,216,0.18);
    border-radius: 10px;
    padding: 14px 18px;
}}
[data-testid="stMetricLabel"] {{ color: {MUTED} !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.1em; }}
[data-testid="stMetricValue"] {{ color: {TEXT} !important; font-weight: 700 !important; }}
[data-testid="stMetricDelta"] svg {{ display: none; }}

/* ── Run button ── */
[data-testid="stSidebar"] .stButton button {{
    background: linear-gradient(135deg, {CYAN} 0%, {PURPLE} 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 10px;
    letter-spacing: 0.04em;
    transition: opacity 0.2s;
}}
[data-testid="stSidebar"] .stButton button:hover {{ opacity: 0.88; }}

/* ── Expander ── */
[data-testid="stExpander"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    background: {CARD} !important;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def kpi(label: str, value: str, delta: str = None, delta_pos: bool = None):
    delta_html = ""
    if delta is not None:
        cls = "pos" if delta_pos else "neg"
        arrow = "▲" if delta_pos else "▼"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {delta}</div>'
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)


def section(icon: str, title: str):
    st.markdown(f'<div class="section-header"><span>{icon}</span><span>{title}</span></div>',
                unsafe_allow_html=True)


def badge(text: str, color: str = "cyan"):
    st.markdown(f'<span class="badge badge-{color}">{text}</span>', unsafe_allow_html=True)


def divider():
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CORE MODELS
# ─────────────────────────────────────────────────────────────────────────────

def amihud_ratio(ret, price, vol):
    dv = (price * vol).replace(0, np.nan)
    return (ret.abs() / dv).mean() * 1e6


def spread_proxy(hi, lo):
    return float(np.log(hi / lo).mean() * 100)


class LiquidityAdjustedBS:
    """Black-Scholes with liquidity premium: r_adj = rf + κ·amihud^α"""

    def __init__(self, S, K, T, sigma, amihud, rf=0.053,
                 kappa=0.02, alpha=0.5, option_type="call"):
        self.S, self.K, self.T = float(S), float(K), float(T)
        self.sigma = float(sigma)
        self.option_type = option_type.lower()
        self.lam = kappa * (max(float(amihud), 0) ** alpha)
        self.r_adj = rf + self.lam
        eps = 1e-10
        self._d1 = ((np.log(S / K + eps) + (self.r_adj + 0.5 * sigma ** 2) * T)
                    / (sigma * np.sqrt(T)))
        self._d2 = self._d1 - sigma * np.sqrt(T)

    def price(self):
        dS = self.S * np.exp(-self.lam * self.T)
        dK = self.K * np.exp(-self.r_adj * self.T)
        return (dS * norm.cdf(self._d1) - dK * norm.cdf(self._d2)
                if self.option_type == "call"
                else dK * norm.cdf(-self._d2) - dS * norm.cdf(-self._d1))

    def delta(self):
        f = np.exp(-self.lam * self.T)
        return f * norm.cdf(self._d1) if self.option_type == "call" \
            else f * (norm.cdf(self._d1) - 1)

    def gamma(self):
        return (np.exp(-self.lam * self.T) * norm.pdf(self._d1)
                / (self.S * self.sigma * np.sqrt(self.T)))

    def vega(self):
        return (self.S * np.exp(-self.lam * self.T)
                * norm.pdf(self._d1) * np.sqrt(self.T) / 100)

    def theta(self):
        d1, d2 = self._d1, self._d2
        t1 = -(self.S * np.exp(-self.lam * self.T) * norm.pdf(d1) * self.sigma
               / (2 * np.sqrt(self.T)))
        if self.option_type == "call":
            t2 = (self.lam * self.S * np.exp(-self.lam * self.T) * norm.cdf(d1)
                  - self.r_adj * self.K * np.exp(-self.r_adj * self.T) * norm.cdf(d2))
        else:
            t2 = (self.lam * self.S * np.exp(-self.lam * self.T) * norm.cdf(-d1)
                  + self.r_adj * self.K * np.exp(-self.r_adj * self.T) * norm.cdf(-d2))
        return (t1 + t2) / 252

    def rho(self):
        v = self.K * self.T * np.exp(-self.r_adj * self.T) / 100
        return v * norm.cdf(self._d2) if self.option_type == "call" \
            else -v * norm.cdf(-self._d2)

    def all_greeks(self):
        return dict(price=self.price(), delta=self.delta(), gamma=self.gamma(),
                    vega=self.vega(), theta=self.theta(), rho=self.rho(),
                    lam=self.lam, r_adj=self.r_adj)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LAYER
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(tickers: tuple, period: str) -> dict:
    raw = yf.download(list(tickers), period=period, interval="1d",
                      auto_adjust=True, progress=False)
    if raw.empty:
        return {}
    close   = raw["Close"].dropna(how="all")
    volume  = raw["Volume"].dropna(how="all")
    high    = raw["High"].dropna(how="all")
    low     = raw["Low"].dropna(how="all")
    returns = close.pct_change().dropna()
    return dict(close=close, volume=volume, high=high, low=low, returns=returns)


def compute_illiquidity(data, tickers):
    rows = []
    for t in tickers:
        if t not in data["close"].columns:
            continue
        r  = data["returns"][t].dropna()
        p  = data["close"][t].reindex(r.index).dropna()
        if len(p) < 2 or len(r) < 2:
            continue
        v  = data["volume"][t].reindex(p.index) if t in data["volume"].columns else pd.Series(np.nan, index=p.index)
        h  = data["high"][t].reindex(p.index)   if t in data["high"].columns   else p
        lo = data["low"][t].reindex(p.index)    if t in data["low"].columns    else p
        r  = r.reindex(p.index)
        rows.append(dict(Ticker=t, Amihud=amihud_ratio(r, p, v),
                         SpreadProxy=spread_proxy(h, lo),
                         AnnVol=r.std() * np.sqrt(252) * 100,
                         AvgPrice=float(p.mean()), LastPrice=float(p.iloc[-1])))
    return pd.DataFrame(rows).set_index("Ticker") if rows else pd.DataFrame()


def compute_portfolio(illiq_df, illiq_classes, T, put_pct, rf, kappa, alpha):
    rows = []
    for ticker, row in illiq_df.iterrows():
        S, sigma, amihud = row["LastPrice"], row["AnnVol"] / 100, row["Amihud"]
        cls = next((c for c, tks in illiq_classes.items() if ticker in tks), "Unknown")
        call = LiquidityAdjustedBS(S, S, T, sigma, amihud, rf, kappa, alpha, "call")
        put  = LiquidityAdjustedBS(S, S * put_pct, T, sigma, amihud, rf, kappa, alpha, "put")
        bs0  = LiquidityAdjustedBS(S, S, T, sigma, 0, rf, kappa, alpha, "call")
        cg, pg = call.all_greeks(), put.all_greeks()
        rows.append(dict(
            Ticker=ticker, Class=cls, S=round(S, 2), Sigma=round(sigma, 4),
            Amihud=round(amihud, 4), Lambda_pct=round(call.lam * 100, 4),
            r_adj_pct=round(call.r_adj * 100, 3),
            CallPrice=round(cg["price"], 4), CallBS_Std=round(bs0.price(), 4),
            LiqImpact_pct=round((cg["price"] - bs0.price()) / bs0.price() * 100, 3) if bs0.price() > 0 else 0,
            CallDelta=round(cg["delta"], 4), CallGamma=round(cg["gamma"], 6),
            CallVega=round(cg["vega"], 4), CallTheta=round(cg["theta"], 4),
            CallRho=round(cg["rho"], 4),
            PutPrice=round(pg["price"], 4), PutDelta=round(pg["delta"], 4),
            PutGamma=round(pg["gamma"], 6),
        ))
    return pd.DataFrame(rows).set_index("Ticker") if rows else pd.DataFrame()


def run_hedge_optimization(portfolio_df, data, hedge_tickers):
    illiq_t = [t for t in portfolio_df.index if t in data["returns"].columns]
    hedge_t = [t for t in hedge_tickers       if t in data["returns"].columns]
    if not illiq_t or not hedge_t:
        return {}
    hedge_ret = data["returns"][hedge_t].dropna()
    illiq_ret = data["returns"][illiq_t].dropna()
    idx = hedge_ret.index.intersection(illiq_ret.index)
    hedge_ret = hedge_ret.loc[idx]
    illiq_ret = illiq_ret.loc[idx]
    delta_w = {t: float(portfolio_df.loc[t, "CallDelta"])
               for t in illiq_t if t in portfolio_df.index}
    n = len(delta_w)
    if n == 0:
        return {}
    ptf_pnl = sum(delta_w[t] * illiq_ret[t] for t in delta_w) / n
    Sigma_h = hedge_ret.cov().values
    cov_ph  = np.array([ptf_pnl.cov(hedge_ret[t]) for t in hedge_t])
    var_p   = float(ptf_pnl.var())
    res = minimize(
        lambda w: var_p + w @ Sigma_h @ w + 2 * w @ cov_ph,
        x0=-np.linalg.solve(Sigma_h + 1e-8 * np.eye(len(hedge_t)), cov_ph),
        method="SLSQP", options={"maxiter": 2000, "ftol": 1e-14},
    )
    w_opt = pd.Series(res.x, index=hedge_t)
    hedge_pnl  = hedge_ret @ w_opt.values
    hedged_pnl = ptf_pnl + hedge_pnl
    return dict(ptf_pnl=ptf_pnl, hedge_pnl=hedge_pnl, hedged_pnl=hedged_pnl,
                weights=w_opt, var_p=var_p, var_hedged=float(res.fun),
                hedge_ratio=1 - float(res.fun) / var_p,
                corr=pd.Series({t: ptf_pnl.corr(hedge_ret[t]) for t in hedge_t}))


def perf_stats(r, rf_daily=0.053 / 252):
    excess = r - rf_daily
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    return dict(
        Sharpe=round(excess.mean() / excess.std() * np.sqrt(252), 3),
        Vol_ann=round(r.std() * np.sqrt(252) * 100, 3),
        MaxDD=round(((cum - peak) / peak).min() * 100, 3),
        VaR95=round(np.percentile(r, 5) * 100, 3),
        Skew=round(float(r.skew()), 3),
        Kurt=round(float(r.kurt()), 3),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 10px 0 20px 0; border-bottom: 1px solid {BORDER}; margin-bottom: 8px;">
        <div style="font-size:1.15rem; font-weight:700; color:{TEXT}; letter-spacing:-0.01em;">
            ⚡ Hedge Lab
        </div>
        <div style="font-size:0.7rem; color:{MUTED}; margin-top:2px;">
            Liquidity-Adjusted Black-Scholes
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Data")
    period = st.selectbox("History period", ["1y", "2y", "3y", "5y"], index=1, label_visibility="collapsed")

    st.markdown("### Asset Universe")
    default_illiquid = {
        "Equity_SmallCap": "IWM, VIOV, SLYV",
        "Real_Estate":     "VNQ, IYR, REM",
        "EM_Debt":         "EMB, EMLC, VWOB",
        "Commodities":     "SLV, PDBC, DJP",
        "HY_Credit":       "HYG, JNK, USHY",
    }
    illiq_classes = {}
    with st.expander("Edit asset classes"):
        for cls, default in default_illiquid.items():
            raw = st.text_input(f"● {cls}", value=default, key=f"cls_{cls}")
            tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
            if tickers:
                illiq_classes[cls] = tickers
                names_preview = "  ·  ".join(
                    f"`{t}` {TICKER_NAMES.get(t, '')}" for t in tickers
                )
                st.markdown(
                    f"<div style='font-size:0.68rem; color:{MUTED}; margin:-8px 0 8px 8px; line-height:1.6;'>"
                    + "  ·  ".join(
                        f"<span style='color:{TEXT}'>{t}</span> "
                        f"<span style='color:{MUTED}'>{TICKER_NAMES.get(t, '')}</span>"
                        for t in tickers
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("### Hedge Instruments")
    hedge_raw = st.text_input("Tickers", value="SPY, QQQ, TLT, GLD, UUP, LQD",
                               label_visibility="collapsed")
    hedge_tickers = [t.strip().upper() for t in hedge_raw.split(",") if t.strip()]

    st.markdown("### Option Parameters")
    c1, c2 = st.columns(2)
    T_months = c1.slider("Maturity (m)", 1, 12, 3)
    put_pct  = c2.slider("Put strike %", 75, 99, 90) / 100
    T = T_months / 12

    st.markdown("### Model Parameters")
    rf    = st.number_input("RF (risk-free)", 0.0, 0.15, 0.053, 0.001, format="%.3f")
    kappa = st.number_input("κ (illiq. sensitivity)", 0.0, 0.20, 0.02, 0.001, format="%.3f")
    alpha = st.slider("α (Amihud exponent)", 0.1, 2.0, 0.5, 0.05)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=True)

    st.markdown(f"""
    <div style="margin-top:24px; padding-top:16px; border-top:1px solid {BORDER};
                font-size:0.68rem; color:{MUTED}; line-height:1.6;">
        Model: Amihud (2002) illiquidity<br>
        Hedge: Min-variance optimization<br>
        Data: Yahoo Finance via yfinance
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex; align-items:flex-end; justify-content:space-between;
            padding-bottom:16px; border-bottom:1px solid {BORDER}; margin-bottom:24px;">
    <div>
        <div class="hero-title">Cross-Asset Hedge Lab</div>
        <div class="hero-sub">
            Liquidity-Adjusted Black-Scholes &nbsp;·&nbsp;
            Amihud Illiquidity &nbsp;·&nbsp;
            Minimum-Variance Optimization
        </div>
    </div>
    <div style="text-align:right; font-size:0.72rem; color:{MUTED};">
        <span class="badge badge-cyan">BETA</span>&nbsp;
        <span class="badge badge-gold">Research</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LANDING PAGE
# ─────────────────────────────────────────────────────────────────────────────
if not run_btn:
    fa, fb, fc, fd = st.columns(4)
    cards = [
        ("🔍", "Illiquidity Scoring", "Amihud (2002) ratio + High-Low spread proxy ranked across all asset classes."),
        ("⚖️", "Liquidity-Adj. B-S", "Premium λ = κ·ILLIQ^α shifts the discount rate, lowering the present value of illiquid underlyings."),
        ("📐", "Full Greek Surface", "Δ, Γ, ν, Θ, ρ all adjusted for the liquidity cost of carry — computable for any moneyness."),
        ("🛡️", "Min-Var Hedge", "Closed-form w* = −Σ⁻¹·cov(r_p, r_h) across liquid ETFs minimises portfolio P&L variance."),
    ]
    for col, (icon, title, desc) in zip([fa, fb, fc, fd], cards):
        col.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    divider()

    col_f, col_e = st.columns([1, 1])
    with col_f:
        st.markdown(f'<div class="section-header">📌 Pricing Formula</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:rgba(22,27,34,0.9); border:1px solid {BORDER};
                    border-radius:10px; overflow:hidden; font-family:'JetBrains Mono','Fira Code',monospace;">

          <!-- Group 1: Liquidity Premium -->
          <div style="padding:10px 16px 6px 16px; background:rgba(0,180,216,0.06);
                      border-bottom:1px solid {BORDER};">
            <div style="font-size:0.62rem; font-weight:700; text-transform:uppercase;
                        letter-spacing:0.12em; color:{MUTED}; margin-bottom:8px;">
              1 — Liquidity Premium
            </div>
            <table style="width:100%; border-collapse:collapse;">
              <tr style="line-height:2.1;">
                <td style="color:{CYAN}; font-size:0.88rem; white-space:nowrap; padding-right:8px;">λ<sub>i</sub></td>
                <td style="color:{MUTED}; font-size:0.88rem; padding:0 8px;">=</td>
                <td style="color:{TEXT}; font-size:0.88rem;">
                  κ &nbsp;·&nbsp; Amihud<sub>i</sub><sup>α</sup>
                </td>
              </tr>
              <tr style="line-height:2.1;">
                <td style="color:{CYAN}; font-size:0.88rem; white-space:nowrap;">r<sub>adj</sub></td>
                <td style="color:{MUTED}; font-size:0.88rem; padding:0 8px;">=</td>
                <td style="color:{TEXT}; font-size:0.88rem;">
                  r<sub>f</sub> &nbsp;+&nbsp; λ<sub>i</sub>
                  <span style="color:{MUTED}; font-size:0.72rem; margin-left:10px;">← illiquidity-adjusted rate</span>
                </td>
              </tr>
            </table>
          </div>

          <!-- Group 2: Option Price -->
          <div style="padding:10px 16px 6px 16px; background:rgba(124,58,237,0.05);
                      border-bottom:1px solid {BORDER};">
            <div style="font-size:0.62rem; font-weight:700; text-transform:uppercase;
                        letter-spacing:0.12em; color:{MUTED}; margin-bottom:8px;">
              2 — Option Price
            </div>
            <table style="width:100%; border-collapse:collapse;">
              <tr style="line-height:2.1;">
                <td style="color:{CYAN}; font-size:0.88rem; white-space:nowrap; padding-right:8px;">C</td>
                <td style="color:{MUTED}; font-size:0.88rem; padding:0 8px;">=</td>
                <td style="color:{TEXT}; font-size:0.88rem; line-height:1.6;">
                  S · e<sup>−λT</sup> · N(d<sub>1</sub>)
                  &nbsp;<span style="color:{MUTED}">−</span>&nbsp;
                  K · e<sup>−r<sub>adj</sub>T</sup> · N(d<sub>2</sub>)
                </td>
              </tr>
            </table>
          </div>

          <!-- Group 3: d1, d2 -->
          <div style="padding:10px 16px 12px 16px;">
            <div style="font-size:0.62rem; font-weight:700; text-transform:uppercase;
                        letter-spacing:0.12em; color:{MUTED}; margin-bottom:8px;">
              3 — Log-moneyness factors
            </div>
            <table style="width:100%; border-collapse:collapse;">
              <tr style="line-height:2.1;">
                <td style="color:{CYAN}; font-size:0.88rem; white-space:nowrap; padding-right:8px;">d<sub>1</sub></td>
                <td style="color:{MUTED}; font-size:0.88rem; padding:0 8px;">=</td>
                <td style="color:{TEXT}; font-size:0.85rem; line-height:1.6;">
                  [ ln(S/K) + (r<sub>adj</sub> + σ²/2)·T ] &nbsp;/&nbsp; (σ√T)
                </td>
              </tr>
              <tr style="line-height:2.1;">
                <td style="color:{CYAN}; font-size:0.88rem; white-space:nowrap;">d<sub>2</sub></td>
                <td style="color:{MUTED}; font-size:0.88rem; padding:0 8px;">=</td>
                <td style="color:{TEXT}; font-size:0.88rem;">
                  d<sub>1</sub> &nbsp;−&nbsp; σ√T
                </td>
              </tr>
            </table>
          </div>

        </div>
        """, unsafe_allow_html=True)

    with col_e:
        st.markdown(f'<div class="section-header">📌 Hedge Optimization</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:rgba(22,27,34,0.9); border:1px solid {BORDER};
                    border-radius:10px; overflow:hidden; font-family:'JetBrains Mono','Fira Code',monospace;">

          <!-- Group 1: Objective -->
          <div style="padding:10px 16px 6px 16px; background:rgba(0,180,216,0.06);
                      border-bottom:1px solid {BORDER};">
            <div style="font-size:0.62rem; font-weight:700; text-transform:uppercase;
                        letter-spacing:0.12em; color:{MUTED}; margin-bottom:8px;">
              1 — Objective
            </div>
            <table style="width:100%; border-collapse:collapse;">
              <tr style="line-height:2.1;">
                <td style="color:{GOLD}; font-size:0.82rem; white-space:nowrap; padding-right:8px;">min<sub>w</sub></td>
                <td style="color:{TEXT}; font-size:0.88rem;">
                  Var&thinsp;( r<sub>ptf</sub> &nbsp;+&nbsp; Σ<sub>j</sub> w<sub>j</sub>·r<sub>j</sub> )
                </td>
              </tr>
            </table>
          </div>

          <!-- Group 2: Variance expansion -->
          <div style="padding:10px 16px 6px 16px; background:rgba(124,58,237,0.05);
                      border-bottom:1px solid {BORDER};">
            <div style="font-size:0.62rem; font-weight:700; text-transform:uppercase;
                        letter-spacing:0.12em; color:{MUTED}; margin-bottom:8px;">
              2 — Variance Expansion
            </div>
            <table style="width:100%; border-collapse:collapse;">
              <tr style="line-height:2.1;">
                <td style="color:{CYAN}; font-size:0.88rem; white-space:nowrap; padding-right:8px;">J(w)</td>
                <td style="color:{MUTED}; font-size:0.88rem; padding:0 8px;">=</td>
                <td style="color:{TEXT}; font-size:0.85rem; line-height:1.6;">
                  σ²<sub>p</sub>
                  &nbsp;+&nbsp; w<sup>⊤</sup>Σ<sub>h</sub>w
                  &nbsp;+&nbsp; 2w<sup>⊤</sup>·cov(r<sub>p</sub>,&thinsp;r<sub>h</sub>)
                </td>
              </tr>
            </table>
          </div>

          <!-- Group 3: Solution -->
          <div style="padding:10px 16px 12px 16px;">
            <div style="font-size:0.62rem; font-weight:700; text-transform:uppercase;
                        letter-spacing:0.12em; color:{MUTED}; margin-bottom:8px;">
              3 — Analytical Solution
            </div>
            <table style="width:100%; border-collapse:collapse;">
              <tr style="line-height:2.1;">
                <td style="color:{CYAN}; font-size:0.88rem; white-space:nowrap; padding-right:8px;">w*</td>
                <td style="color:{MUTED}; font-size:0.88rem; padding:0 8px;">=</td>
                <td style="color:{TEXT}; font-size:0.88rem; line-height:1.6;">
                  −Σ<sub>h</sub><sup>−1</sup>
                  &nbsp;·&nbsp; cov(r<sub>p</sub>,&thinsp;r<sub>h</sub>)
                </td>
              </tr>
              <tr>
                <td colspan="3" style="color:{MUTED}; font-size:0.72rem; padding-top:4px; line-height:1.5;">
                  ↳ &nbsp;Computed numerically via SLSQP for extensibility<br>
                  ↳ &nbsp;Σ<sub>h</sub> = covariance matrix of hedge instruments
                </td>
              </tr>
            </table>
          </div>

        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:32px; text-align:center; color:{MUTED}; font-size:0.8rem;">
        Configure parameters in the sidebar and click <strong style="color:{CYAN}">▶ Run Analysis</strong> to start.
    </div>""", unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# DATA + COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
illiq_flat   = [t for v in illiq_classes.values() for t in v]
all_tickers  = tuple(sorted(set(illiq_flat + hedge_tickers)))

with st.spinner("Fetching market data from Yahoo Finance…"):
    data = fetch_data(all_tickers, period)

if not data:
    st.error("Download failed — check your connection or ticker symbols.")
    st.stop()

available  = set(data["close"].columns)
illiq_ok   = [t for t in illiq_flat    if t in available]
hedge_ok   = [t for t in hedge_tickers if t in available]

if not illiq_ok:
    st.error("No illiquid assets found in the downloaded data.")
    st.stop()

with st.spinner("Computing illiquidity metrics, Greeks & hedge weights…"):
    illiq_df     = compute_illiquidity(data, illiq_ok)
    portfolio_df = compute_portfolio(illiq_df, illiq_classes, T, put_pct, rf, kappa, alpha)
    hedge_res    = run_hedge_optimization(portfolio_df, data, hedge_ok)

# Status bar
st.markdown(f"""
<div style="display:flex; gap:10px; margin-bottom:20px; flex-wrap:wrap; align-items:center;">
    <span class="badge badge-green">✓ {len(illiq_ok)} illiquid assets</span>
    <span class="badge badge-cyan">✓ {len(hedge_ok)} hedge instruments</span>
    <span class="badge badge-gold">✓ {len(data['close'])} trading days</span>
    <span style="color:{MUTED}; font-size:0.72rem;">
        T = {T_months}m &nbsp;|&nbsp;
        Put @ {int(put_pct*100)}% &nbsp;|&nbsp;
        r_f = {rf:.1%} &nbsp;|&nbsp;
        κ = {kappa:.3f} &nbsp;|&nbsp;
        α = {alpha:.2f}
    </span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍  Illiquidity",
    "📐  Option Pricing",
    "🛡️  Hedge Results",
    "📈  Risk Analytics",
    "🔬  Sensitivity",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ILLIQUIDITY
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    section("🔍", "Illiquidity Scores")

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Avg Amihud ratio", f"{illiq_df['Amihud'].mean():.4f}")
    with c2: kpi("Avg Spread Proxy", f"{illiq_df['SpreadProxy'].mean():.3f}%")
    with c3: kpi("Avg Annualised Vol", f"{illiq_df['AnnVol'].mean():.2f}%")
    with c4: kpi("Assets scored", str(len(illiq_df)))

    st.markdown("<br>", unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)

    with ch1:
        fig = go.Figure()
        for cls, tks in illiq_classes.items():
            sub = illiq_df[illiq_df.index.isin(tks)]
            if sub.empty:
                continue
            fig.add_bar(x=sub.index, y=sub["Amihud"], name=cls,
                        marker_color=CLASS_COLORS.get(cls, MUTED),
                        marker_line_color="rgba(0,0,0,0.3)", marker_line_width=0.5)
        fig.update_layout(chart_layout(title="Amihud Illiquidity Ratio",
                                        barmode="group", xaxis_tickangle=-35,
                                        legend=dict(orientation="h", y=-0.28), height=380))
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        fig2 = go.Figure()
        for cls, tks in illiq_classes.items():
            sub = illiq_df[illiq_df.index.isin(tks)]
            if sub.empty:
                continue
            fig2.add_scatter(x=sub["Amihud"], y=sub["AnnVol"],
                             mode="markers+text", text=sub.index,
                             textposition="top right",
                             textfont=dict(size=9, color=TEXT),
                             name=cls,
                             marker=dict(size=11, color=CLASS_COLORS.get(cls, MUTED),
                                         line=dict(width=1.5, color=CARD)))
        fig2.update_layout(chart_layout(title="Illiquidity vs. Annualised Volatility",
                                         xaxis_title="Amihud Ratio",
                                         yaxis_title="Ann. Vol (%)", height=380))
        st.plotly_chart(fig2, use_container_width=True)

    divider()
    section("📋", "Illiquidity Table")
    disp = illiq_df.copy()
    disp.columns = ["Amihud ×1e-6", "Spread Proxy (%)", "Ann. Vol (%)", "Avg Price", "Last Price"]
    disp.insert(0, "Full Name", [TICKER_NAMES.get(t, "—") for t in disp.index])
    disp.insert(1, "Class", [
        next((cls for cls, tks in illiq_classes.items() if t in tks), "—")
        for t in disp.index
    ])
    st.dataframe(
        disp.style
            .format("{:.4f}", subset=["Amihud ×1e-6"])
            .format("{:.3f}", subset=["Spread Proxy (%)"])
            .format("{:.2f}", subset=["Ann. Vol (%)", "Avg Price", "Last Price"])
            .background_gradient(subset=["Amihud ×1e-6"], cmap="YlOrRd"),
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — OPTION PRICING
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    section("📐", f"Option Pricing  ·  T = {T_months}m  ·  Put @ {int(put_pct*100)}%")

    if portfolio_df.empty:
        st.warning("No options could be priced — check asset availability.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Avg Liq. Premium λ", f"{portfolio_df['Lambda_pct'].mean():.4f}%")
        with c2: kpi("Avg Liq. Impact",    f"{portfolio_df['LiqImpact_pct'].mean():.3f}%")
        with c3: kpi("Avg Call Delta",     f"{portfolio_df['CallDelta'].mean():.4f}")
        with c4: kpi("Avg Call Vega",      f"{portfolio_df['CallVega'].mean():.4f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Greeks horizontal bars
        greek_cols   = ["CallDelta", "CallGamma", "CallVega", "CallTheta", "CallRho"]
        greek_labels = ["Δ Delta", "Γ Gamma", "ν Vega", "Θ Theta", "ρ Rho"]

        fig = make_subplots(rows=1, cols=5, subplot_titles=greek_labels, shared_yaxes=True)
        for i, (col, lbl) in enumerate(zip(greek_cols, greek_labels), 1):
            vals = portfolio_df[col]
            colors = [GREEN if v >= 0 else RED for v in vals]
            fig.add_bar(x=vals, y=portfolio_df.index, orientation="h",
                        marker_color=colors, marker_line_width=0,
                        showlegend=False, row=1, col=i)
            fig.update_xaxes(gridcolor="rgba(48,54,61,0.5)", zeroline=True,
                             zerolinecolor=BORDER, zerolinewidth=1,
                             tickfont=dict(color=MUTED, size=9), row=1, col=i)
            fig.update_yaxes(tickfont=dict(color=TEXT, size=9), row=1, col=i)

        for ann in fig.layout.annotations:
            ann.font = dict(color=MUTED, size=11)

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(22,27,34,0.6)",
            title_text="ATM Call Greeks — Liquidity-Adjusted",
            title_font=dict(color=TEXT, size=14),
            height=420,
            hoverlabel=dict(bgcolor=CARD, bordercolor=BORDER, font=dict(color=TEXT)),
            margin=dict(l=80, r=16, t=56, b=16),
        )
        st.plotly_chart(fig, use_container_width=True)

        divider()
        ch1, ch2 = st.columns(2)
        with ch1:
            fig3 = go.Figure(go.Bar(
                x=portfolio_df.index,
                y=portfolio_df["LiqImpact_pct"],
                marker_color=[RED if v < 0 else GOLD for v in portfolio_df["LiqImpact_pct"]],
                marker_line_width=0,
            ))
            fig3.update_layout(chart_layout(title="Liquidity Impact on ATM Call Price (%)",
                                             yaxis_title="(Adj − Std) / Std  ×100",
                                             xaxis_tickangle=-35, height=360))
            fig3.add_hline(y=0, line_color=BORDER, line_width=1)
            st.plotly_chart(fig3, use_container_width=True)

        with ch2:
            fig4 = go.Figure(go.Bar(
                x=portfolio_df.index,
                y=portfolio_df["Lambda_pct"],
                marker_color=[CLASS_COLORS.get(portfolio_df.loc[t, "Class"], MUTED)
                               for t in portfolio_df.index],
                marker_line_width=0,
            ))
            fig4.update_layout(chart_layout(title="Liquidity Premium λ per Asset (%)",
                                             yaxis_title="λ (%)",
                                             xaxis_tickangle=-35, height=360))
            st.plotly_chart(fig4, use_container_width=True)

        with st.expander("📋  Full Pricing Table"):
            show_cols = ["Class", "S", "Sigma", "Amihud", "Lambda_pct", "r_adj_pct",
                         "CallPrice", "CallBS_Std", "LiqImpact_pct",
                         "CallDelta", "CallGamma", "CallVega", "CallTheta",
                         "PutPrice", "PutDelta"]
            st.dataframe(
                portfolio_df[show_cols].rename(columns={
                    "Lambda_pct": "λ (%)", "r_adj_pct": "r_adj (%)", "LiqImpact_pct": "Impact (%)"}),
                use_container_width=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HEDGE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    section("🛡️", "Minimum-Variance Cross-Asset Hedge")

    if not hedge_res:
        st.warning("Hedge optimization could not run — check hedge tickers.")
    else:
        vol_b = np.sqrt(hedge_res["var_p"]) * np.sqrt(252) * 100
        vol_a = np.sqrt(max(hedge_res["var_hedged"], 0)) * np.sqrt(252) * 100
        hr    = hedge_res["hedge_ratio"]

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Vol Before Hedge", f"{vol_b:.2f}%")
        with c2: kpi("Vol After Hedge",  f"{vol_a:.2f}%",
                     delta=f"{abs(vol_a - vol_b):.2f}%", delta_pos=False)
        with c3: kpi("Variance Reduction", f"{hr * 100:.1f}%",
                     delta="annualised", delta_pos=True)
        with c4: kpi("Hedge Instruments", str(len(hedge_res["weights"])))

        st.markdown("<br>", unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)

        with ch1:
            w = hedge_res["weights"].sort_values(key=abs, ascending=False)
            fig = go.Figure(go.Bar(
                x=w.index, y=w.values,
                marker_color=[GREEN if v >= 0 else RED for v in w.values],
                marker_line_width=0,
                text=[f"{v:+.3f}" for v in w.values],
                textposition="outside",
                textfont=dict(color=TEXT, size=10),
            ))
            fig.add_hline(y=0, line_color=BORDER, line_width=1)
            fig.update_layout(chart_layout(title="Optimal Hedge Weights",
                                            yaxis_title="w  (+ long  /  − short)",
                                            xaxis_tickangle=-25, height=380))
            st.plotly_chart(fig, use_container_width=True)

        with ch2:
            corr = hedge_res["corr"].sort_values(ascending=True)
            fig2 = go.Figure(go.Bar(
                x=corr.values, y=corr.index, orientation="h",
                marker_color=[GREEN if v >= 0 else RED for v in corr.values],
                marker_line_width=0,
            ))
            fig2.add_vline(x=0, line_color=BORDER, line_width=1)
            fig2.update_layout(chart_layout(title="Portfolio × Hedge Instrument Correlation",
                                             xaxis_title="Pearson ρ", height=380))
            st.plotly_chart(fig2, use_container_width=True)

        divider()
        section("📈", "Cumulative P&L")

        pnl_raw    = hedge_res["ptf_pnl"]
        pnl_hedged = hedge_res["hedged_pnl"]
        cum_raw    = (1 + pnl_raw).cumprod()
        cum_hedged = (1 + pnl_hedged).cumprod()

        fig3 = go.Figure()
        fig3.add_scatter(x=cum_raw.index, y=cum_raw,
                         name="Unhedged", line=dict(color=RED, width=1.8))
        fig3.add_scatter(x=cum_hedged.index, y=cum_hedged,
                         name="Hedged", line=dict(color=GREEN, width=1.8))
        fig3.add_hline(y=1, line_dash="dot", line_color=BORDER, line_width=1)
        fig3.update_layout(chart_layout(title="Cumulative Portfolio P&L",
                                         yaxis_title="Value (base 1)", height=350,
                                         legend=dict(orientation="h", y=1.08)))
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RISK ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<br>", unsafe_allow_html=True)
    section("📈", "Portfolio Risk Analytics")

    if not hedge_res:
        st.warning("Run the analysis first.")
    else:
        pnl_raw    = hedge_res["ptf_pnl"]
        pnl_hedged = hedge_res["hedged_pnl"]
        sr = perf_stats(pnl_raw)
        sh = perf_stats(pnl_hedged)

        # Stats grid
        col_labels = ["Sharpe", "Vol ann. (%)", "Max DD (%)", "VaR 95% (%)", "Skewness", "Kurtosis"]
        stats_df = pd.DataFrame([sr, sh], index=["Unhedged", "Hedged"], columns=col_labels)
        st.dataframe(
            stats_df.style
                .format("{:.3f}")
                .highlight_min(axis=0, subset=["Vol ann. (%)", "Max DD (%)", "VaR 95% (%)"],
                               props="background-color: rgba(63,185,80,0.15); color: #3fb950")
                .highlight_max(axis=0, subset=["Sharpe"],
                               props="background-color: rgba(63,185,80,0.15); color: #3fb950"),
            use_container_width=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)

        with ch1:
            fig = go.Figure()
            fig.add_histogram(x=pnl_raw * 100, nbinsx=60, opacity=0.65,
                              name="Unhedged", marker_color=RED,
                              histnorm="probability density")
            fig.add_histogram(x=pnl_hedged * 100, nbinsx=60, opacity=0.65,
                              name="Hedged", marker_color=GREEN,
                              histnorm="probability density")
            fig.update_layout(chart_layout(title="Daily Return Distribution",
                                            xaxis_title="Return (%)",
                                            barmode="overlay", height=360,
                                            legend=dict(orientation="h", y=1.08)))
            st.plotly_chart(fig, use_container_width=True)

        with ch2:
            roll = 21
            rv_r = pnl_raw.rolling(roll).std() * np.sqrt(252) * 100
            rv_h = pnl_hedged.rolling(roll).std() * np.sqrt(252) * 100
            fig2 = go.Figure()
            fig2.add_scatter(x=rv_r.index, y=rv_r,
                             name="Unhedged", line=dict(color=RED, width=1.5))
            fig2.add_scatter(x=rv_h.index, y=rv_h,
                             name="Hedged", line=dict(color=GREEN, width=1.5))
            fig2.update_layout(chart_layout(title=f"Rolling {roll}-Day Vol (ann. %)",
                                             yaxis_title="Vol (%)", height=360,
                                             legend=dict(orientation="h", y=1.08)))
            st.plotly_chart(fig2, use_container_width=True)

        divider()
        section("📉", "Drawdown")

        cum_r = (1 + pnl_raw).cumprod()
        cum_h = (1 + pnl_hedged).cumprod()
        dd_r  = (cum_r - cum_r.cummax()) / cum_r.cummax() * 100
        dd_h  = (cum_h - cum_h.cummax()) / cum_h.cummax() * 100

        fig3 = go.Figure()
        fig3.add_scatter(x=dd_r.index, y=dd_r, fill="tozeroy", name="Unhedged",
                         line=dict(color=RED, width=1),
                         fillcolor="rgba(248,81,73,0.18)")
        fig3.add_scatter(x=dd_h.index, y=dd_h, fill="tozeroy", name="Hedged",
                         line=dict(color=GREEN, width=1),
                         fillcolor="rgba(63,185,80,0.18)")
        fig3.update_layout(chart_layout(title="Drawdown Profile",
                                         yaxis_title="Drawdown (%)", height=280,
                                         legend=dict(orientation="h", y=1.08)))
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<br>", unsafe_allow_html=True)
    section("🔬", "Sensitivity Analysis — Liquidity Parameters")

    if portfolio_df.empty:
        st.warning("No options priced.")
    else:
        most_illiq = illiq_df["Amihud"].idxmax()

        if most_illiq not in portfolio_df.index:
            st.info("Reference asset not priced — adjust asset list.")
        else:
            row    = portfolio_df.loc[most_illiq]
            S      = float(row["S"])
            sigma  = float(row["Sigma"])
            amihud = float(row["Amihud"])

            st.markdown(f"""
            <div style="display:flex; gap:10px; margin-bottom:16px; flex-wrap:wrap; align-items:center;">
                <span style="color:{MUTED}; font-size:0.8rem;">Reference asset:</span>
                <span class="badge badge-cyan">{most_illiq}</span>
                <span style="color:{MUTED}; font-size:0.8rem;">Amihud = <strong style="color:{TEXT}">{amihud:.4f}</strong></span>
                <span style="color:{MUTED}; font-size:0.8rem;">σ = <strong style="color:{TEXT}">{sigma*100:.1f}%</strong></span>
            </div>
            """, unsafe_allow_html=True)

            s1, s2 = st.columns(2)

            with s1:
                kappas = np.linspace(0, 0.12, 60)
                line_colors = [CYAN, GOLD, GREEN, RED]
                fig = go.Figure()
                for a, lc in zip([0.25, 0.5, 0.75, 1.0], line_colors):
                    prices = [LiquidityAdjustedBS(S, S, T, sigma, amihud, rf, k, a).price()
                               for k in kappas]
                    fig.add_scatter(x=kappas * 100, y=prices, name=f"α = {a}",
                                    mode="lines", line=dict(color=lc, width=1.8))
                fig.update_layout(chart_layout(title=f"ATM Call Price vs κ  ({most_illiq})",
                                                xaxis_title="κ (%)",
                                                yaxis_title="Call Price", height=370))
                st.plotly_chart(fig, use_container_width=True)

            with s2:
                moneyness = np.linspace(0.78, 1.22, 60)
                amihud_levels = [0, amihud * 0.5, amihud, amihud * 2]
                liq_labels = ["Liquid (λ=0)", "0.5× ILLIQ", "Reference", "2× ILLIQ"]
                fig2 = go.Figure()
                for amv, lbl, lc in zip(amihud_levels, liq_labels, line_colors):
                    deltas = [LiquidityAdjustedBS(S, S * m, T, sigma, amv, rf, kappa, alpha).delta()
                               for m in moneyness]
                    fig2.add_scatter(x=moneyness, y=deltas, name=lbl, mode="lines",
                                     line=dict(color=lc, width=1.8))
                fig2.add_vline(x=1.0, line_dash="dot", line_color=BORDER, line_width=1.2)
                fig2.update_layout(chart_layout(title="Delta vs Moneyness — Illiquidity Levels",
                                                 xaxis_title="Moneyness (K/S)",
                                                 yaxis_title="Delta", height=370))
                st.plotly_chart(fig2, use_container_width=True)

            divider()
            section("🌡️", "Call Price — κ × α Heatmap")

            kp_range = np.linspace(0, 0.10, 30)
            al_range = np.linspace(0.1, 1.5, 30)
            Z = np.array([
                [LiquidityAdjustedBS(S, S, T, sigma, amihud, rf, kp, al).price()
                 for kp in kp_range]
                for al in al_range
            ])
            fig3 = go.Figure(go.Heatmap(
                z=Z,
                x=(kp_range * 100).round(2),
                y=al_range.round(2),
                colorscale=[[0, "#0d1117"], [0.3, "#00b4d8"], [0.7, "#7c3aed"], [1, "#f85149"]],
                colorbar=dict(title=dict(text="Call Price", font=dict(color=MUTED)),
                              tickfont=dict(color=MUTED), bordercolor=BORDER),
                hovertemplate="κ=%{x}%  α=%{y}<br>Price=%{z:.4f}<extra></extra>",
            ))
            fig3.update_layout(chart_layout(xaxis_title="κ (%)", yaxis_title="α", height=400))
            st.plotly_chart(fig3, use_container_width=True)
