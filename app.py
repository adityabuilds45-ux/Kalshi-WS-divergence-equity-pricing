"""
Kalshi CPI Divergence -> ETF Response Terminal
Run locally:  streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import html
import math
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import core

try:  # optional: auto-refresh. App works fine without it.
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

GAMMA_FILE = Path(__file__).with_name("gammas.csv")
SENS_FILE = Path(__file__).with_name("sensitivity_params.csv")

# Tail-risk monitor settings (edit here to change the rules)
TAIL_MIN_OI = 50          # drop Kalshi strikes with open interest below this
TAIL_MAX_SPREAD = 0.15    # drop strikes with bid-ask spread wider than this ($0.15 = 15 cents)
TAIL_THRESHOLD = 0.15     # "tail shock" = CPI at least this many pp above/below consensus
HEDGE_TRIGGER = 1.25      # Kalshi tail move must exceed options-implied move by 25%+
# ETFs allowed to use the ^VIX / sqrt(252) fallback when their options chain is empty/unpriced.
# Default: all nine (as specified). VIX is an equity-volatility index, so for the ETFs NOT in
# EQUITY_VOL_ETFS it is only a rough proxy and is flagged "proxy" in the UI. Remove tickers from
# this set to show "NO SIGNAL" for them instead.
VIX_FALLBACK_TICKERS = set(core.UNIVERSE["Ticker"])
EQUITY_VOL_ETFS = {"QQQ", "XLRE", "KRE"}

st.set_page_config(
    page_title="Kalshi CPI × ETF Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# THEME / CSS  (dark, terminal-style; all colours live in :root variables)
# ============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
:root{
  --bg:#070b12; --panel:#0d1420; --panel2:#111a29; --line:#1b2940;
  --text:#e6edf7; --muted:#7d8ba3; --cyan:#22d3ee; --green:#34d399; --red:#f87171;
  --amber:#fbbf24; --violet:#a78bfa; --blue:#60a5fa;
}
.stApp{
  background:
    radial-gradient(1100px 560px at 88% -8%, rgba(34,211,238,.08), transparent 60%),
    radial-gradient(900px 520px at -8% 108%, rgba(167,139,250,.07), transparent 60%),
    var(--bg);
  font-family:'Space Grotesk',system-ui,-apple-system,'Segoe UI',sans-serif; color:var(--text);
}
.stApp::before{
  content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:linear-gradient(rgba(255,255,255,.02) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(255,255,255,.02) 1px,transparent 1px);
  background-size:46px 46px;
}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.1rem; max-width:1420px;}
footer{visibility:hidden;}
.mono{font-family:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;}

/* ---------- sidebar rail ---------- */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a111c 0%,#070c15 100%); border-right:1px solid var(--line);}
[data-testid="stSidebar"] .block-container{padding-top:1rem;}
.brand{display:flex; align-items:center; gap:10px; padding:4px 2px 14px;}
.brand-mark{width:34px; height:34px; border-radius:9px; display:flex; align-items:center; justify-content:center;
  background:linear-gradient(135deg,#22d3ee,#6366f1); color:#04121f; font-weight:800; font-size:1.05rem;
  box-shadow:0 0 22px rgba(34,211,238,.35);}
.brand-name{font-weight:700; letter-spacing:.06em; font-size:.95rem; line-height:1.1;}
.brand-sub{color:var(--muted); font-size:.68rem; letter-spacing:.14em; text-transform:uppercase;}
.side-label{color:var(--muted); font-size:.66rem; letter-spacing:.16em; text-transform:uppercase; margin:16px 0 6px;}

/* ---------- radios: chips by default, vertical nav in sidebar ---------- */
div[role="radiogroup"]{gap:8px; flex-wrap:wrap;}
div[role="radiogroup"] > label{
  background:var(--panel); border:1px solid var(--line); border-radius:999px; padding:3px 14px; margin:0;
  cursor:pointer; transition:all .15s ease;}
div[role="radiogroup"] > label > div:first-child{display:none;}
div[role="radiogroup"] > label p{font-family:'JetBrains Mono',monospace; font-size:.8rem; letter-spacing:.05em; color:var(--muted); margin:0;}
div[role="radiogroup"] > label:hover{border-color:#2a3d5c;}
div[role="radiogroup"] > label:has(input:checked){
  border-color:var(--cyan); background:rgba(34,211,238,.10); box-shadow:0 0 18px rgba(34,211,238,.20);}
div[role="radiogroup"] > label:has(input:checked) p{color:var(--cyan); font-weight:700;}
[data-testid="stSidebar"] div[role="radiogroup"]{flex-direction:column; gap:3px; flex-wrap:nowrap;}
[data-testid="stSidebar"] div[role="radiogroup"] > label{
  width:100%; border-radius:10px; padding:9px 12px; background:transparent; border-color:transparent;}
[data-testid="stSidebar"] div[role="radiogroup"] > label p{font-family:'Space Grotesk',sans-serif; font-size:.9rem; letter-spacing:.04em; font-weight:500;}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover{background:rgba(255,255,255,.03);}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked){
  background:rgba(34,211,238,.08); border-color:rgba(34,211,238,.30); box-shadow:inset 3px 0 0 var(--cyan);}

/* ---------- top bar / tape ---------- */
.topbar{display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:4px;}
.crumb{font-size:1.15rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase;}
.crumb span{color:var(--muted); font-weight:500;}
.live{display:inline-block; width:9px; height:9px; border-radius:50%; background:var(--green); margin-right:10px;
  box-shadow:0 0 0 0 rgba(52,211,153,.7); animation:pulse 2s infinite;}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(52,211,153,.6)}70%{box-shadow:0 0 0 9px rgba(52,211,153,0)}100%{box-shadow:0 0 0 0 rgba(52,211,153,0)}}
.chip{display:inline-block; padding:4px 11px; margin-left:7px; border-radius:7px; border:1px solid var(--line);
  background:rgba(13,20,32,.8); color:var(--muted); font-family:'JetBrains Mono',monospace; font-size:.72rem;}
.chip b{color:var(--text); font-weight:500;}
.tape{overflow:hidden; white-space:nowrap; border:1px solid var(--line); border-radius:10px; background:rgba(13,20,32,.75);
  padding:8px 0; margin:10px 0 14px;}
.tape-inner{display:inline-block; animation:tape 50s linear infinite;}
.tape:hover .tape-inner{animation-play-state:paused;}
@keyframes tape{from{transform:translateX(0)}to{transform:translateX(-50%)}}
.tk{margin:0 24px; font-family:'JetBrains Mono',monospace; font-size:.8rem;}
.tk b{letter-spacing:.05em; margin-right:7px;} .tk .px{color:var(--muted); margin-right:6px;}
.pos{color:var(--green);} .neg{color:var(--red);} .neu{color:var(--muted);} .amb{color:var(--amber);}
.strip-label{color:var(--muted); font-size:.66rem; letter-spacing:.16em; text-transform:uppercase; margin:2px 0 4px;}

/* ---------- cards ---------- */
.card{background:linear-gradient(180deg,var(--panel2),var(--panel)); border:1px solid var(--line); border-radius:14px;
  padding:14px 16px; box-shadow:0 10px 32px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.03);}
.kpi{min-height:152px; display:flex; flex-direction:column; position:relative; overflow:hidden;}
.kpi::after{content:""; position:absolute; left:0; top:0; height:2px; width:100%;
  background:linear-gradient(90deg,var(--cyan),transparent);}
.kpi.pos::after{background:linear-gradient(90deg,var(--green),transparent);}
.kpi.neg::after{background:linear-gradient(90deg,var(--red),transparent);}
.kpi.amb::after{background:linear-gradient(90deg,var(--amber),transparent);}
.kpi-top{display:flex; justify-content:space-between; align-items:center;}
.kpi-label{font-size:.68rem; letter-spacing:.13em; text-transform:uppercase; color:var(--muted); font-weight:600;}
.kpi-tag{font-family:'JetBrains Mono',monospace; font-size:.62rem; color:var(--cyan); border:1px solid rgba(34,211,238,.35);
  border-radius:5px; padding:1px 6px;}
.kpi-value{font-size:1.75rem; font-weight:700; margin-top:6px; line-height:1.1; letter-spacing:-.01em;}
.kpi.pos .kpi-value{color:var(--green);} .kpi.neg .kpi-value{color:var(--red);}
.kpi-sub{font-size:.76rem; color:var(--muted); margin-top:3px; font-family:'JetBrains Mono',monospace;}
.kpi.pos .kpi-sub{color:var(--green);} .kpi.neg .kpi-sub{color:var(--red);}
.kpi-viz{margin-top:auto; padding-top:8px;}
.spark{width:100%; height:40px; display:block;}
.gauge{position:relative; height:8px; border-radius:6px; background:rgba(125,139,163,.16); margin:14px 0 4px;}
.gmid{position:absolute; left:50%; top:-3px; width:1px; height:14px; background:rgba(230,237,247,.45);}
.gfill{position:absolute; top:0; height:8px; border-radius:6px;}
.track{position:relative; height:34px;}
.track .rail{position:absolute; left:0; right:0; top:11px; height:4px; border-radius:4px; background:rgba(125,139,163,.2);}
.track .dot{position:absolute; top:6px; width:14px; height:14px; border-radius:50%; margin-left:-7px; border:2px solid #0d1420;}
.track-lbl{position:absolute; left:0; right:0; top:22px; display:flex; justify-content:space-between;
  font-family:'JetBrains Mono',monospace; font-size:.62rem; color:var(--muted);}
.odds-bar{height:6px; border-radius:6px; background:rgba(125,139,163,.16); margin-top:10px; overflow:hidden;}
.odds-fill{height:6px; border-radius:6px;}
.etf-head{display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;}
.etf-sym{font-size:2.1rem; font-weight:700; letter-spacing:.04em; line-height:1;}
.etf-name{color:var(--muted); font-size:.95rem; margin-top:4px;}
.etf-exp{color:var(--text); font-size:.86rem; max-width:520px; opacity:.85;}

/* ---------- panels (st.container(key="p_...")) ---------- */
[class*="st-key-p_"]{background:linear-gradient(180deg,var(--panel2),var(--panel)); border:1px solid var(--line);
  border-radius:14px; padding:14px 18px 8px; box-shadow:0 10px 32px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.03);}
.panel-title{font-size:.72rem; letter-spacing:.15em; text-transform:uppercase; color:var(--muted); font-weight:600; margin-bottom:6px;}
.panel-title i{color:var(--cyan); font-style:normal; margin-right:6px;}
.section-h{font-size:1.25rem; font-weight:700; letter-spacing:.04em; margin:6px 0 2px;}
.section-p{color:var(--muted); font-size:.9rem; margin-bottom:10px;}

/* ---------- ETF table ---------- */
.tbl-wrap{overflow-x:auto;}
.tbl{width:100%; border-collapse:collapse; font-size:.86rem; color:var(--text);}
.tbl td{color:var(--text);}
.tbl th{color:var(--muted); font-weight:600; text-transform:uppercase; font-size:.66rem; letter-spacing:.1em; text-align:left;
  padding:8px 10px; border-bottom:1px solid var(--line); white-space:nowrap;}
.tbl td{padding:10px; border-bottom:1px solid rgba(27,41,64,.55); vertical-align:middle;}
.tbl .num{text-align:right; font-family:'JetBrains Mono',monospace; white-space:nowrap;}
.tbl th.num{text-align:right;}
.tbl tr.sel td{background:rgba(34,211,238,.06);}
.tbl tr.sel td:first-child{box-shadow:inset 3px 0 0 var(--cyan);}
.sym{font-weight:700; letter-spacing:.05em;}
.sub-muted{color:var(--muted); font-size:.78rem;}
.pill{display:inline-block; padding:2px 9px; border-radius:6px; font-family:'JetBrains Mono',monospace; font-size:.78rem;}
.pill.pos{background:rgba(52,211,153,.13); color:var(--green);}
.pill.neg{background:rgba(248,113,113,.13); color:var(--red);}
.pill.neu{background:rgba(125,139,163,.14); color:var(--muted);}
.pill.cy{background:rgba(34,211,238,.12); color:var(--cyan);}

/* ---------- tail-risk monitor ---------- */
.pill.amb{background:rgba(251,191,36,.14); color:var(--amber);}
.banner{display:flex; align-items:center; gap:14px; border-radius:14px; padding:14px 16px; margin-bottom:10px; border:1px solid var(--line);}
.banner-tk{font-family:'JetBrains Mono',monospace; font-weight:700; letter-spacing:.06em; padding:6px 10px; border-radius:8px; background:rgba(0,0,0,.28);}
.banner-text{font-weight:700; letter-spacing:.02em; font-size:.98rem; line-height:1.25;}
.banner-detail{font-family:'JetBrains Mono',monospace; font-size:.72rem; opacity:.8; margin-top:3px;}
.banner.go{background:linear-gradient(90deg,rgba(52,211,153,.18),rgba(52,211,153,.05)); border-color:rgba(52,211,153,.55); color:var(--green); box-shadow:0 0 28px rgba(52,211,153,.16);}
.banner.stand{background:linear-gradient(90deg,rgba(251,191,36,.14),rgba(251,191,36,.04)); border-color:rgba(251,191,36,.45); color:var(--amber);}
.banner.na{background:rgba(125,139,163,.08); border-color:var(--line); color:var(--muted);}
.banner.compact{padding:10px 12px; gap:10px; margin-bottom:8px; border-radius:12px; min-height:72px;}
.banner.compact .banner-tk{padding:5px 8px; font-size:.82rem;}
.banner.compact .banner-text{font-size:.8rem;}
.banner.compact .banner-detail{font-size:.62rem;}
.banner.pick{outline:1px solid rgba(34,211,238,.7); outline-offset:2px;}
.rng{position:relative; height:10px; min-width:170px; border-radius:6px; background:rgba(125,139,163,.14);}
.rng-mid{position:absolute; left:50%; top:-2px; width:1px; height:14px; background:rgba(230,237,247,.35);}
.rng-seg{position:absolute; top:2px; height:6px; border-radius:4px; background:linear-gradient(90deg,var(--cyan),var(--violet)); opacity:.85;}
.rng-dot{position:absolute; top:-1px; width:12px; height:12px; margin-left:-6px; border-radius:50%; background:#e6edf7; border:2px solid #0d1420;}

/* ---------- inputs ---------- */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"]{background-color:var(--panel2) !important; border-color:var(--line) !important;}
.stButton > button, .stDownloadButton > button{border:1px solid var(--line); background:var(--panel2); color:var(--text); border-radius:10px;}
.stButton > button:hover, .stDownloadButton > button:hover{border-color:var(--cyan); color:var(--cyan);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# colour constants shared with charts
C_CYAN, C_GREEN, C_RED = "#22d3ee", "#34d399", "#f87171"
C_AMBER, C_VIOLET, C_MUTED = "#fbbf24", "#a78bfa", "#8896ae"


# ============================================================================
# Helpers
# ============================================================================
def esc(x) -> str:
    return html.escape(str(x))


def isnan(x) -> bool:
    try:
        return x is None or bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def fnum(x, d=2, signed=False, suffix="") -> str:
    if isnan(x):
        return "—"
    return f"{x:+.{d}f}{suffix}" if signed else f"{x:.{d}f}{suffix}"


def tone_of(x) -> str:
    if isnan(x) or abs(x) < 1e-12:
        return "neu"
    return "pos" if x > 0 else "neg"


def to_et(ts):
    try:
        return pd.Timestamp(ts).tz_convert("America/New_York")
    except Exception:
        return None


def fmt_close(ts) -> str:
    t = to_et(ts)
    return t.strftime("%b %d, %Y %I:%M %p ET") if t is not None and not pd.isna(t) else "n/a"


def now_et() -> str:
    try:
        return pd.Timestamp.now(tz="America/New_York").strftime("%b %d · %I:%M:%S %p ET")
    except Exception:
        return dt.datetime.now(dt.timezone.utc).strftime("%b %d · %H:%M:%S UTC")


def panel(key: str):
    """A styled panel. Falls back to a plain container on older Streamlit."""
    try:
        return st.container(key=f"p_{key}")
    except TypeError:
        return st.container()


def panel_title(text: str, icon: str = "◆") -> None:
    st.markdown(f'<div class="panel-title"><i>{esc(icon)}</i>{esc(text)}</div>', unsafe_allow_html=True)


# ---- tiny inline-SVG / HTML visuals ----------------------------------------
def spark_svg(values, color=C_CYAN, uid="a", w=140, h=40) -> str:
    vals = [float(v) for v in values if not isnan(v)]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    n = len(vals)
    pts = [(i * w / (n - 1), (h - 3) - (v - lo) / rng * (h - 6)) for i, v in enumerate(vals)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return (
        f'<svg class="spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
        f'<defs><linearGradient id="g{uid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{color}" stop-opacity=".35"/>'
        f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
        f'<polygon points="0,{h} {line} {w},{h}" fill="url(#g{uid})"/>'
        f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.8" '
        f'stroke-linejoin="round" vector-effect="non-scaling-stroke"/></svg>'
    )


def minibars_svg(probs, highlight=None, color=C_CYAN, w=140, h=40) -> str:
    probs = [float(p) for p in probs]
    if not probs:
        return ""
    n, mx = len(probs), (max(probs) or 1.0)
    bw = w / n
    rects = []
    for i, p in enumerate(probs):
        bh = max(p / mx * (h - 4), 1.0)
        hl = i == highlight
        rects.append(
            f'<rect x="{i * bw + 1:.1f}" y="{h - bh:.1f}" width="{max(bw - 2, 1):.1f}" height="{bh:.1f}" rx="1.5" '
            f'fill="{C_AMBER if hl else color}" opacity="{1 if hl else .6}"/>'
        )
    return f'<svg class="spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none">{"".join(rects)}</svg>'


def gauge_html(value, scale, pos_color, neg_color) -> str:
    if isnan(value) or scale <= 0:
        return '<div class="gauge"><div class="gmid"></div></div>'
    frac = min(abs(value) / scale, 1.0) * 50
    if value >= 0:
        fill = f'<div class="gfill" style="left:50%;width:{frac:.1f}%;background:{pos_color};box-shadow:0 0 12px {pos_color}"></div>'
    else:
        fill = f'<div class="gfill" style="right:50%;width:{frac:.1f}%;background:{neg_color};box-shadow:0 0 12px {neg_color}"></div>'
    return f'<div class="gauge"><div class="gmid"></div>{fill}</div>'


def track_html(lo, hi, marks) -> str:
    span = (hi - lo) or 1.0
    dots = "".join(
        f'<span class="dot" style="left:{min(max((v - lo) / span, 0), 1) * 100:.1f}%;background:{c};'
        f'box-shadow:0 0 10px {c}"></span>' for v, c in marks
    )
    return (f'<div class="track"><div class="rail"></div>{dots}'
            f'<div class="track-lbl"><span>{lo:.1f}%</span><span>{hi:.1f}%</span></div></div>')


def card(label, value, sub="", tone="", viz="", tag="") -> str:
    tag_html = f'<span class="kpi-tag">{esc(tag)}</span>' if tag else ""
    return (f'<div class="card kpi {tone}"><div class="kpi-top"><span class="kpi-label">{esc(label)}</span>{tag_html}</div>'
            f'<div class="kpi-value mono">{esc(value)}</div><div class="kpi-sub">{esc(sub)}</div>'
            f'<div class="kpi-viz">{viz}</div></div>')


def pill(text, tone="neu") -> str:
    return f'<span class="pill {tone}">{esc(text)}</span>'


# ---- charts (dark-themed Altair) ---------------------------------------------
def style_chart(chart):
    return (chart.configure(background="transparent")
            .configure_axis(labelColor=C_MUTED, titleColor=C_MUTED, gridColor="#16233a",
                            domainColor="#1b2940", tickColor="#1b2940",
                            labelFont="JetBrains Mono", titleFont="Space Grotesk")
            .configure_view(strokeWidth=0)
            .configure_legend(labelColor=C_MUTED, titleColor=C_MUTED, orient="top"))


def show(chart) -> None:
    st.altair_chart(style_chart(chart), theme=None)


def dist_chart(implied, consensus):
    dd = implied.dist.copy()
    tol = implied.step / 2
    dd["CPI print"] = dd["outcome"].map(lambda v: f"{v:.2f}")
    dd["vs consensus"] = np.where(dd["outcome"] > consensus + tol, "Hotter",
                                  np.where(dd["outcome"] < consensus - tol, "Cooler", "In line"))
    return alt.Chart(dd).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X("CPI print:N", sort=list(dict.fromkeys(dd["CPI print"])), title="CPI print (%)",
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y("probability:Q", title="Probability", axis=alt.Axis(format="%")),
        color=alt.Color("vs consensus:N", title=None,
                        scale=alt.Scale(domain=["Cooler", "In line", "Hotter"],
                                        range=[C_CYAN, C_VIOLET, C_AMBER])),
        tooltip=[alt.Tooltip("CPI print:N", title="Print (%)"),
                 alt.Tooltip("probability:Q", title="Probability", format=".1%"), "vs consensus"],
    ).properties(width="container", height=330)


def consensus_chart(consensus, expected):
    cdf = pd.DataFrame({"Source": ["Bloomberg consensus", "Kalshi expected"], "CPI (%)": [consensus, expected]})
    base = alt.Chart(cdf).encode(
        x=alt.X("Source:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("CPI (%):Q", title="CPI (%)"),
        color=alt.Color("Source:N", legend=None,
                        scale=alt.Scale(domain=["Bloomberg consensus", "Kalshi expected"],
                                        range=[C_VIOLET, C_CYAN])),
        tooltip=["Source", alt.Tooltip("CPI (%):Q", format=".3f")],
    )
    return (base.mark_bar(size=70, cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            + base.mark_text(dy=-10, fontWeight="bold", fontSize=13, color="#e6edf7")
            .encode(text=alt.Text("CPI (%):Q", format=".3f"))).properties(width="container", height=330)


def car_chart(eq, selected):
    cd = eq.dropna(subset=["Est. CAR (%)"])
    return alt.Chart(cd).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("Ticker:N", sort=alt.EncodingSortField(field="Est. CAR (%)", order="ascending"), title=None),
        x=alt.X("Est. CAR (%):Q", title="Estimated CAR (%)"),
        color=alt.condition("datum['Est. CAR (%)'] >= 0", alt.value(C_GREEN), alt.value(C_RED)),
        opacity=alt.condition(f"datum.Ticker == '{selected}'", alt.value(1.0), alt.value(0.45)),
        tooltip=["Ticker", "Name", "Group", alt.Tooltip("Est. CAR (%):Q", format=".3f")],
    ).properties(width="container", height=max(26 * len(cd) + 30, 120))


def perf_chart(hist, selected, tickers):
    frames = []
    for t in tickers:
        s = hist.get(t)
        if s is None or len(s) < 2:
            continue
        s = s.dropna()
        if len(s) < 2 or s.iloc[0] == 0:
            continue
        frames.append(pd.DataFrame({"Date": s.index, "Ticker": t, "Rebased": s.values / s.iloc[0] * 100.0}))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    enc = dict(x=alt.X("Date:T", title=None, axis=alt.Axis(format="%b %y")),
               y=alt.Y("Rebased:Q", scale=alt.Scale(zero=False), title="Rebased (start = 100)"))
    tip = ["Ticker", alt.Tooltip("Rebased:Q", format=".1f"), alt.Tooltip("Date:T", format="%b %d, %Y")]
    base = alt.Chart(df)
    others = (base.transform_filter(f"datum.Ticker != '{selected}'")
              .mark_line(strokeWidth=1, opacity=0.28, color="#64748b").encode(detail="Ticker:N", tooltip=tip, **enc))
    top = (base.transform_filter(f"datum.Ticker == '{selected}'")
           .mark_line(strokeWidth=2.6, color=C_CYAN).encode(tooltip=tip, **enc))
    return alt.layer(others, top).properties(width="container", height=330)


# ---- ETF table -----------------------------------------------------------------
def etf_table(eq: pd.DataFrame, selected: str) -> str:
    head = ("<tr><th>Ticker</th><th>Fund</th><th>Group</th><th class='num'>Price</th><th class='num'>Day</th>"
            "<th class='num'>γ₁</th><th class='num'>Est. CAR</th></tr>")
    rows = []
    for r in eq.to_dict("records"):
        price = f"${r['Price']:,.2f}" if not isnan(r["Price"]) else "—"
        day = pill(fnum(r["Day %"], 2, True, "%"), tone_of(r["Day %"])) if not isnan(r["Day %"]) else pill("—")
        car = pill(fnum(r["Est. CAR (%)"], 3, True, "%"), tone_of(r["Est. CAR (%)"]))
        cls = ' class="sel"' if r["Ticker"] == selected else ""
        rows.append(
            f'<tr{cls}><td><span class="sym">{esc(r["Ticker"])}</span></td>'
            f'<td>{esc(r["Name"])}</td><td class="sub-muted">{esc(r["Group"])}</td>'
            f'<td class="num">{price}</td><td class="num">{day}</td>'
            f'<td class="num">{fnum(r["gamma1"], 2, True)}</td><td class="num">{car}</td></tr>'
        )
    return f'<div class="tbl-wrap"><table class="tbl"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'


# ---- cached data ---------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def cached_markets(series: str) -> list:
    return core.fetch_open_markets(series)


@st.cache_data(ttl=60, show_spinner=False)
def cached_market(tickers: tuple):
    return core.fetch_market(list(tickers), "6mo")


@st.cache_data(ttl=120, show_spinner=False)
def cached_options_all(tickers: tuple, vix_tickers: tuple):
    return core.fetch_options_moves(list(tickers), vix_fallback_tickers=list(vix_tickers))


# ============================================================================
# Session state: coefficients
# ============================================================================
ss = st.session_state
if "gamma_base" not in ss:
    ss.gamma_base = core.default_gammas(GAMMA_FILE)  # Ticker, gamma1, source
    ss.gamma_override = {}                            # {ticker: user-typed value}
    ss.gamma_ver = 0
    ss.gamma_sig = None
if "qr_active" not in ss:
    ss.qr_active = core.default_sensitivity(SENS_FILE)  # Ticker, ols_m, ols_b, q05_m, q05_b, q95_m, q95_b
    ss.qr_ver = 0


def get_active() -> pd.DataFrame:
    a = ss.gamma_base.copy()
    for t, v in ss.gamma_override.items():
        m = a["Ticker"] == t
        a.loc[m, "gamma1"] = float(v)
        a.loc[m, "source"] = "user-edited"
    return a


# ============================================================================
# Sidebar: navigation rail + parameters
# ============================================================================
UNI = core.UNIVERSE
names = dict(zip(UNI.Ticker, UNI.Name))
tickers = list(UNI.Ticker)

NAV = {
    "▣  Dashboard": "dash",
    "◫  ETF universe": "etf",
    "◉  Kalshi market": "kalshi",
    "⚙  Coefficients": "coef",
    "☰  Methodology": "method",
}

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">K</div><div>'
        '<div class="brand-name">CPI SIGNAL</div><div class="brand-sub">Kalshi × ETF terminal</div></div></div>',
        unsafe_allow_html=True)
    st.markdown('<div class="side-label">Navigate</div>', unsafe_allow_html=True)
    nav_label = st.radio("Navigate", list(NAV), label_visibility="collapsed", key="section")
    page = NAV.get(nav_label, "dash")

    st.markdown('<div class="side-label">CPI parameters</div>', unsafe_allow_html=True)
    series_label = st.selectbox("Kalshi CPI series", list(core.CPI_SERIES) + ["Custom…"])
    if series_label == "Custom…":
        series = st.text_input("Series ticker", value="KXCPI",
                               help="The series ticker from the Kalshi market URL.").strip().upper()
    else:
        series = core.CPI_SERIES[series_label]

    markets, kalshi_error = [], None
    try:
        with st.spinner("Loading Kalshi markets…"):
            markets = cached_markets(series)
    except core.DataError as e:
        kalshi_error = str(e)
    except Exception as e:  # never let an unexpected error white-screen the app
        kalshi_error = f"Unexpected error talking to Kalshi: {e}"

    events = core.list_events(markets)
    event_ticker = None
    if not events.empty:
        labels = {r.event_ticker: f"{r.event_ticker} · closes {fmt_close(r.close_time)}"
                  for r in events.itertuples()}
        event_ticker = st.selectbox("CPI release", list(labels), format_func=labels.get,
                                    help="Defaults to the next release to close.")
    elif not kalshi_error:
        kalshi_error = (f"No open markets found for series “{series}”. "
                        "Check the ticker on kalshi.com, or pick a different series.")

    consensus = st.number_input("Bloomberg consensus (%)", min_value=-5.0, max_value=15.0,
                                value=0.20, step=0.10, format="%.2f",
                                help="Must match the series type (m/m vs y/y) selected above.")

    with st.expander("Advanced"):
        step_choice = st.selectbox("CPI rounding step", ["Auto", 0.1, 0.05, 0.01],
                                   help="BLS publishes CPI to one decimal. Auto infers it from the strike spacing.")
        max_spread_c = st.slider("Max bid-ask spread to trust (¢)", 5, 60, 20, 5,
                                 help="Wider than this, the app ignores the midpoint and uses the last traded price.")
        auto = st.toggle("Auto-refresh", value=False)
        refresh_s = st.select_slider("Refresh every (sec)", [60, 120, 300], value=60, disabled=not auto)

    if st.button("🔄  Refresh data now"):
        st.cache_data.clear()
        st.rerun()

if auto and st_autorefresh is not None:
    st_autorefresh(interval=refresh_s * 1000, key="auto_refresh")
elif auto:
    st.sidebar.caption("Auto-refresh needs the `streamlit-autorefresh` package.")

# ============================================================================
# Kalshi -> implied CPI
# ============================================================================
ladder = pd.DataFrame(columns=core.LADDER_COLUMNS)
implied = None
calc_error = None
if event_ticker and not kalshi_error:
    try:
        ladder = core.build_ladder(markets, event_ticker, max_spread_c / 100.0)
        implied = core.implied_distribution(ladder, None if step_choice == "Auto" else float(step_choice))
    except core.DataError as e:
        calc_error = str(e)
    except Exception as e:
        calc_error = f"Couldn't compute the implied distribution: {e}"

divergence = (implied.expected - consensus) if implied else None

# ============================================================================
# ETF market data
# ============================================================================
try:
    quotes, hist = cached_market(tuple(tickers))
except Exception:
    quotes, hist = pd.DataFrame({"Ticker": tickers}), {}
for c in ("Price", "Prev Close", "Day %", "As of"):
    if c not in quotes.columns:
        quotes[c] = np.nan

# ============================================================================
# Header, ticker tape, ETF selector
# ============================================================================
event_row = events[events.event_ticker == event_ticker].iloc[0] if event_ticker else None
chips = [f"SERIES <b>{esc(series or '—')}</b>"]
if event_row is not None:
    chips += [f"EVENT <b>{esc(event_row.event_ticker)}</b>", f"CLOSES <b>{esc(fmt_close(event_row.close_time))}</b>"]
chips += [f"UPDATED <b>{esc(now_et())}</b>"]
st.markdown(
    '<div class="topbar"><div class="crumb"><span class="live"></span>Kalshi CPI × ETF response '
    '<span>/ terminal</span></div><div>' + "".join(f'<span class="chip">{c}</span>' for c in chips) + "</div></div>",
    unsafe_allow_html=True)

tape_items = []
for r in quotes.to_dict("records"):
    if isnan(r.get("Price")):
        tape_items.append(f'<span class="tk"><b>{esc(r["Ticker"])}</b><span class="px">—</span></span>')
        continue
    d = r.get("Day %")
    arrow = "▲" if (not isnan(d) and d > 0) else ("▼" if (not isnan(d) and d < 0) else "•")
    tape_items.append(
        f'<span class="tk"><b>{esc(r["Ticker"])}</b><span class="px">{r["Price"]:,.2f}</span>'
        f'<span class="{tone_of(d)}">{arrow} {fnum(abs(d) if not isnan(d) else float("nan"), 2, False, "%")}</span></span>')
tape = "".join(tape_items)
st.markdown(f'<div class="tape"><div class="tape-inner">{tape}{tape}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="strip-label">Select ETF</div>', unsafe_allow_html=True)
selected = st.radio("ETF", tickers, horizontal=True, label_visibility="collapsed", key="etf")

banner = st.container()


# ============================================================================
# Pages
# ============================================================================
def build_eq(active: pd.DataFrame) -> pd.DataFrame:
    eq = (UNI.merge(quotes[["Ticker", "Price", "Day %", "As of"]], on="Ticker", how="left")
             .merge(active[["Ticker", "gamma1", "source"]], on="Ticker", how="left"))
    eq["Est. CAR (%)"] = [core.estimated_car(divergence, g) for g in eq["gamma1"]]
    return eq


def odds_cards(implied):
    pv = core.prob_vs_consensus(implied.dist, consensus, implied.step)
    c = st.columns(4, gap="small")

    def odds(label, p, color, sub):
        return (f'<div class="card kpi"><div class="kpi-top"><span class="kpi-label">{esc(label)}</span></div>'
                f'<div class="kpi-value mono">{p*100:.0f}%</div><div class="kpi-sub">{esc(sub)}</div>'
                f'<div class="kpi-viz"><div class="odds-bar"><div class="odds-fill" '
                f'style="width:{p*100:.1f}%;background:{color};box-shadow:0 0 12px {color}"></div></div></div></div>')

    c[0].markdown(odds("Hotter than consensus", pv["above"], C_AMBER, "P(print > consensus)"), unsafe_allow_html=True)
    c[1].markdown(odds("In line", pv["inline"], C_VIOLET, "P(print = consensus)"), unsafe_allow_html=True)
    c[2].markdown(odds("Cooler than consensus", pv["below"], C_CYAN, "P(print < consensus)"), unsafe_allow_html=True)
    c[3].markdown(card("Implied σ", f"{implied.std:.3f} pp", "spread of Kalshi distribution",
                       viz=minibars_svg(implied.dist["probability"].tolist(), color=C_VIOLET)),
                  unsafe_allow_html=True)


# ============================================================================
# Institutional Tail-Risk & Relative Value Monitor
# ============================================================================
def range_bar_html(lo, mid, hi, scale) -> str:
    def pos(v):
        return min(max((v / scale + 1.0) / 2.0 * 100.0, 0.0), 100.0)
    a, b, m = pos(lo), pos(hi), pos(mid)
    return (f'<div class="rng"><div class="rng-mid"></div>'
            f'<div class="rng-seg" style="left:{a:.1f}%;width:{max(b - a, 0.8):.1f}%"></div>'
            f'<div class="rng-dot" style="left:{m:.1f}%"></div></div>')


def banner_html(tkr: str, state: str, detail: str, picked: bool = False) -> str:
    text = {"go": "GREEN LIGHT: Protection Underpriced — Hedge Recommended",
            "stand": "STAND DOWN: Options Already Price In Shock"}.get(state, "NO SIGNAL: required data unavailable")
    return (f'<div class="banner {state} compact{" pick" if picked else ""}"><span class="banner-tk">{esc(tkr)}</span><div>'
            f'<div class="banner-text">{esc(text)}</div><div class="banner-detail">{esc(detail)}</div></div></div>')


def options_sub(om) -> str:
    if om.source in ("options", "options_last"):
        basis = "mid" if om.source == "options" else "last trade"
        return f"ATM {om.strike:g} straddle ({basis}) · exp {om.expiry} · {om.dte} DTE"
    if om.source == "vix_fallback":
        return "fallback: ^VIX ÷ √252 (1σ daily)"
    return "options and VIX data unavailable"


def page_tail_monitor(eq):
    st.write("")
    st.markdown(
        '<div class="section-h">Institutional Tail-Risk &amp; Relative Value Monitor</div>'
        '<div class="section-p">Cleaner Kalshi tail metrics, an options-market price check, and a 90% '
        'prediction interval per ETF. Additive: the metrics above are unchanged.</div>', unsafe_allow_html=True)

    # ---- 1. editable sensitivity parameters ------------------------------------------------
    with st.expander("Edit Quantile & OLS Sensitivity Parameters"):
        st.caption("Model per ETF: **response (%) = m × Divergence (pp) + b**, evaluated at the current divergence "
                   "(Kalshi expected − consensus). Edits update this whole section instantly and last for your "
                   "browser session; to change the defaults permanently, edit `sensitivity_params.csv` in the repo.")
        key = f"qr_editor_{ss.qr_ver}"
        if key not in ss:                      # fresh widget (first visit / returning to this page)
            ss.qr_seed = ss.qr_active.copy()   # seed from the latest edits so nothing is lost
        num = lambda label, help_: st.column_config.NumberColumn(label, help=help_, format="%.7f", step=0.0001)
        edited = st.data_editor(
            ss.qr_seed, key=key, hide_index=True, num_rows="fixed", disabled=["Ticker"],
            column_config={
                "Ticker": st.column_config.TextColumn("ETF"),
                "ols_m": num("OLS m", "Conditional mean (OLS) slope"),
                "ols_b": num("OLS b", "Conditional mean (OLS) intercept"),
                "q05_m": num("5th pct m", "q = 0.05 slope"),
                "q05_b": num("5th pct b", "q = 0.05 intercept"),
                "q95_m": num("95th pct m", "q = 0.95 slope"),
                "q95_b": num("95th pct b", "q = 0.95 intercept"),
            })
        ss.qr_active = core.normalize_sensitivity(edited, fallback=ss.qr_active)
        b1, b2, _ = st.columns([1, 1, 3])
        if b1.button("↺  Reset to defaults", key="qr_reset"):
            ss.qr_active = core.default_sensitivity(SENS_FILE)
            ss.qr_ver += 1
            st.rerun()
        b2.download_button("⬇  Download CSV", ss.qr_active.to_csv(index=False),
                           file_name="sensitivity_params.csv", mime="text/csv", key="qr_dl")
    params = ss.qr_active

    # ---- 2. clean Kalshi tail metrics -------------------------------------------------------
    tm, tm_err = None, None
    if event_ticker and not kalshi_error:
        try:
            tm = core.tail_metrics(markets, event_ticker, consensus, TAIL_MIN_OI, TAIL_MAX_SPREAD, TAIL_THRESHOLD,
                                   None if step_choice == "Auto" else float(step_choice))
        except core.DataError as e:
            tm_err = str(e)
        except Exception as e:
            tm_err = f"Couldn't compute tail metrics: {e}"
    else:
        tm_err = "Kalshi data is unavailable (see the message at the top)."

    st.markdown('<div class="strip-label" style="margin-top:14px">Clean Kalshi tail metrics</div>', unsafe_allow_html=True)
    k1, k2 = st.columns(2, gap="small")
    if tm is not None:
        k1.markdown(card(
            "Market uncertainty (σ)", f"{tm.std:.3f} pp", f"{tm.n_used} of {tm.n_total} strikes pass liquidity filters",
            tag="KALSHI σ", viz=minibars_svg(tm.dist["probability"].tolist(), color=C_VIOLET)), unsafe_allow_html=True)
        k2.markdown(card(
            "Tail shock probability", f"{tm.tail_prob * 100:.1f}%",
            f"↑ {tm.prob_up * 100:.1f}%  ·  ↓ {tm.prob_down * 100:.1f}%  (|CPI − consensus| ≥ {TAIL_THRESHOLD:.2f})",
            tone="amb", tag="TAIL",
            viz=f'<div class="odds-bar"><div class="odds-fill" style="width:{min(tm.tail_prob * 100, 100):.1f}%;'
                f'background:{C_AMBER};box-shadow:0 0 12px {C_AMBER}"></div></div>'), unsafe_allow_html=True)
        notes = [f"Filters: open interest ≥ {TAIL_MIN_OI}, bid-ask spread ≤ {TAIL_MAX_SPREAD * 100:.0f}¢, "
                 "probabilities forced non-increasing across strikes."]
        if not tm.oi_filter_applied:
            notes.append("Kalshi returned no open-interest data, so the open-interest filter was skipped.")
        if tm.monotone_adjusted:
            notes.append("Some strike prices were not monotone and were smoothed.")
        if not tm.covers_down:
            notes.append("Liquid strikes do not reach the lower tail threshold: the downside tail may be understated.")
        if not tm.covers_up:
            notes.append("Liquid strikes do not reach the upper tail threshold: the upside tail may be understated.")
        st.caption("  \n".join(notes))
    else:
        k1.markdown(card("Market uncertainty (σ)", "—", "unavailable", tag="KALSHI σ"), unsafe_allow_html=True)
        k2.markdown(card("Tail shock probability", "—", "unavailable", tone="amb", tag="TAIL"), unsafe_allow_html=True)
        st.warning(tm_err)

    # ---- 3. options market price check: ALL ETFs -------------------------------------------
    st.markdown('<div class="strip-label" style="margin-top:16px">Options market price check · relative value filter · '
                'all 9 ETFs</div>', unsafe_allow_html=True)
    try:
        opts = cached_options_all(tuple(tickers), tuple(sorted(VIX_FALLBACK_TICKERS)))
    except Exception:
        opts = {}
    ols_slope = params.set_index("Ticker")["ols_m"].to_dict()
    cond = None if tm is None else (0.0 if isnan(tm.cond_shock) else float(tm.cond_shock))
    nan = float("nan")
    sig = []
    for tkr in tickers:
        om = opts.get(tkr) or core.OptionsMove(tkr, nan, "unavailable", note="lookup failed")
        slope = abs(float(ols_slope.get(tkr, nan)))
        kal = nan if cond is None else slope * cond
        ready = not (isnan(kal) or isnan(om.move_pct) or om.move_pct <= 0)
        ratio = kal / om.move_pct if ready else nan
        state = "na" if not ready else ("go" if kal >= HEDGE_TRIGGER * om.move_pct else "stand")
        proxy = om.source == "vix_fallback" and tkr not in EQUITY_VOL_ETFS
        if ready:
            detail = (f"Kalshi tail move {kal:.2f}% vs options-implied {om.move_pct:.2f}%  →  {ratio:.2f}× "
                      f"(trigger {HEDGE_TRIGGER:.2f}×)" + ("  ·  VIX proxy" if proxy else ""))
        else:
            detail = om.note or "waiting on Kalshi or options data"
        sig.append(dict(t=tkr, om=om, slope=slope, kal=kal, ratio=ratio, state=state, proxy=proxy, detail=detail))

    n_go = sum(1 for r in sig if r["state"] == "go")
    n_na = sum(1 for r in sig if r["state"] == "na")
    st.markdown(f'<div class="sub-muted" style="margin:2px 0 8px">{pill(f"{n_go} of {len(sig)} GREEN LIGHT", "pos" if n_go else "neu")} '
                f'&nbsp;{pill(f"{len(sig) - n_go - n_na} STAND DOWN", "amb")} '
                + (f'&nbsp;{pill(f"{n_na} NO SIGNAL", "neu")}' if n_na else "") + '</div>', unsafe_allow_html=True)
    for i in range(0, len(sig), 3):
        grid = st.columns(3, gap="small")
        for col, r in zip(grid, sig[i:i + 3]):
            col.markdown(banner_html(r["t"], r["state"], r["detail"], picked=(r["t"] == selected)), unsafe_allow_html=True)

    with panel("optsig"):
        panel_title("Kalshi tail move vs. options-implied move", "▤")
        head = ("<tr><th>ETF</th><th class='num'>Options-implied</th><th>Basis</th><th class='num'>|OLS m|</th>"
                "<th class='num'>Kalshi tail move</th><th class='num'>Kalshi ÷ options</th><th>Signal</th></tr>")
        rows = []
        for r in sig:
            om = r["om"]
            if om.source in ("options", "options_last"):
                basis = f"ATM {om.strike:g} straddle · exp {esc(om.expiry)} · {om.dte}d" + (" · last trade" if om.source == "options_last" else "")
            elif om.source == "vix_fallback":
                basis = "^VIX ÷ √252" + (" " + pill("proxy", "amb") if r["proxy"] else "")
            else:
                basis = '<span class="sub-muted">unavailable</span>'
            signal = {"go": pill("GREEN LIGHT", "pos"), "stand": pill("STAND DOWN", "amb")}.get(r["state"], pill("NO SIGNAL", "neu"))
            sel_cls = ' class="sel"' if r["t"] == selected else ""
            rows.append(
                f'<tr{sel_cls}><td><span class="sym">{esc(r["t"])}</span></td>'
                f'<td class="num">{fnum(om.move_pct, 2, False, "%")}</td><td class="sub-muted">{basis}</td>'
                f'<td class="num">{fnum(r["slope"], 2)}</td><td class="num">{fnum(r["kal"], 2, False, "%")}</td>'
                f'<td class="num">{fnum(r["ratio"], 2, False, "×")}</td><td>{signal}</td></tr>')
        st.markdown(f'<div class="tbl-wrap"><table class="tbl"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>',
                    unsafe_allow_html=True)
    st.caption("Kalshi tail move = |OLS slope m| × the average CPI miss (pp) given a tail shock (|CPI − consensus| ≥ "
               f"{TAIL_THRESHOLD:.2f}), from the cleaned Kalshi distribution. GREEN LIGHT when it is at least "
               f"{(HEDGE_TRIGGER - 1) * 100:.0f}% above the options-implied move. The ^VIX ÷ √252 fallback is a 1σ daily "
               "equity-volatility move (an ATM straddle is roughly 0.8σ); for non-equity ETFs it is only a rough "
               "proxy and is flagged “proxy”. Limit it with `VIX_FALLBACK_TICKERS` in `app.py`.")

    # ---- 4. dynamic 90% prediction interval table -------------------------------------------
    with panel("pi"):
        panel_title("Dynamic 90% prediction interval · estimated ETF response (%)", "▤")
        if divergence is None:
            st.info("Needs the Kalshi divergence (Kalshi expected − consensus) — see the message at the top.")
        else:
            pi = core.prediction_intervals(params, divergence).merge(UNI[["Ticker", "Name"]], on="Ticker", how="left")
            vals = pi[["lower", "upper"]].to_numpy(dtype=float)
            scale = max(float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 0.1, 0.1)
            head = ("<tr><th>ETF</th><th>Fund</th><th class='num'>5th pct lower</th><th class='num'>Cond. mean (OLS)</th>"
                    "<th class='num'>95th pct upper</th><th>Range</th><th>Guardrail</th></tr>")
            rows = []
            for r in pi.to_dict("records"):
                sel_cls = ' class="sel"' if r["Ticker"] == selected else ""
                rows.append(
                    f'<tr{sel_cls}><td><span class="sym">{esc(r["Ticker"])}</span></td><td>{esc(r["Name"])}</td>'
                    f'<td class="num">{pill(fnum(r["lower"], 3, True, "%"), tone_of(r["lower"]))}</td>'
                    f'<td class="num">{pill(fnum(r["y_ols"], 3, True, "%"), "cy")}</td>'
                    f'<td class="num">{pill(fnum(r["upper"], 3, True, "%"), tone_of(r["upper"]))}</td>'
                    f'<td>{range_bar_html(r["lower"], r["y_ols"], r["upper"], scale)}</td>'
                    f'<td>{pill("re-ordered", "amb") if r["adjusted"] else pill("ok")}</td></tr>')
            st.markdown(f'<div class="tbl-wrap"><table class="tbl"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>',
                        unsafe_allow_html=True)
            st.caption(f"Evaluated at the current divergence of {divergence:+.3f} pp. Non-crossing guardrail: "
                       "lower = min(5th pct, OLS, 95th pct), upper = max(5th pct, OLS, 95th pct); "
                       "“re-ordered” means the raw lines had crossed.")



def page_dashboard(eq, sel):
    series_short = series_label.split("(")[0].strip() if series_label != "Custom…" else series
    cols = st.columns(5, gap="small")
    lo = float(ladder["strike"].min()) if len(ladder) else consensus - 0.3
    hi = float(ladder["strike"].max()) if len(ladder) else consensus + 0.3

    cols[0].markdown(card(
        "Bloomberg consensus", f"{consensus:.2f}%", series_short, tag="INPUT",
        viz=track_html(lo, hi, [(consensus, C_VIOLET)] + ([(implied.expected, C_CYAN)] if implied else []))),
        unsafe_allow_html=True)

    if implied:
        near = int(np.argmin(np.abs(implied.dist["outcome"].to_numpy() - consensus)))
        cols[1].markdown(card(
            "Kalshi expected CPI", f"{implied.expected:.3f}%", f"{implied.n_strikes} strikes · σ {implied.std:.2f}",
            tag="LIVE", viz=minibars_svg(implied.dist["probability"].tolist(), highlight=near)),
            unsafe_allow_html=True)
    else:
        cols[1].markdown(card("Kalshi expected CPI", "—", "no data", tag="LIVE"), unsafe_allow_html=True)

    cols[2].markdown(card(
        "Divergence", fnum(divergence, 3, True, " pp"), "Kalshi − consensus", tone="amb", tag="Δ",
        viz=gauge_html(divergence, 0.30, C_AMBER, C_CYAN)), unsafe_allow_html=True)

    scale = max(0.10, float(np.nanmax(np.abs(eq["Est. CAR (%)"].to_numpy(dtype=float)))) if eq["Est. CAR (%)"].notna().any() else 0.10)
    tone = tone_of(sel["Est. CAR (%)"])
    cols[3].markdown(card(
        f"Est. CAR · {selected}", fnum(sel["Est. CAR (%)"], 3, True, "%"), f"γ₁ = {fnum(sel['gamma1'], 2, True)}",
        tone=tone if tone != "neu" else "", tag=selected,
        viz=gauge_html(sel["Est. CAR (%)"], scale, C_GREEN, C_RED)), unsafe_allow_html=True)

    h = hist.get(selected)
    px = f"${sel['Price']:,.2f}" if not isnan(sel["Price"]) else "—"
    cols[4].markdown(card(
        f"{selected} price", px, f"{fnum(sel['Day %'], 2, True, '%')} day",
        tone=tone_of(sel["Day %"]) if tone_of(sel["Day %"]) != "neu" else "", tag="6M",
        viz=spark_svg(h.tail(90).tolist(), C_GREEN if tone_of(sel["Day %"]) != "neg" else C_RED, uid="px") if h is not None else ""),
        unsafe_allow_html=True)

    st.write("")
    if implied is not None:
        left, right = st.columns([3, 2], gap="small")
        with left:
            with panel("dist"):
                panel_title("Kalshi-implied distribution of the CPI print")
                show(dist_chart(implied, consensus))
        with right:
            with panel("cons"):
                panel_title("Consensus vs. Kalshi expectation")
                show(consensus_chart(consensus, implied.expected))
        st.write("")
        odds_cards(implied)
    else:
        st.info("Waiting on Kalshi data — see the message above. ETF quotes are still available below.")

    st.write("")
    with panel("table"):
        panel_title("ETF universe · estimated CAR", "▤")
        st.markdown(etf_table(eq, selected), unsafe_allow_html=True)

    page_tail_monitor(eq)


def page_etf(eq, sel):
    st.markdown(
        f'<div class="card"><div class="etf-head"><div><div class="etf-sym mono">{esc(selected)}</div>'
        f'<div class="etf-name">{esc(sel["Name"])}</div></div>'
        f'<div>{pill(sel["Group"], "cy")}<div class="etf-exp" style="margin-top:8px">{esc(sel["Exposure"])}</div></div></div></div>',
        unsafe_allow_html=True)
    st.write("")
    h = hist.get(selected)
    c = st.columns(4, gap="small")
    tone_d = tone_of(sel["Day %"])
    c[0].markdown(card("Last price", f"${sel['Price']:,.2f}" if not isnan(sel["Price"]) else "—",
                       f"as of {sel['As of']}" if isinstance(sel["As of"], str) and sel["As of"] else "", tag="6M",
                       viz=spark_svg(h.tail(90).tolist(), C_GREEN if tone_d != "neg" else C_RED, uid="pe") if h is not None else ""),
                  unsafe_allow_html=True)
    c[1].markdown(card("Day change", fnum(sel["Day %"], 2, True, "%"), "vs prior close",
                       tone=tone_d if tone_d != "neu" else "",
                       viz=gauge_html(sel["Day %"], 3.0, C_GREEN, C_RED)), unsafe_allow_html=True)
    c[2].markdown(card("Sensitivity γ₁", fnum(sel["gamma1"], 3, True), f"source: {sel['source']}",
                       viz=gauge_html(sel["gamma1"], 2.0, C_CYAN, C_VIOLET)), unsafe_allow_html=True)
    tone_c = tone_of(sel["Est. CAR (%)"])
    scale = max(0.10, float(np.nanmax(np.abs(eq["Est. CAR (%)"].to_numpy(dtype=float)))) if eq["Est. CAR (%)"].notna().any() else 0.10)
    c[3].markdown(card("Est. CAR", fnum(sel["Est. CAR (%)"], 3, True, "%"), "divergence × γ₁",
                       tone=tone_c if tone_c != "neu" else "", viz=gauge_html(sel["Est. CAR (%)"], scale, C_GREEN, C_RED)),
                  unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([3, 2], gap="small")
    with left:
        with panel("perf"):
            panel_title(f"6-month price performance · {selected} vs. universe (rebased, unadjusted close)")
            pc = perf_chart(hist, selected, tickers)
            if pc is None:
                st.info("Price history is unavailable right now (Yahoo may be throttling). Try Refresh in a minute.")
            else:
                show(pc)
    with right:
        with panel("carbars"):
            panel_title("Estimated CAR across the universe")
            if eq["Est. CAR (%)"].notna().any():
                show(car_chart(eq, selected))
            else:
                st.info("Needs the Kalshi expectation — see the message at the top.")
    st.write("")
    with panel("table2"):
        panel_title("ETF universe", "▤")
        sort_by = st.radio("Sort by", ["Universe order", "Est. CAR (most negative first)", "Day change"],
                           horizontal=True, label_visibility="collapsed", key="sort_by")
        view = eq.copy()
        if sort_by.startswith("Est. CAR"):
            view = view.sort_values("Est. CAR (%)", na_position="last")
        elif sort_by == "Day change":
            view = view.sort_values("Day %", ascending=False, na_position="last")
        st.markdown(etf_table(view, selected), unsafe_allow_html=True)


def page_kalshi():
    st.markdown('<div class="section-h">Kalshi market</div>'
                '<div class="section-p">The contract ladder behind the expected CPI, plus data-quality notes.</div>',
                unsafe_allow_html=True)
    if implied is None:
        st.info("No usable Kalshi data to display yet.")
        return
    used = ladder.dropna(subset=["prob"])
    n_last = int((used["price_source"] == "last").sum())
    notes = [f"**{len(used)}** of **{len(ladder)}** contracts had a usable price.",
             f"Rounding step used: **{implied.step:g}**."]
    if n_last:
        notes.append(f"**{n_last}** contract(s) had a wide bid-ask spread (>{max_spread_c}¢), "
                     "so the last traded price was used instead of the midpoint.")
    if implied.monotone_adjusted:
        notes.append("Prices were **not perfectly monotone** (a higher strike priced above a lower one); "
                     "they were smoothed so probabilities stay valid.")
    thin = int((used["volume"] < 50).sum())
    if thin:
        notes.append(f"{thin} contract(s) have very low volume (<50): treat the expectation as noisy.")
    with panel("notes"):
        panel_title("Data quality", "✓")
        st.markdown("\n".join(f"- {n}" for n in notes))

    st.write("")
    with panel("ladder"):
        panel_title("Contract ladder · P(CPI above strike)", "▤")
        lad = ladder.copy()
        lad["P(> strike) %"] = lad["prob"] * 100
        lad["Spread (¢)"] = lad["spread"] * 100
        lad["Status"] = np.where(lad["prob"].isna(), "excluded (no usable price)", "used")
        st.dataframe(
            lad[["ticker", "strike", "P(> strike) %", "price_source", "Spread (¢)", "volume", "open_interest", "Status"]]
            .rename(columns={"ticker": "Contract", "strike": "Strike (%)", "price_source": "Price source",
                             "volume": "Volume", "open_interest": "Open interest"}),
            hide_index=True,
            column_config={"P(> strike) %": st.column_config.NumberColumn(format="%.1f"),
                           "Spread (¢)": st.column_config.NumberColumn(format="%.0f"),
                           "Strike (%)": st.column_config.NumberColumn(format="%.2f")})
    st.write("")
    with panel("impl"):
        panel_title("Implied distribution of the print", "▤")
        out = implied.dist.rename(columns={"outcome": "CPI print (%)", "probability": "Probability"})
        out["Probability"] = out["Probability"] * 100
        st.dataframe(out, hide_index=True,
                     column_config={"Probability": st.column_config.NumberColumn("Probability (%)", format="%.1f"),
                                    "CPI print (%)": st.column_config.NumberColumn(format="%.2f")})


def page_coefficients():
    st.markdown('<div class="section-h">Sensitivity coefficients γ₁</div>'
                '<div class="section-p">Estimated % CAR per +1 percentage-point divergence, per ETF. '
                'Edits apply instantly across the dashboard. Make them permanent by editing '
                '<span class="mono">gammas.csv</span> in your repo.</div>', unsafe_allow_html=True)

    with panel("gup"):
        panel_title("Load from file", "⇪")
        up = st.file_uploader("Upload coefficients (CSV with columns: ticker, gamma1)", type=["csv"], key="gamma_upload")
        if up is not None:
            sig = (up.name, up.size)
            if ss.gamma_sig != sig:
                ss.gamma_sig = sig
                try:
                    ss.gamma_base = core.normalize_gammas(pd.read_csv(up))
                    ss.gamma_override = {}
                    ss.gamma_ver += 1
                    st.success("Coefficients loaded. ETFs missing from your file kept the placeholder value.")
                except Exception as e:
                    st.error(f"Couldn't use that file: {e}")
        else:
            ss.gamma_sig = None

    st.write("")
    with panel("gedit"):
        panel_title("Edit coefficients", "✎")
        base = ss.gamma_base
        grid = st.columns(3, gap="medium")
        for i, t in enumerate(tickers):
            base_g = float(base.loc[base.Ticker == t, "gamma1"].iloc[0])
            cur = float(ss.gamma_override.get(t, base_g))
            with grid[i % 3]:
                v = st.number_input(f"{t} · {names[t]}", value=cur, step=0.05, format="%.3f",
                                    key=f"gin_{ss.gamma_ver}_{t}")
            if v is not None and not isnan(v):
                if abs(float(v) - base_g) > 1e-12:
                    ss.gamma_override[t] = float(v)
                else:
                    ss.gamma_override.pop(t, None)

        b1, b2, _ = st.columns([1, 1, 2])
        if b1.button("↺  Reset to file values"):
            ss.gamma_override = {}
            ss.gamma_ver += 1
            st.rerun()
        active_now = get_active()
        b2.download_button("⬇  Download CSV", active_now.rename(columns={"Ticker": "ticker"}).to_csv(index=False),
                           file_name="gammas.csv", mime="text/csv")


def page_method():
    st.markdown('<div class="section-h">Methodology</div>'
                '<div class="section-p">How every number on the dashboard is built.</div>', unsafe_allow_html=True)
    with panel("m1"):
        panel_title("1 · Kalshi-implied expected CPI", "∑")
        st.markdown(
            "Kalshi lists contracts of the form *“CPI above X%?”*. A YES price of 60¢ is read as a 60% probability. "
            "CPI is published rounded to one decimal, so *above 0.2* means a print of 0.3 or higher. "
            "Differencing the ladder gives the probability of each print:")
        st.latex(r"P(X=t_{i+1}) = s_i - s_{i+1},\quad P(X=t_1)=1-s_1,\quad P(X=t_n+\Delta)=s_n")
        st.latex(r"E^{\text{Kalshi}}[\text{CPI}] = \sum_k x_k\,p_k")
    st.write("")
    with panel("m2"):
        panel_title("2 · Divergence and estimated CAR", "Δ")
        st.latex(r"\text{Divergence} = E^{\text{Kalshi}}[\text{CPI}] - \text{Consensus}")
        st.latex(r"\widehat{\text{CAR}}_i = \gamma_{1,i}\times \text{Divergence}")
        st.markdown("Quotes and price history are **display only**; they never enter the CAR calculation.")
    st.write("")
    with panel("m3"):
        panel_title("3 · Assumptions & limits", "!")
        st.markdown(
            """
- **Price handling.** Midpoint of bid/ask when the spread is tight; the last traded price when the book is wide
  (a 3¢ bid / 97¢ ask midpoint of 50% would be meaningless). Unusable contracts are excluded and flagged.
- Probabilities are forced to be non-increasing in the strike, then normalized to sum to 1.
- Mass below the lowest strike sits at that strike; mass above the highest strike sits at highest + one step.
- Thinly traded strikes are noisy; check the *Kalshi market* page.
- γ₁ is a **fixed linear** sensitivity that **you** supply. The app applies it and does not estimate it.
- Quotes come from Yahoo Finance via `yfinance`, may be delayed ~15 min, and the performance chart uses
  **unadjusted** closes (distributions are not reinvested).
- Estimate γ₁ by regressing each ETF's event-window abnormal return on the divergence across CPI releases:
  `CAR_i,e = γ₀ + γ₁·Divergence_e + u`.
""")


# ============================================================================
# Router
# ============================================================================
if page == "coef":
    page_coefficients()  # may update ss.gamma_override, so evaluate coefficients afterwards

active = get_active()
eq = build_eq(active)
sel = eq[eq.Ticker == selected].iloc[0]

if page == "dash":
    page_dashboard(eq, sel)
elif page == "etf":
    page_etf(eq, sel)
elif page == "kalshi":
    page_kalshi()
elif page == "method":
    page_method()

# ---- banners (rendered into the slot above the page content) --------------------
n_placeholder = int((active["source"] == "placeholder").sum())
with banner:
    if kalshi_error:
        st.error(f"**Kalshi data unavailable.** {kalshi_error}")
    if calc_error:
        st.warning(f"**Couldn't build the Kalshi expectation.** {calc_error}")
    if n_placeholder:
        st.warning(
            f"**{n_placeholder} of {len(active)} γ₁ coefficients are placeholders (−0.50), not estimates.** "
            "CAR figures are illustrative until you load your own on the *Coefficients* page.")
    if implied is not None and len(ladder):
        lo_k, hi_k = float(ladder["strike"].min()), float(ladder["strike"].max())
        if consensus < lo_k - 3 * implied.step or consensus > hi_k + 3 * implied.step:
            st.warning(
                f"Your consensus ({consensus:.2f}%) is far outside Kalshi's strike range ({lo_k:.1f}%–{hi_k:.1f}%). "
                "Make sure it matches the selected series (m/m vs y/y, headline vs core).")
    if quotes["Price"].isna().all():
        st.warning("Live quotes are unavailable right now (Yahoo Finance may be throttling). "
                   "Kalshi/CAR figures are unaffected. Try *Refresh data now* in a minute.")

st.caption("For academic research and education only; not investment advice. "
           "Data: Kalshi public API and Yahoo Finance (may be delayed or incomplete).")
