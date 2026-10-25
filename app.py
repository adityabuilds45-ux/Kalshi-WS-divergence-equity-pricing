"""
Kalshi CPI Divergence & Bank CAR Dashboard
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

st.set_page_config(
    page_title="Kalshi CPI Divergence | Bank CAR",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling (works in both light and dark mode: translucent surfaces, fixed accents)
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
.block-container {padding-top: 1.5rem; max-width: 1280px;}
.hero {background: linear-gradient(135deg,#0b1f3f 0%,#1d4ed8 100%);
       border-radius: 18px; padding: 26px 32px; margin-bottom: 14px;}
.hero-title {color:#fff; font-size: 1.85rem; font-weight: 750; letter-spacing:-0.02em; line-height:1.2;}
.hero-sub {color:#cbd5e1; font-size: .98rem; margin: 6px 0 14px;}
.pill {display:inline-block; padding:4px 12px; margin:0 8px 6px 0; border-radius:999px;
       background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.2);
       color:#fff; font-size:.78rem; font-weight:500;}
.kpi {border:1px solid rgba(128,128,128,.25); border-left:5px solid #2563eb; border-radius:14px;
      padding:16px 18px; background:rgba(128,128,128,.06); height:100%;}
.kpi.pos {border-left-color:#16a34a;} .kpi.neg {border-left-color:#dc2626;}
.kpi.amber {border-left-color:#f59e0b;} .kpi.gray {border-left-color:#94a3b8;}
.kpi-label {font-size:.74rem; text-transform:uppercase; letter-spacing:.07em; opacity:.7; font-weight:650;}
.kpi-value {font-size:1.85rem; font-weight:750; line-height:1.15; margin-top:4px;}
.kpi.sm .kpi-value {font-size:1.35rem;}
.kpi-sub {font-size:.8rem; opacity:.65; margin-top:3px;}
.section-title {font-size:1.05rem; font-weight:650; margin:20px 0 8px;}
.stTabs [data-baseweb="tab-list"] {gap:6px;}
.stTabs [data-baseweb="tab"] {padding:8px 16px; font-weight:600;}
footer {visibility:hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x)) or bool(pd.isna(x))


def fnum(x, d=2, signed=False, suffix="") -> str:
    if isnan(x):
        return "—"
    return f"{x:+.{d}f}{suffix}" if signed else f"{x:.{d}f}{suffix}"


def kpi(label: str, value: str, sub: str = "", tone: str = "", small: bool = False) -> str:
    cls = " ".join(c for c in ("kpi", tone, "sm" if small else "") if c)
    return (f'<div class="{cls}"><div class="kpi-label">{html.escape(label)}</div>'
            f'<div class="kpi-value">{html.escape(value)}</div>'
            f'<div class="kpi-sub">{html.escape(sub)}</div></div>')


def car_tone(x) -> str:
    if isnan(x) or abs(x) < 1e-12:
        return "gray"
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
        return pd.Timestamp.now(tz="America/New_York").strftime("%b %d, %I:%M:%S %p ET")
    except Exception:
        return dt.datetime.now(dt.timezone.utc).strftime("%b %d, %H:%M:%S UTC")


@st.cache_data(ttl=60, show_spinner=False)
def cached_markets(series: str) -> list:
    return core.fetch_open_markets(series)


@st.cache_data(ttl=60, show_spinner=False)
def cached_quotes(tickers: tuple) -> pd.DataFrame:
    return core.fetch_quotes(list(tickers))


# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------
UNI = core.UNIVERSE
names = dict(zip(UNI.Ticker, UNI.Name))
tickers = list(UNI.Ticker)

with st.sidebar:
    st.markdown("### ⚙️ Controls")

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

    selected = st.selectbox("Equity to analyze", tickers, index=tickers.index("WAL"),
                            format_func=lambda t: f"{t} — {names[t]}")

    with st.expander("Advanced"):
        step_choice = st.selectbox("CPI rounding step", ["Auto", 0.1, 0.05, 0.01],
                                   help="BLS publishes CPI to one decimal. Auto infers it from the strike spacing.")
        max_spread_c = st.slider("Max bid-ask spread to trust (¢)", 5, 60, 20, 5,
                                 help="Wider than this, the app ignores the midpoint and uses the last traded price.")
        auto = st.toggle("Auto-refresh", value=False)
        refresh_s = st.select_slider("Refresh every (sec)", [60, 120, 300], value=60, disabled=not auto)

    if st.button("🔄 Refresh data now"):
        st.cache_data.clear()
        st.rerun()

if auto and st_autorefresh is not None:
    st_autorefresh(interval=refresh_s * 1000, key="auto_refresh")
elif auto:
    st.sidebar.caption("Auto-refresh needs the `streamlit-autorefresh` package.")

# ----------------------------------------------------------------------------
# Kalshi -> implied CPI
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# Hero + banner slot + tabs
# ----------------------------------------------------------------------------
event_row = events[events.event_ticker == event_ticker].iloc[0] if event_ticker else None
pills = [f"Series: {series or '—'}"]
if event_row is not None:
    pills += [f"Event: {event_row.event_ticker}", f"Closes {fmt_close(event_row.close_time)}"]
pills += [f"Updated {now_et()}"]
st.markdown(
    '<div class="hero"><div class="hero-title">Kalshi CPI Divergence → Bank CAR Impact</div>'
    '<div class="hero-sub">Prediction-market expectations vs. Bloomberg consensus, translated into an '
    "estimated cumulative abnormal return for regional &amp; money-center banks.</div>"
    + "".join(f'<span class="pill">{html.escape(p)}</span>' for p in pills)
    + "</div>",
    unsafe_allow_html=True,
)
banner = st.container()

tab_dash, tab_eq, tab_k, tab_coef, tab_m = st.tabs(
    ["📊 Dashboard", "🏦 All equities", "🎲 Kalshi market", "🧮 Coefficients (γ₁)", "📘 Methodology"]
)

# ---- Coefficients first (everything else depends on them) -------------------
if "gamma_base" not in st.session_state:
    st.session_state.gamma_base = core.default_gammas(GAMMA_FILE)
    st.session_state.gamma_ver = 0
    st.session_state.gamma_sig = None

with tab_coef:
    st.markdown(
        "γ₁ is the **estimated % CAR per +1 percentage-point divergence** for each equity. "
        "Estimate it from your own event study, then paste or upload it here. "
        "Edits apply instantly across the dashboard."
    )
    up = st.file_uploader("Upload coefficients (CSV with columns: ticker, gamma1)", type=["csv"],
                          key="gamma_upload")
    if up is not None:
        sig = (up.name, up.size)
        if st.session_state.gamma_sig != sig:
            st.session_state.gamma_sig = sig
            try:
                st.session_state.gamma_base = core.normalize_gammas(pd.read_csv(up))
                st.session_state.gamma_ver += 1
                st.success("Coefficients loaded. Tickers missing from your file kept the placeholder value.")
            except Exception as e:
                st.error(f"Couldn't use that file: {e}")

    base = st.session_state.gamma_base
    edited = st.data_editor(
        base, key=f"gamma_editor_{st.session_state.gamma_ver}", hide_index=True,
        disabled=["Ticker", "source"], num_rows="fixed",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "gamma1": st.column_config.NumberColumn("γ₁ (% CAR per +1pp)", format="%.3f", step=0.01),
            "source": st.column_config.TextColumn("Source"),
        },
    )
    active = edited.copy()
    active["gamma1"] = pd.to_numeric(active["gamma1"], errors="coerce")
    bad = active["gamma1"].isna()
    active.loc[bad, "gamma1"] = base.loc[bad, "gamma1"]
    changed = ~np.isclose(active["gamma1"].astype(float).values, base["gamma1"].astype(float).values)
    active.loc[changed, "source"] = "user-edited"
    st.download_button("⬇️ Download coefficients (CSV)",
                       active.rename(columns={"Ticker": "ticker"}).to_csv(index=False),
                       file_name="gammas.csv", mime="text/csv")

n_placeholder = int((active["source"] == "placeholder").sum())
with banner:
    if kalshi_error:
        st.error(f"**Kalshi data unavailable.** {kalshi_error}")
    if calc_error:
        st.warning(f"**Couldn't build the Kalshi expectation.** {calc_error}")
    if n_placeholder:
        st.warning(
            f"**{n_placeholder} of {len(active)} γ₁ coefficients are placeholders (−0.50), not estimates.** "
            "CAR figures are illustrative until you load your own in the *Coefficients* tab."
        )
    if implied is not None:
        lo, hi = ladder["strike"].min(), ladder["strike"].max()
        if consensus < lo - 3 * implied.step or consensus > hi + 3 * implied.step:
            st.warning(
                f"Your consensus ({consensus:.2f}%) is far outside Kalshi's strike range "
                f"({lo:.1f}%–{hi:.1f}%). Make sure it matches the selected series (m/m vs y/y, headline vs core)."
            )

# ---- Quotes + per-equity table -------------------------------------------------
try:
    quotes = cached_quotes(tuple(tickers))
except Exception:
    quotes = pd.DataFrame({"Ticker": tickers})
for c in ("Price", "Prev Close", "Day %", "As of"):
    if c not in quotes.columns:
        quotes[c] = np.nan

eq = UNI.merge(quotes[["Ticker", "Price", "Day %", "As of"]], on="Ticker", how="left") \
        .merge(active[["Ticker", "gamma1", "source"]], on="Ticker", how="left")
eq["Est. CAR (%)"] = [core.estimated_car(divergence, g) for g in eq["gamma1"]]
kre_car = eq.loc[eq.Ticker == "KRE", "Est. CAR (%)"].iloc[0]
eq["vs KRE (pp)"] = eq["Est. CAR (%)"] - kre_car

sel = eq[eq.Ticker == selected].iloc[0]
if quotes["Price"].isna().all():
    banner.warning("Live quotes are unavailable right now (Yahoo Finance may be throttling). "
                   "Kalshi/CAR figures are unaffected — try *Refresh data now* in a minute.")

# ============================================================================
# Dashboard tab
# ============================================================================
with tab_dash:
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.markdown(kpi("Bloomberg consensus", f"{consensus:.2f}%", "Your input", "gray"), unsafe_allow_html=True)
    c2.markdown(kpi("Kalshi expected CPI", fnum(implied.expected if implied else None, 3, suffix="%"),
                    f"{implied.n_strikes} strikes used" if implied else "no data"),
                unsafe_allow_html=True)
    c3.markdown(kpi("Divergence", fnum(divergence, 3, signed=True, suffix=" pp"),
                    "Kalshi − consensus", "amber"), unsafe_allow_html=True)
    c4.markdown(kpi(f"Est. CAR · {selected}", fnum(sel["Est. CAR (%)"], 3, signed=True, suffix="%"),
                    f"γ₁ = {fnum(sel['gamma1'], 2, signed=True)}", car_tone(sel["Est. CAR (%)"])),
                unsafe_allow_html=True)

    st.markdown(f'<div class="section-title">{html.escape(selected)} · '
                f'{html.escape(sel["Name"])}</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4, gap="medium")
    s1.markdown(kpi("Last price", f"${sel['Price']:,.2f}" if not isnan(sel["Price"]) else "—",
                    f"as of {sel['As of']}" if sel["As of"] else "", small=True), unsafe_allow_html=True)
    s2.markdown(kpi("Day change", fnum(sel["Day %"], 2, signed=True, suffix="%"), "vs prior close",
                    car_tone(sel["Day %"]), small=True), unsafe_allow_html=True)
    s3.markdown(kpi("Group", sel["Group"], sel["Role"], "gray", small=True), unsafe_allow_html=True)
    s4.markdown(kpi("CAR vs KRE", fnum(sel["vs KRE (pp)"], 3, signed=True, suffix=" pp"),
                    "relative to sector ETF", car_tone(sel["vs KRE (pp)"]), small=True), unsafe_allow_html=True)

    if implied is not None:
        pv = core.prob_vs_consensus(implied.dist, consensus, implied.step)
        st.markdown('<div class="section-title">Market-implied odds vs. consensus</div>', unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4, gap="medium")
        p1.markdown(kpi("Hotter than consensus", f"{pv['above']*100:.0f}%", "P(print > consensus)", "amber", True),
                    unsafe_allow_html=True)
        p2.markdown(kpi("In line", f"{pv['inline']*100:.0f}%", "P(print = consensus)", "gray", True),
                    unsafe_allow_html=True)
        p3.markdown(kpi("Cooler than consensus", f"{pv['below']*100:.0f}%", "P(print < consensus)", "", True),
                    unsafe_allow_html=True)
        p4.markdown(kpi("Implied σ", f"{implied.std:.3f} pp", "spread of Kalshi distribution", "gray", True),
                    unsafe_allow_html=True)

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown('<div class="section-title">Consensus vs. Kalshi expectation</div>', unsafe_allow_html=True)
            cdf = pd.DataFrame({"Source": ["Bloomberg consensus", "Kalshi expected"],
                                "CPI (%)": [consensus, implied.expected]})
            base_c = alt.Chart(cdf).encode(
                x=alt.X("Source:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("CPI (%):Q", title="CPI (%)"),
                color=alt.Color("Source:N", legend=None,
                                scale=alt.Scale(domain=["Bloomberg consensus", "Kalshi expected"],
                                                range=["#94a3b8", "#2563eb"])),
                tooltip=["Source", alt.Tooltip("CPI (%):Q", format=".3f")],
            )
            chart = (base_c.mark_bar(size=80, cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                     + base_c.mark_text(dy=-9, fontWeight="bold", fontSize=13)
                     .encode(text=alt.Text("CPI (%):Q", format=".3f"))
                     ).properties(width="container", height=320)
            st.altair_chart(chart)
        with right:
            st.markdown('<div class="section-title">Kalshi-implied distribution of the print</div>',
                        unsafe_allow_html=True)
            dd = implied.dist.copy()
            tol = implied.step / 2
            dd["CPI print"] = dd["outcome"].map(lambda v: f"{v:.2f}")
            dd["vs consensus"] = np.where(dd["outcome"] > consensus + tol, "Hotter",
                                          np.where(dd["outcome"] < consensus - tol, "Cooler", "In line"))
            chart2 = alt.Chart(dd).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                x=alt.X("CPI print:N", sort=list(dict.fromkeys(dd["CPI print"])), title="CPI print (%)",
                        axis=alt.Axis(labelAngle=0)),
                y=alt.Y("probability:Q", title="Probability", axis=alt.Axis(format="%")),
                color=alt.Color("vs consensus:N", title=None,
                                scale=alt.Scale(domain=["Cooler", "In line", "Hotter"],
                                                range=["#38bdf8", "#94a3b8", "#f97316"])),
                tooltip=[alt.Tooltip("CPI print:N", title="Print (%)"),
                         alt.Tooltip("probability:Q", title="Probability", format=".1%"), "vs consensus"],
            ).properties(width="container", height=320)
            st.altair_chart(chart2)
    else:
        st.info("Waiting on Kalshi data — see the message above. Equity quotes are still available in the "
                "*All equities* tab.")

# ============================================================================
# All equities tab
# ============================================================================
with tab_eq:
    st.markdown('<div class="section-title">Estimated CAR across the tracking universe</div>',
                unsafe_allow_html=True)
    sort_by = st.radio("Sort by", ["Universe order", "Est. CAR (most negative first)", "Day change"],
                       horizontal=True, label_visibility="collapsed")
    view = eq.copy()
    if sort_by.startswith("Est. CAR"):
        view = view.sort_values("Est. CAR (%)", na_position="last")
    elif sort_by == "Day change":
        view = view.sort_values("Day %", ascending=False, na_position="last")
    show = view[["Ticker", "Name", "Group", "Price", "Day %", "gamma1", "Est. CAR (%)", "vs KRE (pp)"]] \
        .rename(columns={"gamma1": "γ₁", "Day %": "Day (%)", "Price": "Price ($)"})
    st.dataframe(
        show, hide_index=True,
        column_config={
            "Price ($)": st.column_config.NumberColumn(format="%.2f"),
            "Day (%)": st.column_config.NumberColumn(format="%.2f"),
            "γ₁": st.column_config.NumberColumn(format="%.2f"),
            "Est. CAR (%)": st.column_config.NumberColumn(format="%.3f"),
            "vs KRE (pp)": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    cd = eq.dropna(subset=["Est. CAR (%)"])
    if not cd.empty:
        chart3 = alt.Chart(cd).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("Ticker:N", sort=alt.EncodingSortField(field="Est. CAR (%)", order="ascending"), title=None),
            x=alt.X("Est. CAR (%):Q", title="Estimated CAR (%)"),
            color=alt.Color("Group:N", title="Group"),
            tooltip=["Ticker", "Name", "Group", alt.Tooltip("Est. CAR (%):Q", format=".3f")],
        ).properties(width="container", height=26 * len(cd) + 40)
        st.altair_chart(chart3)
    st.caption("Roles: " + " · ".join(f"**{r.Ticker}** {r.Role.lower()}" for r in UNI.itertuples()))

# ============================================================================
# Kalshi market tab
# ============================================================================
with tab_k:
    if implied is None:
        st.info("No usable Kalshi data to display yet.")
    else:
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
            notes.append(f"{thin} contract(s) have very low volume (<50) — treat the expectation as noisy.")
        st.markdown("\n".join(f"- {n}" for n in notes))

        st.markdown('<div class="section-title">Contract ladder: P(CPI above strike)</div>', unsafe_allow_html=True)
        lad = ladder.copy()
        lad["P(> strike) %"] = lad["prob"] * 100
        lad["Spread (¢)"] = lad["spread"] * 100
        lad["Status"] = np.where(lad["prob"].isna(), "excluded (no usable price)", "used")
        st.dataframe(
            lad[["ticker", "strike", "P(> strike) %", "price_source", "Spread (¢)", "volume",
                 "open_interest", "Status"]].rename(columns={
                     "ticker": "Contract", "strike": "Strike (%)", "price_source": "Price source",
                     "volume": "Volume", "open_interest": "Open interest"}),
            hide_index=True,
            column_config={"P(> strike) %": st.column_config.NumberColumn(format="%.1f"),
                           "Spread (¢)": st.column_config.NumberColumn(format="%.0f"),
                           "Strike (%)": st.column_config.NumberColumn(format="%.2f")},
        )
        st.markdown('<div class="section-title">Implied distribution</div>', unsafe_allow_html=True)
        out = implied.dist.rename(columns={"outcome": "CPI print (%)", "probability": "Probability"})
        out["Probability"] = out["Probability"] * 100
        st.dataframe(out, hide_index=True,
                     column_config={"Probability": st.column_config.NumberColumn("Probability (%)", format="%.1f"),
                                    "CPI print (%)": st.column_config.NumberColumn(format="%.2f")})

# ============================================================================
# Coefficients tab handled above; Methodology tab
# ============================================================================
with tab_m:
    st.markdown("### How the numbers are built")
    st.markdown(
        "**1. Kalshi-implied expected CPI.** Kalshi lists contracts of the form *“CPI above X%?”*. "
        "A YES price of 60¢ is read as a 60% probability. CPI is published rounded to one decimal, "
        "so *above 0.2* means a print of 0.3 or higher. Differencing the ladder gives the probability of each print:"
    )
    st.latex(r"P(X=t_{i+1}) = s_i - s_{i+1},\quad P(X=t_1)=1-s_1,\quad P(X=t_n+\Delta)=s_n")
    st.latex(r"E^{\text{Kalshi}}[\text{CPI}] = \sum_k x_k\,p_k")
    st.markdown("**2. Divergence and CAR.**")
    st.latex(r"\text{Divergence} = E^{\text{Kalshi}}[\text{CPI}] - \text{Consensus}")
    st.latex(r"\widehat{\text{CAR}}_i = \gamma_{1,i}\times \text{Divergence}")
    st.markdown(
        """
**Price handling.** Midpoint of bid/ask when the spread is tight; the last traded price when the book is wide
(a 3¢ bid / 97¢ ask midpoint of 50% would be meaningless). Contracts with no usable price are excluded and flagged.
Probabilities are forced to be non-increasing in the strike, then normalized to sum to 1.

**Assumptions & limits.**
- Kalshi “above” contracts resolve strictly above the strike on the BLS one-decimal print.
- Mass below the lowest strike is placed at that strike; mass above the highest strike at the highest strike + one step.
- Thinly traded strikes are noisy; check the *Kalshi market* tab.
- γ₁ is a **fixed linear** sensitivity. The app applies the coefficients you provide; it does not estimate them.
- Quotes come from Yahoo Finance via `yfinance` and may be delayed ~15 minutes.

**Estimating γ₁.** For each release *e* and equity *i*, compute CAR over your event window from a market-model
benchmark, then regress `CAR_i,e = γ₀ + γ₁·Divergence_e + controls + u`. Use the fitted γ₁ per equity, or a pooled
panel with release-date clustered standard errors.
"""
    )

st.caption("For academic research and education only; not investment advice. "
           "Data: Kalshi public API and Yahoo Finance (may be delayed or incomplete).")
