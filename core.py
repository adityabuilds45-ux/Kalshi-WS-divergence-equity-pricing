"""
core.py - data + math layer for the Kalshi CPI divergence dashboard.

No Streamlit imports here, so everything can be unit-tested offline.

Pipeline
--------
1. Pull open Kalshi markets for a CPI series (public endpoint, no API key).
2. Turn each "CPI above X" contract into an implied probability P(CPI > X).
3. Difference the ladder into a discrete distribution and take its mean.
4. Divergence = Kalshi expected CPI - consensus.
5. Estimated CAR = Divergence * gamma_1.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# --------------------------------------------------------------------------
# ETF universe
# --------------------------------------------------------------------------
UNIVERSE = pd.DataFrame(
    [
        ("TIP", "iShares TIPS Bond ETF", "Inflation-Linked Bonds",
         "Intermediate-term US TIPS (inflation-protected Treasuries)"),
        ("GLD", "SPDR Gold Shares", "Commodity (Gold)",
         "Gold bullion; classic inflation / real-rate hedge"),
        ("XLRE", "Real Estate Select Sector SPDR Fund", "Equity (Real Estate)",
         "US REITs and real estate; rate-sensitive"),
        ("UUP", "Invesco DB US Dollar Index Bullish Fund", "Currency (USD)",
         "Long US dollar vs. a basket of major currencies"),
        ("VTIP", "Vanguard Short-Term Inflation-Protected Securities ETF", "Inflation-Linked Bonds",
         "Short-term US TIPS (low duration)"),
        ("KRE", "SPDR S&P Regional Banking ETF", "Equity (Regional Banks)",
         "US regional bank equities"),
        ("QQQ", "Invesco QQQ Trust", "Equity (Nasdaq-100)",
         "Large-cap growth / tech; long-duration equities"),
        ("TLT", "iShares 20+ Year Treasury Bond ETF", "Treasuries (Long)",
         "Long-duration US Treasuries; most rate-sensitive"),
        ("SHY", "iShares 1-3 Year Treasury Bond ETF", "Treasuries (Short)",
         "Short-term US Treasuries; front-end rates"),
    ],
    columns=["Ticker", "Name", "Group", "Exposure"],
)

# Kalshi CPI series. Use "Custom" in the UI for anything not listed here.
CPI_SERIES = {
    "Headline CPI, m/m  (KXCPI)": "KXCPI",
    "Core CPI, m/m  (KXCPICORE)": "KXCPICORE",
    "Headline CPI, y/y  (KXCPIYOY)": "KXCPIYOY",
    "Core CPI, y/y  (KXCPICOREYOY)": "KXCPICOREYOY",
}


class DataError(RuntimeError):
    """Raised for any recoverable data problem; the UI shows the message."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _to_float(x) -> float | None:
    """Parse numbers defensively; return None for blanks / NaN / junk."""
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _dollars(m: dict, base: str) -> float | None:
    """
    Kalshi has exposed prices both as integer cents (`yes_bid`: 56) and as
    dollar strings (`yes_bid_dollars`: "0.5600"). Accept either, return dollars.
    """
    v = _to_float(m.get(f"{base}_dollars"))
    if v is not None:
        return v
    c = _to_float(m.get(base))
    return None if c is None else c / 100.0


def _count(m: dict, base: str) -> float | None:
    """Contract counts: Kalshi's fixed-point `<base>_fp` strings first, legacy integer `<base>` as fallback."""
    v = _to_float(m.get(f"{base}_fp"))
    return v if v is not None else _to_float(m.get(base))



def _get_json(url: str, params: dict | None = None, retries: int = 3, timeout: int = 15):
    last = "unknown error"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"Accept": "application/json"})
            if r.status_code == 429:
                last = "rate limited (HTTP 429)"
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            last = str(e)[:200]
            time.sleep(0.7 * (attempt + 1))
    raise DataError(f"Kalshi request failed after {retries} attempts: {last}")


# --------------------------------------------------------------------------
# Kalshi
# --------------------------------------------------------------------------
def fetch_open_markets(series_ticker: str) -> list[dict]:
    """All open markets in a series (follows pagination, guarded)."""
    series_ticker = (series_ticker or "").strip().upper()
    if not series_ticker:
        raise DataError("Enter a Kalshi series ticker (e.g. KXCPI).")

    markets: list[dict] = []
    cursor = None
    for _ in range(25):  # hard stop so a bad cursor can never loop forever
        params = {"series_ticker": series_ticker, "status": "open", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        data = _get_json(f"{KALSHI_BASE}/markets", params)
        if not isinstance(data, dict):
            raise DataError("Kalshi returned an unexpected response format.")
        markets.extend(data.get("markets") or [])
        cursor = data.get("cursor")
        if not cursor:
            break
    return markets


def list_events(markets: list[dict]) -> pd.DataFrame:
    """One row per event (one CPI release), soonest close first."""
    rows = [(m.get("event_ticker"), m.get("close_time") or m.get("expiration_time"))
            for m in markets if m.get("event_ticker")]
    if not rows:
        return pd.DataFrame(columns=["event_ticker", "close_time", "n_markets"])
    df = pd.DataFrame(rows, columns=["event_ticker", "close_time"])
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
    out = (df.groupby("event_ticker", as_index=False)
             .agg(close_time=("close_time", "min"), n_markets=("event_ticker", "size"))
             .sort_values("close_time", na_position="last")
             .reset_index(drop=True))
    return out


def implied_probability(m: dict, max_spread: float = 0.20) -> tuple[float | None, str, float | None]:
    """
    Implied P(YES) for one contract, in [0, 1].

    - Tight market (spread <= max_spread): bid/ask midpoint.
    - Wide / empty book: fall back to last traded price (a midpoint of a
      3c bid and 97c ask would be a meaningless 50%).
    - Nothing usable: None (contract is excluded and flagged in the UI).

    Returns (probability, source, spread).
    """
    bid = _dollars(m, "yes_bid")
    ask = _dollars(m, "yes_ask")
    last = _dollars(m, "last_price")

    spread = None
    if bid is not None and ask is not None and ask > 0 and ask >= bid:
        spread = ask - bid
        if spread <= max_spread + 1e-9:  # tolerance: 0.73-0.58 == 0.15000000000000002
            return min(max((bid + ask) / 2, 0.0), 1.0), "mid", spread
    if last is not None and last > 0:
        return min(max(last, 0.0), 1.0), "last", spread
    return None, "none", spread


_TICKER_STRIKE = re.compile(r"-T(-?\d+(?:\.\d+)?)$")


def _strike_of(m: dict) -> float | None:
    s = _to_float(m.get("floor_strike"))
    if s is not None:
        return s
    hit = _TICKER_STRIKE.search(str(m.get("ticker") or ""))
    return float(hit.group(1)) if hit else None


LADDER_COLUMNS = ["ticker", "title", "strike", "prob", "price_source", "spread", "volume", "open_interest"]


def build_ladder(markets: list[dict], event_ticker: str, max_spread: float = 0.20) -> pd.DataFrame:
    """
    The 'CPI above X' ladder for one event. Only 'greater'-type contracts are
    used; bucket ('between') and 'less' contracts are ignored.
    """
    rows = []
    for m in markets:
        if m.get("event_ticker") != event_ticker:
            continue
        stype = str(m.get("strike_type") or "").lower()
        if stype and stype != "greater":
            continue
        strike = _strike_of(m)
        if strike is None:
            continue
        prob, src, spread = implied_probability(m, max_spread)
        oi = _count(m, "open_interest")
        rows.append({
            "ticker": m.get("ticker"),
            "title": m.get("title") or m.get("yes_sub_title") or "",
            "strike": strike,
            "prob": prob,
            "price_source": src,
            "spread": spread,
            "volume": _count(m, "volume") or 0.0,
            "open_interest": oi if oi is not None else float("nan"),  # NaN = field absent from API response
        })
    if not rows:
        return pd.DataFrame(columns=LADDER_COLUMNS)
    df = pd.DataFrame(rows)
    # one row per strike: keep the most traded duplicate
    df = (df.sort_values(["strike", "volume"], ascending=[True, False])
            .drop_duplicates("strike")
            .reset_index(drop=True))
    return df[LADDER_COLUMNS]


# --------------------------------------------------------------------------
# Expected value math
# --------------------------------------------------------------------------
@dataclass
class Implied:
    expected: float
    std: float
    dist: pd.DataFrame  # columns: outcome, probability
    step: float
    n_strikes: int
    monotone_adjusted: bool


def implied_distribution(ladder: pd.DataFrame, step: float | None = None) -> Implied:
    """
    Convert survival probabilities s_i = P(CPI > t_i) into a discrete distribution.

    CPI is published rounded to `step` (0.1), so 'above t_i' means a print of at
    least t_i + step. With strikes t_1 < ... < t_n:

        P(X = t_1)               = 1 - s_1       (print at or below lowest strike)
        P(X = t_{i+1})           = s_i - s_{i+1}
        P(X = t_n + step)        = s_n           (print above highest strike)

    If two strikes are further apart than `step`, the mass between them is placed
    at the mean of the possible prints in that gap.
    """
    usable = ladder.dropna(subset=["strike", "prob"]).sort_values("strike").drop_duplicates("strike")
    if len(usable) < 2:
        raise DataError(
            "Need at least 2 priced strikes to build a distribution "
            f"(found {len(usable)}). The market may be too illiquid right now."
        )

    t = usable["strike"].to_numpy(dtype=float)
    s_raw = np.clip(usable["prob"].to_numpy(dtype=float), 0.0, 1.0)
    s = np.minimum.accumulate(s_raw)  # survival function must be non-increasing
    adjusted = bool(not np.allclose(s, s_raw))

    if step is None or step <= 0:
        diffs = np.diff(t)
        diffs = diffs[diffs > 1e-9]
        step = float(np.round(np.median(diffs), 4)) if diffs.size else 0.1
        if step <= 0:
            step = 0.1

    outcomes = [t[0]]
    probs = [1.0 - s[0]]
    for i in range(len(t) - 1):
        lo, hi = t[i], t[i + 1]
        outcomes.append(hi if (hi - lo) <= step * 1.5 else (lo + step + hi) / 2.0)
        probs.append(s[i] - s[i + 1])
    outcomes.append(t[-1] + step)
    probs.append(s[-1])

    p = np.clip(np.array(probs, dtype=float), 0.0, None)
    total = p.sum()
    if total <= 0:
        raise DataError("Contract prices imply an empty distribution.")
    p = p / total
    x = np.array(outcomes, dtype=float)

    mean = float(np.dot(x, p))
    std = float(np.sqrt(max(np.dot(p, (x - mean) ** 2), 0.0)))
    dist = pd.DataFrame({"outcome": np.round(x, 6), "probability": p})
    return Implied(mean, std, dist, float(step), int(len(t)), adjusted)


def prob_vs_consensus(dist: pd.DataFrame, consensus: float, step: float) -> dict:
    """P(print above / in line with / below the consensus figure)."""
    tol = max(step, 1e-9) / 2.0
    x, p = dist["outcome"].to_numpy(), dist["probability"].to_numpy()
    above = float(p[x > consensus + tol].sum())
    below = float(p[x < consensus - tol].sum())
    return {"above": above, "inline": max(0.0, 1.0 - above - below), "below": below}


def estimated_car(divergence: float | None, gamma: float | None) -> float:
    """CAR (%) = divergence (pp) * gamma_1 (% CAR per 1pp). NaN if either is missing."""
    if divergence is None or gamma is None:
        return float("nan")
    try:
        return float(divergence) * float(gamma)
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------
# Quotes (yfinance)
# --------------------------------------------------------------------------
def _closes(raw, ticker: str) -> pd.Series:
    empty = pd.Series(dtype=float)
    try:
        if raw is None or getattr(raw, "empty", True):
            return empty
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker in raw.columns.get_level_values(0):
                s = raw[ticker]["Close"]
            elif ticker in raw.columns.get_level_values(1):
                s = raw.xs(ticker, axis=1, level=1)["Close"]
            else:
                return empty
        else:
            s = raw["Close"]
        return pd.to_numeric(s, errors="coerce").dropna()
    except Exception:
        return empty


def _quote_row(ticker: str, closes: pd.Series) -> dict:
    nan = float("nan")
    if closes is None or len(closes) == 0:
        return {"Ticker": ticker, "Price": nan, "Prev Close": nan, "Day %": nan, "As of": ""}
    last = float(closes.iloc[-1])
    prev = float(closes.iloc[-2]) if len(closes) > 1 else nan
    chg = (last / prev - 1.0) * 100.0 if prev == prev and prev != 0 else nan
    try:
        asof = str(pd.Timestamp(closes.index[-1]).date())
    except Exception:
        asof = ""
    return {"Ticker": ticker, "Price": last, "Prev Close": prev, "Day %": chg, "As of": asof}


def _naive_index(closes: pd.Series) -> pd.Series:
    """Daily close series with a tz-naive DatetimeIndex (clean for charting)."""
    s = closes.copy()
    idx = pd.to_datetime(s.index, errors="coerce")
    try:
        idx = idx.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    s.index = idx
    return s[~s.index.isna()]


def fetch_market(tickers: list[str], period: str = "6mo") -> tuple[pd.DataFrame, dict]:
    """
    One Yahoo request for everything the UI needs: latest quotes AND daily close
    history (for sparklines / performance chart). NEVER raises: a ticker that
    can't be fetched shows NaN / is missing from the history dict.
    """
    import yfinance as yf  # imported lazily so unit tests don't need it

    tickers = list(dict.fromkeys(t.strip().upper() for t in tickers if t and t.strip()))
    closes: dict[str, pd.Series] = {t: pd.Series(dtype=float) for t in tickers}

    try:
        raw = yf.download(tickers, period=period, interval="1d", auto_adjust=False,
                          progress=False, group_by="ticker", threads=True)
        for t in tickers:
            closes[t] = _closes(raw, t)
    except Exception:
        pass

    for t in tickers:
        if len(closes[t]) == 0:
            try:
                h = yf.Ticker(t).history(period=period, interval="1d", auto_adjust=False)
                closes[t] = pd.to_numeric(h["Close"], errors="coerce").dropna()
            except Exception:
                pass

    quotes = pd.DataFrame([_quote_row(t, closes[t]) for t in tickers])
    hist = {t: _naive_index(closes[t]) for t in tickers if len(closes[t]) > 1}
    return quotes, hist


def fetch_quotes(tickers: list[str]) -> pd.DataFrame:
    """Quotes only (kept for backwards compatibility)."""
    return fetch_market(tickers, period="10d")[0]


# --------------------------------------------------------------------------
# Sensitivity coefficients (gamma_1)
# --------------------------------------------------------------------------
PLACEHOLDER_GAMMA = -0.50  # NOT an estimate - replace with your event-study results


def normalize_gammas(df: pd.DataFrame, default_source: str = "uploaded") -> pd.DataFrame:
    """
    Validate a coefficient table and align it to UNIVERSE.
    Accepts columns ticker, gamma1 (and optional source), any case.
    Missing / non-numeric tickers fall back to the placeholder so the app never breaks.
    Raises DataError if the required columns are absent.
    """
    d = df.copy()
    d.columns = [str(c).strip().lower().replace("γ", "gamma").replace("₁", "1") for c in d.columns]
    if "ticker" not in d.columns:
        raise DataError("Coefficient file needs a 'ticker' column.")
    gcol = next((c for c in d.columns if c in ("gamma1", "gamma_1", "gamma", "coef", "coefficient")), None)
    if gcol is None:
        raise DataError("Coefficient file needs a 'gamma1' column.")

    d["Ticker"] = d["ticker"].astype(str).str.strip().str.upper()
    d["gamma1"] = pd.to_numeric(d[gcol], errors="coerce")
    d["source"] = d["source"].astype(str) if "source" in d.columns else default_source
    d = d.drop_duplicates("Ticker", keep="last")[["Ticker", "gamma1", "source"]]

    out = UNIVERSE[["Ticker"]].merge(d, on="Ticker", how="left")
    missing = out["gamma1"].isna()
    out.loc[missing, "gamma1"] = PLACEHOLDER_GAMMA
    out.loc[missing, "source"] = "placeholder"
    out["source"] = out["source"].fillna(default_source)
    return out


def default_gammas(path) -> pd.DataFrame:
    """Load gammas.csv; if it is missing or broken, return all-placeholder values."""
    try:
        return normalize_gammas(pd.read_csv(path), default_source="file")
    except Exception:
        out = UNIVERSE[["Ticker"]].copy()
        out["gamma1"] = PLACEHOLDER_GAMMA
        out["source"] = "placeholder"
        return out


# --------------------------------------------------------------------------
# Tail-Risk & Relative Value Monitor
# --------------------------------------------------------------------------
SENS_COLS = ["ols_m", "ols_b", "q05_m", "q05_b", "q95_m", "q95_b"]

# Model per ETF:  response (%) = m * Divergence (pp) + b
# (ols = conditional mean, q05 = 5th percentile, q95 = 95th percentile)
DEFAULT_SENSITIVITY = {
    #        ols_m,     ols_b,      q05_m,     q05_b,     q95_m,      q95_b
    "SHY":  (-1.13533,  0.0238787,  -1.2333,   -0.199291, -1.273,     0.1983),
    "TLT":  (-5.20376,  0.0640881,  -3.55091,  -0.818909, -7.233,     0.9364),
    "QQQ":  (-8.299,    0.178553,   -7.704,    -1.2662,   -15.21826,  1.86853),
    "TIP":  (-1.80826,  0.0498414,  -1.29042,  -0.349958, -1.71767,   0.450067),
    "VTIP": (-0.58354,  0.0270654,  -0.430788, -0.173321, -1.0035,    0.27095),
    "UUP":  (3.20372,   -0.0817419, 4.51495,   -0.69659,  1.94001,    0.521099),
    "XLRE": (-8.04231,  0.188113,   -5.343,    -1.5213,   -11.18605,  1.5869),
    "GLD":  (-5.82199,  -0.0202403, -6.906,    -1.0783,   -5.99563,   1.01503),
    "KRE":  (-5.33347,  0.452446,   -5.33347,  -1.82645,  -5.33347,   3.52465),
}


def sensitivity_defaults() -> pd.DataFrame:
    """Built-in default parameter table, in UNIVERSE order."""
    return pd.DataFrame(
        [{"Ticker": t, **dict(zip(SENS_COLS, DEFAULT_SENSITIVITY[t]))} for t in UNIVERSE["Ticker"]]
    )


def normalize_sensitivity(df: pd.DataFrame, fallback: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Align a user-edited parameter table to UNIVERSE. Cells that are blank or
    non-numeric keep their value from `fallback` (default: built-in defaults),
    so a half-typed edit can never break the app.
    """
    out = (fallback.copy() if fallback is not None else sensitivity_defaults())
    d = df.copy()
    d.columns = ["Ticker" if str(c).strip().lower() == "ticker" else str(c).strip().lower() for c in d.columns]
    if "Ticker" not in d.columns:
        return out
    d["Ticker"] = d["Ticker"].astype(str).str.strip().str.upper()
    d = d.drop_duplicates("Ticker", keep="last").set_index("Ticker")
    for c in SENS_COLS:
        if c in d.columns:
            vals = out["Ticker"].map(pd.to_numeric(d[c], errors="coerce"))
            out[c] = vals.where(vals.notna(), out[c]).astype(float)
    return out[["Ticker"] + SENS_COLS]


def default_sensitivity(path) -> pd.DataFrame:
    """Load sensitivity_params.csv; fall back to the built-in defaults if missing/broken."""
    try:
        return normalize_sensitivity(pd.read_csv(path))
    except Exception:
        return sensitivity_defaults()


def prediction_intervals(params: pd.DataFrame, divergence: float | None) -> pd.DataFrame:
    """
    90% prediction interval per ETF at the given divergence (pp).
    Non-crossing guardrail:  lower = min(y05, y_ols, y95),  upper = max(y05, y_ols, y95).
    """
    nan = float("nan")
    rows = []
    for r in params.to_dict("records"):
        def line(m, b):
            m, b = _to_float(m), _to_float(b)
            if divergence is None or m is None or b is None or isnan_(divergence):
                return nan
            return m * float(divergence) + b

        y_ols, y05, y95 = line(r["ols_m"], r["ols_b"]), line(r["q05_m"], r["q05_b"]), line(r["q95_m"], r["q95_b"])
        if any(isnan_(v) for v in (y_ols, y05, y95)):
            lower = upper = nan
            adjusted = False
        else:
            lower, upper = min(y05, y_ols, y95), max(y05, y_ols, y95)
            adjusted = not (lower == y05 and upper == y95)
        rows.append({"Ticker": r["Ticker"], "y_ols": y_ols, "y_q05": y05, "y_q95": y95,
                     "lower": lower, "upper": upper, "adjusted": adjusted})
    return pd.DataFrame(rows)


def isnan_(x) -> bool:
    try:
        return x is None or bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


@dataclass
class TailMetrics:
    std: float             # std dev of the cleaned Kalshi distribution (pp)
    expected: float        # mean of the cleaned distribution (info only; headline E[CPI] is untouched)
    tail_prob: float       # P(|CPI - consensus| >= threshold)
    prob_up: float
    prob_down: float
    cond_shock: float      # E[|CPI - consensus| | tail event] in pp (NaN if tail prob == 0)
    n_used: int
    n_total: int
    step: float
    dist: pd.DataFrame
    covers_up: bool        # do liquid strikes reach the upper tail threshold?
    covers_down: bool
    oi_filter_applied: bool
    monotone_adjusted: bool


def tail_metrics(markets: list[dict], event_ticker: str, consensus: float,
                 min_open_interest: float = 50, max_spread: float = 0.15,
                 tail_threshold: float = 0.15, step: float | None = None) -> TailMetrics:
    """
    Separate, stricter read of the Kalshi ladder for risk metrics (does NOT touch the headline
    expected CPI):
      * drop strikes with open interest < min_open_interest
      * drop strikes whose bid-ask spread is wider than max_spread (or has no two-sided quote)
      * enforce non-increasing probabilities across strikes (inside implied_distribution)
    """
    full = build_ladder(markets, event_ticker, max_spread=max_spread)
    if full.empty:
        raise DataError("No CPI contracts found for this release.")

    keep = full["prob"].notna() & full["spread"].notna() & (full["spread"] <= max_spread + 1e-9)
    oi_known = bool(full["open_interest"].notna().any())
    if oi_known:
        keep &= full["open_interest"].fillna(0) >= min_open_interest
    clean = full[keep]
    try:
        imp = implied_distribution(clean, step)
    except DataError:
        raise DataError(
            f"Only {len(clean)} of {len(full)} strikes pass the liquidity filters "
            f"(open interest ≥ {min_open_interest:g}, spread ≤ {max_spread * 100:.0f}¢); need at least 2."
        )

    x = imp.dist["outcome"].to_numpy(dtype=float)
    p = imp.dist["probability"].to_numpy(dtype=float)
    d = x - float(consensus)
    eps = 1e-9
    up, down = d >= tail_threshold - eps, d <= -tail_threshold + eps
    prob_up, prob_down = float(p[up].sum()), float(p[down].sum())
    tail = prob_up + prob_down
    cond = float((p[up | down] * np.abs(d[up | down])).sum() / tail) if tail > 1e-12 else float("nan")

    lo, hi = float(clean["strike"].min()), float(clean["strike"].max())
    return TailMetrics(
        std=imp.std, expected=imp.expected, tail_prob=tail, prob_up=prob_up, prob_down=prob_down,
        cond_shock=cond, n_used=int(len(clean)), n_total=int(len(full)), step=imp.step, dist=imp.dist,
        covers_up=hi + imp.step >= consensus + tail_threshold - eps,
        covers_down=lo <= consensus - tail_threshold + eps,
        oi_filter_applied=oi_known, monotone_adjusted=imp.monotone_adjusted,
    )


@dataclass
class OptionsMove:
    ticker: str
    move_pct: float            # options-implied move, % of spot
    source: str                # "options" | "options_last" | "vix_fallback" | "unavailable"
    spot: float | None = None
    expiry: str = ""
    dte: int | None = None
    strike: float | None = None
    call: float | None = None
    put: float | None = None
    note: str = ""


def _opt_price(row: dict) -> tuple[float | None, str]:
    bid, ask, last = _to_float(row.get("bid")), _to_float(row.get("ask")), _to_float(row.get("lastPrice"))
    if bid is not None and ask is not None and bid > 0 and ask >= bid:
        return (bid + ask) / 2.0, "mid"
    if last is not None and last > 0:
        return last, "last"
    return None, "none"


def _spot(t) -> float | None:
    try:
        fi = t.fast_info
        for k in ("last_price", "lastPrice"):
            try:
                v = _to_float(fi[k])
            except Exception:
                v = None
            if v:
                return v
    except Exception:
        pass
    try:
        c = pd.to_numeric(t.history(period="5d")["Close"], errors="coerce").dropna()
        return float(c.iloc[-1]) if len(c) else None
    except Exception:
        return None


def fetch_vix() -> float | None:
    """Latest ^VIX level (annualized %, e.g. 18.5) or None."""
    import yfinance as yf  # lazy: unit tests inject a fake

    try:
        v = pd.to_numeric(yf.Ticker("^VIX").history(period="5d")["Close"], errors="coerce").dropna()
        return float(v.iloc[-1]) if len(v) else None
    except Exception:
        return None


def _try_options(yf, ticker: str, today=None) -> tuple[OptionsMove | None, str]:
    """Nearest-expiration ATM straddle for one ticker. Returns (move, "") or (None, reason). Never raises."""
    try:
        t = yf.Ticker(ticker)
        spot = _spot(t)
        today = today or pd.Timestamp.now(tz="America/New_York").date()
        future = [e for e in (list(t.options or [])) if pd.Timestamp(e).date() >= today]
        if spot and future:
            exp = future[0]
            chain = t.option_chain(exp)
            calls, puts = chain.calls, chain.puts
            common = sorted(set(calls["strike"]).intersection(set(puts["strike"])))
            if common:
                k = min(common, key=lambda s: abs(float(s) - spot))
                c, cb = _opt_price(calls[calls["strike"] == k].iloc[0].to_dict())
                p, pb = _opt_price(puts[puts["strike"] == k].iloc[0].to_dict())
                if c is not None and p is not None:
                    pct = (c + p) / spot * 100.0
                    if 0 < pct < 50:
                        return OptionsMove(
                            ticker, pct, "options" if (cb == "mid" and pb == "mid") else "options_last",
                            spot, str(exp), (pd.Timestamp(exp).date() - today).days, float(k), c, p, ""), ""
        return None, "options chain empty or unpriced"
    except Exception as e:
        return None, f"options lookup failed ({type(e).__name__})"


def _vix_or_unavailable(ticker: str, note: str, vix: float | None, allow: bool) -> OptionsMove:
    if not allow:
        return OptionsMove(ticker, float("nan"), "unavailable", note=f"{note}; VIX fallback disabled for this ETF")
    if vix:
        return OptionsMove(ticker, vix / math.sqrt(252.0), "vix_fallback", note=f"{note}; ^VIX = {vix:.2f}")
    return OptionsMove(ticker, float("nan"), "unavailable", note=f"{note}; ^VIX unavailable")


def fetch_options_move(ticker: str, today=None, allow_vix_fallback: bool = True) -> OptionsMove:
    """
    Nearest-expiration ATM straddle as % of spot:  (ATM call + ATM put) / spot.
    Falls back to ^VIX / sqrt(252) when the chain is empty or unpriced (e.g. after hours).
    NEVER raises.
    """
    import yfinance as yf

    ticker = ticker.strip().upper()
    mv, note = _try_options(yf, ticker, today)
    if mv is not None:
        return mv
    return _vix_or_unavailable(ticker, note, fetch_vix() if allow_vix_fallback else None, allow_vix_fallback)


def fetch_options_moves(tickers: list[str], vix_fallback_tickers=None, today=None,
                        max_workers: int = 5) -> dict[str, OptionsMove]:
    """
    Options-implied move for many tickers at once: option chains are fetched in parallel and
    ^VIX is fetched at most ONCE (shared by every ticker that needs the fallback).
    `vix_fallback_tickers`: which tickers may use the VIX fallback (default: all).
    NEVER raises.
    """
    from concurrent.futures import ThreadPoolExecutor

    import yfinance as yf

    tickers = list(dict.fromkeys(t.strip().upper() for t in tickers if t and t.strip()))
    if not tickers:
        return {}
    allowed = set(tickers) if vix_fallback_tickers is None else {t.strip().upper() for t in vix_fallback_tickers}

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(tickers)))) as ex:
        results = list(ex.map(lambda t: _try_options(yf, t, today), tickers))

    out: dict[str, OptionsMove] = {}
    need: list[tuple[str, str]] = []
    for t, (mv, note) in zip(tickers, results):
        if mv is not None:
            out[t] = mv
        else:
            need.append((t, note))
    vix = fetch_vix() if any(t in allowed for t, _ in need) else None
    for t, note in need:
        out[t] = _vix_or_unavailable(t, note, vix, t in allowed)
    return out
