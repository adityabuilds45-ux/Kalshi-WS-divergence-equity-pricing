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
# Equity universe (ASP 2 tracking list)
# --------------------------------------------------------------------------
UNIVERSE = pd.DataFrame(
    [
        ("KRE", "SPDR S&P Regional Banking ETF", "Benchmark ETF", "Primary sector-wide baseline signal"),
        ("WAL", "Western Alliance Bancorp", "Regional Bank", "High-sensitivity regional deposit proxy"),
        ("FLG", "Flagstar Financial", "Regional Bank", "CRE-heavy regional shock signal"),
        ("KEY", "KeyCorp", "Regional Bank", "Mid-cap regional rate sensitivity"),
        ("ZION", "Zions Bancorporation", "Regional Bank", "High-beta regional lender"),
        ("FITB", "Fifth Third Bancorp", "Regional Bank", "Broad Midwest regional control"),
        ("CFG", "Citizens Financial Group", "Regional Bank", "East Coast regional benchmark"),
        ("OZK", "Bank OZK", "Niche Regional", "CRE / construction loan exposure"),
        ("CUBI", "Customers Bancorp", "Small Regional", "Small-cap high-beta regional"),
        ("SOFI", "SoFi Technologies", "Consumer Fintech", "High-beta growth / consumer credit signal"),
        ("JPM", "JPMorgan Chase", "Megabank (Money-Center)", "Defensive control (flight-to-quality)"),
        ("BAC", "Bank of America", "Megabank (Money-Center)", "Large consumer/deposit control"),
        ("GS", "Goldman Sachs", "Investment Bank", "Non-NIM capital markets control"),
    ],
    columns=["Ticker", "Name", "Group", "Role"],
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
        if spread <= max_spread:
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
        rows.append({
            "ticker": m.get("ticker"),
            "title": m.get("title") or m.get("yes_sub_title") or "",
            "strike": strike,
            "prob": prob,
            "price_source": src,
            "spread": spread,
            "volume": _to_float(m.get("volume")) or 0.0,
            "open_interest": _to_float(m.get("open_interest")) or 0.0,
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


def fetch_quotes(tickers: list[str]) -> pd.DataFrame:
    """
    Latest price + day change for each ticker. NEVER raises: a ticker that
    can't be fetched simply shows NaN, so one bad symbol can't break the app.
    Batch download first (fewer Yahoo requests), then per-ticker fallback.
    """
    import yfinance as yf  # imported lazily so unit tests don't need it

    tickers = list(dict.fromkeys(t.strip().upper() for t in tickers if t and t.strip()))
    closes: dict[str, pd.Series] = {t: pd.Series(dtype=float) for t in tickers}

    try:
        raw = yf.download(tickers, period="10d", interval="1d", auto_adjust=False,
                          progress=False, group_by="ticker", threads=True)
        for t in tickers:
            closes[t] = _closes(raw, t)
    except Exception:
        pass

    for t in tickers:
        if len(closes[t]) == 0:
            try:
                h = yf.Ticker(t).history(period="10d", interval="1d", auto_adjust=False)
                closes[t] = pd.to_numeric(h["Close"], errors="coerce").dropna()
            except Exception:
                pass

    return pd.DataFrame([_quote_row(t, closes[t]) for t in tickers])


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
