import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import core  # noqa: E402


def ladder(strikes, probs):
    return pd.DataFrame({"strike": strikes, "prob": probs})


# ---------------------------------------------------------------- expected value
def test_expected_value_worked_example():
    imp = core.implied_distribution(ladder([0.1, 0.2, 0.3], [0.9, 0.6, 0.2]))
    assert imp.expected == pytest.approx(0.27)
    assert imp.dist["probability"].sum() == pytest.approx(1.0)
    assert imp.step == pytest.approx(0.1)
    assert list(imp.dist["outcome"]) == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert not imp.monotone_adjusted


def test_non_monotone_prices_are_smoothed():
    imp = core.implied_distribution(ladder([0.1, 0.2, 0.3], [0.9, 0.5, 0.6]))
    assert imp.monotone_adjusted
    assert (imp.dist["probability"] >= 0).all()
    assert imp.dist["probability"].sum() == pytest.approx(1.0)


def test_gap_wider_than_step_uses_mean_of_possible_prints():
    # strikes 0.1 and 0.4 with step forced to 0.1 -> possible prints 0.2, 0.3, 0.4 -> mean 0.3
    imp = core.implied_distribution(ladder([0.1, 0.4], [0.8, 0.2]), step=0.1)
    row = imp.dist.iloc[1]
    assert row["outcome"] == pytest.approx(0.3)
    assert row["probability"] == pytest.approx(0.6)


def test_needs_two_strikes():
    with pytest.raises(core.DataError):
        core.implied_distribution(ladder([0.2], [0.5]))
    with pytest.raises(core.DataError):
        core.implied_distribution(ladder([0.2, 0.3], [None, 0.5]))


def test_all_probability_mass_certain():
    imp = core.implied_distribution(ladder([0.1, 0.2, 0.3], [1.0, 1.0, 1.0]))
    assert imp.expected == pytest.approx(0.4)  # certain to print above top strike


def test_prob_vs_consensus():
    imp = core.implied_distribution(ladder([0.1, 0.2, 0.3], [0.9, 0.6, 0.2]))
    pv = core.prob_vs_consensus(imp.dist, 0.2, imp.step)
    assert pv["above"] == pytest.approx(0.6)   # 0.3 (40%) + 0.4 (20%)
    assert pv["inline"] == pytest.approx(0.3)
    assert pv["below"] == pytest.approx(0.1)


def test_estimated_car():
    assert core.estimated_car(0.07, -0.5) == pytest.approx(-0.035)
    assert math.isnan(core.estimated_car(None, -0.5))
    assert math.isnan(core.estimated_car(0.1, None))


# ---------------------------------------------------------------- pricing
def test_mid_price_when_tight():
    p, src, sp = core.implied_probability({"yes_bid": 55, "yes_ask": 60, "last_price": 40})
    assert (round(p, 3), src) == (0.575, "mid")
    assert sp == pytest.approx(0.05)


def test_wide_spread_falls_back_to_last():
    p, src, _ = core.implied_probability({"yes_bid": 3, "yes_ask": 97, "last_price": 3})
    assert (p, src) == (pytest.approx(0.03), "last")


def test_no_price_returns_none():
    p, src, _ = core.implied_probability({"yes_bid": 0, "yes_ask": 0, "last_price": 0})
    assert p is None and src == "none"


def test_dollar_string_fields_supported():
    p, src, _ = core.implied_probability({"yes_bid_dollars": "0.5600", "yes_ask_dollars": "0.6000"})
    assert p == pytest.approx(0.58) and src == "mid"


# ---------------------------------------------------------------- ladder / events
MARKETS = [
    {"ticker": "KXCPI-26SEP-T0.1", "event_ticker": "KXCPI-26SEP", "close_time": "2026-10-14T12:25:00Z",
     "strike_type": "greater", "floor_strike": 0.1, "yes_bid": 88, "yes_ask": 92, "volume": 10},
    {"ticker": "KXCPI-26SEP-T0.2", "event_ticker": "KXCPI-26SEP", "close_time": "2026-10-14T12:25:00Z",
     "strike_type": "greater", "floor_strike": None, "yes_bid": 58, "yes_ask": 62, "volume": 20},  # strike from ticker
    {"ticker": "KXCPI-26SEP-B0.2", "event_ticker": "KXCPI-26SEP", "close_time": "2026-10-14T12:25:00Z",
     "strike_type": "between", "floor_strike": 0.2, "yes_bid": 30, "yes_ask": 34},                  # ignored
    {"ticker": "KXCPI-26OCT-T0.1", "event_ticker": "KXCPI-26OCT", "close_time": "2026-11-12T12:25:00Z",
     "strike_type": "greater", "floor_strike": 0.1, "yes_bid": 70, "yes_ask": 74},
]


def test_list_events_sorted_by_close():
    ev = core.list_events(MARKETS)
    assert list(ev["event_ticker"]) == ["KXCPI-26SEP", "KXCPI-26OCT"]
    assert list(ev["n_markets"]) == [3, 1]


def test_build_ladder_filters_and_parses():
    lad = core.build_ladder(MARKETS, "KXCPI-26SEP")
    assert list(lad["strike"]) == [0.1, 0.2]
    assert lad["prob"].tolist() == pytest.approx([0.90, 0.60])


def test_empty_inputs_do_not_crash():
    assert core.list_events([]).empty
    assert core.build_ladder([], "X").empty


# ---------------------------------------------------------------- network layer
def test_fetch_follows_pagination(monkeypatch):
    pages = [{"markets": [{"a": 1}], "cursor": "c1"}, {"markets": [{"a": 2}], "cursor": ""}]
    calls = []

    def fake(url, params=None, **kw):
        calls.append(dict(params))
        return pages[len(calls) - 1]

    monkeypatch.setattr(core, "_get_json", fake)
    out = core.fetch_open_markets(" kxcpi ")
    assert len(out) == 2
    assert calls[0]["series_ticker"] == "KXCPI" and calls[1]["cursor"] == "c1"


def test_fetch_blank_series_errors():
    with pytest.raises(core.DataError):
        core.fetch_open_markets("  ")


def test_get_json_gives_friendly_error(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(core.requests, "get", boom)
    monkeypatch.setattr(core.time, "sleep", lambda s: None)
    with pytest.raises(core.DataError, match="Kalshi request failed"):
        core._get_json("http://x", retries=2)


# ---------------------------------------------------------------- quotes
def test_closes_from_multiindex_download():
    idx = pd.date_range("2026-10-20", periods=3)
    cols = pd.MultiIndex.from_product([["WAL", "KRE"], ["Close", "Open"]])
    raw = pd.DataFrame(np.arange(12).reshape(3, 4).astype(float), index=idx, columns=cols)
    s = core._closes(raw, "WAL")
    assert len(s) == 3
    row = core._quote_row("WAL", s)
    assert row["Price"] == s.iloc[-1] and row["Day %"] == pytest.approx((s.iloc[-1] / s.iloc[-2] - 1) * 100)


def test_quote_row_handles_empty():
    row = core._quote_row("XYZ", pd.Series(dtype=float))
    assert math.isnan(row["Price"]) and row["Ticker"] == "XYZ"


# ---------------------------------------------------------------- coefficients
def test_normalize_gammas_fills_missing_with_placeholder():
    df = pd.DataFrame({"Ticker": ["wal", "jpm"], "gamma1": [-1.2, "oops"]})
    out = core.normalize_gammas(df)
    assert len(out) == len(core.UNIVERSE)
    assert out.set_index("Ticker").loc["WAL", "gamma1"] == pytest.approx(-1.2)
    assert out.set_index("Ticker").loc["JPM", "source"] == "placeholder"
    assert out.set_index("Ticker").loc["KRE", "gamma1"] == core.PLACEHOLDER_GAMMA


def test_normalize_gammas_requires_columns():
    with pytest.raises(core.DataError):
        core.normalize_gammas(pd.DataFrame({"foo": [1]}))
    with pytest.raises(core.DataError):
        core.normalize_gammas(pd.DataFrame({"ticker": ["WAL"]}))


def test_default_gammas_survives_missing_file(tmp_path):
    out = core.default_gammas(tmp_path / "nope.csv")
    assert len(out) == len(core.UNIVERSE) and (out["source"] == "placeholder").all()


def test_shipped_gammas_csv_is_valid():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gammas.csv")
    out = core.default_gammas(path)
    assert set(out["Ticker"]) == set(core.UNIVERSE["Ticker"])
