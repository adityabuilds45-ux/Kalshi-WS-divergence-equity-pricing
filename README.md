# Kalshi CPI Divergence → ETF Response Terminal

Interactive Streamlit dashboard for a finance research paper.

1. Pulls **live Kalshi "CPI above X%" contracts** (public API, no key) and computes the
   probability-weighted **expected CPI**.
2. You enter the **Bloomberg consensus**.
3. **Divergence = Kalshi expected − consensus.**
4. **Estimated CAR = Divergence × γ₁** for each ETF in the universe:
   **TIP, GLD, XLRE, UUP, VTIP, KRE, QQQ, TLT, SHY**.

## Using the dashboard

- **Left rail:** Dashboard · ETF universe · Kalshi market · Coefficients · Methodology.
- **ETF chips** under the ticker tape switch the selected ETF on every page.
- **Sidebar parameters:** Kalshi series, CPI release, consensus input, advanced options, refresh.

## Institutional Tail-Risk & Relative Value Monitor (bottom of the Dashboard page)

An additive section; the original metrics, charts and inputs are unchanged.

1. **Edit Quantile & OLS Sensitivity Parameters** (expander): an editable table of `m` (slope) and `b` (intercept)
   for the conditional mean (OLS), 5th percentile and 95th percentile of every ETF. Edits update the whole section
   instantly and last for the browser session. Defaults load from `sensitivity_params.csv`; to change them
   permanently, edit that file on GitHub (or use the *Download CSV* button, then paste it in).
   Model: **response (%) = m × Divergence (pp) + b**.
2. **Clean Kalshi tail metrics**, computed separately from the headline expected CPI: strikes with open interest
   under 50 or a bid-ask spread wider than 15¢ are dropped, and probabilities are forced non-increasing across
   strikes. Cards: *Market uncertainty* (std dev of the cleaned distribution) and *Tail shock probability*
   (chance CPI prints at least 0.15 pp above or below your consensus).
3. **Options market price check for all 9 ETFs**: nearest-expiration ATM straddle ÷ spot (`yfinance`), fetched
   in parallel, with automatic fallback to `^VIX / sqrt(252)` (fetched once) when a chain is empty or unpriced.
   Each ETF gets its own banner: *GREEN LIGHT: Protection Underpriced — Hedge Recommended* when its Kalshi tail
   move is at least 25% above the options-implied move, otherwise *STAND DOWN: Options Already Price In Shock*
   (or *NO SIGNAL* if data is missing). A comparison table shows every number behind the banners.
   **Kalshi tail move = |OLS slope m| × the average CPI miss (pp) given a tail shock**, from the cleaned distribution.
   VIX is an equity-volatility index, so for non-equity ETFs the fallback is only a rough proxy and is flagged
   "proxy"; remove tickers from `VIX_FALLBACK_TICKERS` in `app.py` to show NO SIGNAL for them instead.
4. **Dynamic 90% prediction interval table** for all 9 ETFs, with a non-crossing guardrail
   (`lower = min(q05, ols, q95)`, `upper = max(q05, ols, q95)`).

The rule thresholds (50 open interest, 15¢ spread, 0.15 pp tail, 1.25× trigger) are constants at the top of `app.py`.

## Project layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (theme/CSS, navigation, pages, charts) |
| `core.py` | Data + math (Kalshi, yfinance, expected value, γ₁ handling). No Streamlit imports |
| `gammas.csv` | Sensitivity coefficients γ₁. **Currently placeholders (−0.50)** |
| `sensitivity_params.csv` | Default OLS / 5th / 95th percentile slope & intercept per ETF for the tail-risk monitor |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Dark theme / server settings |
| `tests/test_core.py` | Offline unit tests |
| `.github/workflows/tests.yml` | Runs tests automatically on every push |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501.

## Put it on GitHub (no command line needed)

1. github.com → **New repository** → name it `kalshi-cpi-dashboard` → Create.
2. **uploading an existing file** → drag in **everything** from this folder, including the hidden
   `.streamlit` and `.github` folders (unzip first, then drag from the unzipped folder).
3. **Commit changes**.

Or with git:
```bash
git init && git add . && git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<you>/kalshi-cpi-dashboard.git
git push -u origin main
```

## Deploy free on Streamlit Community Cloud

1. https://share.streamlit.io → sign in with GitHub (authorize the repo).
2. **Create app** → repo, branch `main`, main file path `app.py`.
3. (Optional) **Advanced settings** → Python 3.12.
4. **Deploy.** You get a public URL like `https://<name>.streamlit.app`.

Every `git push` redeploys automatically. Free apps sleep after inactivity and wake on the next visit.
Other hosts (Render, Railway, Fly.io, Hugging Face Spaces) work with:
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## Setting your γ₁ coefficients

The values in `gammas.csv` are **placeholders, not estimates**; the app shows a warning until you replace them.

- **Temporary (in the app):** *Coefficients* page → type new values per ETF, or upload a CSV, then download the result.
  Edits last for your browser session only.
- **Permanent:** edit `gammas.csv` on GitHub (pencil icon → **Commit changes**). Columns: `ticker,gamma1,source`.
  Change `source` from `placeholder` (e.g. to `event study, 1h`) to clear the warning.

Units: **% CAR per +1 percentage-point divergence.**

## Troubleshooting

| Symptom | Fix |
|---|---|
| "No open markets found for series …" | Check the series ticker on kalshi.com; use *Custom…* in the sidebar |
| Kalshi expected value looks off | See the *Kalshi market* page: thin or wide-spread strikes make it noisy; tune the spread slider |
| Quotes show "—" / no price history | Yahoo is throttling shared cloud IPs. Wait a minute, click *Refresh data now*, or `pip install -U yfinance` |
| Consensus warning banner | Your consensus is on a different basis (m/m vs y/y, headline vs core) than the selected series |
| Navigation / chips look unstyled | Styling targets Streamlit's page structure; update with `pip install -U streamlit` and check CSS at the top of `app.py` |

## Customizing

- **ETFs:** edit `UNIVERSE` in `core.py` and add matching rows to `gammas.csv`.
- **Kalshi series:** edit `CPI_SERIES` in `core.py`.
- **Look & feel:** the CSS block (colour variables in `:root`) at the top of `app.py`; theme in `.streamlit/config.toml`.

## Tests

```bash
pip install pytest
pytest -q
```

*For academic research and education only; not investment advice.*
