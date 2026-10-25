# Kalshi CPI Divergence → Bank CAR Dashboard

Interactive Streamlit dashboard for a finance research paper.

1. Pulls **live Kalshi "CPI above X%" contracts** (public API, no key) and computes the
   probability-weighted **expected CPI**.
2. You enter the **Bloomberg consensus**.
3. **Divergence = Kalshi expected − consensus.**
4. **Estimated CAR = Divergence × γ₁** for any equity in the tracking list
   (KRE, WAL, FLG, KEY, ZION, FITB, CFG, OZK, CUBI, SOFI, JPM, BAC, GS).

## Project layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (tabs, cards, charts) |
| `core.py` | Data + math (Kalshi, yfinance, expected value, γ₁ handling). No Streamlit imports |
| `gammas.csv` | Sensitivity coefficients γ₁. **Currently placeholders (−0.50)** |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Theme / server settings |
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

1. Sign in at github.com → **New repository** → name it `kalshi-cpi-dashboard` → Create.
2. Click **uploading an existing file**, then drag in **everything** from this folder
   (including the `.streamlit`, `tests` and `.github` folders; if your file picker hides
   dotfolders, unzip first and drag from the unzipped folder in Finder/Explorer).
3. Click **Commit changes**.

Or with git:
```bash
git init && git add . && git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<you>/kalshi-cpi-dashboard.git
git push -u origin main
```

## Deploy free on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub (authorize access to the repo).
2. **Create app** → pick the repo, branch `main`, main file path `app.py`.
3. (Optional) **Advanced settings** → choose Python 3.12.
4. **Deploy.** First build takes a couple of minutes. You get a public URL like
   `https://<name>.streamlit.app` you can cite in the paper.

Every `git push` redeploys automatically. Free apps sleep after inactivity and wake on the next visit.

Other hosts (Render, Railway, Fly.io, Hugging Face Spaces) work too. Start command:
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## Setting your γ₁ coefficients

The values in `gammas.csv` are **placeholders, not estimates**, and the app says so in a banner until
you replace them. Two ways:

- **Permanent:** edit `gammas.csv` (columns `ticker,gamma1,source`), change `source` to e.g. `event-study 2021-2026`, push.
- **On the fly:** *Coefficients* tab → upload a CSV or edit cells directly, then download the result.

Units: **% CAR per +1 percentage-point divergence.** Estimate by regressing each equity's event-window CAR on
the divergence across CPI releases (see the *Methodology* tab).

## Troubleshooting

| Symptom | Fix |
|---|---|
| "No open markets found for series …" | Check the series ticker on kalshi.com; use *Custom…* in the sidebar |
| Kalshi expected value looks off | See the *Kalshi market* tab: thin or wide-spread strikes make it noisy; tune the spread slider |
| Quotes show "—" | Yahoo is throttling shared cloud IPs. Wait a minute, click *Refresh data now*, or `pip install -U yfinance` |
| Consensus warning banner | Your consensus is on a different basis (m/m vs y/y, headline vs core) than the selected series |
| Auto-refresh does nothing | Install `streamlit-autorefresh` (in requirements.txt) and switch it on under *Advanced* |

## Customizing

- **Tickers:** edit `UNIVERSE` in `core.py` and add matching rows to `gammas.csv`. Keep `KRE`, the "vs KRE" column uses it as the sector baseline.
- **Kalshi series:** edit `CPI_SERIES` in `core.py`.
- **Look & feel:** CSS block at the top of `app.py`, theme in `.streamlit/config.toml`.

## Tests

```bash
pip install pytest
pytest -q
```

*For academic research and education only; not investment advice.*
