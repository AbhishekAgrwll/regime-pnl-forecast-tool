"""Generate the synthetic dataset used by the tool.

The original tool ran on a broker's internal daily P&L and volume data, which
is confidential and is not included here. This script produces a dataset with
the same structure so the tool can be run end to end:

- one row per instrument per trading day
- daily OHLC prices from a geometric Brownian motion
- two client segments (A and B), each with lots traded and the broker's P&L

How the synthetic P&L is built (a simple model of a broker that internalises
client flow):

    P&L = spread income per lot * lots
          - client net position * price move
          + noise (Student-t, so the tails are fat)

Client net position follows an AR(1) process, so clients lean one way for
several days at a time. Lots rise on high-range days. None of the parameters
are taken from real data. Units are arbitrary.

Usage:
    python scripts/make_synthetic_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
START, END = "2024-01-02", "2025-07-31"
OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic_flow.csv"

# start price, annual vol, base lots for segment A and segment B, P&L per lot-unit move
INSTRUMENTS = {
    "XAUUSD": dict(s0=2050.0, vol=0.15, lots_a=40.0, lots_b=25.0, scale=60.0),
    "EURUSD": dict(s0=1.09, vol=0.07, lots_a=30.0, lots_b=35.0, scale=50.0),
    "USDJPY": dict(s0=142.0, vol=0.10, lots_a=20.0, lots_b=30.0, scale=45.0),
    "DE40": dict(s0=16700.0, vol=0.16, lots_a=15.0, lots_b=10.0, scale=55.0),
    "USOIL": dict(s0=72.0, vol=0.30, lots_a=25.0, lots_b=15.0, scale=40.0),
}


def simulate_instrument(name, p, dates, rng):
    n = len(dates)
    sigma_d = p["vol"] / np.sqrt(252)

    # Close-to-close log returns with mild volatility clustering
    vol_state = np.exp(np.convolve(rng.normal(0, 0.25, n), np.ones(10) / 10, mode="same"))
    log_ret = rng.normal(0, sigma_d, n) * vol_state
    close = p["s0"] * np.exp(np.cumsum(log_ret))
    prev_close = np.concatenate([[p["s0"]], close[:-1]])
    open_ = prev_close * np.exp(rng.normal(0, sigma_d * 0.2, n))
    rng_frac = np.abs(rng.normal(0, sigma_d, n)) * vol_state + sigma_d * 0.5
    high = np.maximum(open_, close) * (1 + rng_frac / 2)
    low = np.minimum(open_, close) * (1 - rng_frac / 2)

    day_move = (close - open_) / open_
    rows = {"Date": dates, "Instrument": name, "Open": open_, "High": high, "Low": low, "Close": close}

    for seg, base in (("A", p["lots_a"]), ("B", p["lots_b"])):
        activity = 1 + 3 * (high - low) / open_ / sigma_d * 0.2
        lots = base * activity * np.exp(rng.normal(0, 0.3, n))

        # Client net positioning: AR(1) in [-1, 1] as a share of lots
        net = np.zeros(n)
        for t in range(1, n):
            net[t] = 0.85 * net[t - 1] + rng.normal(0, 0.2)
        net = np.tanh(net)

        spread_income = 0.35 * p["scale"] * lots / base
        market_pnl = -net * lots * day_move / sigma_d * p["scale"] * 0.3
        noise = rng.standard_t(4, n) * p["scale"] * 0.4
        rows[f"Lots_{seg}"] = lots.round(2)
        rows[f"PnL_{seg}"] = (spread_income + market_pnl + noise).round(2)

    df = pd.DataFrame(rows)
    df["Lots_Total"] = (df["Lots_A"] + df["Lots_B"]).round(2)
    df["PnL_Total"] = (df["PnL_A"] + df["PnL_B"]).round(2)
    return df


def main():
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range(START, END)
    frames = [simulate_instrument(k, v, dates, rng) for k, v in INSTRUMENTS.items()]
    out = pd.concat(frames, ignore_index=True)
    for col in ("Open", "High", "Low", "Close"):
        out[col] = out[col].round(5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out):,} rows to {OUT}")


if __name__ == "__main__":
    main()
