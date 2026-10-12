"""Regime-conditioned P&L forecasting and risk for a broker's client flow.

Pipeline:
1. Load daily data per instrument (prices, lots and P&L for two client segments).
2. Label every day with one of four regimes: high or low intraday range,
   crossed with an up or down day.
3. Within each instrument, segment and regime, fit a linear model of daily P&L
   on lots traded, the day's return and its intraday range, and keep the residuals.
4. Monte Carlo: draw a day's drivers from the chosen regime's history, predict
   P&L from the regime's model and add a residual drawn from the same regime.
5. Historical VaR and expected shortfall from past windows that began in a chosen regime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_PATH = Path(__file__).resolve().parent / "data" / "synthetic_flow.csv"
SEGMENTS = ("A", "B")
FEATURES = ("Daily_Return", "Intraday_Range")
MIN_OBS = 10


# ---------------------------------------------------------------------------
# 1-2. Data and regimes
# ---------------------------------------------------------------------------
def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values(["Instrument", "Date"]).reset_index(drop=True)
    df["Daily_Return"] = (df["Close"] - df["Open"]) / df["Open"]
    df["Intraday_Range"] = (df["High"] - df["Low"]) / df["Open"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=list(FEATURES))
    return add_regimes(df)


def add_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Median split of intraday range (per instrument) crossed with the sign of the day's return."""
    vol = df.groupby("Instrument")["Intraday_Range"].transform(
        lambda x: np.where(x > x.median(), "High_Vol", "Low_Vol")
    )
    direction = np.where(df["Daily_Return"] >= 0, "Up", "Down")
    df["Regime"] = pd.Series(vol, index=df.index) + "_" + direction
    return df


# ---------------------------------------------------------------------------
# 3. Regime models
# ---------------------------------------------------------------------------
@dataclass
class RegimeModel:
    model: LinearRegression
    residuals: np.ndarray
    drivers: np.ndarray  # rows of [lots, Daily_Return, Intraday_Range] observed in this regime
    r2: float


@dataclass
class ModelSet:
    models: dict = field(default_factory=dict)  # (instrument, segment, regime) -> RegimeModel

    def get(self, instrument: str, segment: str, regime: str) -> RegimeModel | None:
        return self.models.get((instrument, segment, regime))


def fit_models(df: pd.DataFrame) -> ModelSet:
    out = ModelSet()
    for (inst, regime), sub in df.groupby(["Instrument", "Regime"]):
        for seg in SEGMENTS:
            cols = [f"Lots_{seg}", *FEATURES]
            data = sub[cols + [f"PnL_{seg}"]].dropna()
            if len(data) < MIN_OBS:
                continue
            X, y = data[cols].to_numpy(), data[f"PnL_{seg}"].to_numpy()
            m = LinearRegression().fit(X, y)
            out.models[(inst, seg, regime)] = RegimeModel(
                model=m, residuals=y - m.predict(X), drivers=X, r2=m.score(X, y)
            )
    return out


def model_summary(models: ModelSet) -> pd.DataFrame:
    rows = [
        {"Instrument": i, "Segment": s, "Regime": r, "n": len(rm.residuals), "R2": round(rm.r2, 3)}
        for (i, s, r), rm in models.models.items()
    ]
    return pd.DataFrame(rows).sort_values(["Instrument", "Segment", "Regime"])


# ---------------------------------------------------------------------------
# 4. Monte Carlo
# ---------------------------------------------------------------------------
def _simulate_day(rm: RegimeModel, n: int, rng: np.random.Generator) -> np.ndarray:
    """n draws of one day's P&L: resampled drivers through the regime model plus a resampled residual."""
    X = rm.drivers[rng.integers(0, len(rm.drivers), n)]
    return rm.model.predict(X) + rm.residuals[rng.integers(0, len(rm.residuals), n)]


def simulate_paths(
    df: pd.DataFrame,
    models: ModelSet,
    instrument: str,
    days: int,
    n_sims: int,
    starting_regime: str | None = None,
    segments: tuple[str, ...] = SEGMENTS,
    seed: int | None = None,
) -> np.ndarray:
    """Cumulative P&L paths, shape (days + 1, n_sims).

    Day 1 uses `starting_regime` if given. Every other day's regime is drawn from
    the instrument's historical regime frequencies.
    """
    rng = np.random.default_rng(seed)
    hist_regimes = df.loc[df["Instrument"] == instrument, "Regime"].to_numpy()
    daily = np.zeros((days, n_sims))
    for d in range(days):
        if d == 0 and starting_regime is not None:
            regimes = np.full(n_sims, starting_regime)
        else:
            regimes = hist_regimes[rng.integers(0, len(hist_regimes), n_sims)]
        for regime in np.unique(regimes):
            idx = np.flatnonzero(regimes == regime)
            for seg in segments:
                rm = models.get(instrument, seg, regime)
                if rm is not None:
                    daily[d, idx] += _simulate_day(rm, len(idx), rng)
    return np.vstack([np.zeros(n_sims), daily.cumsum(axis=0)])


def simulate_book(df, models, days=252, n_sims=1000, seed=None) -> np.ndarray:
    """All instruments, both segments. Instruments are simulated independently."""
    total = None
    for i, inst in enumerate(sorted(df["Instrument"].unique())):
        paths = simulate_paths(df, models, inst, days, n_sims, seed=None if seed is None else seed + i)
        total = paths if total is None else total + paths
    return total


def summarise(final_pnl: np.ndarray) -> dict:
    return {
        "median": float(np.median(final_pnl)),
        "mean": float(np.mean(final_pnl)),
        "q1": float(np.percentile(final_pnl, 25)),
        "q3": float(np.percentile(final_pnl, 75)),
        "p5": float(np.percentile(final_pnl, 5)),
        "p95": float(np.percentile(final_pnl, 95)),
    }


# ---------------------------------------------------------------------------
# 5. Regime-conditional historical VaR and expected shortfall
# ---------------------------------------------------------------------------
def regime_var(
    df: pd.DataFrame,
    instrument: str,
    regime: str,
    horizon: int,
    confidence: float = 0.95,
    segment: str = "Total",
) -> dict | None:
    """VaR and expected shortfall of `horizon`-day cumulative P&L, using every historical
    window of consecutive trading days that started on a day in `regime`.
    No distributional assumption: the quantile is read from the empirical outcomes.
    """
    sub = df[df["Instrument"] == instrument].reset_index(drop=True)
    pnl = sub[f"PnL_{segment}"].to_numpy()
    # forward sum over t .. t+horizon-1
    fwd = pd.Series(pnl[::-1]).rolling(horizon).sum().to_numpy()[::-1]
    starts = (sub["Regime"].to_numpy() == regime) & ~np.isnan(fwd)
    outcomes = fwd[starts]
    if len(outcomes) < 20:
        return None
    var = np.percentile(outcomes, 100 * (1 - confidence))
    es = outcomes[outcomes <= var].mean()
    return {"outcomes": outcomes, "var": float(var), "es": float(es), "n_windows": int(len(outcomes))}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_paths(paths: np.ndarray, title: str, max_paths: int = 300):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(paths[:, :max_paths], color="#4b3f8c", alpha=0.08, lw=0.8)
    for q, ls in ((50, "-"), (5, "--"), (95, "--")):
        ax.plot(np.percentile(paths, q, axis=1), color="#d9480f", ls=ls, lw=1.6,
                label="median" if q == 50 else ("5th / 95th percentile" if q == 5 else None))
    ax.set(title=title, xlabel="Trading days ahead", ylabel="Cumulative P&L (synthetic units)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_distribution(final_pnl: np.ndarray, title: str):
    s = summarise(final_pnl)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(final_pnl, bins=50, color="#1c7c7d", alpha=0.75)
    ax.axvline(s["median"], color="#d9480f", ls="--", label=f"Median {s['median']:,.0f}")
    ax.axvspan(s["q1"], s["q3"], color="#d9480f", alpha=0.12, label="Interquartile range")
    ax.set(title=title, xlabel="Cumulative P&L (synthetic units)", ylabel="Simulations")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_var(result: dict, title: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(result["outcomes"], bins=40, color="#9c2b2b", alpha=0.7, label="Historical window outcomes")
    ax.axvline(result["var"], color="black", ls="--", lw=2, label=f"VaR {result['var']:,.0f}")
    ax.axvline(result["es"], color="#e8a317", lw=2, label=f"Expected shortfall {result['es']:,.0f}")
    ax.set(title=title, xlabel="Cumulative P&L over horizon (synthetic units)", ylabel="Windows")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    data = load_data()
    ms = fit_models(data)
    print(data.groupby("Instrument")["Regime"].value_counts().unstack())
    print(model_summary(ms).to_string(index=False))
    book = simulate_book(data, ms, days=252, n_sims=1000, seed=0)
    print({k: round(v) for k, v in summarise(book[-1]).items()})
