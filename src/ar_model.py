"""
ar_model.py – AR(1)-Schätzung auf ersten Differenzen ΔY
=========================================================
Für jeden Ticker × Kennzahl:
  1. Zeitreihe Y(t) laden
  2. Erste Differenz bilden: ΔY(t) = Y(t) − Y(t−1)
  3. AR(1) per OLS schätzen: ΔY(t) = c + φ₁·ΔY(t−1) + ε(t)
  4. Volatilität: σ = std(ΔY)
  5. Half-Life berechnen: HL = log(0.5) / log(|φ₁|)

Ergebnis: Feature-Matrix mit (ticker, ratio, phi1, phi1_se, phi1_pval,
          sigma_diff, half_life, c, n_obs)
"""

import logging
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import (
    RATIO_NAMES, AR_ORDER, DIFF_ORDER,
    OUT_AR, ensure_output_dirs,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# AR(1)-Schätzung für eine Einzelreihe
# ─────────────────────────────────────────────
def _fit_ar1_on_diff(series: pd.Series) -> dict | None:
    """Schätzt AR(1) auf der ersten Differenz einer Zeitreihe.

    Args:
        series: Level-Zeitreihe Y(t), indiziert auf Quartal.

    Returns:
        Dict mit Schätzergebnissen oder None bei zu wenig Daten / Fehler.
    """
    # Erste Differenz
    diff = series.diff(DIFF_ORDER).dropna()

    # Brauchen mindestens AR_ORDER + 3 Beobachtungen für sinnvolle OLS
    if len(diff) < AR_ORDER + 3:
        return None

    # Volatilität der Differenzen
    sigma_diff = float(diff.std(ddof=1))

    # Konstante Reihe → φ₁ undefiniert
    if sigma_diff < 1e-15:
        return None

    # OLS: ΔY(t) = c + φ₁·ΔY(t−1)
    y = diff.iloc[AR_ORDER:]
    x = diff.shift(AR_ORDER).iloc[AR_ORDER:]

    # NaN-Bereinigung (sollte nach diff+shift nur am Rand sein)
    mask = y.notna() & x.notna()
    y = y[mask].values
    x = x[mask].values

    n_obs = len(y)
    if n_obs < AR_ORDER + 3:
        return None

    X = sm.add_constant(x)

    try:
        model = sm.OLS(y, X).fit()
    except Exception:
        return None

    phi1 = float(model.params[1])
    phi1_se = float(model.bse[1])
    phi1_pval = float(model.pvalues[1])
    c = float(model.params[0])
    r_squared = float(model.rsquared)

    # Half-Life: Perioden bis 50 % Mean-Reversion
    # Nur sinnvoll wenn |φ₁| < 1 und φ₁ > 0
    if 0 < phi1 < 1:
        half_life = float(np.log(0.5) / np.log(abs(phi1)))
    else:
        half_life = np.nan

    return {
        "phi1": phi1,
        "phi1_se": phi1_se,
        "phi1_pval": phi1_pval,
        "c": c,
        "r_squared": r_squared,
        "sigma_diff": sigma_diff,
        "half_life": half_life,
        "n_obs": n_obs,
    }


# ─────────────────────────────────────────────
# Gesamtes Panel schätzen
# ─────────────────────────────────────────────
def estimate_ar_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Schätzt AR(1) auf ΔY für jeden Ticker × Kennzahl.

    Args:
        panel: DataFrame mit Spalten [ticker, quarter, <RATIO_NAMES>].

    Returns:
        Feature-Matrix mit einer Zeile pro (ticker, ratio).
    """
    results: list[dict] = []
    tickers = sorted(panel["ticker"].unique())

    for ticker in tickers:
        df_t = panel[panel["ticker"] == ticker].set_index("quarter").sort_index()

        for ratio in RATIO_NAMES:
            series = df_t[ratio].copy()

            # NaN-Lücken: nur schätzen wenn zusammenhängende Reihe lang genug
            n_valid = series.notna().sum()
            if n_valid < AR_ORDER + DIFF_ORDER + 3:
                continue

            fit = _fit_ar1_on_diff(series)
            if fit is None:
                continue

            results.append({"ticker": ticker, "ratio": ratio, **fit})

    features = pd.DataFrame(results)
    logger.info("AR(1)-Schätzung: %d Ergebnisse (%d Ticker × %d Kennzahlen, mit Lücken)",
                len(features), features["ticker"].nunique(),
                features["ratio"].nunique())

    return features


# ─────────────────────────────────────────────
# Zusammenfassung pro Kennzahl
# ─────────────────────────────────────────────
def summarize_by_ratio(features: pd.DataFrame) -> pd.DataFrame:
    """Deskriptive Statistik der AR-Parameter, gruppiert nach Kennzahl."""
    summary = (
        features
        .groupby("ratio")
        .agg(
            n=("phi1", "count"),
            phi1_mean=("phi1", "mean"),
            phi1_median=("phi1", "median"),
            phi1_std=("phi1", "std"),
            phi1_signif_pct=("phi1_pval", lambda s: (s < 0.05).mean() * 100),
            sigma_mean=("sigma_diff", "mean"),
            sigma_median=("sigma_diff", "median"),
            hl_median=("half_life", "median"),
        )
        .round(4)
    )
    return summary


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    ensure_output_dirs()

    # Panel laden
    panel = pd.read_csv(OUT_AR.parent / "ratios" / "ratio_panel.csv")
    logger.info("Panel geladen: %d Zeilen, %d Ticker",
                len(panel), panel["ticker"].nunique())

    # AR(1) schätzen
    features = estimate_ar_features(panel)

    # Speichern
    out_detail = OUT_AR / "ar_features.csv"
    features.to_csv(out_detail, index=False)
    logger.info("Detail-Ergebnisse: %s", out_detail)

    # Zusammenfassung
    summary = summarize_by_ratio(features)
    out_summary = OUT_AR / "ar_summary_by_ratio.csv"
    summary.to_csv(out_summary)
    logger.info("Zusammenfassung: %s", out_summary)

    print(f"\n{'='*70}")
    print(f"  AR(1) auf ΔY – Ergebnisse")
    print(f"  {features['ticker'].nunique()} Ticker, {len(features)} Schätzungen")
    print(f"{'='*70}")
    print(f"\nZusammenfassung nach Kennzahl:")
    print(summary.to_string())
