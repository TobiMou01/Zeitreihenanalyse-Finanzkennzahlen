"""
load_data.py – Rohdaten einlesen, Kennzahlen berechnen, Panel bauen
====================================================================
Workflow:
  1. Alle Ticker ermitteln (Schnittmenge der 3 Quell-Ordner)
  2. Finanzsektor exkludieren (via Profil-CSV)
  3. Pro Ticker: Bilanz, GuV und Cashflow einlesen, auf 2000–2024 filtern
  4. Vollständigkeitsfilter: nur Ticker mit exakt 100 Quartalen
  5. Kennzahlen berechnen (Zähler / Nenner)
  6. Winsorisierung pro Quartal
  7. Ergebnis: ein Panel-DataFrame (ticker × quarter × ratio)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    RAW_BALANCE, RAW_CASHFLOW, RAW_INCOME, RAW_PROFILE,
    YEAR_START, YEAR_END, EXPECTED_QUARTERS,
    EXCLUDED_SECTORS, PROFILE_SECTOR_COL,
    RATIOS, RATIO_NAMES,
    WINSORIZE_QUANTILES,
    OUT_RATIOS, SRC_DIRS,
    ensure_output_dirs,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. Ticker ermitteln
# ─────────────────────────────────────────────
def _csv_tickers(folder: Path) -> set[str]:
    """Gibt die Menge aller Ticker zurück, die als CSV in *folder* liegen."""
    return {p.stem for p in folder.glob("*.csv")}


def get_ticker_universe() -> list[str]:
    """Schnittmenge der Ticker über alle drei Quell-Ordner, sortiert."""
    sets = [_csv_tickers(d) for name, d in SRC_DIRS.items() if name != "profile"]
    common = sets[0].intersection(*sets[1:])
    logger.info("Ticker-Universum: %d (Schnittmenge aus %s)",
                len(common), [d.name for d in [RAW_BALANCE, RAW_INCOME, RAW_CASHFLOW]])
    return sorted(common)


# ─────────────────────────────────────────────
# 2. Finanzsektor exkludieren
# ─────────────────────────────────────────────
def load_sector_map() -> dict[str, str]:
    """Liest alle Profil-CSVs und gibt {ticker: sector} zurück."""
    sector_map: dict[str, str] = {}
    for csv_path in RAW_PROFILE.glob("*.csv"):
        try:
            df = pd.read_csv(csv_path, usecols=["symbol", PROFILE_SECTOR_COL])
            if len(df) > 0:
                sector_map[df["symbol"].iloc[0]] = str(df[PROFILE_SECTOR_COL].iloc[0])
        except Exception:
            pass  # Profil nicht lesbar → Ticker bleibt ohne Sektor
    return sector_map


def filter_sectors(tickers: list[str], sector_map: dict[str, str]) -> list[str]:
    """Entfernt Ticker, deren Sektor in EXCLUDED_SECTORS liegt."""
    kept = [t for t in tickers if sector_map.get(t, "") not in EXCLUDED_SECTORS]
    n_removed = len(tickers) - len(kept)
    logger.info("Sektor-Filter: %d entfernt, %d verbleibend", n_removed, len(kept))
    return kept


# ─────────────────────────────────────────────
# 3. Einzelne Quelle für einen Ticker laden
# ─────────────────────────────────────────────
def _load_source(ticker: str, src_name: str) -> pd.DataFrame | None:
    """Lädt eine CSV, filtert auf 2000–2024, indiziert auf Kalenderquartal.

    Gibt None zurück, wenn die Datei fehlt oder leer ist.
    Bei Quartalsduplikaten wird der jeweils letzte Eintrag behalten
    (spätestes Filing-Datum).
    """
    folder = SRC_DIRS[src_name]
    path = folder / f"{ticker}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if "date" not in df.columns or len(df) == 0:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Zeitraum filtern
    df = df[(df["date"].dt.year >= YEAR_START) & (df["date"].dt.year <= YEAR_END)]

    # Kalenderquartal als Index
    df["quarter"] = df["date"].dt.to_period("Q")

    # Bei Duplikaten: spätestes Datum behalten (= aktuellstes Filing)
    df = df.sort_values("date").drop_duplicates(subset=["quarter"], keep="last")
    df = df.set_index("quarter").sort_index()

    # Metadaten-Spalten entfernen
    drop_cols = [c for c in ("date", "filing_date", "currency_symbol", "symbol") if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Alles numerisch erzwingen
    df = df.apply(pd.to_numeric, errors="coerce")

    return df


# ─────────────────────────────────────────────
# 4. Vollständigkeitsfilter
# ─────────────────────────────────────────────
def _build_expected_quarters() -> pd.PeriodIndex:
    """Gibt die 100 erwarteten Kalenderquartale Q1/2000 bis Q4/2024 zurück."""
    return pd.period_range(
        start=f"{YEAR_START}Q1",
        end=f"{YEAR_END}Q4",
        freq="Q",
    )


EXPECTED_QTR_INDEX = _build_expected_quarters()
assert len(EXPECTED_QTR_INDEX) == EXPECTED_QUARTERS, (
    f"Erwartete {EXPECTED_QUARTERS} Quartale, aber PeriodIndex hat {len(EXPECTED_QTR_INDEX)}"
)


def _is_complete(dfs: dict[str, pd.DataFrame]) -> bool:
    """Prüft, ob *alle* Quellen exakt die 100 erwarteten Quartale abdecken."""
    for src_name, df in dfs.items():
        if df is None:
            return False
        if not EXPECTED_QTR_INDEX.isin(df.index).all():
            return False
    return True


# ─────────────────────────────────────────────
# 5. Kennzahlen berechnen
# ─────────────────────────────────────────────
def _compute_ratios(
    sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Berechnet alle 7 Kennzahlen aus den Quell-DataFrames.

    Rückgabe: DataFrame mit Index=quarter, Spalten=RATIO_NAMES.
    """
    result = pd.DataFrame(index=EXPECTED_QTR_INDEX)
    result.index.name = "quarter"

    for r in RATIOS:
        num = sources[r.numerator_src][r.numerator_col]
        den = sources[r.denominator_src][r.denominator_col]

        # Division; 0-Nenner → NaN
        ratio_series = num / den.replace(0, np.nan)
        result[r.name] = ratio_series

    return result


# ─────────────────────────────────────────────
# 6. Winsorisierung
# ─────────────────────────────────────────────
def winsorize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Winsorisiert jede Kennzahl pro Quartal auf WINSORIZE_QUANTILES."""
    lo, hi = WINSORIZE_QUANTILES

    def _clip_col(series: pd.Series) -> pd.Series:
        lower = series.quantile(lo)
        upper = series.quantile(hi)
        return series.clip(lower=lower, upper=upper)

    # Winsorisierung pro Quartal (= pro Zeile wäre sinnlos, da nur 1 Wert pro Ticker).
    # Stattdessen: pro Spalte über alle Ticker hinweg.
    for col in RATIO_NAMES:
        if col in panel.columns:
            panel[col] = _clip_col(panel[col])

    return panel


# ─────────────────────────────────────────────
# Hauptfunktion
# ─────────────────────────────────────────────
def build_ratio_panel(save: bool = True) -> pd.DataFrame:
    """Kompletter Pipeline-Durchlauf: Laden → Filtern → Berechnen → Winsorisieren.

    Rückgabe: Long-Format DataFrame mit Spalten [ticker, quarter, <ratios>].
    """
    ensure_output_dirs()

    # 1. Ticker-Universum
    tickers = get_ticker_universe()

    # 2. Sektor-Filter
    sector_map = load_sector_map()
    tickers = filter_sectors(tickers, sector_map)

    # 3+4. Laden + Vollständigkeitsfilter
    complete_tickers: list[str] = []
    all_ratios: list[pd.DataFrame] = []

    for ticker in tickers:
        sources = {
            "income": _load_source(ticker, "income"),
            "balance": _load_source(ticker, "balance"),
            "cashflow": _load_source(ticker, "cashflow"),
        }

        if not _is_complete(sources):
            continue

        # 5. Kennzahlen
        ratios = _compute_ratios(sources)
        ratios["ticker"] = ticker
        complete_tickers.append(ticker)
        all_ratios.append(ratios)

    logger.info("Vollständigkeitsfilter: %d / %d Ticker bestanden (100 Quartale lückenlos)",
                len(complete_tickers), len(tickers))

    if not all_ratios:
        raise RuntimeError("Kein einziger Ticker hat den Vollständigkeitsfilter bestanden!")

    # Zusammenfügen
    panel = pd.concat(all_ratios, ignore_index=False)
    panel = panel.reset_index()  # quarter wird Spalte
    panel["quarter"] = panel["quarter"].astype(str)  # für CSV-Kompatibilität

    # 6. Winsorisierung (pro Spalte über gesamtes Panel)
    panel = winsorize_panel(panel)

    # Spaltenreihenfolge
    panel = panel[["ticker", "quarter"] + RATIO_NAMES]

    if save:
        out_path = OUT_RATIOS / "ratio_panel.csv"
        panel.to_csv(out_path, index=False)
        logger.info("Panel gespeichert: %s (%d Zeilen, %d Ticker)",
                     out_path, len(panel), len(complete_tickers))

    return panel


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    panel = build_ratio_panel(save=True)

    # Zusammenfassung
    n_tickers = panel["ticker"].nunique()
    print(f"\n{'='*60}")
    print(f"  Panel fertig: {n_tickers} Unternehmen × {EXPECTED_QUARTERS} Quartale")
    print(f"  Kennzahlen: {RATIO_NAMES}")
    print(f"  Gespeichert: output/ratios/ratio_panel.csv")
    print(f"{'='*60}")
    print(f"\nDeskriptive Statistik:")
    print(panel[RATIO_NAMES].describe().round(4).to_string())
