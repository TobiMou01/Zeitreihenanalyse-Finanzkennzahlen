"""
config.py – Zentrale Konfiguration der Analyse-Pipeline
========================================================
Masterarbeit: Zeitreihenbasierte Analyse der Dynamik von Finanzkennzahlen
Autor: Tobias Mourier, TH Wildau, 2026

Methodische Entscheidungen:
  - AR(1) auf ersten Differenzen ΔY (nicht Levels)
  - Fixer Zeitraum 2000–2024 für alle Unternehmen
  - Nur Unternehmen mit lückenlosem Quartalsdatensatz (100 Quartale)
  - Einfache Volatilität σ(ΔY) als zweite Dimension (kein GARCH)
"""

from pathlib import Path
from dataclasses import dataclass


# ─────────────────────────────────────────────
# Pfade
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Rohdaten-Unterordner (je eine CSV pro Ticker)
RAW_BALANCE = RAW_DIR / "balance_sheets"
RAW_INCOME = RAW_DIR / "income_statements"
RAW_CASHFLOW = RAW_DIR / "cashflows"
RAW_PROFILE = RAW_DIR / "profiles"

# Output-Unterordner
OUT_RATIOS = OUTPUT_DIR / "ratios"        # berechnete Kennzahlen-Panels
OUT_AR = OUTPUT_DIR / "ar"                # AR(1)-Ergebnisse
OUT_CLUSTER = OUTPUT_DIR / "clustering"   # Klassifikation
OUT_PLOTS = OUTPUT_DIR / "plots"          # Visualisierungen


# ─────────────────────────────────────────────
# Zeitraum & Vollständigkeitsfilter
# ─────────────────────────────────────────────
YEAR_START = 2000
YEAR_END = 2024

EXPECTED_QUARTERS = (YEAR_END - YEAR_START + 1) * 4  # = 100
"""Q1 2000 bis Q4 2024 = 25 Jahre × 4 = 100 Quartale.
Nur Unternehmen, die für *alle* 100 Quartale einen Datenpunkt haben,
kommen in die Analyse. Harter Vollständigkeitsfilter – keine Lücken."""


# ─────────────────────────────────────────────
# Sektor-Exklusion
# ─────────────────────────────────────────────
EXCLUDED_SECTORS = frozenset([
    "Financial Services",
    "Banks",
    "Insurance",
])
"""Finanzsektor wird exkludiert, da fundamental andere Bilanzstruktur
(Leverage = Geschäftsmodell, nicht Risikofaktor). Vgl. Fama & French (1992)."""

PROFILE_SECTOR_COL = "sector"


# ─────────────────────────────────────────────
# Kennzahlen-Definitionen
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class RatioDef:
    """Definition einer Finanzkennzahl."""
    name: str               # interner Kurzname
    label: str              # Anzeigename für Plots/Tabellen
    category: str           # Profitabilität / Liquidität / Kapitalstruktur
    numerator_col: str      # Spalte im Quell-CSV (Zähler)
    numerator_src: str      # "income" | "balance" | "cashflow"
    denominator_col: str    # Spalte im Quell-CSV (Nenner)
    denominator_src: str    # "income" | "balance" | "cashflow"


RATIOS: list[RatioDef] = [
    # --- Profitabilität ---
    RatioDef(
        name="ROA",
        label="Return on Assets",
        category="Profitabilität",
        numerator_col="netIncome",
        numerator_src="income",
        denominator_col="totalAssets",
        denominator_src="balance",
    ),
    RatioDef(
        name="ROE",
        label="Return on Equity",
        category="Profitabilität",
        numerator_col="netIncome",
        numerator_src="income",
        denominator_col="totalStockholderEquity",
        denominator_src="balance",
    ),
    RatioDef(
        name="EBIT_margin",
        label="EBIT-Marge",
        category="Profitabilität",
        numerator_col="ebit",
        numerator_src="income",
        denominator_col="totalRevenue",
        denominator_src="income",
    ),
    RatioDef(
        name="fcf_margin",
        label="Free-Cashflow-Marge",
        category="Profitabilität",
        numerator_col="freeCashFlow",
        numerator_src="cashflow",
        denominator_col="totalRevenue",
        denominator_src="income",
    ),
    # --- Liquidität ---
    RatioDef(
        name="current_ratio",
        label="Current Ratio",
        category="Liquidität",
        numerator_col="totalCurrentAssets",
        numerator_src="balance",
        denominator_col="totalCurrentLiabilities",
        denominator_src="balance",
    ),
    # --- Kapitalstruktur ---
    RatioDef(
        name="debt_to_equity",
        label="Debt-to-Equity",
        category="Kapitalstruktur",
        numerator_col="shortLongTermDebtTotal",
        numerator_src="balance",
        denominator_col="totalStockholderEquity",
        denominator_src="balance",
    ),
    RatioDef(
        name="equity_ratio",
        label="Eigenkapitalquote",
        category="Kapitalstruktur",
        numerator_col="totalStockholderEquity",
        numerator_src="balance",
        denominator_col="totalAssets",
        denominator_src="balance",
    ),
]

RATIO_NAMES: list[str] = [r.name for r in RATIOS]


# ─────────────────────────────────────────────
# AR(1)-Modell auf ersten Differenzen
# ─────────────────────────────────────────────
AR_ORDER = 1
"""Lag-Ordnung des AR-Modells. Literaturstandard für Earnings-Persistenz."""

DIFF_ORDER = 1
"""Differenzierungsordnung: ΔY(t) = Y(t) − Y(t−1).
Erste Differenzen machen nicht-stationäre Reihen stationär und
verschieben den Fokus von Niveaus auf Veränderungsraten."""


# ─────────────────────────────────────────────
# Volatilitätsmaß (zweite Dimension für FF1)
# ─────────────────────────────────────────────
# Einfache Standardabweichung der Differenzen σ(ΔY).
# Kein GARCH – bei Quartalsdaten und heterogener Datenlänge
# liefert GARCH(1,1) nur bei ~20 % der Unternehmen
# signifikante Ergebnisse (vgl. Zhu et al. 2025).
# σ(ΔY) ist robust, immer berechenbar und interpretierbar.
VOL_MEASURE = "std_diff"


# ─────────────────────────────────────────────
# Klassifikation (FF1) – Median-Quadranten
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class QuadrantLabels:
    """Beschriftung der vier Median-Quadranten (φ₁ × σ)."""
    low_phi_low_vol: str = "STABIL-DYNAMISCH"     # geringe Persistenz, geringe Volatilität
    low_phi_high_vol: str = "VOLATIL-DYNAMISCH"    # geringe Persistenz, hohe Volatilität
    high_phi_low_vol: str = "STABIL-TRÄGE"         # hohe Persistenz, geringe Volatilität
    high_phi_high_vol: str = "VOLATIL-TRÄGE"       # hohe Persistenz, hohe Volatilität


QUADRANT_LABELS = QuadrantLabels()


# ─────────────────────────────────────────────
# Winsorisierung / Ausreißerbehandlung
# ─────────────────────────────────────────────
WINSORIZE_QUANTILES = (0.01, 0.99)
"""Kennzahlen werden pro Quartal auf das 1.–99. Perzentil gewinsorisiert,
um extreme Ausreißer (z. B. negative Equity → D/E = −∞) zu begrenzen."""


# ─────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────
def ensure_output_dirs() -> None:
    """Legt alle Output-Unterordner an, falls sie nicht existieren."""
    for d in (OUT_RATIOS, OUT_AR, OUT_CLUSTER, OUT_PLOTS):
        d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Quelldatei-Mapping (Kurzname → Pfad)
# ─────────────────────────────────────────────
SRC_DIRS: dict[str, Path] = {
    "income": RAW_INCOME,
    "balance": RAW_BALANCE,
    "cashflow": RAW_CASHFLOW,
    "profile": RAW_PROFILE,
}
