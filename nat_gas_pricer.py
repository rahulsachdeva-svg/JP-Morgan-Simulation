"""
Natural Gas Price Estimator
============================
Quantitative Research – Commodity Trading Desk

Takes a date as input and returns an estimated natural gas price,
using cubic spline interpolation over historical data and a
trend + seasonal (Fourier) extrapolation for future dates.

Usage:
    python nat_gas_pricer.py              # interactive prompt
    python nat_gas_pricer.py 2025-06-15   # CLI date argument
"""

import sys
import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ── 1. Load & clean data ──────────────────────────────────────────────────────

def load_data(path: str = "Nat_Gas.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = ["Date", "Price"]
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
    df = df.sort_values("Date").reset_index(drop=True)
    df["Price"] = df["Price"].astype(float)
    return df


# ── 2. Build the pricing model ────────────────────────────────────────────────

class NatGasPricer:
    """
    Interpolates within the historical range with a CubicSpline.
    Extrapolates outside that range with a fitted trend + seasonal model:

        P(t) = a + b*t + c*sin(2π*t/T + φ₁) + d*sin(4π*t/T + φ₂)

    where t is measured in years from the dataset start date and T=1.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._t0 = df["Date"].min()
        self._t_years = self._date_to_years(df["Date"])
        self._prices = df["Price"].values

        # Cubic spline for interpolation (covers the historical range)
        self._spline = CubicSpline(self._t_years, self._prices)

        # Trend + 2-harmonic seasonal model for extrapolation
        self._fit_extrapolation_model()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _date_to_years(self, dates) -> np.ndarray:
        """Convert dates to fractional years from t0."""
        delta = (pd.to_datetime(dates) - self._t0).dt.total_seconds()
        return (delta / (365.25 * 24 * 3600)).values

    def _seasonal_trend(self, t, a, b, c1, phi1, c2, phi2):
        """Trend + two Fourier harmonics (annual + semi-annual)."""
        return (
            a
            + b * t
            + c1 * np.sin(2 * np.pi * t + phi1)
            + c2 * np.sin(4 * np.pi * t + phi2)
        )

    def _fit_extrapolation_model(self):
        p0 = [10, 0.3, 0.5, 0, 0.2, 0]
        self._popt, _ = curve_fit(
            self._seasonal_trend,
            self._t_years,
            self._prices,
            p0=p0,
            maxfev=10_000,
        )

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def data_start(self) -> pd.Timestamp:
        return self._t0

    @property
    def data_end(self) -> pd.Timestamp:
        return self.df["Date"].max()

    def price(self, date) -> float:
        """Return estimated price for a given date (string or Timestamp)."""
        d = pd.to_datetime(date)
        t = self._date_to_years(pd.Series([d]))[0]

        t_min, t_max = self._t_years[0], self._t_years[-1]

        if t_min <= t <= t_max:
            # Inside the historical window → cubic spline
            return float(self._spline(t))
        else:
            # Outside the historical window → seasonal-trend model
            return float(self._seasonal_trend(t, *self._popt))

    def price_curve(self, start=None, end=None, freq="D") -> pd.DataFrame:
        """Return a DataFrame of daily estimated prices over a date range."""
        if start is None:
            start = self.data_start
        if end is None:
            end = self.data_end + pd.DateOffset(years=1)
        dates = pd.date_range(start, end, freq=freq)
        prices = [self.price(d) for d in dates]
        return pd.DataFrame({"Date": dates, "EstimatedPrice": prices})


# ── 3. Visualisation ──────────────────────────────────────────────────────────

def plot_analysis(pricer: NatGasPricer, save_path: str = "/mnt/user-data/outputs/nat_gas_analysis.png"):
    """
    4-panel figure:
      1. Full price curve (history + 1-yr extrapolation)
      2. Seasonal (monthly) box-plot
      3. Year-over-year comparison
      4. Residuals: spline vs seasonal model
    """
    df = pricer.df
    curve = pricer.price_curve()
    hist_curve = curve[curve["Date"] <= pricer.data_end]
    future_curve = curve[curve["Date"] > pricer.data_end]

    # Colour palette
    DARK = "#0d1b2a"
    BLUE = "#2196f3"
    CYAN = "#00e5ff"
    ORANGE = "#ff6b35"
    GREEN = "#69f0ae"
    GRAY = "#546e7a"
    WHITE = "#eceff1"

    fig = plt.figure(figsize=(18, 14), facecolor=DARK)
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35,
                  left=0.07, right=0.97, top=0.92, bottom=0.07)

    ax1 = fig.add_subplot(gs[0, :])   # full-width top
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(DARK)
        ax.tick_params(colors=WHITE, labelsize=9)
        ax.xaxis.label.set_color(WHITE)
        ax.yaxis.label.set_color(WHITE)
        ax.title.set_color(WHITE)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRAY)

    # ── Panel 1: Price curve ─────────────────────────────────────────────────
    ax1.plot(hist_curve["Date"], hist_curve["EstimatedPrice"],
             color=CYAN, lw=1.5, label="Spline (historical)", zorder=3)
    ax1.plot(future_curve["Date"], future_curve["EstimatedPrice"],
             color=ORANGE, lw=2, ls="--", label="Extrapolated (+1yr)", zorder=3)
    ax1.scatter(df["Date"], df["Price"],
                color=WHITE, s=28, zorder=5, label="Monthly snapshot", edgecolors=CYAN, lw=0.6)
    ax1.axvline(pricer.data_end, color=GRAY, lw=1.2, ls=":", alpha=0.7)
    ax1.text(pricer.data_end + pd.Timedelta(days=5),
             df["Price"].min() + 0.1,
             "Extrapolation →", color=GRAY, fontsize=8)
    ax1.set_title("Natural Gas Price – Historical & Extrapolated", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Price ($/MMBtu)", fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax1.legend(fontsize=8, framealpha=0.3, labelcolor=WHITE,
               facecolor="#1c2e40", edgecolor=GRAY)
    ax1.grid(axis="y", color=GRAY, alpha=0.25, lw=0.7)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Panel 2: Seasonal box-plot ───────────────────────────────────────────
    df_s = df.copy()
    df_s["Month"] = df_s["Date"].dt.month
    month_groups = [df_s[df_s["Month"] == m]["Price"].values for m in range(1, 13)]
    bp = ax2.boxplot(month_groups, patch_artist=True, medianprops=dict(color=CYAN, lw=2))
    colors_cycle = [BLUE if m in [11,12,1,2] else GREEN if m in [6,7,8] else ORANGE
                    for m in range(1, 13)]
    for patch, col in zip(bp["boxes"], colors_cycle):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    for whisker in bp["whiskers"]:
        whisker.set(color=GRAY, lw=1)
    for cap in bp["caps"]:
        cap.set(color=GRAY, lw=1)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=8)
    ax2.set_title("Seasonal Price Distribution by Month", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Price ($/MMBtu)", fontsize=9)
    ax2.grid(axis="y", color=GRAY, alpha=0.25, lw=0.7)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=BLUE, alpha=0.6, label="Winter (Nov–Feb)"),
        Patch(facecolor=GREEN, alpha=0.6, label="Summer (Jun–Aug)"),
        Patch(facecolor=ORANGE, alpha=0.6, label="Shoulder months"),
    ]
    ax2.legend(handles=legend_elements, fontsize=7.5, framealpha=0.3,
               labelcolor=WHITE, facecolor="#1c2e40", edgecolor=GRAY)

    # ── Panel 3: YoY overlaid line ───────────────────────────────────────────
    df_s["Year"] = df_s["Date"].dt.year
    palette = [BLUE, GREEN, ORANGE, CYAN]
    years = sorted(df_s["Year"].unique())
    for i, yr in enumerate(years):
        sub = df_s[df_s["Year"] == yr].copy()
        ax3.plot(sub["Month"], sub["Price"],
                 marker="o", markersize=4, lw=1.8,
                 color=palette[i % len(palette)], label=str(yr), alpha=0.85)
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"], fontsize=8)
    ax3.set_title("Year-over-Year Seasonal Overlay", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Price ($/MMBtu)", fontsize=9)
    ax3.set_xlabel("Month", fontsize=9)
    ax3.legend(fontsize=8, framealpha=0.3, labelcolor=WHITE,
               facecolor="#1c2e40", edgecolor=GRAY)
    ax3.grid(color=GRAY, alpha=0.25, lw=0.7)

    fig.suptitle("Natural Gas Price Analysis  |  Quantitative Research Desk",
                 fontsize=15, fontweight="bold", color=WHITE, y=0.97)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  Chart saved → {save_path}")


# ── 4. Main entry-point ───────────────────────────────────────────────────────

def main():
    # Load
    try:
        df = load_data("/mnt/user-data/uploads/Nat_Gas.csv")
    except FileNotFoundError:
        df = load_data("Nat_Gas.csv")

    pricer = NatGasPricer(df)

    print("\n" + "="*56)
    print("  Natural Gas Price Estimator – Commodity Desk Tool")
    print("="*56)
    print(f"  Historical range : {pricer.data_start.date()} → {pricer.data_end.date()}")
    print(f"  Extrapolation to : {(pricer.data_end + pd.DateOffset(years=1)).date()}")
    print("="*56)

    # Generate chart
    plot_analysis(pricer)

    # Date to price
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
    else:
        date_str = input("\n  Enter a date (YYYY-MM-DD): ").strip()

    try:
        est_price = pricer.price(date_str)
        d = pd.to_datetime(date_str)
        within = pricer.data_start <= d <= pricer.data_end
        tag = "interpolated" if within else "extrapolated"
        print(f"\n  Estimated price on {d.date()} : ${est_price:.4f} /MMBtu  [{tag}]")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n  Done.\n")
    return pricer   # useful for imports / notebooks


if __name__ == "__main__":
    main()
