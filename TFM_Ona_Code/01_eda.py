"""
01_EDA.py — Exploratory Data Analysis
TFM: Demand Forecasting for Inventory Management Using Machine Learning
Author: Ona Mas i Serra
Dataset: Datos.xlsx (4 sheets: Venta, Calendario, Promociones, Stock)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from style_config import (apply_style, PRIMARY, SECONDARY, ACCENT, NEUTRAL,
                          PROMO_YES, PROMO_NO, CAT_PALETTE, UOC_CMAP,
                          AIR_BLUE, DARK_BLUE, CORAL, AMBER, SLATE)

# ── Configuration ──────────────────────────────────────────────────
DATA_PATH = Path(__file__).resolve().parent.parent / "Datos.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()
plt.rcParams.update({"figure.figsize": (14, 6)})

def save_fig(name):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}.png", bbox_inches="tight")
    plt.close()
    print(f"   → Saved {name}.png")

# ══════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

venta = pd.read_excel(DATA_PATH, sheet_name="Venta")
calendario = pd.read_excel(DATA_PATH, sheet_name="Calendario")
promociones = pd.read_excel(DATA_PATH, sheet_name="Promociones")
stock = pd.read_excel(DATA_PATH, sheet_name="Stock")

# Parse idSecuencia as date
def parse_seq(s):
    return pd.to_datetime(s.astype(str), format="%Y%m%d")

venta["fecha"] = parse_seq(venta["idSecuencia"])
calendario["fecha"] = parse_seq(calendario["idSecuencia"])
stock["fecha"] = parse_seq(stock["idSecuencia"])
promociones["fecha_ini"] = parse_seq(promociones["idSecuenciaIni"])
promociones["fecha_fin"] = parse_seq(promociones["idSecuenciaFin"])

print(f"Venta:       {venta.shape}  |  {venta['fecha'].min().date()} → {venta['fecha'].max().date()}")
print(f"Calendario:  {calendario.shape}  |  {calendario['fecha'].min().date()} → {calendario['fecha'].max().date()}")
print(f"Stock:       {stock.shape}  |  {stock['fecha'].min().date()} → {stock['fecha'].max().date()}")
print(f"Promociones: {promociones.shape}  |  {promociones['fecha_ini'].min().date()} → {promociones['fecha_fin'].max().date()}")
print(f"Products:    {venta['producto'].nunique()}")
print(f"Days:        {venta['fecha'].nunique()}")


# ══════════════════════════════════════════════════════════════════
# 2. MERGE INTO SINGLE DATAFRAME
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. MERGING DATA")
print("=" * 60)

# Merge venta + stock
df = venta.merge(stock, on=["producto", "idSecuencia", "fecha"], how="left")

# Merge calendario
df = df.merge(calendario[["fecha", "bolOpen", "bolHoliday"]], on="fecha", how="left")

# Build promo flag: for each row, check if product is in promo on that date
# Efficient approach: expand promo ranges per product
promo_flags = []
for _, row in promociones.iterrows():
    dates = pd.date_range(row["fecha_ini"], row["fecha_fin"], freq="D")
    promo_flags.append(pd.DataFrame({
        "producto": row["producto"],
        "fecha": dates,
        "en_promo": 1
    }))

promo_df = pd.concat(promo_flags, ignore_index=True).drop_duplicates(subset=["producto", "fecha"])
df = df.merge(promo_df, on=["producto", "fecha"], how="left")
df["en_promo"] = df["en_promo"].fillna(0).astype(int)

# Add calendar features
df["year"] = df["fecha"].dt.year
df["month"] = df["fecha"].dt.month
df["day_of_week"] = df["fecha"].dt.dayofweek  # 0=Mon, 6=Sun
df["day_name"] = df["fecha"].dt.day_name()
df["week_of_year"] = df["fecha"].dt.isocalendar().week.astype(int)

df = df.sort_values(["producto", "fecha"]).reset_index(drop=True)

print(f"Merged shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Promo rows: {df['en_promo'].sum():,} ({100*df['en_promo'].mean():.1f}%)")
print(f"Nulls:\n{df.isnull().sum()}")
print(f"\nSample:\n{df.head(10)}")


# ══════════════════════════════════════════════════════════════════
# 3. DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. DESCRIPTIVE STATISTICS")
print("=" * 60)

print("\n── udsVenta (Sales) ──")
print(df["udsVenta"].describe())
print(f"Negative values: {(df['udsVenta'] < 0).sum()}")
print(f"Zero values:     {(df['udsVenta'] == 0).sum()} ({100*(df['udsVenta']==0).mean():.1f}%)")

print("\n── udsStock ──")
print(df["udsStock"].describe())
print(f"Negative values: {(df['udsStock'] < 0).sum()}")
print(f"Zero values:     {(df['udsStock'] == 0).sum()} ({100*(df['udsStock']==0).mean():.1f}%)")

# Stock breaks: sales=0 AND stock=0
stock_breaks = (df["udsVenta"] == 0) & (df["udsStock"] == 0)
print(f"\nStock breaks (sales=0 & stock=0): {stock_breaks.sum()} ({100*stock_breaks.mean():.1f}%)")

# Per-product summary
prod_stats = df.groupby("producto").agg(
    mean_sales=("udsVenta", "mean"),
    std_sales=("udsVenta", "std"),
    total_sales=("udsVenta", "sum"),
    zero_pct=("udsVenta", lambda x: (x == 0).mean() * 100),
    promo_pct=("en_promo", "mean"),
    n_days=("fecha", "nunique")
).reset_index()

print(f"\n── Per-product summary ──")
print(prod_stats.describe())

# ══════════════════════════════════════════════════════════════════
# 4. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. GENERATING PLOTS")
print("=" * 60)

# 4.1 Distribution of daily sales (overall)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["udsVenta"], bins=100, edgecolor="white", alpha=0.7, color=PRIMARY)
axes[0].set_title("Distribution of Daily Sales (udsVenta)")
axes[0].set_xlabel("Units Sold")
axes[0].set_ylabel("Frequency")
axes[0].set_xlim(-10, df["udsVenta"].quantile(0.99))

axes[1].hist(df["udsVenta"][df["udsVenta"] > 0], bins=100, edgecolor="white", alpha=0.7, color=SECONDARY)
axes[1].set_title("Distribution of Daily Sales (excluding zeros)")
axes[1].set_xlabel("Units Sold")
axes[1].set_ylabel("Frequency")
axes[1].set_xlim(0, df["udsVenta"].quantile(0.99))
save_fig("01_sales_distribution")

# 4.2 Sales by day of week
dow_sales = df.groupby("day_name")["udsVenta"].mean()
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow_sales = dow_sales.reindex(dow_order)

fig, ax = plt.subplots(figsize=(10, 5))
dow_sales.plot(kind="bar", ax=ax, color=PRIMARY, edgecolor="white")
ax.set_title("Average Daily Sales by Day of Week")
ax.set_ylabel("Mean Units Sold")
ax.set_xlabel("")
plt.xticks(rotation=45)
save_fig("02_sales_by_day_of_week")

# 4.3 Sales by month
monthly = df.groupby("month")["udsVenta"].mean()
fig, ax = plt.subplots(figsize=(10, 5))
monthly.plot(kind="bar", ax=ax, color=PRIMARY, edgecolor="white")
ax.set_title("Average Daily Sales by Month")
ax.set_ylabel("Mean Units Sold")
ax.set_xlabel("Month")
save_fig("03_sales_by_month")

# 4.4 Aggregate daily sales over time
daily_total = df.groupby("fecha")["udsVenta"].sum()
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(daily_total.index, daily_total.values, linewidth=0.5, alpha=0.7, color=PRIMARY)
ax.set_title("Total Daily Sales Over Time (All Products)")
ax.set_ylabel("Total Units Sold")
ax.set_xlabel("Date")
save_fig("04_daily_sales_timeseries")

# 4.5 Promo vs Non-promo sales
promo_comp = df.groupby("en_promo")["udsVenta"].agg(["mean", "median", "std", "count"])
promo_comp.index = ["Non-Promo", "Promo"]
print(f"\n── Promo vs Non-Promo Sales ──")
print(promo_comp)

fig, ax = plt.subplots(figsize=(8, 5))
df.groupby("en_promo")["udsVenta"].mean().plot(
    kind="bar", ax=ax, color=[PROMO_NO, PROMO_YES], edgecolor="white"
)
ax.set_xticklabels(["Non-Promo", "Promo"], rotation=0)
ax.set_title("Average Sales: Promo vs Non-Promo")
ax.set_ylabel("Mean Units Sold")
save_fig("05_promo_vs_nonpromo")

# 4.6 Open vs Closed days
open_comp = df.groupby("bolOpen")["udsVenta"].agg(["mean", "median", "count"])
open_comp.index = ["Closed", "Open"]
print(f"\n── Open vs Closed Day Sales ──")
print(open_comp)

# 4.7 Top 20 products by total sales
top20 = prod_stats.nlargest(20, "total_sales")
fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(top20["producto"].astype(str), top20["total_sales"], color=PRIMARY, edgecolor="white")
ax.set_title("Top 20 Products by Total Sales")
ax.set_xlabel("Total Units Sold")
ax.set_ylabel("Product ID")
ax.invert_yaxis()
save_fig("06_top20_products")

# 4.8 Correlation heatmap
corr_cols = ["udsVenta", "udsStock", "bolOpen", "bolHoliday", "en_promo", "month", "day_of_week"]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap=UOC_CMAP, center=0, ax=ax)
ax.set_title("Correlation Matrix")
save_fig("07_correlation_heatmap")


# ══════════════════════════════════════════════════════════════════
# 5. AUTOCORRELATION ANALYSIS (sample products)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. AUTOCORRELATION ANALYSIS")
print("=" * 60)

# Pick 4 representative products: high-volume, medium, low, and one with promos
sorted_prods = prod_stats.sort_values("total_sales", ascending=False)
sample_products = [
    sorted_prods.iloc[0]["producto"],    # highest sales
    sorted_prods.iloc[len(sorted_prods)//4]["producto"],  # Q1
    sorted_prods.iloc[len(sorted_prods)//2]["producto"],  # median
    prod_stats[prod_stats["promo_pct"] > 0].sort_values("total_sales", ascending=False).iloc[0]["producto"]  # top promo product
]
sample_products = [int(p) for p in sample_products]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, prod_id in zip(axes.flat, sample_products):
    series = df[df["producto"] == prod_id].set_index("fecha")["udsVenta"].sort_index()
    # Compute autocorrelation for lags 1-60
    acf_vals = [series.autocorr(lag=i) for i in range(1, 61)]
    ax.bar(range(1, 61), acf_vals, color=PRIMARY, alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(y=1.96/np.sqrt(len(series)), color=CORAL, linestyle="--", linewidth=0.8, label="95% CI")
    ax.axhline(y=-1.96/np.sqrt(len(series)), color=CORAL, linestyle="--", linewidth=0.8)
    ax.set_title(f"Product {prod_id} (mean={series.mean():.1f})")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("ACF")
    ax.legend(fontsize=8)

plt.suptitle("Autocorrelation Functions — Sample Products", fontsize=14)
save_fig("08_autocorrelation_sample")

# ══════════════════════════════════════════════════════════════════
# 6. SEASONALITY CHECK
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. SEASONALITY PATTERNS")
print("=" * 60)

# Weekly pattern (aggregate)
weekly_pattern = df.groupby("day_of_week")["udsVenta"].mean()
print("Weekly pattern (0=Mon):")
print(weekly_pattern)

# Monthly pattern by year
monthly_year = df.groupby(["year", "month"])["udsVenta"].mean().unstack(level=0)
fig, ax = plt.subplots(figsize=(12, 5))
# Explicit colours: 2 years → dark blue + light blue; 3+ → add teal, etc.
_year_colors = [PRIMARY, SECONDARY, ACCENT, NEUTRAL][:monthly_year.shape[1]]
monthly_year.plot(ax=ax, marker="o", color=_year_colors)
ax.set_title("Average Daily Sales by Month and Year")
ax.set_xlabel("Month")
ax.set_ylabel("Mean Units Sold")
ax.legend(title="Year")
save_fig("09_monthly_by_year")

# ══════════════════════════════════════════════════════════════════
# 7. DATA QUALITY SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. DATA QUALITY SUMMARY")
print("=" * 60)

neg_sales = (df["udsVenta"] < 0).sum()
neg_stock = (df["udsStock"] < 0).sum()
zero_sales = (df["udsVenta"] == 0).sum()
missing_cal = df["bolOpen"].isnull().sum()

print(f"Total rows:           {len(df):,}")
print(f"Products:             {df['producto'].nunique()}")
print(f"Date range:           {df['fecha'].min().date()} → {df['fecha'].max().date()}")
print(f"Days:                 {df['fecha'].nunique()}")
print(f"Negative sales:       {neg_sales:,}")
print(f"Negative stock:       {neg_stock:,}")
print(f"Zero sales:           {zero_sales:,} ({100*zero_sales/len(df):.1f}%)")
print(f"Stock breaks:         {stock_breaks.sum():,} ({100*stock_breaks.mean():.1f}%)")
print(f"Missing calendario:   {missing_cal}")
print(f"Promo observations:   {df['en_promo'].sum():,} ({100*df['en_promo'].mean():.1f}%)")

# Save merged dataframe for next step
PROCESSED_DIR = Path(__file__).resolve().parent / "output" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
df.to_parquet(PROCESSED_DIR / "merged_raw.parquet", index=False)
print(f"\n✓ Merged dataframe saved to output/processed/merged_raw.parquet")

print("\n" + "=" * 60)
print("EDA COMPLETE — All plots saved to output/eda/")
print("=" * 60)
