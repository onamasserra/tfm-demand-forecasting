"""
Shared visual style configuration for all TFM scripts.
Ensures consistent colors, fonts, and styling across all plots.

Palette derived from UOC brand colours (TFUOC.cls):
  darkblueUOC  RGB(1, 0, 115)   → #010073
  lightblueUOC RGB(147,234,252)  → #93EAFC
  airblueuoc   rgb(0.36,0.54,0.66) → #5C8AA8
  greyUOC      RGB(239,239,239) → #EFEFEF
"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ── UOC brand colours ───────────────────────────────────────────
DARK_BLUE   = "#010073"   # darkblueUOC — headings, primary
LIGHT_BLUE  = "#93EAFC"   # lightblueUOC — accents, fills
AIR_BLUE    = "#5C8AA8"   # airblueuoc — mid-tone
GREY        = "#EFEFEF"   # greyUOC — backgrounds

# ── Extended palette (harmonious with UOC brand) ────────────────
TEAL        = "#1A8A7D"   # complementary green-teal
AMBER       = "#D4850E"   # warm contrast for highlights
CORAL       = "#C0392B"   # red accent (errors, warnings)
SLATE       = "#6B7280"   # neutral grey for baselines

# ── General-purpose palette (for non-model plots) ──────────────
# When a plot needs only 2 colours, ALWAYS use PRIMARY + SECONDARY
# (the two UOC brand colours). Extra colours only when 3+ are needed.
PRIMARY     = DARK_BLUE    # main bars, single-series plots
SECONDARY   = LIGHT_BLUE   # second series, paired with PRIMARY
ACCENT      = TEAL         # highlights, best values (3rd colour)
WARM        = AMBER        # warm contrast (4th colour)
ALERT       = CORAL        # red accent for reference lines, warnings
NEUTRAL     = SLATE        # baselines, less important
PROMO_YES   = LIGHT_BLUE   # promotional  (UOC secondary)
PROMO_NO    = DARK_BLUE    # non-promotional (UOC primary)

# ── Consistent model color mapping ─────────────────────────────
MODEL_COLORS = {
    "XGBoost":        DARK_BLUE,   # primary brand colour
    "LightGBM":       TEAL,        # complementary teal
    "Random Forest":  AIR_BLUE,    # mid-tone blue
    "RandomForest":   AIR_BLUE,    # alias
    "ARIMA":          AMBER,       # warm contrast
    "ARIMA*":         AMBER,       # alias for annotated ARIMA
    "LSTM":           CORAL,       # red accent
    "Naive (lag-7)":  SLATE,       # neutral baseline
    "Naive_lag7":     SLATE,       # alias
}

# ── Categorical palette for bar charts (7 distinct colours) ────
CAT_PALETTE = [DARK_BLUE, AIR_BLUE, TEAL, AMBER, CORAL, LIGHT_BLUE, SLATE]

# ── Sequential colourmap for heatmaps ──────────────────────────
UOC_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "uoc_diverging",
    [CORAL, "#FFFFFF", DARK_BLUE],
    N=256,
)

# ── Seaborn / matplotlib global style ──────────────────────────
def apply_style():
    """Call this at the top of every script to set consistent styling."""
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "figure.dpi": 150,
        "figure.figsize": (12, 6),
        "axes.edgecolor": "#D1D5DB",
        "axes.linewidth": 0.8,
        "axes.prop_cycle": plt.cycler(color=CAT_PALETTE),
        "grid.alpha": 0.4,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


def get_model_color(name):
    """Return the consistent color for a model name."""
    return MODEL_COLORS.get(name, PRIMARY)


def get_model_colors(names):
    """Return a list of colors for a list of model names."""
    return [get_model_color(n) for n in names]
