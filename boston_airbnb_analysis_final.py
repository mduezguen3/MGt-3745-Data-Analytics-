"""
Group Project 2 - Data Analytics
Marketing Track: Boston Airbnb
MGT-3745, Spring 2026

RQ1: How does reputation (review sentiment, ratings, volume) relate to
     listing performance (price, occupancy, revenue proxy)?
RQ2: Which host and listing segments are most sensitive to reputation?

Components:
  1. Data cleaning & preprocessing
  2. Text analysis - VADER (L18) + LDA topic modeling (L19)
  3. Data wrangling & construction
  4. Data visualizations
  5. Regression analysis (L19-L22)

AI transparency:
  Component 1 - Claude (chat) helped us talk through cleaning decisions.
    All pandas code was written and tested by us.
  Component 2 - Claude (chat) explained how VADER and LDA work at a high
    level. We wrote the vectorizer setup, LDA fitting, and all aggregation.
  Component 3 - Claude (coding agent) gave us a starting skeleton for the
    merges. We verified every join and variable by hand.
  Components 4 & 5 - Claude (chat) suggested some plot types and model
    specs to consider. All interpretation is ours.
"""

import os
import glob
import warnings
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from scipy import stats

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 10})

# change this if the folder is somewhere else on your machine
BASE_DIR = (
    r"C:\Users\Mert\OneDrive - Georgia Institute of Technology"
    r"\Spring 2026\MGT 3745\marketing_boston_handout\marketing_boston_handout"
)
LISTINGS_DIR = os.path.join(BASE_DIR, "listings")
REVIEWS_DIR  = os.path.join(BASE_DIR, "reviews")
CALENDAR_DIR = os.path.join(BASE_DIR, "calendar")
OUT_DIR      = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

def extract_date(filepath):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(filepath))
    return pd.to_datetime(m.group(1)) if m else pd.NaT

# every 4th scrape keeps it from taking forever; flip to 1 for a full run
SCRAPE_STRIDE = 4


# =============================================================================
# DATA VALIDATION - sanity check before we do anything
# =============================================================================
print("\n" + "="*70)
print("DATA VALIDATION")
print("="*70)

listing_files_all = sorted(glob.glob(os.path.join(LISTINGS_DIR, "*.csv")))
review_files_all  = sorted(glob.glob(os.path.join(REVIEWS_DIR, "*.csv")))
cal_files_all     = sorted(glob.glob(os.path.join(CALENDAR_DIR, "*.csv")))

print(f"  listings folder : {len(listing_files_all)} files found")
print(f"  reviews folder  : {len(review_files_all)} files found")
print(f"  calendar folder : {len(cal_files_all)} files found")

if not listing_files_all:
    raise FileNotFoundError(f"No listing CSVs found in {LISTINGS_DIR}")
if not review_files_all:
    raise FileNotFoundError(f"No review CSVs found in {REVIEWS_DIR}")

# peek at one listing file to confirm expected columns exist
_sample_listing = pd.read_csv(listing_files_all[0], nrows=3, low_memory=False)
print(f"\n  Sample listing file: {os.path.basename(listing_files_all[0])}")
print(f"  Columns found ({len(_sample_listing.columns)}): {list(_sample_listing.columns[:10])} ...")

REQUIRED_LISTING_COLS = ["id", "price", "room_type", "neighbourhood_cleansed",
                         "review_scores_rating", "availability_30"]
missing_cols = [c for c in REQUIRED_LISTING_COLS if c not in _sample_listing.columns]
if missing_cols:
    print(f"  WARNING: expected columns not found: {missing_cols}")
else:
    print(f"  All required listing columns present")

_sample_review = pd.read_csv(review_files_all[0], nrows=3, low_memory=False)
print(f"\n  Sample review file: {os.path.basename(review_files_all[0])}")
print(f"  Columns: {list(_sample_review.columns)}")

_test_date = extract_date(listing_files_all[0])
print(f"\n  Date parsed from first listing filename: {_test_date}")
if pd.isna(_test_date):
    print("  WARNING: could not parse date from filename - check naming pattern")

dates_found = [d for d in [extract_date(f) for f in listing_files_all] if not pd.isna(d)]
if dates_found:
    print(f"  Scrape date range: {min(dates_found).date()} to {max(dates_found).date()}")

del _sample_listing, _sample_review


# =============================================================================
# COMPONENT 1 - DATA CLEANING & PREPROCESSING
# =============================================================================
print("\n" + "="*70)
print("COMPONENT 1 - DATA CLEANING & PREPROCESSING")
print("="*70)

listing_files_sampled = listing_files_all[::SCRAPE_STRIDE]
print(f"  Listing scrapes available : {len(listing_files_all)}")
print(f"  Listing scrapes used      : {len(listing_files_sampled)}")

LISTING_COLS = [
    "id", "host_id", "host_since", "host_is_superhost",
    "host_listings_count", "host_total_listings_count",
    "neighbourhood_cleansed", "latitude", "longitude",
    "property_type", "room_type", "accommodates",
    "bedrooms", "beds", "bathrooms_text",
    "price", "minimum_nights", "maximum_nights",
    "has_availability", "availability_30", "availability_60",
    "availability_90", "availability_365",
    "number_of_reviews", "number_of_reviews_ltm", "number_of_reviews_l30d",
    "first_review", "last_review",
    "review_scores_rating", "review_scores_accuracy",
    "review_scores_cleanliness", "review_scores_checkin",
    "review_scores_communication", "review_scores_location",
    "review_scores_value", "reviews_per_month",
    "instant_bookable", "calculated_host_listings_count",
]

frames = []
for fp in listing_files_sampled:
    try:
        df = pd.read_csv(fp, usecols=lambda c: c in LISTING_COLS, low_memory=False)
        df["scrape_date"] = extract_date(fp)
        frames.append(df)
    except Exception as e:
        print(f"  WARNING: {os.path.basename(fp)}: {e}")

listings_raw = pd.concat(frames, ignore_index=True)
print(f"  Raw rows after stacking: {len(listings_raw):,}")

# price comes in as "$125.00" - strip everything non-numeric
def parse_price(s):
    if pd.isna(s):
        return np.nan
    cleaned = re.sub(r"[^\d.]", "", str(s))
    return float(cleaned) if cleaned else np.nan

listings_raw["price_num"] = listings_raw["price"].apply(parse_price)

p99    = listings_raw["price_num"].quantile(0.99)
before = len(listings_raw)
listings_raw = listings_raw[listings_raw["price_num"].between(1, p99)]
print(f"  Dropped {before - len(listings_raw):,} rows (price = 0, NaN, or above 99th pct)")
print(f"  99th pct price cutoff: ${p99:,.0f}")

listings_raw["host_since"] = pd.to_datetime(listings_raw["host_since"], errors="coerce")
listings_raw["host_tenure_days"] = (
    listings_raw["scrape_date"] - listings_raw["host_since"]
).dt.days

# t/f strings to 1/0
for col in ["host_is_superhost", "has_availability", "instant_bookable"]:
    listings_raw[col] = (
        listings_raw[col]
        .map({"t": 1, "f": 0, True: 1, False: 0, 1: 1, 0: 0})
        .fillna(0).astype(int)
    )

def parse_baths(s):
    if pd.isna(s):
        return np.nan
    nums = re.findall(r"\d+\.?\d*", str(s))
    return float(nums[0]) if nums else np.nan

listings_raw["bathrooms"] = listings_raw["bathrooms_text"].apply(parse_baths)

key_cols = ["price_num", "review_scores_rating", "bedrooms",
            "host_tenure_days", "neighbourhood_cleansed"]
miss = listings_raw[key_cols].isna().mean().mul(100).round(1)
print("\n  Missing-value rates (%) in key columns:")
print(miss.to_string())

listings_raw.dropna(subset=["price_num", "scrape_date"], inplace=True)
# new listings have no reviews yet so rating NaN is expected - keep and flag
listings_raw["has_rating"] = listings_raw["review_scores_rating"].notna().astype(int)

print(f"\n  Clean rows       : {len(listings_raw):,}")
print(f"  Unique listings  : {listings_raw['id'].nunique():,}")
print(f"  Unique hosts     : {listings_raw['host_id'].nunique():,}")
print(f"  Scrape dates     : {sorted(listings_raw['scrape_date'].dt.date.unique())}")

# Inside Airbnb re-exports everything on every scrape so each file is
# cumulative. Load all of them then deduplicate by review id.
review_files = sorted(glob.glob(os.path.join(REVIEWS_DIR, "*.csv")))

rev_frames = []
for fp in review_files:
    try:
        df = pd.read_csv(fp, usecols=["listing_id", "id", "date", "comments"],
                         low_memory=False)
        rev_frames.append(df)
    except Exception as e:
        print(f"  WARNING: {os.path.basename(fp)}: {e}")

reviews_raw = pd.concat(rev_frames, ignore_index=True)
reviews_raw.drop_duplicates(subset=["id"], inplace=True)
reviews_raw["date"] = pd.to_datetime(reviews_raw["date"], errors="coerce")
reviews_raw.dropna(subset=["date"], inplace=True)

reviews_raw["comments"] = reviews_raw["comments"].astype(str).str.strip()
reviews_raw = reviews_raw[reviews_raw["comments"].str.len() > 10]

print(f"\n  Unique reviews after dedup: {len(reviews_raw):,}")
print(f"  Review date range: {reviews_raw['date'].min().date()} to {reviews_raw['date'].max().date()}")


# =============================================================================
# COMPONENT 2 - TEXT ANALYSIS
# Part A: VADER sentiment (L18)
# Part B: LDA topic modeling (L19)
# =============================================================================
print("\n" + "="*70)
print("COMPONENT 2 - TEXT ANALYSIS (VADER + LDA)")
print("="*70)

# ---- Part A: VADER ----
# we went with VADER because it was built for short informal text like this.
# handles caps, punctuation, etc. without needing labeled training data.
# compound score is [-1, +1] where >= 0.05 is positive.
analyzer = SentimentIntensityAnalyzer()

def vader_compound(text):
    return analyzer.polarity_scores(str(text))["compound"]

print("\n  Running VADER (batched for speed)...")
# running .apply() row by row on 640k reviews is slow - chunk it instead
_texts   = reviews_raw["comments"].tolist()
_chunk   = 50000
_results = []
for _i in range(0, len(_texts), _chunk):
    _batch = _texts[_i:_i+_chunk]
    _results.extend(analyzer.polarity_scores(t)["compound"] for t in _batch)
    print(f"    scored {min(_i+_chunk, len(_texts)):,} / {len(_texts):,}")
reviews_raw["sentiment"] = _results
del _texts, _results

def classify_sentiment(score):
    if score >= 0.05:  return "positive"
    if score <= -0.05: return "negative"
    return "neutral"

reviews_raw["sentiment_label"] = reviews_raw["sentiment"].apply(classify_sentiment)

dist = reviews_raw["sentiment_label"].value_counts(normalize=True).mul(100).round(1)
print("  Sentiment breakdown:")
print(dist.to_string())
print(f"  Mean compound score: {reviews_raw['sentiment'].mean():.4f}")

reviews_raw["year_month"]    = reviews_raw["date"].dt.to_period("M")
reviews_raw["year_month_ts"] = reviews_raw["year_month"].dt.to_timestamp()

review_agg = (
    reviews_raw
    .groupby(["listing_id", "year_month"])
    .agg(
        review_count   = ("id",             "count"),
        mean_sentiment = ("sentiment",       "mean"),
        pct_positive   = ("sentiment_label", lambda x: (x == "positive").mean()),
        pct_negative   = ("sentiment_label", lambda x: (x == "negative").mean()),
    )
    .reset_index()
    .rename(columns={"listing_id": "id"})
)

cumulative_sent = (
    reviews_raw
    .sort_values("date")
    .groupby("listing_id")
    .agg(
        cum_reviews        = ("id",             "count"),
        cum_mean_sentiment = ("sentiment",       "mean"),
        cum_pct_positive   = ("sentiment_label", lambda x: (x == "positive").mean()),
        cum_pct_negative   = ("sentiment_label", lambda x: (x == "negative").mean()),
    )
    .reset_index()
    .rename(columns={"listing_id": "id"})
)
print(f"\n  Monthly aggregates shape  : {review_agg.shape}")
print(f"  Cumulative sentiment shape: {cumulative_sent.shape}")

# ---- Part B: LDA ----
# VADER tells us positive/negative but not what guests actually talk about.
# LDA finds latent themes without any labels.
print("\n  Running LDA...")

docs = (
    reviews_raw
    .groupby("listing_id", as_index=False)
    .agg(
        text         = ("comments", lambda s: " ".join(s.astype(str))),
        review_count = ("id",       "count"),
    )
)
print(f"  Listings with review text: {len(docs):,}")

# generic words would swamp every topic and make them all look the same
EXTRA_STOP = {
    "boston", "airbnb", "place", "stay", "host", "home", "room",
    "apartment", "house", "great", "good", "nice", "clean", "ll",
    "ve", "re", "don", "didn", "doesn", "won", "isn", "aren",
    "wasn", "haven", "nt", "just", "really", "very", "also",
}
stop_words = sorted(ENGLISH_STOP_WORDS | EXTRA_STOP)

vectorizer = CountVectorizer(
    max_df      = 0.5,
    min_df      = 10,
    stop_words  = stop_words,
    ngram_range = (1, 1),
    max_features= 2000,
)
X = vectorizer.fit_transform(docs["text"])
feature_names = vectorizer.get_feature_names_out()
print(f"  Document-term matrix: {X.shape}")

n_topics = 5
lda = LatentDirichletAllocation(
    n_components    = n_topics,
    learning_method = "online",
    max_iter        = 20,
    random_state    = 42,
    n_jobs          = -1,
)
doc_topic = lda.fit_transform(X)
print(f"  Doc-topic matrix: {doc_topic.shape}")

print("\n  Top 12 words per topic:")
TOPIC_NAMES = {}
for topic_idx in range(n_topics):
    w         = lda.components_[topic_idx]
    top_idx   = w.argsort()[:-13:-1]
    top_terms = [feature_names[j] for j in top_idx]
    print(f"  Topic {topic_idx}: {', '.join(top_terms)}")
    TOPIC_NAMES[topic_idx] = f"Topic_{topic_idx}"

# update these after you run and read the output above
TOPIC_NAMES = {
    0: "Location & Neighborhood",
    1: "Cleanliness & Amenities",
    2: "Host Communication",
    3: "Value & Price",
    4: "Overall Experience",
}

dominant = doc_topic.argmax(axis=1)
docs = docs.assign(
    dominant_topic = dominant,
    topic_weight   = doc_topic.max(axis=1),
)
for t in range(n_topics):
    docs[f"lda_topic_{t}"] = doc_topic[:, t]

print("\n  Dominant topic distribution:")
print(docs["dominant_topic"].map(TOPIC_NAMES).value_counts().to_string())

docs_with_hood = docs.merge(
    listings_raw[["id", "neighbourhood_cleansed"]]
    .rename(columns={"id": "listing_id"})
    .drop_duplicates("listing_id"),
    on="listing_id", how="left"
)
docs_with_hood["topic_name"] = docs_with_hood["dominant_topic"].map(TOPIC_NAMES)

top_hoods_for_ct = docs_with_hood["neighbourhood_cleansed"].value_counts().head(8).index
ct = pd.crosstab(
    docs_with_hood.loc[
        docs_with_hood["neighbourhood_cleansed"].isin(top_hoods_for_ct),
        "neighbourhood_cleansed"
    ],
    docs_with_hood.loc[
        docs_with_hood["neighbourhood_cleansed"].isin(top_hoods_for_ct),
        "topic_name"
    ],
    normalize="index",
).round(3)

print("\n  Topic share by neighbourhood (rows sum to 1):")
print(ct.to_string())

lda_features = docs[
    ["listing_id"] + [f"lda_topic_{t}" for t in range(n_topics)]
    + ["dominant_topic", "topic_weight"]
]


# =============================================================================
# COMPONENT 3 - DATA WRANGLING & CONSTRUCTION
# =============================================================================
print("\n" + "="*70)
print("COMPONENT 3 - DATA WRANGLING & CONSTRUCTION")
print("="*70)

# unit of observation: listing x scrape_date

panel = listings_raw.merge(cumulative_sent, on="id", how="left")
_matched   = panel["cum_mean_sentiment"].notna().sum()
_unmatched = panel["cum_mean_sentiment"].isna().sum()
print(f"  After merging cumulative sentiment : {len(panel):,} rows")
print(f"    joined (have sentiment)          : {_matched:,}")
print(f"    no reviews yet (NaN sentiment)   : {_unmatched:,}  ({_unmatched/len(panel):.1%} of panel)")

panel["scrape_ym"]               = panel["scrape_date"].dt.to_period("M")
review_agg["year_month_str"]     = review_agg["year_month"].astype(str)
panel["scrape_ym_str"]           = panel["scrape_ym"].astype(str)

panel = panel.merge(
    review_agg.rename(columns={"year_month_str": "scrape_ym_str"}),
    on=["id", "scrape_ym_str"],
    how="left",
    suffixes=("", "_monthly"),
)
_m2_matched   = panel["review_count"].notna().sum()
_m2_unmatched = panel["review_count"].isna().sum()
print(f"  After merging monthly review agg   : {len(panel):,} rows")
print(f"    scrape-months with reviews       : {_m2_matched:,}")
print(f"    scrape-months with no reviews    : {_m2_unmatched:,}  ({_m2_unmatched/len(panel):.1%}) - expected for quiet months")

panel = panel.merge(
    lda_features.rename(columns={"listing_id": "id"}),
    on="id", how="left"
)
_m3_matched   = panel["dominant_topic"].notna().sum()
_m3_unmatched = panel["dominant_topic"].isna().sum()
print(f"  After merging LDA features         : {len(panel):,} rows")
print(f"    listings with LDA topic          : {_m3_matched:,}")
print(f"    listings with no reviews/topic   : {_m3_unmatched:,}  ({_m3_unmatched/len(panel):.1%})")

# occupancy and revenue proxy
# calendar CSVs are hundreds of MB per file and contain one row per listing per
# night for a 365-day forward window. Loading all of them would be impractical
# for a class project. availability_30 in the listings snapshot already summarizes
# exactly what we need: how many of the next 30 days are open. We use that.
print("\n  Calendar data approach:")
print("  We derive occupancy from availability_30 in the listings snapshots rather")
print("  than loading the raw calendar CSVs (which are 200-400 MB each).")
print("  availability_30 = days open in next 30-day window from each scrape.")
print("  occupancy_proxy_30 = 1 - (availability_30 / 30), clipped to [0, 1].")
print("  revenue_proxy_30  = price_num x estimated booked nights (30 - availability_30).")
print("  Limitations: does not capture dynamic nightly pricing, discounts, or")
print("  actual booking vs. blocked-by-host distinctions.")

panel["occupancy_proxy_30"] = 1 - (panel["availability_30"] / 30).clip(0, 1)
panel["revenue_proxy_30"]   = panel["price_num"] * (30 - panel["availability_30"].clip(0, 30))

# standardize both components then average - neither rating nor sentiment
# should dominate just because their raw scales are different
for col in ["review_scores_rating", "cum_mean_sentiment"]:
    mu = panel[col].mean()
    sd = panel[col].std()
    panel[f"{col}_z"] = (panel[col] - mu) / sd

panel["reputation_index"] = panel[["review_scores_rating_z", "cum_mean_sentiment_z"]].mean(axis=1)

panel["host_tier"] = pd.cut(
    panel["host_tenure_days"],
    bins   = [-1, 365, 365*3, 365*6, np.inf],
    labels = ["<1yr", "1-3yr", "3-6yr", "6yr+"],
)

panel["size_bucket"] = pd.cut(
    panel["accommodates"].fillna(2),
    bins   = [0, 2, 4, 6, np.inf],
    labels = ["1-2", "3-4", "5-6", "7+"],
)

panel["multi_listing_host"] = (panel["calculated_host_listings_count"] > 1).astype(int)
panel["log_price"]          = np.log1p(panel["price_num"])
panel["log_revenue_proxy"]  = np.log1p(panel["revenue_proxy_30"])

panel      = pd.get_dummies(panel, columns=["room_type"], prefix="rt", drop_first=False)
room_dummies = [c for c in panel.columns if c.startswith("rt_")]

print(f"\n  Final panel shape: {panel.shape}")
print(f"  Sample rows:")
print(panel[["id", "scrape_date", "price_num", "occupancy_proxy_30",
             "cum_mean_sentiment", "reputation_index", "host_tier",
             "dominant_topic"]].head(3).to_string())

panel.to_csv(os.path.join(OUT_DIR, "panel_analysis_ready.csv"), index=False)
print(f"\n  Saved panel_analysis_ready.csv")



warnings.filterwarnings("ignore")


# ── Design System (light theme) ───────────────────────────────────────────────
BG        = "#FAFAFA"
PANEL_BG  = "#FFFFFF"
GRID      = "#E8E8E8"
BORDER    = "#D0D0D0"
TEXT      = "#1A1A2E"
TEXT_SUB  = "#5A6272"
BLUE      = "#2563EB"
RED       = "#DC2626"
GREEN     = "#16A34A"
ORANGE    = "#EA580C"
PURPLE    = "#7C3AED"
TEAL      = "#0891B2"
GOLD      = "#D97706"
PINK      = "#DB2777"

# note on significance: *** p<0.01, ** p<0.05, * p<0.10
SIG_NOTE = "*** p<0.01   ** p<0.05   * p<0.10   HC3 robust SE"

matplotlib.rcParams.update({
    "figure.dpi": 180,
    "font.family": "DejaVu Sans",
    "axes.labelsize": 9.5,
    "axes.titlesize": 12,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
})

def style_ax(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(colors=TEXT_SUB)
    ax.xaxis.label.set_color(TEXT_SUB)
    ax.yaxis.label.set_color(TEXT_SUB)
    ax.set_title(ax.get_title(), color=TEXT, fontweight="bold", pad=12)
    ax.grid(color=GRID, linewidth=0.7, linestyle="-", alpha=1)
    ax.set_axisbelow(True)

def save(name):
    plt.savefig(os.path.join(OUT_DIR, name), bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✓ {name}")

def fig_tag(ax, tag):
    ax.text(1.0, -0.10, tag, transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=TEXT_SUB)

# ── Figure data: use panel already computed above ─────────────────────────────
df = panel.copy()
df["scrape_ym_ts"] = pd.to_datetime(df["scrape_ym"].astype(str), errors="coerce")

def get_rt(row):
    for col, lbl in [("rt_Entire home/apt","Entire home/apt"),
                     ("rt_Private room","Private room"),
                     ("rt_Shared room","Shared room"),
                     ("rt_Hotel room","Hotel room")]:
        if col in row.index and row[col] == 1:
            return lbl
    return None
df["room_type"] = df.apply(get_rt, axis=1)

print(f"\n  Panel ready for figures: {len(df):,} rows")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Price by Room Type
# ═══════════════════════════════════════════════════════════════════════════════
rt_order  = ["Entire home/apt","Private room","Shared room","Hotel room"]
rt_colors = [BLUE, GREEN, GOLD, RED]
rt_data   = [df[df["room_type"]==rt]["price_num"].dropna() for rt in rt_order]
rt_counts = [len(d) for d in rt_data]

fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot(rt_data, patch_artist=True, widths=0.55,
    medianprops=dict(color="white", linewidth=2.5),
    whiskerprops=dict(color=TEXT_SUB, linewidth=1.2),
    capprops=dict(color=TEXT_SUB, linewidth=1.2),
    flierprops=dict(marker="o", color=TEXT_SUB, alpha=0.25, markersize=2.5),
    showfliers=True)
for patch, color in zip(bp["boxes"], rt_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.78)
    patch.set_edgecolor("white")
    patch.set_linewidth(1.2)

ax.set_xticks(range(1,5))
ax.set_xticklabels(
    [f"{rt}\n(n={c:,})" for rt,c in zip(rt_order,rt_counts)], color=TEXT_SUB)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
ax.set_ylabel("Nightly Price")
ax.set_title("Nightly Price Distribution by Room Type — Boston Airbnb")
style_ax(ax, fig)

medians = [d.median() for d in rt_data]
for i,(med,color) in enumerate(zip(medians, rt_colors),1):
    ax.text(i, med+10, f"Median: ${med:,.0f}",
            ha="center", va="bottom", color=color, fontsize=8, fontweight="bold")

ax.text(0.02, 0.97, "Outliers shown above whiskers (cap at 99th pct)",
        transform=ax.transAxes, color=TEXT_SUB, fontsize=7.5, va="top")
fig_tag(ax, "Fig 1")
plt.tight_layout()
save("fig1_price_by_room_type.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Sentiment Distribution (cumulative listing-level VADER)
# ═══════════════════════════════════════════════════════════════════════════════
sent = df["cum_mean_sentiment"].dropna()
mean_s  = sent.mean()
pct_pos = (sent >= 0.05).mean()*100
pct_neg = (sent <= -0.05).mean()*100
pct_neu = 100 - pct_pos - pct_neg

fig, ax = plt.subplots(figsize=(9, 5))
n, bins, patches = ax.hist(sent, bins=55, edgecolor="white", linewidth=0.4)
for patch, left in zip(patches, bins[:-1]):
    if left >= 0.05:
        patch.set_facecolor(GREEN); patch.set_alpha(0.75)
    elif left <= -0.05:
        patch.set_facecolor(RED);   patch.set_alpha(0.75)
    else:
        patch.set_facecolor(GOLD);  patch.set_alpha(0.75)

ax.axvline(0.05,  color=GREEN, linewidth=1.2, linestyle="--", alpha=0.7)
ax.axvline(-0.05, color=RED,   linewidth=1.2, linestyle="--", alpha=0.7)
ax.axvline(mean_s, color=TEXT, linewidth=2.0, linestyle="-",
           label=f"Mean = {mean_s:.3f}")

legend_h = [
    mpatches.Patch(color=GREEN, alpha=0.75, label=f"Positive ≥ 0.05  ({pct_pos:.1f}% of listings)"),
    mpatches.Patch(color=GOLD,  alpha=0.75, label=f"Neutral            ({pct_neu:.1f}% of listings)"),
    mpatches.Patch(color=RED,   alpha=0.75, label=f"Negative ≤ −0.05 ({pct_neg:.1f}% of listings)"),
    plt.Line2D([0],[0], color=TEXT, lw=2, label=f"Mean = {mean_s:.3f}"),
]
ax.legend(handles=legend_h, framealpha=0.9, edgecolor=BORDER)
ax.set_xlabel("Cumulative VADER Compound Score per Listing  (−1 to +1)")
ax.set_ylabel("Number of Listings")
ax.set_title("Distribution of Cumulative Review Sentiment by Listing (VADER)")
style_ax(ax, fig)
fig_tag(ax, "Fig 2")
plt.tight_layout()
save("fig2_sentiment_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Sentiment Over Time (22 scrape-month observations)
# ═══════════════════════════════════════════════════════════════════════════════
monthly = (df.dropna(subset=["cum_mean_sentiment","scrape_ym_ts"])
           .groupby("scrape_ym_ts")
           .agg(mean_sent=("cum_mean_sentiment","mean"), n=("id","count"))
           .reset_index()
           .sort_values("scrape_ym_ts"))
monthly = monthly[monthly["n"] >= 20]

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(monthly["scrape_ym_ts"], monthly["mean_sent"],
        color=BLUE, linewidth=2.2, marker="o", markersize=5,
        markerfacecolor=BLUE, markeredgecolor="white", markeredgewidth=1, zorder=3)
ax.fill_between(monthly["scrape_ym_ts"], monthly["mean_sent"],
                monthly["mean_sent"].min()-0.002,
                alpha=0.10, color=BLUE)

ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"),
           alpha=0.08, color=RED, zorder=1, label="COVID-19 period")
ax.text(pd.Timestamp("2020-09-15"), monthly["mean_sent"].max()-0.001,
        "COVID-19", color=RED, fontsize=8, ha="center", alpha=0.9)

mean_line = monthly["mean_sent"].mean()
ax.axhline(mean_line, color=GOLD, linestyle="--", linewidth=1.2, alpha=0.8,
           label=f"Period average = {mean_line:.3f}")

ax.set_xlabel("Scrape Month")
ax.set_ylabel("Mean Cumulative VADER Sentiment")
ax.set_ylim(monthly["mean_sent"].min()-0.01, monthly["mean_sent"].max()+0.01)
ax.set_title("Mean Listing Reputation Sentiment Over Time — Boston Airbnb  (2015–2023)")
ax.legend(framealpha=0.9, edgecolor=BORDER)
ax.text(0.02, 0.05, f"n = {len(monthly)} scrape-month observations",
        transform=ax.transAxes, color=TEXT_SUB, fontsize=7.5)
style_ax(ax, fig)
fig_tag(ax, "Fig 3")
plt.tight_layout()
save("fig3_sentiment_over_time.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Review Score vs. Price (rating is 0–100 scale in this dataset)
# ═══════════════════════════════════════════════════════════════════════════════
plot4 = df.dropna(subset=["review_scores_rating","price_num"])
plot4 = plot4[plot4["review_scores_rating"] > 20].sample(min(6000, len(plot4)), random_state=42)

m, b, r, p_val, _ = stats.linregress(plot4["review_scores_rating"], plot4["price_num"])
x_rng = np.linspace(plot4["review_scores_rating"].min(),
                    plot4["review_scores_rating"].max(), 200)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.scatter(plot4["review_scores_rating"], plot4["price_num"],
           alpha=0.07, s=5, color=BLUE, rasterized=True)
ax.plot(x_rng, m*x_rng+b, color=RED, linewidth=2.5, zorder=5,
        label=f"OLS  r = {r:.3f}  (p {'< 0.001' if p_val < 0.001 else f'= {p_val:.3f}'})")

ax.set_xlabel("Overall Review Score  (0–100 scale, higher = better)")
ax.set_ylabel("Nightly Price ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
ax.set_title("Review Score vs. Nightly Price")
ax.legend(framealpha=0.9, edgecolor=BORDER)
ax.text(0.02, 0.96,
        "Note: most listings cluster between 80–100\n(ceiling effect — Airbnb review inflation)",
        transform=ax.transAxes, color=TEXT_SUB, fontsize=7.5, va="top",
        bbox=dict(facecolor="white", edgecolor=BORDER, alpha=0.8, boxstyle="round,pad=0.3"))
style_ax(ax, fig)
fig_tag(ax, "Fig 4")
plt.tight_layout()
save("fig4_rating_vs_price.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Sentiment vs. Occupancy Proxy
# ═══════════════════════════════════════════════════════════════════════════════
plot5 = df.dropna(subset=["cum_mean_sentiment","occupancy_proxy_30"]).sample(
    min(6000, len(df)), random_state=42)
m2, b2, r2, p2, _ = stats.linregress(plot5["cum_mean_sentiment"], plot5["occupancy_proxy_30"])
x2 = np.linspace(plot5["cum_mean_sentiment"].min(), plot5["cum_mean_sentiment"].max(), 200)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.scatter(plot5["cum_mean_sentiment"], plot5["occupancy_proxy_30"],
           alpha=0.08, s=5, color=TEAL, rasterized=True)
ax.plot(x2, m2*x2+b2, color=RED, linewidth=2.5, zorder=5,
        label=f"OLS  r = {r2:.3f}  (p {'< 0.001' if p2 < 0.001 else f'= {p2:.3f}'})")

ax.set_xlabel("Cumulative Mean VADER Sentiment (per listing)")
ax.set_ylabel("30-Day Occupancy Proxy  (0 = fully open, 1 = fully booked)")
ax.set_title("Cumulative Sentiment vs. Occupancy Proxy")
ax.legend(framealpha=0.9, edgecolor=BORDER)
ax.text(0.02, 0.96,
        "Occupancy proxy = 1 − (availability_30 / 30)",
        transform=ax.transAxes, color=TEXT_SUB, fontsize=7.5, va="top")
style_ax(ax, fig)
fig_tag(ax, "Fig 5")
plt.tight_layout()
save("fig5_sentiment_vs_occupancy.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Reputation Index by Neighbourhood
# ═══════════════════════════════════════════════════════════════════════════════
top_hoods = df["neighbourhood_cleansed"].value_counts().head(15).index
hood_df = (df[df["neighbourhood_cleansed"].isin(top_hoods)]
           .groupby("neighbourhood_cleansed")
           .agg(mean_rep=("reputation_index","mean"), n=("id","count"))
           .sort_values("mean_rep").reset_index())

bar_colors = [RED if v < 0 else GREEN for v in hood_df["mean_rep"]]
alphas = [0.55 + 0.4*abs(v)/hood_df["mean_rep"].abs().max() for v in hood_df["mean_rep"]]

fig, ax = plt.subplots(figsize=(9, 7))
for i,(row,color,alpha) in enumerate(zip(hood_df.itertuples(), bar_colors, alphas)):
    ax.barh(i, row.mean_rep, color=color, alpha=alpha, edgecolor="none", height=0.65)
    offset = 0.004 if row.mean_rep >= 0 else -0.004
    ha     = "left"                if row.mean_rep >= 0 else "right"
    ax.text(row.mean_rep + offset, i,
            f"{row.mean_rep:+.3f}  (n={row.n:,})",
            va="center", ha=ha, color=TEXT_SUB, fontsize=7.8)

ax.axvline(0, color=BORDER, linewidth=1.5)
ax.set_yticks(range(len(hood_df)))
ax.set_yticklabels(hood_df["neighbourhood_cleansed"], fontsize=9, color=TEXT)
ax.set_xlabel("Mean Reputation Index (standardized composite of rating + VADER sentiment)")
ax.set_title("Mean Reputation Index by Neighbourhood — Top 15 by Listing Count")

legend_h = [mpatches.Patch(color=GREEN, alpha=0.75, label="Above average reputation"),
            mpatches.Patch(color=RED,   alpha=0.75, label="Below average reputation")]
ax.legend(handles=legend_h, framealpha=0.9, edgecolor=BORDER, loc="lower right")
style_ax(ax, fig)
fig_tag(ax, "Fig 6")
plt.tight_layout()
save("fig6_reputation_by_neighbourhood.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — LDA Topic Distribution
# ═══════════════════════════════════════════════════════════════════════════════
TOPIC_NAMES = {0:"Location &\nNeighborhood", 1:"Cleanliness &\nAmenities",
               2:"Host\nCommunication", 3:"Value &\nPrice", 4:"Overall\nExperience"}
topic_counts = (df.dropna(subset=["dominant_topic"])
                .groupby("dominant_topic")["id"].nunique()
                .reset_index())
topic_counts["name"] = topic_counts["dominant_topic"].map(TOPIC_NAMES)
topic_counts = topic_counts.sort_values("id", ascending=False)

topic_colors = [BLUE, GREEN, ORANGE, GOLD, PURPLE]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(range(len(topic_counts)), topic_counts["id"],
              color=topic_colors[:len(topic_counts)], alpha=0.82,
              edgecolor="white", linewidth=1.2, width=0.6)
ax.set_xticks(range(len(topic_counts)))
ax.set_xticklabels(topic_counts["name"], fontsize=9, color=TEXT)
for bar, val in zip(bars, topic_counts["id"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
            f"{val:,}", ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")
ax.set_ylabel("Number of Unique Listings with Reviews")
ax.set_title("Listings by Dominant Review Topic  (LDA, 5 Topics — L19)")
ax.set_ylim(0, topic_counts["id"].max()*1.13)
ax.text(0.02, 0.97,
        "Dominant topic = highest LDA weight per listing\nOnly listings with ≥1 review included",
        transform=ax.transAxes, color=TEXT_SUB, fontsize=7.5, va="top")
style_ax(ax, fig)
fig_tag(ax, "Fig 7")
plt.tight_layout()
save("fig7_lda_topic_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Sentiment Share by Room Type
# ═══════════════════════════════════════════════════════════════════════════════
rt_sent = (df.dropna(subset=["room_type","cum_mean_sentiment"])
           .assign(sent_label=lambda d: d["cum_mean_sentiment"].apply(
               lambda s: "Positive" if s>=0.05 else ("Negative" if s<=-0.05 else "Neutral")))
           .groupby(["room_type","sent_label"])["id"].count()
           .reset_index(name="count"))
rt_sent["total"] = rt_sent.groupby("room_type")["count"].transform("sum")
rt_sent["pct"]   = rt_sent["count"] / rt_sent["total"] * 100
pivot = rt_sent.pivot(index="room_type", columns="sent_label", values="pct").fillna(0)
for col in ["Positive","Neutral","Negative"]:
    if col not in pivot.columns:
        pivot[col] = 0
pivot = pivot[["Positive","Neutral","Negative"]]

fig, ax = plt.subplots(figsize=(9.5, 5))
x = np.arange(len(pivot))
w = 0.26
b1 = ax.bar(x-w, pivot["Positive"], w, color=GREEN,  alpha=0.80, label="Positive  (≥ 0.05)", edgecolor="white", linewidth=0.8)
b2 = ax.bar(x,   pivot["Neutral"],  w, color=GOLD,   alpha=0.80, label="Neutral",            edgecolor="white", linewidth=0.8)
b3 = ax.bar(x+w, pivot["Negative"], w, color=RED,    alpha=0.80, label="Negative  (≤ −0.05)",edgecolor="white", linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(pivot.index, fontsize=9, color=TEXT)
ax.set_ylabel("Share of Listings with Reviews (%)")
ax.set_title("Sentiment Label Share by Room Type  (Listing-Level Cumulative Sentiment)")
ax.legend(framealpha=0.9, edgecolor=BORDER)
ax.set_ylim(0, 100)

for bars_group in [b1, b2, b3]:
    for rect in bars_group:
        h = rect.get_height()
        if h > 4:
            ax.text(rect.get_x()+rect.get_width()/2, h+0.8,
                    f"{h:.0f}%", ha="center", va="bottom", color=TEXT_SUB, fontsize=7.2)
style_ax(ax, fig)
fig_tag(ax, "Fig 8")
plt.tight_layout()
save("fig8_sentiment_by_room_type.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9 — Occupancy by Host Tenure Tier
# ═══════════════════════════════════════════════════════════════════════════════
tier_order = ["<1yr","1-3yr","3-6yr","6yr+"]
tier_data  = (df.dropna(subset=["host_tier","occupancy_proxy_30"])
              .groupby("host_tier", observed=True)
              .agg(mean_occ=("occupancy_proxy_30","mean"),
                   sem_occ=("occupancy_proxy_30", lambda x: x.sem()),
                   n=("id","count"))
              .reindex(tier_order).reset_index())

blue_alphas = [0.45, 0.62, 0.78, 0.95]

fig, ax = plt.subplots(figsize=(8, 5))
for i,(row,alpha) in enumerate(zip(tier_data.itertuples(), blue_alphas)):
    ax.bar(i, row.mean_occ, color=BLUE, alpha=alpha, edgecolor="white",
           linewidth=1.2, width=0.55)
    ax.errorbar(i, row.mean_occ, yerr=row.sem_occ, fmt="none",
                color=TEXT, capsize=6, linewidth=1.8, capthick=1.8)
    ax.text(i, row.mean_occ + row.sem_occ + 0.012,
            f"{row.mean_occ:.3f}\n(n={row.n:,})",
            ha="center", va="bottom", color=TEXT, fontsize=8, fontweight="bold")

ax.set_xticks(range(4))
ax.set_xticklabels(tier_order, fontsize=9.5, color=TEXT)
ax.set_ylabel("Mean 30-Day Occupancy Proxy")
ax.set_ylim(0, tier_data["mean_occ"].max()*1.20)
ax.set_title("30-Day Occupancy Proxy by Host Tenure Tier")
ax.text(0.02, 0.05,
        "Error bars = ±1 SEM  |  Occupancy proxy = 1 − (availability_30 / 30)",
        transform=ax.transAxes, color=TEXT_SUB, fontsize=7.5)
style_ax(ax, fig)
fig_tag(ax, "Fig 9")
plt.tight_layout()
save("fig9_occupancy_by_host_tier.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 10 — Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
heat_cols = ["price_num","occupancy_proxy_30","revenue_proxy_30",
             "review_scores_rating","cum_mean_sentiment","reputation_index",
             "accommodates","host_tenure_days","calculated_host_listings_count",
             "number_of_reviews"]
heat_labels = ["Price","Occupancy\nProxy","Revenue\nProxy","Review\nScore (0–100)",
               "VADER\nSentiment","Reputation\nIndex","Accommodates",
               "Host Tenure\n(days)","Host Listings\nCount","Review\nCount"]

corr = df[heat_cols].dropna().corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(10.5, 9))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL_BG)

cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
            center=0, vmin=-0.8, vmax=0.8,
            ax=ax, linewidths=0.5, linecolor="#F0F0F0",
            square=True, cbar_kws={"shrink":0.7, "label":"Pearson r"},
            annot_kws={"size":8.5, "color":TEXT})

ax.set_xticklabels(heat_labels, rotation=45, ha="right", fontsize=8.5, color=TEXT)
ax.set_yticklabels(heat_labels, rotation=0, fontsize=8.5, color=TEXT)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title("Correlation Heatmap — Key Panel Variables  (lower triangle)",
             color=TEXT, fontweight="bold", pad=12, fontsize=12)
ax.text(0.02, -0.10, "Fig 10", transform=ax.transAxes,
        ha="left", va="top", fontsize=8, color=TEXT_SUB)
plt.tight_layout()
save("fig10_correlation_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 11 — Coefficient Plot (Model 3: rating + sentiment + controls)
# Real values from regression output
# ═══════════════════════════════════════════════════════════════════════════════
coef_data = [
    ("VADER Sentiment",       0.2958,  0.2442,  0.3474, "***"),
    ("Accommodates",          0.1724,  0.1671,  0.1777, "***"),
    ("Multi-Listing Host",   -0.0484, -0.0666, -0.0302, "***"),
    ("Is Superhost",         -0.0569, -0.0777, -0.0361, "***"),
    ("Minimum Nights",       -0.0019, -0.0023, -0.0015, "***"),
    ("Host Tenure (days)",    0.0001,  0.0001,  0.0001, "***"),
    ("Review Score (0–100)", -0.0010, -0.0012, -0.0008, "***"),
]
# sort by coef
coef_data.sort(key=lambda x: x[1])

labels   = [d[0] for d in coef_data]
coefs    = [d[1] for d in coef_data]
lows     = [d[2] for d in coef_data]
highs    = [d[3] for d in coef_data]
stars    = [d[4] for d in coef_data]
colors11 = [GREEN if c > 0 else RED for c in coefs]

fig, ax = plt.subplots(figsize=(9.5, 5.5))
for i,(c,lo,hi,star,color) in enumerate(zip(coefs, lows, highs, stars, colors11)):
    ax.plot([lo,hi],[i,i], color=color, linewidth=2.2, alpha=0.55, solid_capstyle="round")
    ax.scatter(c, i, color=color, s=80, zorder=5,
               edgecolors="white", linewidth=1.2)
    offset = (highs[-1]-lows[0])*0.015
    ax.text(hi+offset, i, star, va="center", color=color,
            fontsize=9, fontweight="bold")

ax.axvline(0, color=BORDER, linewidth=1.5)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9.5, color=TEXT)
ax.set_xlabel("OLS Coefficient  (95% Confidence Interval, HC3 Robust Standard Errors)")
ax.set_title("Coefficient Plot — Model 3: log(Price) ~ Review Score + VADER Sentiment + Controls")
ax.text(0.02, -0.10, SIG_NOTE, transform=ax.transAxes,
        ha="left", va="top", fontsize=7.5, color=TEXT_SUB)

legend_h = [
    plt.Line2D([0],[0], color=GREEN, lw=2, marker="o", ms=7,
               markerfacecolor=GREEN, markeredgecolor="white", label="Positive effect on log(price)"),
    plt.Line2D([0],[0], color=RED,   lw=2, marker="o", ms=7,
               markerfacecolor=RED,   markeredgecolor="white", label="Negative effect on log(price)"),
]
ax.legend(handles=legend_h, framealpha=0.9, edgecolor=BORDER, loc="lower right")
style_ax(ax, fig)
plt.tight_layout()
save("fig11_coef_plot.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 12 — Partial Regression (proper residual-on-residual)
# ═══════════════════════════════════════════════════════════════════════════════
pr_df = df.dropna(subset=["log_price","reputation_index",
                           "accommodates","host_tenure_days","minimum_nights"]).sample(
    min(8000, len(df)), random_state=42)

controls = ["accommodates","host_tenure_days","minimum_nights"]
X_c  = add_constant(pr_df[controls].astype(float))
e_y  = OLS(pr_df["log_price"].astype(float), X_c).fit().resid
e_ri = OLS(pr_df["reputation_index"].astype(float), X_c).fit().resid

m_pr, b_pr, r_pr, p_pr, _ = stats.linregress(e_ri, e_y)
x_pr = np.linspace(np.percentile(e_ri,1), np.percentile(e_ri,99), 200)

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(e_ri, e_y, alpha=0.06, s=5, color=BLUE, rasterized=True)
ax.plot(x_pr, m_pr*x_pr+b_pr, color=RED, linewidth=2.5, zorder=5,
        label=f"Partial slope = {m_pr:+.4f}   r = {r_pr:.3f}   (p {'< 0.001' if p_pr < 0.001 else f'= {p_pr:.3f}'})")

ax.set_xlabel("Residual of Reputation Index  |  Controls")
ax.set_ylabel("Residual of log(Price)  |  Controls")
ax.set_title("Partial Regression — Reputation Index → log(Price)\n"
             "After partialling out: accommodates, host tenure, minimum nights")
ax.legend(framealpha=0.9, edgecolor=BORDER)
ax.text(0.02, 0.97,
        "Each axis = residual after removing effect of control variables.\n"
        "Slope = unique contribution of reputation index to log(price).",
        transform=ax.transAxes, color=TEXT_SUB, fontsize=7.5, va="top")
style_ax(ax, fig)
fig_tag(ax, "Fig 12")
plt.tight_layout()
save("fig12_partial_regression.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 13 — Within-Listing Price Variation
# ═══════════════════════════════════════════════════════════════════════════════
panel_reg = df.dropna(subset=["log_price","reputation_index","scrape_date"]).copy()
counts_p  = panel_reg.groupby("id")["scrape_date"].count()
panel_reg = panel_reg[panel_reg["id"].isin(counts_p[counts_p>=2].index)]

sample_ids = (panel_reg.groupby("id")["log_price"].std()
              .dropna().nlargest(8).index.tolist())
line_colors13 = [BLUE, RED, GREEN, ORANGE, PURPLE, TEAL, GOLD, PINK]

fig, ax = plt.subplots(figsize=(10, 5))
for lid, color in zip(sample_ids, line_colors13):
    sub = panel_reg[panel_reg["id"]==lid].sort_values("scrape_date")
    ax.plot(sub["scrape_date"], sub["log_price"], marker="o", markersize=6,
            linewidth=2, alpha=0.85, color=color,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.2,
            label=f"Listing {lid}")

ax.set_xlabel("Scrape Date")
ax.set_ylabel("log(Price)")
ax.set_title("Within-Listing log(Price) Variation Over Time\n"
             "8 highest-variance listings — demonstrates why listing fixed effects matter")
ax.text(0.02, 0.97,
        "Each line = one listing observed across multiple scrapes.\n"
        "Substantial within-listing variation justifies the listing FE approach.",
        transform=ax.transAxes, color=TEXT_SUB, fontsize=7.5, va="top")
style_ax(ax, fig)
fig_tag(ax, "Fig 13")
plt.tight_layout()
save("fig13_within_listing_price_variation.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 14 — log(Price) by Scrape Quarter (violin)
# ═══════════════════════════════════════════════════════════════════════════════
panel_reg["scrape_period"] = panel_reg["scrape_date"].dt.to_period("Q").astype(str)
quarters = sorted(panel_reg["scrape_period"].unique())
qt_data  = [panel_reg[panel_reg["scrape_period"]==q]["log_price"].dropna().values
            for q in quarters if len(panel_reg[panel_reg["scrape_period"]==q]) > 10]
quarters_valid = [q for q in quarters if len(panel_reg[panel_reg["scrape_period"]==q]) > 10]

fig, ax = plt.subplots(figsize=(13, 5))
parts = ax.violinplot(qt_data, positions=range(len(qt_data)),
                      showmedians=True, showextrema=False, widths=0.75)
for pc in parts["bodies"]:
    pc.set_facecolor(BLUE)
    pc.set_alpha(0.30)
    pc.set_edgecolor(BLUE)
    pc.set_linewidth(1.0)
parts["cmedians"].set_color(RED)
parts["cmedians"].set_linewidth(2)

ax.set_xticks(range(len(quarters_valid)))
ax.set_xticklabels(quarters_valid, rotation=45, ha="right", fontsize=7.5, color=TEXT_SUB)
ax.set_ylabel("log(Price)")
ax.set_title("log(Price) Distribution by Scrape Quarter  (2015 Q4 – 2023 Q1)\n"
             "Violin = full distribution  |  Red line = median  |  Motivates time fixed effects")
ax.text(0.02, 0.04, f"Pooled n = {len(panel_reg):,} observations  across {len(quarters_valid)} quarters",
        transform=ax.transAxes, color=TEXT_SUB, fontsize=7.5)
style_ax(ax, fig)
fig_tag(ax, "Fig 14")
plt.tight_layout()
save("fig14_price_by_scrape_quarter.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 15 — Two OLS Lines: Reputation vs. log(Price) by Superhost
# ═══════════════════════════════════════════════════════════════════════════════
reg_df = (df.dropna(subset=["reputation_index","log_price",
                              "host_is_superhost","accommodates",
                              "host_tenure_days","occupancy_proxy_30",
                              "neighbourhood_cleansed"])
          .sort_values("scrape_date")
          .groupby("id").last().reset_index())

plot15 = reg_df.dropna(subset=["reputation_index","log_price"]).sample(
    min(5000, len(reg_df)), random_state=42)

fig, ax = plt.subplots(figsize=(9.5, 5.5))
for val, label, color in [(1,"Superhost",RED),(0,"Non-Superhost",BLUE)]:
    sub = plot15[plot15["host_is_superhost"]==val]
    ax.scatter(sub["reputation_index"], sub["log_price"],
               alpha=0.07, s=6, color=color, rasterized=True)
    m_s = smf.ols("log_price ~ reputation_index", data=sub).fit()
    b_s = m_s.params["reputation_index"]
    p_s = m_s.pvalues["reputation_index"]
    star_s = "***" if p_s < 0.01 else "**" if p_s < 0.05 else "*" if p_s < 0.10 else "(ns)"
    x_s = np.linspace(sub["reputation_index"].quantile(0.02),
                       sub["reputation_index"].quantile(0.98), 200)
    y_s = m_s.params["Intercept"] + b_s * x_s
    ax.plot(x_s, y_s, color=color, linewidth=2.8,
            label=f"{label}  β = {b_s:+.4f}{star_s}  (n={len(sub):,})")

ax.set_xlabel("Reputation Index  (standardized composite of rating + VADER sentiment)")
ax.set_ylabel("log(Price)")
ax.set_title("Reputation Index → log(Price) by Superhost Status\n"
             "Subsample OLS — L20/L21 approach")
ax.legend(framealpha=0.9, edgecolor=BORDER)
ax.text(0.02, -0.12, SIG_NOTE, transform=ax.transAxes,
        ha="left", va="top", fontsize=7.5, color=TEXT_SUB)
style_ax(ax, fig)
plt.tight_layout()
save("fig15_subsample_ols_superhost.png")

print("\nAll 15 figures generated.")
print(f"Output: {OUT_DIR}")
# =============================================================================
# COMPONENT 5 - REGRESSION ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("COMPONENT 5 - REGRESSION ANALYSIS")
print("="*70)

# most-recent scrape per listing = clean cross-section
# avoids inflating t-stats by having the same listing appear many times
reg_df = (
    panel.sort_values("scrape_date")
    .groupby("id")
    .last()
    .reset_index()
)
reg_df = reg_df.dropna(subset=[
    "log_price", "reputation_index", "review_scores_rating",
    "cum_mean_sentiment", "accommodates", "host_tenure_days",
    "neighbourhood_cleansed", "occupancy_proxy_30",
])
print(f"  Cross-section sample: {len(reg_df):,} listings")

# add month-of-year dummy to capture seasonality in the cross-section
# Boston peaks in summer/fall; this absorbs those listing-level seasonal patterns
reg_df["scrape_month"] = reg_df["scrape_date"].dt.month

m1 = smf.ols("log_price ~ reputation_index", data=reg_df).fit(cov_type="HC3")

m2 = smf.ols(
    "log_price ~ reputation_index + accommodates + host_tenure_days "
    "+ host_is_superhost + multi_listing_host + minimum_nights",
    data=reg_df
).fit(cov_type="HC3")

# does the structured rating or the unstructured text do more work?
m3 = smf.ols(
    "log_price ~ review_scores_rating + cum_mean_sentiment "
    "+ accommodates + host_tenure_days + host_is_superhost "
    "+ multi_listing_host + minimum_nights",
    data=reg_df
).fit(cov_type="HC3")

m4 = smf.ols(
    "log_price ~ reputation_index + accommodates + host_tenure_days "
    "+ host_is_superhost + multi_listing_host + minimum_nights "
    "+ C(neighbourhood_cleansed)",
    data=reg_df
).fit(cov_type="HC3")

# model 4 with seasonality control added - month dummies absorb Boston's
# strong summer/fall seasonal price patterns
m4s = smf.ols(
    "log_price ~ reputation_index + accommodates + host_tenure_days "
    "+ host_is_superhost + multi_listing_host + minimum_nights "
    "+ C(neighbourhood_cleansed) + C(scrape_month)",
    data=reg_df
).fit(cov_type="HC3")
_rep_no_season = m4.params.get("reputation_index", np.nan)
_rep_season    = m4s.params.get("reputation_index", np.nan)
print(f"\n  Seasonality check: reputation coef without month dummies = {_rep_no_season:+.4f}")
print(f"                     reputation coef with month dummies    = {_rep_season:+.4f}")
print(f"  (If these are close the result is not driven by seasonal patterns)")

m5 = smf.ols(
    "occupancy_proxy_30 ~ reputation_index + accommodates + host_tenure_days "
    "+ host_is_superhost + multi_listing_host + minimum_nights",
    data=reg_df
).fit(cov_type="HC3")

# * notation in patsy includes both main effects and the interaction
m6 = smf.ols(
    "log_price ~ reputation_index * host_is_superhost + accommodates "
    "+ host_tenure_days + multi_listing_host + minimum_nights",
    data=reg_df
).fit(cov_type="HC3")

lda_cols_avail = [c for c in reg_df.columns if c.startswith("lda_topic_")]
if lda_cols_avail:
    lda_formula = (
        "log_price ~ reputation_index + "
        + " + ".join(lda_cols_avail[:-1])
        + " + accommodates + host_tenure_days + host_is_superhost + minimum_nights"
    )
    m7 = smf.ols(lda_formula, data=reg_df).fit(cov_type="HC3")
else:
    m7 = None

def fmt_coef_table(models, model_names, show_vars):
    rows = []
    for var in show_vars:
        coef_row = {"Variable": var}
        se_row   = {"Variable": f"  ({var})"}
        for name, m in zip(model_names, models):
            if m is None:
                coef_row[name] = se_row[name] = ""
                continue
            if var in m.params:
                coef  = m.params[var]
                se    = m.bse[var]
                pv    = m.pvalues[var]
                stars = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.10 else ""
                coef_row[name] = f"{coef:+.4f}{stars}"
                se_row[name]   = f"({se:.4f})"
            else:
                coef_row[name] = "-"
                se_row[name]   = ""
        rows.append(coef_row)
        rows.append(se_row)
    n_row  = {"Variable": "N"}
    r2_row = {"Variable": "R2"}
    for name, m in zip(model_names, models):
        if m is None:
            n_row[name] = r2_row[name] = ""
        else:
            n_row[name]  = f"{int(m.nobs):,}"
            r2_row[name] = f"{m.rsquared:.3f}"
    rows += [n_row, r2_row]
    return pd.DataFrame(rows).set_index("Variable")

SHOW_VARS   = ["Intercept", "reputation_index", "review_scores_rating",
               "cum_mean_sentiment", "reputation_index:host_is_superhost",
               "host_is_superhost", "accommodates", "host_tenure_days",
               "multi_listing_host", "minimum_nights"]
MODELS      = [m1, m2, m3, m4, m5, m6]
MODEL_NAMES = ["(1) Baseline", "(2) Controls", "(3) Sep. text",
               "(4) Nbhd FE",  "(5) Occupancy", "(6) Superhost x"]

reg_table = fmt_coef_table(MODELS, MODEL_NAMES, SHOW_VARS)
print("\n  --- REGRESSION TABLE ---")
print("  DV: log(price) for (1)-(4) and (6); occupancy_proxy_30 for (5)")
print("  HC3 robust SE; *** p<0.01, ** p<0.05, * p<0.10\n")
print(reg_table.to_string())
reg_table.to_csv(os.path.join(OUT_DIR, "regression_table.csv"))
print(f"\n  Saved regression_table.csv")

print("\n  --- Reputation sensitivity by room type ---")
for rt_label, rt_col in [("Entire home/apt", "rt_Entire home/apt"),
                          ("Private room",    "rt_Private room"),
                          ("Shared room",     "rt_Shared room"),
                          ("Hotel room",      "rt_Hotel room")]:
    if rt_col not in reg_df.columns:
        continue
    sub = reg_df[reg_df[rt_col] == 1]
    if len(sub) < 50:
        continue
    m_sub = smf.ols(
        "log_price ~ reputation_index + accommodates + host_tenure_days "
        "+ host_is_superhost + minimum_nights",
        data=sub
    ).fit(cov_type="HC3")
    coef = m_sub.params.get("reputation_index", np.nan)
    pv   = m_sub.pvalues.get("reputation_index", np.nan)
    print(f"    {rt_label:25s} n={len(sub):5,}  b={coef:+.4f}  p={pv:.3f}  R2={m_sub.rsquared:.3f}")

print("\n  --- Reputation sensitivity by host tier ---")
for tier in ["<1yr", "1-3yr", "3-6yr", "6yr+"]:
    sub = reg_df[reg_df["host_tier"].astype(str) == tier]
    if len(sub) < 30:
        continue
    m_tier = smf.ols(
        "log_price ~ reputation_index + accommodates + minimum_nights",
        data=sub
    ).fit(cov_type="HC3")
    coef = m_tier.params.get("reputation_index", np.nan)
    pv   = m_tier.pvalues.get("reputation_index", np.nan)
    print(f"    Tier {tier:8s} n={len(sub):5,}  b={coef:+.4f}  p={pv:.3f}  R2={m_tier.rsquared:.3f}")


# =============================================================================
# PANEL DATA - L21: pooled OLS -> listing FE -> two-way FE
# =============================================================================
print("\n" + "="*70)
print("PANEL DATA ANALYSIS (L21)")
print("="*70)

panel_reg = panel.dropna(subset=[
    "log_price", "reputation_index", "accommodates",
    "host_tenure_days", "host_is_superhost",
    "neighbourhood_cleansed", "scrape_date"
]).copy()

panel_reg["scrape_period"] = panel_reg["scrape_date"].dt.to_period("Q").astype(str)

counts        = panel_reg.groupby("id")["scrape_date"].count()
multi_obs_ids = counts[counts >= 2].index
panel_reg     = panel_reg[panel_reg["id"].isin(multi_obs_ids)]
print(f"  Obs (listings with 2+ scrapes): {len(panel_reg):,}")
print(f"  Unique listings               : {panel_reg['id'].nunique():,}")
print(f"  Quarters covered              : {panel_reg['scrape_period'].nunique()}")

# note: within-listing and quarterly variation plots (fig 13 & 14) were already
# saved earlier in the polished figures section - no need to duplicate here

# helper: safely print a summary2 table regardless of column names
def _print_summary(model, label):
    print(f"\n  --- {label} ---")
    try:
        tbl = model.summary2().tables[1]
        print(tbl.to_string())
    except Exception as e:
        print(f"  (could not print summary: {e})")
    print(f"  R2 = {model.rsquared:.3f}  N = {int(model.nobs):,}")

p_pooled = smf.ols(
    "log_price ~ reputation_index + accommodates + host_tenure_days "
    "+ host_is_superhost + minimum_nights",
    data=panel_reg
).fit(cov_type="HC3")
_print_summary(p_pooled, "(1) Pooled OLS")

# listing FE via within-demeaning - same coefficients as C(id) dummies
# but avoids building an 11k-column matrix which crashes statsmodels
_fe_vars  = ["log_price", "reputation_index", "accommodates",
             "host_tenure_days", "host_is_superhost", "minimum_nights"]
_means    = panel_reg.groupby("id")[_fe_vars].transform("mean")
_demeaned = panel_reg[_fe_vars].subtract(_means)

p_fe_listing = smf.ols(
    "log_price ~ reputation_index + accommodates + host_tenure_days "
    "+ host_is_superhost + minimum_nights",
    data=_demeaned
).fit(cov_type="HC3")
_print_summary(p_fe_listing, "(2) Listing Fixed Effects (within-demeaned)")

# two-way FE: demean by listing AND time period
_period_means  = panel_reg.groupby("scrape_period")[_fe_vars].transform("mean")
_grand_means   = panel_reg[_fe_vars].mean()
_demeaned_2way = panel_reg[_fe_vars].subtract(_means).subtract(_period_means).add(_grand_means)

p_fe_twoway = smf.ols(
    "log_price ~ reputation_index + accommodates + host_tenure_days "
    "+ host_is_superhost + minimum_nights",
    data=_demeaned_2way
).fit(cov_type="HC3")
_print_summary(p_fe_twoway, "(3) Two-Way FE (Listing + Time, within-demeaned)")

# panel summary table - build manually to avoid summary_col endog_names issues
print("\n  --- Panel Summary (reputation_index coefficient across specs) ---")
_panel_rows = []
for _name, _mod, _lfe, _tfe in [
    ("Pooled OLS",   p_pooled,     "No",  "No"),
    ("Listing FE",   p_fe_listing, "Yes", "No"),
    ("Two-Way FE",   p_fe_twoway,  "Yes", "Yes"),
]:
    _c  = _mod.params.get("reputation_index", np.nan)
    _p  = _mod.pvalues.get("reputation_index", np.nan)
    _st = "***" if _p < 0.01 else "**" if _p < 0.05 else "*" if _p < 0.10 else ""
    _panel_rows.append({
        "Model": _name,
        "reputation_index": f"{_c:+.4f}{_st}",
        "p-value": f"{_p:.3f}",
        "R2": f"{_mod.rsquared:.3f}",
        "N": f"{int(_mod.nobs):,}",
        "Listing FE": _lfe,
        "Time FE": _tfe,
    })
_panel_tbl = pd.DataFrame(_panel_rows).set_index("Model")
print(_panel_tbl.to_string())
_panel_tbl.to_csv(os.path.join(OUT_DIR, "panel_regression_summary.csv"))
print("  Saved panel_regression_summary.csv")

# sensitivity check - does reputation coefficient hold across all cross-section models?
print("\n  --- Sensitivity Check (reputation_index across models) ---")
_sens_rows = []
for _name, _mod in zip(["(1) Baseline", "(2) Controls", "(3) Sep.text",
                         "(4) Nbhd FE", "(5) Occupancy", "(6) Superhost x"],
                        [m1, m2, m3, m4, m5, m6]):
    _c  = _mod.params.get("reputation_index", np.nan)
    _p  = _mod.pvalues.get("reputation_index", np.nan)
    _st = "***" if not np.isnan(_p) and _p < 0.01 else \
          "**"  if not np.isnan(_p) and _p < 0.05 else \
          "*"   if not np.isnan(_p) and _p < 0.10 else ""
    _dv = _mod.model.formula.split("~")[0].strip()
    _nfe = "Yes" if "C(neighbourhood_cleansed)" in _mod.model.formula else "No"
    _sens_rows.append({
        "Model": _name,
        "rep_index coef": f"{_c:+.4f}{_st}" if not np.isnan(_c) else "-",
        "p": f"{_p:.3f}" if not np.isnan(_p) else "-",
        "R2": f"{_mod.rsquared:.3f}",
        "N": f"{int(_mod.nobs):,}",
        "DV": _dv,
        "Nbhd FE": _nfe,
    })
print(pd.DataFrame(_sens_rows).set_index("Model").to_string())
# =============================================================================
# LPM vs. LOGIT (L20/L21)
# =============================================================================
print("\n" + "="*70)
print("LPM vs. LOGISTIC REGRESSION (L20/L21)")
print("="*70)

reg_df["is_high_reputation"] = (reg_df["reputation_index"] > 0).astype(int)
print(f"  High-reputation listings: {reg_df['is_high_reputation'].mean():.1%}")

lpm = smf.ols(
    "is_high_reputation ~ accommodates + host_tenure_days "
    "+ host_is_superhost + multi_listing_host + minimum_nights + number_of_reviews",
    data=reg_df
).fit(cov_type="HC3")
print("\n  --- LPM ---")
print(lpm.summary2().tables[1].to_string())

lpm_preds    = lpm.fittedvalues
out_of_range = ((lpm_preds < 0) | (lpm_preds > 1)).sum()
print(f"\n  Predictions outside [0,1]: {out_of_range:,} ({out_of_range/len(lpm_preds):.1%})")

logit_m = smf.logit(
    "is_high_reputation ~ accommodates + host_tenure_days "
    "+ host_is_superhost + multi_listing_host + minimum_nights + number_of_reviews",
    data=reg_df
).fit(disp=False)
print("\n  --- Logit (log-odds) ---")
print(logit_m.summary2().tables[1].to_string())

margeff = logit_m.get_margeff()
print("\n  --- Marginal Effects (probability scale) ---")
print(margeff.summary())

# LPM vs logit comparison table
try:
    lpm_coefs  = lpm.params.drop("Intercept")
    marg_frame = margeff.summary_frame()
    marg_coefs = pd.Series(margeff.margeff, index=marg_frame.index)
    common = lpm_coefs.index.intersection(marg_coefs.index)
    if len(common) == 0:
        # index names differ slightly - try stripping 'x' prefix
        marg_coefs.index = [i.lstrip("x") for i in marg_coefs.index]
        common = lpm_coefs.index.intersection(marg_coefs.index)
    compare_df = pd.DataFrame({
        "LPM coef":        lpm_coefs.reindex(common).round(4),
        "Logit marg. eff": marg_coefs.reindex(common).round(4),
    })
    print("\n  LPM vs. Logit marginal effects comparison:")
    print(compare_df.to_string())
except Exception as e:
    print(f"\n  (LPM/Logit comparison skipped: {e})")


# =============================================================================
# CAUSALITY SEGMENT - L22
# =============================================================================
print("\n" + "="*70)
print("CAUSALITY SEGMENT (L22)")
print("="*70)
print("""
What we found:
  Reputation index is positively and significantly associated with both
  price and occupancy across all specs, including with neighbourhood FE.
  Superhosts show a stronger reputation-price relationship than non-superhosts.
  LDA results suggest cleanliness and host communication topics cluster in
  higher-reputation listings.

What the regressions cannot show:

  Reverse causality - higher-quality listings may attract guests who were
  already going to leave positive reviews. We cannot separate "sentiment
  causes higher price" from "quality causes both." A coefficient is an
  association, not a causal arrow (L22).

  Selection into reviews - only booked listings get reviewed. Zero-booking
  listings with no reviews are excluded from text analysis entirely, which
  pulls our sentiment estimates toward better-performing properties.

  Omitted variable bias - professional photos, a well-designed interior,
  or prime micro-location within a neighbourhood all drive both sentiment
  and price. Neighbourhood FE removes area-level noise but not
  within-neighbourhood unobserved quality.

  Market shocks - the Boston data spans 2015-2023. COVID, regulation
  changes, and platform updates all happened during this window. Pooled
  OLS treats time trends as if they are reputation effects.

What would move us closer to causal:
  Listing FE (done above) - within-listing changes in sentiment predicting
  within-listing price changes removes time-invariant unobserved quality.
  If the coefficient holds after listing FE, that is a stronger signal.

  Event study - find listings hit by a sudden negative review and test
  whether price or occupancy drops the following month vs. control listings.
  That shock would be plausibly exogenous.

  IV - instrument cumulative sentiment with reviewer harshness (average
  score on all other listings that reviewer has visited). That is exogenous
  to any individual listing's quality.

Honest summary:
  We have robust associations that survive multiple control specs. We
  cannot claim causation with this design, but the patterns are consistent
  with reputation mattering, especially for experienced hosts and
  entire-home listings.
""")

print("\n" + "="*70)
print("DONE - all outputs saved to:", OUT_DIR)
print("="*70)
