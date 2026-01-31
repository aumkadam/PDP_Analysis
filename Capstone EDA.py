#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# In[8]:


# Option B: raw string (keeps backslashes as-is)
CSV_PATH = r"C:\Users\vg362\Downloads\novica_final.csv"
FIG_DIR = "eda_figures"
OUT_DIR = "eda_outputs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# In[9]:


df = pd.read_csv(CSV_PATH, low_memory=False)

# Helper: save figure with safe filename
def savefig(name: str):
    fname = os.path.join(FIG_DIR, f"{name}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    return fname

# Helper: check required columns exist
def have_cols(cols):
    missing = [c for c in cols if c not in df.columns]
    return len(missing) == 0, missing


# In[10]:


for col in df.columns:
    if df[col].dtype == object and set(df[col].dropna().astype(str).str.lower().unique()).issubset({"true","false","0","1"}):
        df[col] = df[col].astype(str).str.lower().map({"true": True, "1": True, "false": False, "0": False})


# In[13]:


# A) Data Quality & Structure
# ============================

# 1) Missingness Bar (Top 30)
missing_pct = df.isna().mean().sort_values(ascending=False) * 100
plt.figure(figsize=(10, 8))
topN = missing_pct.head(30)
plt.barh(topN.index, topN.values)
plt.gca().invert_yaxis()
plt.xlabel("Missing (%)")
plt.title("Top Columns by Missingness (%)")


# In[18]:


# 2) Data Type Summary Table

dtypes_df = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.astype(str).values,
    "n_missing": df.isna().sum().values,
    "missing_pct": (df.isna().mean().values * 100).round(2)
})
dtypes_df.head(20)  # Show first 20 columns


# In[21]:


# 3) Descriptive Statistics Table (numeric)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
desc = df[num_cols].describe().T if num_cols else pd.DataFrame()
if not desc.empty:
    # add skew/kurtosis
    desc["skew"] = df[num_cols].skew(numeric_only=True)
    desc["kurtosis"] = df[num_cols].kurtosis(numeric_only=True)
desc.round(2)


# In[30]:


#B) Distribution plots
def hist_if(col, bins=30):
    ok, miss = have_cols([col])
    if ok:
        plt.figure(figsize=(8, 6))
        plt.hist(df[col].dropna(), bins=bins)
        plt.title(f"Histogram – {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()     # 👈 displays the plot inline
        
for col in ["price", "quantity_sold_365days", "rating_numeric"]:
    hist_if(col)


# In[31]:


##What it shows

X-axis: product price (in USD)

Y-axis: number of SKUs (products) in each price range

Pattern you can see

The distribution is heavily right-skewed.

Most products are clustered between $20 – $100, with only a few very high-priced outliers reaching $700+.

Business interpretation

NOVICA’s catalog on Amazon is mostly affordable or mid-tier handmade items, 
while only a few premium products are priced high.

That long tail suggests strong price segmentation — 
an opportunity to test whether premium listings 
justify higher prices through richer PDP content (images, copy, or A+).


2. Histogram – Quantity Sold (365 Days)
 What it shows

X-axis: units sold in the past year

Y-axis: number of products with that sales level

Pattern you can see

Again, a sharp right-skew.

Most SKUs sold only 1–5 units in a year, while a tiny fraction sold hundreds or even thousands.

Business interpretation

Sales are extremely concentrated: a small 
portion of products drives most revenue.

This validates NOVICA’s client challenge — 
comparing top-selling vs slow-selling SKUs 
to see which PDP elements (A+, keywords, visuals) differentiate them.

3. Histogram – Rating Numeric
What it shows

X-axis: average customer rating (1 – 5 stars)

Y-axis: number of products in each rating bucket

Pattern you can see

Ratings are skewed toward 5 stars.

Many products cluster between 4.5 – 5, and
there are missing or unrated listings (hence some low bars or gaps).

Business interpretation

NOVICA products receive mostly positive reviews, 
but incomplete feedback coverage (≈ 30 % missing ratings) limits quantitative sentiment analysis.

This also hints at potential visibility or 
newness issues — new products might not have accumulated reviews yet.


# In[33]:


# C) Categorical Comparisons
# =========================

# 7) Boxplot – price ~ classification
ok, miss = have_cols(["price", "classification"])
if ok:
    # Keep only rows with both
    sub = df[["price", "classification"]].dropna()
    if not sub.empty:
        # Matplotlib boxplot needs grouped arrays
        groups = [g["price"].dropna().values for _, g in sub.groupby("classification")]
        labels = [str(k) for k, _ in sub.groupby("classification")]
        plt.figure(figsize=(12, 6))
        plt.boxplot(groups, vert=True, labels=labels, showfliers=False)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("price")
        plt.title("Boxplot – price by classification")
       


# In[34]:


### What it shows

X-axis: classification → each product category (e.g., Jewelry, Wall Décor, Apparel, Rugs, etc.)

Y-axis: price → selling price in USD

Each vertical box summarizes the price spread within that category:

Orange line (center) = median price

Box edges = 25th and 75th percentiles (middle 50 % of products)

Whiskers / dots = price outliers (very cheap or premium products)

 What patterns you can see

Price variability across categories:
Some boxes (like certain tall ones in the middle) have much longer ranges, meaning that category has a wide price diversity — both budget and high-end options.

Premium categories:
Categories whose boxes sit higher on the Y-axis (e.g., Rugs, Home Décor, or Fine Art) represent higher-priced segments.

Consistent low-price categories:
Others (e.g., Jewelry or Accessories) have short, lower boxes — more standardized, low-to-mid prices.

Presence of outliers:
Several whiskers and isolated dots show a few very high-priced SKUs — possible luxury or large-format handcrafted items.

 Business interpretation

NOVICA’s pricing strategy varies strongly by product type.

Some categories (like Jewelry) are crowded in low-price ranges, suggesting potential price compression or heavy competition.

Others (e.g., Rugs, Wall Décor) have room for premium positioning — high perceived value handmade goods that justify larger margins.

These findings help the team later analyze whether price + content richness (A+, images, keywords) drive success differently per category.


# In[35]:


# 8) Bar – Avg quantity_sold_365days by classification
ok, miss = have_cols(["quantity_sold_365days", "classification"])
if ok:
    grp = df.groupby("classification", dropna=True)["quantity_sold_365days"].mean().sort_values(ascending=False)
    if not grp.empty:
        plt.figure(figsize=(12, 6))
        plt.bar(grp.index.astype(str), grp.values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Avg quantity_sold_365days")
        plt.title("Average Sales by Classification")


# In[37]:


##9) Countplot – has_aplus / has_video
for flag in ["has_aplus", "has_video"]:
    if flag in df.columns:
        counts = df[flag].value_counts(dropna=False)
        plt.figure(figsize=(6, 5))
        plt.bar(counts.index.astype(str), counts.values)
        plt.title(f"Count – {flag}")
        plt.xlabel(flag); plt.ylabel("Count")


# In[38]:


##10) Boxplot – quantity_sold_365days ~ has_aplus
ok, miss = have_cols(["quantity_sold_365days", "has_aplus"])
if ok:
    sub = df[["quantity_sold_365days", "has_aplus"]].dropna()
    if not sub.empty:
        groups = [g["quantity_sold_365days"].values for _, g in sub.groupby("has_aplus")]
        labels = [str(k) for k, _ in sub.groupby("has_aplus")]
        plt.figure(figsize=(7, 5))
        plt.boxplot(groups, vert=True, labels=labels, showfliers=False)
        plt.ylabel("quantity_sold_365days")
        plt.title("Boxplot – Sales by A+ Content")
      


# In[39]:


# ===========================================
# D) Relationships Between Key PDP Features
# ===========================================

def scatter_if(x, y, sample_n=5000):
    ok, miss = have_cols([x, y])
    if ok:
        sub = df[[x, y]].dropna()
        if sub.shape[0] > sample_n:
            sub = sub.sample(sample_n, random_state=42)
        if not sub.empty:
            plt.figure(figsize=(7, 5))
            plt.scatter(sub[x], sub[y], s=10, alpha=0.6)
            plt.xlabel(x); plt.ylabel(y)
            plt.title(f"Scatter – {x} vs {y}")
           

scatter_if("price", "quantity_sold_365days")
scatter_if("rating_numeric", "quantity_sold_365days")
scatter_if("unique_keyword_count", "quantity_sold_365days")


# In[42]:


if len(num_cols) >= 2:
    corr = df[num_cols].corr()   # <- removed numeric_only=True
    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr, aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.show()


# In[43]:


# E) Textual / Keyword Insight (Qual EDA)
# ======================================

# 15) Word Cloud – top_keywords (optional; falls back to frequency bar)
def plot_keyword_wordcloud_or_freq(col="top_keywords", max_words=150):
    if col not in df.columns:
        return
    # Flatten keyword lists. Assume comma-separated or space-separated.
    series = df[col].dropna().astype(str)
    tokens = []
    for s in series:
        # split by comma first, else whitespace
        parts = [p.strip() for p in s.replace(";", ",").split(",")]
        if len(parts) == 1:
            parts = s.split()
        tokens.extend([p for p in parts if p])
    if not tokens:
        return
    # Try wordcloud
    used_wordcloud = False
    try:
        from wordcloud import WordCloud
        text = " ".join(tokens)
        wc = WordCloud(width=1200, height=600, background_color="white", collocations=False).generate(text)
        plt.figure(figsize=(12, 6))
        plt.imshow(wc)
        plt.axis("off")
        plt.title("Word Cloud – top_keywords")
        
        used_wordcloud = True
    except Exception:
        used_wordcloud = False
    if not used_wordcloud:
        # Fallback: frequency bar (Top 30)
        from collections import Counter
        cnt = Counter([t.lower() for t in tokens])
        top = cnt.most_common(30)
        words, vals = zip(*top)
        plt.figure(figsize=(12, 6))
        plt.bar(words, vals)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Frequency")
        plt.title("Top Keyword Frequencies (fallback)")
       

plot_keyword_wordcloud_or_freq("top_keywords")


# In[44]:


# 16) Frequency Bar – sentiment_keywords (Top 30)
if "sentiment_keywords" in df.columns:
    series = df["sentiment_keywords"].dropna().astype(str)
    tokens = []
    for s in series:
        parts = [p.strip() for p in s.replace(";", ",").split(",")]
        if len(parts) == 1:
            parts = s.split()
        tokens.extend([p for p in parts if p])
    if tokens:
        from collections import Counter
        cnt = Counter([t.lower() for t in tokens])
        top = cnt.most_common(30)
        words, vals = zip(*top)
        plt.figure(figsize=(12, 6))
        plt.bar(words, vals)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Frequency")
        plt.title("Top Sentiment Keyword Frequencies")


# In[45]:


##17) Countplot – Boolean flags (has_quality_words, has_material_words, has_benefit_words)
bool_flags = [c for c in ["has_quality_words", "has_material_words", "has_benefit_words"] if c in df.columns]
if bool_flags:
   proportions = {}
   for c in bool_flags:
       if df[c].dropna().size > 0:
           true_rate = (df[c] == True).mean() * 100
           proportions[c] = true_rate
   if proportions:
       plt.figure(figsize=(8, 5))
       plt.bar(list(proportions.keys()), list(proportions.values()))
       plt.xticks(rotation=30, ha="right")
       plt.ylabel("% True")
       plt.title("Boolean Linguistic Flags – % of listings using each")


# In[47]:


# F) Outlier & Constraint Checks
# ===============================

# 18) Boxplot – quantity_sold_365days
if "quantity_sold_365days" in df.columns:
    plt.figure(figsize=(6, 5))
    plt.boxplot(df["quantity_sold_365days"].dropna().values, vert=True, showfliers=True)
    plt.ylabel("quantity_sold_365days")
    plt.title("Boxplot – Sales (Outlier Check)")
   

# 19) Scatter – price vs reviews_count_numeric
scatter_if("price", "reviews_count_numeric")


# In[ ]:




