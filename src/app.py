# ============================================================
# Linear Regression Project
#
# Problem Statement:
# An insurance company wants to predict the health insurance
# premium (charges) for each customer based on their data.
# A team of doctors assembled a dataset from industry data
# and a dedicated study. This is a regression problem.
#
# Features:
#   age      — Age of primary beneficiary (numeric)
#   sex      — Gender of primary beneficiary (categorical)
#   bmi      — Body mass index (numeric)
#   children — Number of kids (numeric)
#   smoker   — Is the person a smoker? (categorical)
#   region   — U.S. region: northeast, southeast, southwest, northwest (categorical)
#   charges  — Annual insurance premium in USD (numeric) — TARGET
# ============================================================

# ------------------------------------------------------------
# EDA Pipeline
# 1. Load the Dataset
# 2. High level view — shape, info, categorical vs numerical
# 3. Data cleaning — nulls, duplicates, noise columns
# 4. Descriptive statistics
# 5. Univariate analysis — graphing individual variables
# 6. Verbal analysis of univariate graphs
# 7. Multivariate analysis — heatmap, boxplots, scatter plots
# 8. Verbal analysis of multivariate graphs
# 9. Feature engineering — outlier handling, encoding, scaling
# 10. Feature selection — SelectKBest with f_regression
# 11. Train/test split and save
# ------------------------------------------------------------

# ============================================================
# Step 1: Load the Dataset
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils import db_connect
# engine = db_connect()

df = pd.read_csv("https://breathecode.herokuapp.com/asset/internal-link?id=416&path=medical_insurance_cost.csv")

print(df.head())

# ============================================================
# Step 2: High Level View
# ============================================================

# Dimensions
print(f"Shape: {df.shape}")

# Data types and non-null counts
df.info()

# Numerical columns:  age, bmi, children, charges
# Categorical columns: sex, smoker, region

# ============================================================
# Step 3: Data Cleaning
# ============================================================

# Check for duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")

rows_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
rows_after = len(df)

print(f"Rows before: {rows_before}")
print(f"Rows after:  {rows_after}")
print(f"Removed:     {rows_before - rows_after}")

# Check for nulls
print("\nWe already learned from 'df.info()' that there are no null values!")

# ============================================================
# Step 4: Descriptive Statistics
# ============================================================

# By default, df.describe() only gives us info on the numerical columns
print(df.describe())

# All columns including categorical
print(df.describe(include="all"))

# ============================================================
# Step 5: Univariate Analysis
# ============================================================

cat_cols = ["sex", "smoker", "region"]

# testing the enumerate() function
print(list(enumerate(cat_cols)))

# Numerical columns
num_cols = ["age", "bmi", "children", "charges"]

# categorical columns
cat_cols = ["sex", "smoker", "region"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    axes[i].hist(df[col], bins=30, edgecolor="black")
    axes[i].set_title(col)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")

plt.suptitle("Numerical Columns", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

# Categorical columns — bar charts
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for i, col in enumerate(cat_cols):
    df[col].value_counts().plot(kind="bar", ax=axes[i], edgecolor="black")
    axes[i].set_title(col)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
    axes[i].tick_params(axis="x", rotation=30)

plt.suptitle("Categorical Columns — Value Counts", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# ============================================================
# Step 6: Univariate Analysis — Findings
#
# - Dataset: ~1,338 rows, 7 columns. No nulls found.
# - age: Fairly uniform distribution across 18–64. No extreme skew.
# - bmi: Roughly bell-shaped, centered around 30 (clinical obesity
#        threshold). Some high-end outliers.
# - children: Right-skewed — most customers have 0–2 children.
#             Values go up to 5.
# - charges: Heavily right-skewed with a long tail. A secondary peak
#            at higher values is visible — likely driven by the smoker
#            subgroup. This is the target variable and will not be
#            capped or transformed.
# - sex: Roughly balanced — slightly more males than females.
# - smoker: Strong imbalance — approximately 80% non-smokers.
# - region: Fairly balanced across four U.S. regions; southeast
#           slightly overrepresented.
# ============================================================

# ============================================================
# Step 7: Multivariate Analysis
# ============================================================

# First we encode categorical columns so the correlation heatmap math works.

# Encode categoricals for correlation analysis
# not changing the original dataframe, making a copy
df_encoded = df.copy()

# factorizing all categorical columns — changing values to numbers
df_encoded["sex"]    = pd.factorize(df_encoded["sex"])[0]
df_encoded["smoker"] = pd.factorize(df_encoded["smoker"])[0]
df_encoded["region"] = pd.factorize(df_encoded["region"])[0]

# Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))

sns.heatmap(df_encoded.corr(), annot=True, fmt=".2f", ax=ax, cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Boxplots — numerical features vs smoker (strongest categorical predictor)
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for i, col in enumerate(["age", "bmi", "children"]):
    df.boxplot(column=col, by="smoker", ax=axes[i])
    axes[i].set_title(f"{col} by smoker")
    axes[i].set_xlabel("smoker")

plt.suptitle("")
plt.tight_layout()
plt.show()

# Scatter plots — numerical features vs target (charges)
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for i, col in enumerate(["age", "bmi", "children"]):
    axes[i].scatter(df[col], df["charges"], alpha=0.3, edgecolors="none")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("charges")
    axes[i].set_title(f"{col} vs charges")

plt.suptitle("Scatter Plots — Numerical Features vs Charges", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# Charges by smoker — the most impactful categorical split
fig, ax = plt.subplots(figsize=(7, 5))
df.boxplot(column="charges", by="smoker", ax=ax)
ax.set_title("charges by smoker")
ax.set_xlabel("smoker")
ax.set_ylabel("charges")
plt.suptitle("")
plt.tight_layout()
plt.show()

# ============================================================
# Step 8: Multivariate Analysis — Findings
#
# Strongest correlations with charges:
#   - smoker   — by far the single most predictive feature.
#                Smokers pay dramatically higher premiums.
#   - age      — moderate positive correlation.
#   - bmi      — moderate positive correlation; interaction with
#                smoker amplifies it significantly.
#   - children — weak positive correlation.
#   - region   — very weak signal; close to noise.
#   - sex      — negligible correlation with charges.
#
# Interaction effects observed:
#   - BMI and smoking together are highly predictive — high BMI
#     smokers form a distinct cluster with very high charges.
#     Consider engineering a bmi_smoker interaction feature.
#
# Multicollinearity:
#   - No severe multicollinearity detected among numerical features.
#
# Feature candidates for removal:
#   - sex    — near-zero correlation with charges. SelectKBest decides.
#   - region — low predictive value.
# ============================================================

# ============================================================
# Step 9: Feature Engineering
# ============================================================

# Outlier handling (IQR method on bmi)
# charges is the target

bmi_stats = df["bmi"].describe()
bmi_iqr = bmi_stats["75%"] - bmi_stats["25%"]
upper_bmi = bmi_stats["75%"] + 1.5 * bmi_iqr

print(f"BMI upper limit (IQR): {upper_bmi:.2f}")
print(f"BMI max before:        {bmi_stats['max']:.2f}")

df = df[df["bmi"] <= upper_bmi].reset_index(drop=True)
print(f"Rows remaining after BMI outlier removal: {len(df)}")

# Encode categorical columns
df["sex"] = pd.factorize(df["sex"])[0]
df["smoker"] = pd.factorize(df["smoker"])[0]

# Multi-class: one-hot encode region (drop_first avoids dummy variable trap)
df = pd.get_dummies(df, columns=["region"], drop_first=True)

print("Columns after encoding:")
print(list(df.columns))
print(df.head())

# Engineer interaction feature: bmi × smoker
# Multivariate analysis showed this interaction is highly predictive
df["bmi_smoker"] = df["bmi"] * df["smoker"]

# Feature scaling: MinMaxScaler on all features (not target)
from sklearn.preprocessing import MinMaxScaler

target = "charges"
feature_cols = ["age", "bmi", "children", "sex", "smoker", "region_northwest", "region_southeast", "region_southwest"]

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df[feature_cols]),
    columns=feature_cols,
    index=df.index
)

# Add target back unscaled
df_scaled[target] = df[target].values

print(f"Shape after scaling: {df_scaled.shape}")
print(df_scaled.head())

# ============================================================
# Step 10: Feature Selection
# ============================================================

from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.model_selection import train_test_split

X = df_scaled.drop(target, axis=1)
y = df_scaled[target]

# Train/test split BEFORE SelectKBest to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# f_regression is the correct scoring function for continuous targets
# (chi2 is for classification; f_regression measures linear correlation with y)
selection_model = SelectKBest(f_regression, k=7)
selection_model.fit(X_train, y_train)

# Selected feature names
ix = selection_model.get_support()
selected_features = X_train.columns.values[ix]
print(f"Selected features: {selected_features}")

# Build reduced feature sets
X_train_sel = pd.DataFrame(
    selection_model.transform(X_train),
    columns=selected_features
)
X_test_sel = pd.DataFrame(
    selection_model.transform(X_test),
    columns=selected_features
)

# ============================================================
# Step 11: Save Results
# ============================================================

import os

# Re-attach target column
X_train_sel["charges"] = list(y_train)
X_test_sel["charges"]  = list(y_test)

# Save to ../data/processed/
base_dir      = os.path.dirname(os.path.abspath("__file__"))
processed_dir = os.path.join(base_dir, "../data/processed")
os.makedirs(processed_dir, exist_ok=True)

X_train_sel.to_csv(os.path.join(processed_dir, "clean_train_data.csv"), index=False)
X_test_sel.to_csv(os.path.join(processed_dir,  "clean_test_data.csv"),  index=False)

print(f"Train set shape: {X_train_sel.shape}")
print(f"Test set shape:  {X_test_sel.shape}")
print("Files saved successfully.")

print(X_train_sel.head())

# ============================================================
# Linear Regression Model
# ============================================================

train_data = pd.read_csv("../data/processed/clean_train_data.csv")
test_data = pd.read_csv("../data/processed/clean_test_data.csv")

print(train_data.head())

X_train = train_data.drop(["charges"], axis=1)
y_train = train_data["charges"]
X_test = test_data.drop(["charges"], axis=1)
y_test = test_data["charges"]

# This is the actual "Machine Learning" part of the project — we use the
# sklearn library and LinearRegression() with the data from the EDA.
# This is surprisingly the easiest part: once EDA is done, we plug the
# data into the model and optimize if possible.

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Here we can see the model parameters
print(f"Intercept (a): {model.intercept_}")
print(f"Coefficients (b1, b2): {model.coef_}")

y_pred = model.predict(X_test)
print(y_pred)

from sklearn.metrics import mean_squared_error, r2_score

print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"R2 Score: {r2_score(y_test, y_pred)}")

# We don't have any hyperparams to optimize in this case, so this R2 score
# is the final evaluation of the model. About 80% — not terrible.
