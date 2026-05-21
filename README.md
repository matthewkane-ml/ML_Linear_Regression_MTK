# Linear Regression — Health Insurance Premium Prediction

> Regression pipeline predicting annual insurance charges from patient demographics: full EDA revealing the outsized effect of smoking, interaction feature engineering, and a Linear Regression model achieving R² ≈ 0.80.

---

## Problem

An insurance company wants to predict the annual health insurance premium (`charges`) for each customer based on their demographic and health data. A dataset assembled by a team of doctors from industry data is used to train the model. This is a supervised regression problem.

## Dataset

- **Source:** Medical Insurance Cost dataset (~1,338 rows)
- **Target:** `charges` — annual insurance premium in USD (continuous)
- **Features:**

| Feature | Type | Description |
|---|---|---|
| age | Numerical | Age of the primary beneficiary |
| sex | Categorical | Gender of the primary beneficiary |
| bmi | Numerical | Body mass index |
| children | Numerical | Number of dependents |
| smoker | Categorical | Is the person a smoker? |
| region | Categorical | U.S. residential region (NE / SE / SW / NW) |

## EDA & Preprocessing Pipeline

| Step | Action |
|---|---|
| Duplicates | 1 duplicate removed |
| Nulls | None found |
| Outlier handling | BMI capped at IQR upper bound (very few extreme values removed) |
| Encoding | `sex` and `smoker` → pd.factorize(); `region` → one-hot (drop_first to avoid dummy trap) |
| Interaction feature | `bmi_smoker = bmi × smoker` — multivariate analysis showed high-BMI smokers form a distinct high-charges cluster |
| Scaling | MinMaxScaler on all feature columns |
| Feature selection | SelectKBest (f_regression, k=7) — split before selection to prevent data leakage |
| Split | 80/20 train/test |

**Key EDA finding:** `smoker` is by far the most predictive single feature. Smokers pay dramatically higher premiums — the charges distribution has a pronounced secondary peak driven entirely by this group. The interaction term `bmi_smoker` captures that this effect amplifies at high BMI.

**Correlation with charges (approximate):**

| Feature | Signal |
|---|---|
| smoker | Very strong |
| age | Moderate positive |
| bmi | Moderate positive (amplified by smoker interaction) |
| children | Weak positive |
| region | Negligible |
| sex | Near-zero |

## Model Results

- **Model:** `sklearn.linear_model.LinearRegression`
- **R² score:** ≈ **0.80** (80% of variance in insurance charges explained)
- **Note:** Linear Regression has no hyperparameters to tune — the R² score is the final evaluation.

## Key Takeaways

- **One feature dominates:** Smoker status alone accounts for the largest share of explainable variance. Any model that doesn't capture this distinction will fail badly.
- **Interaction features matter:** `bmi_smoker` is more predictive than BMI alone because the relationship isn't additive — a high-BMI non-smoker pays much less than a high-BMI smoker. Linear models need this engineered explicitly; tree-based models would find it automatically.
- **R² ≈ 0.80 has a ceiling:** The remaining 20% of variance likely comes from claim history, pre-existing conditions, and lifestyle factors not captured in these 6 features — not a modelling failure.

## Tech Stack

`Python` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## Run It Locally

```bash
git clone https://github.com/matthewkane-ml/ML_Linear_Regression_MTK.git
cd ML_Linear_Regression_MTK
pip install -r requirements.txt
python src/app.py
```

## What I'd Do Next

- Try **Ridge or Lasso regression** to add regularization and compare whether penalising weak coefficients improves generalisation
- Engineer a **smoker × age** interaction to test whether the smoking penalty grows with age
- Compare against a **Gradient Boosting Regressor** — tree-based methods find non-linear relationships and interactions automatically, which would likely push R² above 0.85

---

**Author:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [GitHub portfolio](https://github.com/matthewkane-ml)
