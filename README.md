# CCS6521 - Data Mining & Machine Learning Coursework

Churn prediction pipeline for a FinTech company. Predicts whether a user is
"unengaged / churned" based on their demographics, device, transactions and
notification activity.

## Project layout

```
.
├── data/churn_train/                Raw CSVs (devices, notifications,
│                                      transactions_{1,2,3}, users)
├── src/
│   ├── config.py                    Paths, snapshot-date logic, lookback windows
│   ├── data_loader.py               Load + concat the four tables
│   ├── eda.py                       Step 1: Exploratory Data Analysis
│   ├── preprocessing.py             Step 2: Clean, encode, feature engineer
│   ├── engagement.py                Step 3: Heuristic + ML churn classification
│   └── main.py                      End-to-end orchestrator
├── outputs/
│   ├── figures/                     13 PNG plots (EDA + modelling)
│   ├── tables/                      CSV / JSON summaries, predictions, feature matrix
│   └── models/                      Pickled best Random Forest
├── report/
│   ├── report.md                    Markdown source of the report
│   ├── build_docx.py                Generates the formatted Word submission
│   └── CCS6521_Churn_Report.docx    Formatted submission (Calibri 11, 1.5 line, page numbers)
└── requirements.txt
```

## Running

```bash
pip install -r requirements.txt
python -m src.main             # runs EDA → preprocessing → modelling (~22 s)
python report/build_docx.py    # rebuilds the Word report from outputs/figures
```

The first script reads CSVs from `data/churn_train/`, runs EDA → preprocessing
→ modelling, and writes all artefacts to `outputs/`. The second rebuilds the
submission Word document from those artefacts. Tested on Python 3.12.

## Headline results

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Heuristic - predict churn if days-inactive ≥ 28 | n/a | n/a | 0.797 | 0.840 | 0.757 |
| Logistic Regression (median impute, scale, balanced) | 0.893 | 0.862 | 0.797 | 0.818 | 0.776 |
| **Random Forest** (300 trees, min_leaf=20, balanced) | **0.901** | **0.875** | **0.811** | 0.798 | 0.826 |

On the 15,544 eligible users, the Random Forest at threshold 0.5 flags
**7,425 (47.8 %)** as churn-risks; actual churn is 7,196 (46.3 %).

## Pipeline summary

1. **EDA** - descriptive statistics, missing-value audit, distribution plots,
   correlation heatmap, outlier detection.
2. **Preprocessing** - combine the four tables into one user-level feature
   matrix; clip `amount_usd` outliers; impute marketing-flag NaNs as 0;
   engineer transactional KPIs (volume, value, success-rate, currency
   diversity, MCC diversity, recency, tenure-normalised rates) and
   notification KPIs (sent/failed by channel).
3. **Engagement / churn label**  - 
   *snapshot = max(transactions.created_date)*,
   *cutoff   = snapshot − 28 days*.
   A user is **engaged** if they have at least one *COMPLETED* transaction in
   `(cutoff, snapshot]`, otherwise **churned**.
   Features are computed strictly from data ≤ cutoff to prevent leakage.
   Users registered fewer than 28 days before cutoff are excluded (insufficient
   history).
4. **Models** - heuristic baseline (recency-only rule) plus a Logistic
   Regression and Random Forest, evaluated with ROC-AUC, PR-AUC, F1 and a
   confusion matrix on a stratified held-out test set.
