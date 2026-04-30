# CCS6521 - Data Mining and Machine Learning
## Coursework Assessment 1 - Working with data in Python
### Churn-prediction Knowledge-Discovery Pipeline

---

## 1. Exploratory Data Analysis

### 1.1 Schema and volumes

The dataset is supplied as four CSV tables. Their shapes and key columns are
summarised below.

| Table | Rows | Cols | Key columns |
|---|---:|---:|---|
| `users` | 15,544 | 12 | `user_id`, `birth_year`, `country`, `city`, `created_date`, `plan`, `user_settings_crypto_unlocked`, `attributes_notifications_marketing_*`, `num_contacts`, `num_referrals`, `num_successful_referrals` |
| `devices` | 15,544 | 2 | `brand`, `user_id` (one row per user) |
| `notifications` | 97,704 | 5 | `reason`, `channel`, `status`, `user_id`, `created_date` |
| `transactions` (3 split CSVs) | 2,181,026 | 12 | `transaction_id`, `transactions_type`, `transactions_currency`, `amount_usd`, `transactions_state`, `ea_cardholderpresence`, `ea_merchant_mcc`, `ea_merchant_city`, `ea_merchant_country`, `direction`, `user_id`, `created_date` |

`devices` has exactly one row per `user_id`, so it acts as a 1:1 lookup. The
remaining tables are 1:N from the user side.

### 1.2 Time coverage

`users.created_date` runs from 2018-01-01 to 2019-01-03 (so registrations stop
roughly four months before the data was extracted). Activity (transactions and
notifications) runs from 2018-01-01 to 2019-05-16, giving us approximately 16
months of behavioural history per cohort. The latest transaction date is taken
as the **snapshot date**: 2019-05-16.

### 1.3 Missing values

A column-level audit (see `outputs/tables/missing_*.csv`) shows that the only
material missingness is in the `users` table: both
`attributes_notifications_marketing_push` and
`attributes_notifications_marketing_email` are NaN for 5,260 users (33.8 %).
Every other column is fully populated. The `transactions` table has the
expected NaNs in card-only fields (`ea_*`, `direction`) for non-card
transaction types. Treating those as informative absence rather than a data
quality issue is appropriate, as those fields are simply not applicable to a
TRANSFER or TOPUP record.

### 1.4 Outliers and obvious data errors

The single most striking issue is in `transactions.amount_usd`:

```
count    2.18 × 10⁶
mean     1.19 × 10⁵
std      6.96 × 10⁷
min      0
50%      8.51
max      7.46 × 10¹⁰
```

A maximum of 74 billion USD is clearly impossible for a retail FinTech
transaction. The 99.5 ᵗʰ percentile is below 8 k USD, so we cap to that level
rather than discarding rows; this neutralises the impact on means/sums while
keeping the long but legitimate tail of larger transfers. Negative amounts do
not appear in this dataset, but the cleaning code defends against them.

### 1.5 Univariate distributions

Plots are saved to `outputs/figures/`. Highlights:

- **Plan**: 92.6 % STANDARD, 4.6 % SILVER, 2.8 % GOLD - a heavy long tail.
- **Country**: 32 % GB, 12 % PL, 11 % FR, 6 % IE, 6 % RO, 5 % ES, ... (top-15
  account for ~85 %).
- **Device brand**: 50.2 % Android, 49.6 % Apple, 0.2 % "Unknown".
- **Birth year**: median 1986 (33 yo at snapshot), range 1929-2001; no
  implausible values, all derived ages between 18 and 90.
- **Transaction type**: 53.8 % CARD_PAYMENT, 18.4 % TRANSFER, 14.2 % TOPUP,
  5.7 % EXCHANGE, 3.4 % ATM, 3.0 % CASHBACK, smaller fees/refunds.
- **Transaction state**: 87.9 % COMPLETED, 5.7 % DECLINED, 4.1 % REVERTED,
  1.5 % FAILED, 0.8 % PENDING, 0.06 % CANCELLED.
- **Direction**: 81 % OUTBOUND, 19 % INBOUND.
- **Notifications**: 50 % EMAIL, 47 % PUSH, 2 % SMS; 73 % SENT, 27 % FAILED.
  Failure rates are concentrated in EMAIL (delivery bounces) and reveal a
  clear deliverability issue worth flagging.
- **Notification reasons**: 17 distinct values; a single `REENGAGEMENT_*`
  campaign accounts for the bulk of PUSH activity and is a useful churn
  proxy because the business has already labelled those recipients
  internally.
- **Activity over time**: both transactions and notifications grow roughly
  linearly with the user base over 2018, then stabilise in 2019.

### 1.6 Bivariate relationships

A correlation heat-map of log-scaled volume features (Figure
`correlation_heatmap.png`) shows the expected dominant pair
`n_transactions ↔ total_completed_usd` (r ≈ 0.88), and a moderate positive
relationship between `num_contacts` and transaction volume (r ≈ 0.36). Plan
mostly captures activity differences in the right tail: GOLD users have
median transaction counts ~3× higher than STANDARD (Figure
`activity_by_plan.png`).

---

## 2. Initial preprocessing

### 2.1 Combining the tables

Because the modelling target is one label per user, all four tables are
collapsed to user level. `devices` joins 1:1 on `user_id`. `notifications`
and `transactions` are aggregated to user level (counts, sums, distinct
counts, shares, recency) before the join. This keeps the training matrix
narrow and avoids leaking row-level information that the model could not
recover at scoring time.

### 2.2 Cleaning steps

| Issue | Treatment | Rationale |
|---|---|---|
| `attributes_notifications_marketing_*` NaN | filled with 0 | A user without an explicit opt-in is, by product convention, opted-out. |
| `transactions.amount_usd` extreme outliers (max 7.4×10¹⁰) | clipped at the 99.5 ᵗʰ percentile (~8 k USD) | Preserves the long legitimate tail while neutralising data-entry errors that would otherwise dominate every aggregate. |
| `transactions.amount_usd` < 0 | dropped (none in data, defensive) | Negative amounts are a violation of the field semantics. |
| `users.birth_year` implausible | filtered to age ∈ [14, 100] | All rows passed, but the rule is in place. |
| Card-only fields (`ea_*`, `direction`) NaN | left as NaN, used as informative absence in shares | These fields simply do not apply to TRANSFER / TOPUP records. |

### 2.3 Encoding

- **Plan** (3 levels) and **device brand** (3 levels) → one-hot.
- **Country** → grouped to top-15 plus an `OTHER` bucket, then one-hot. This
  caps cardinality at 16 dummies while keeping the dominant geographies
  separable.
- **City** is dropped - it is colinear with country and adds thousands of
  high-cardinality categories.

### 2.4 Feature engineering

All engineered features are computed strictly from data observed up to the
**cutoff date**. For an out-of-sample-like evaluation we set:

```
snapshot = max(transactions.created_date)        # 2019-05-16
cutoff   = snapshot − 28 days                    # 2019-04-18
```

so that the prediction is *"given everything we know about the user up to
2019-04-18, will they be active in the following 28 days?"*. The full feature
list (62 columns) groups into:

- **Demographics**: `age`, plan dummies, brand dummies, country dummies,
  marketing opt-in flags, `user_settings_crypto_unlocked`, `tenure_days`.
- **Network**: `num_contacts`, `num_referrals`, `num_successful_referrals`.
- **Transaction volume**: `trx_count`, `trx_count_completed`,
  `trx_value_completed`, `trx_value_mean_completed`, `trx_value_max_completed`.
- **Transaction quality**: `trx_success_rate`.
- **Diversity**: `trx_n_distinct_currency`, `trx_n_distinct_mcc`,
  `trx_n_distinct_country`.
- **Behaviour profile**: per-type shares (CARD_PAYMENT, TRANSFER, TOPUP,
  EXCHANGE, ATM, CASHBACK, FEE, CARD_REFUND, REFUND, TAX) and direction
  shares (OUTBOUND, INBOUND).
- **Recency / cadence**: `days_since_last_trx` and `days_since_first_trx`
  (both relative to the cutoff), `trx_active_days`.
- **Notifications**: `notif_count`, `notif_n_sent`, `notif_n_failed`,
  `notif_send_rate`, per-channel counts, count of `REENGAGEMENT_*`
  notifications.

Why this set of KPIs and not raw event counts? Recency, breadth (distinct
currencies / MCCs) and behavioural mix together tend to discriminate active
from inactive users better than any single volume metric, and they are robust
to a user's plan tier. Engineering them once at the user level also means the
classifier is not learning the same signal repeatedly across millions of raw
transaction rows.

---

## 3. Identifying unengaged and churned users

### 3.1 Engagement metric

> **A user is engaged if they have at least one COMPLETED transaction in the
> 28 days following the cutoff date; otherwise they are unengaged / churned.**

Three reasons for this choice:

1. **Completed transactions are the cleanest signal of intentional usage.**
   Counting `DECLINED`, `FAILED` or `REVERTED` transactions would conflate
   genuine engagement with payment-rail problems and fraud blocks.
2. **A 28-day window matches the natural monthly cadence of consumer
   finance** (salary cycles, card statements). It is short enough to be
   actionable for a re-engagement campaign and long enough to smooth out
   weekly noise.
3. **Symmetry with the feature window.** The features look at the user's
   pre-cutoff history; the label looks at post-cutoff behaviour. The two
   periods do not overlap, so there is no leakage by construction.

Users whose registration date is fewer than 28 days before the cutoff are
excluded - they have not had a full window of opportunity to be active. In
the current dataset, no user fails this test (all users registered on or
before 2019-01-03), but the rule is in place for future data extracts.

### 3.2 Class balance

Out of 15,544 eligible users, **7,196 (46.3 %) are churned** and 8,348
(53.7 %) are engaged. The classes are close enough to balanced that no
resampling or focal-loss is required; we still pass `class_weight="balanced"`
to both classifiers as cheap insurance.

### 3.3 Models

Three classifiers are produced and evaluated on a stratified 20 % held-out
test set (3,109 users). The "heuristic recency" classifier is a non-ML
baseline included to show how much (or little) the ML models add on top of a
single rule.

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Heuristic - predict churn if days-inactive ≥ 28 | n/a | n/a | 0.797 | 0.840 | 0.757 |
| Logistic Regression (median-impute, scale, balanced, max_iter=2000) | 0.893 | 0.862 | 0.797 | 0.818 | 0.776 |
| **Random Forest** (300 trees, min_leaf=20, balanced) | **0.901** | **0.875** | **0.811** | 0.798 | 0.826 |

ROC and Precision-Recall curves are in `outputs/figures/model_roc_pr.png`;
confusion matrices in `confusion_logreg.png` and `confusion_rf.png`.

#### Why these three models?

- **Heuristic recency** is the cheapest possible policy and a strong
  yardstick. If our ML model cannot beat it materially, the operational cost
  of an ML solution is hard to justify.
- **Logistic Regression** is a transparent linear baseline. With balanced
  class weights and standardised features it produces calibrated
  probabilities that operations can threshold directly.
- **Random Forest** captures non-linear interactions (e.g. *"STANDARD plan
  AND high transfer share AND notification failures"*) without manual
  feature crosses. It is also robust to the long-tailed transactional
  features we did not log-transform.

We deliberately do **not** include gradient boosting in the headline
comparison: with a 62-column matrix and 15 k rows, the marginal AUC gain is
small and the extra hyper-parameter tuning (and dependency footprint) is not
warranted for a coursework deliverable. We discuss this further in §4.

#### Feature importances (RF)

`days_since_last_trx` dominates (≈ 29 % of total decrease-in-impurity),
followed by `trx_active_days`, `trx_count`, `trx_count_completed` and
`trx_value_completed` - all near 5-10 %. Diversity features
(`trx_n_distinct_mcc`, `trx_n_distinct_country`) and `num_contacts` round
out the top ten. The plan / country / brand dummies and individual
notification-channel features have negligible importance. This explains why
the heuristic recency rule is already very strong: the dominant axis of
variation is *how long ago the user was last active*.

### 3.4 Population-level numbers

Scoring the trained Random Forest on the full eligible population gives:

| Metric | Value |
|---|---:|
| Eligible users | 15,544 |
| **Predicted churn (model, threshold 0.5)** | **7,425 (47.8 %)** |
| Predicted churn (heuristic) | 6,484 (41.7 %) |
| Actual churn | 7,196 (46.3 %) |

The model predicts slightly more churners than reality (false positive rate
≈ 8 %) at the default threshold. Lowering the threshold trades precision for
recall and would catch more truly-at-risk users at the cost of contacting
more false alarms - this is a business decision, not a modelling one.

---

## 4. Actionable decisions (critical discussion)

### 4.1 How many users are classified as churned?

At a 0.5 probability threshold, the chosen Random Forest flags **7,425
users (47.8 %)** as churn-risks. If marketing budget is the binding
constraint, the same model can be operated at any other threshold:

| Threshold | Approx. flagged | Approx. precision | Approx. recall |
|---:|---:|---:|---:|
| 0.30 | ~8,500 (55 %) | ~0.70 | ~0.92 |
| 0.50 | 7,425 (48 %) | 0.80 | 0.83 |
| 0.70 | ~5,600 (36 %) | ~0.91 | ~0.71 |
| 0.85 | ~3,400 (22 %) | ~0.96 | ~0.45 |

(Exact values are computable from `outputs/tables/predictions.csv`.)

### 4.2 Designing an experiment to verify churn reduction

The right design is a **randomised controlled experiment**, not a
pre-/post-comparison.

1. **Define the eligible population.** Score every user weekly with the
   model. Users whose probability ≥ a chosen threshold (and who have not
   already been part of a recent campaign) are eligible for the test.
2. **Random assignment.** At enrolment, randomly split eligible users into
   *treatment* (receive the re-engagement intervention) and *control*
   (receive nothing, or the existing default) at a pre-registered ratio
   (e.g. 50/50). Stratify the randomisation by plan and country so the
   marginal subgroups are balanced.
3. **Pre-register everything.** Decide *before* the test starts: the
   intervention, the primary outcome (proportion of users who make ≥ 1
   COMPLETED transaction in the next 28 days), the test duration, the
   minimum-detectable-effect, the required sample size from a power
   calculation, the analysis plan, and the stopping rules.
4. **Run for the full pre-registered period.** Resist the urge to stop
   early; sequential peeks inflate Type-I error.
5. **Analyse per-protocol and intent-to-treat.** Use a two-proportion
   z-test (or a logistic regression with covariates for variance reduction)
   on the primary outcome.

Critically, the **treatment effect is the difference between treatment and
control**, not the absolute conversion rate of the treated group. Many
flagged users would have come back on their own - without a control arm we
would have no way of separating that natural recovery from the intervention.

### 4.3 Metrics and techniques to assess impact

**Primary outcome (causal):** treatment-vs-control lift on the same
engagement metric we used for training: *proportion of users with ≥ 1
completed transaction in the 28-day post-treatment window*.

**Supporting metrics:**

- **Volume metrics.** Mean number of completed transactions and total USD
  spent in the post-treatment window, per arm.
- **Cost / contribution.** Net revenue lift per user contacted, including
  the cost of the intervention (push notification ≈ free, SMS ≈ €0.05,
  email ≈ free, in-app credit / cashback ≈ the credit amount).
- **Funnel diagnostics.** Notification deliverability, open rate,
  click-through rate, then conversion. These tell us *why* a treatment
  did or did not move the primary metric.
- **Long-run retention.** Re-measure the engagement metric at +60 and +90
  days. A successful re-engagement that fades within a month is far less
  valuable than one that durably restores the user.
- **Heterogeneous effects.** Slice the lift by plan, country, device, and
  pre-treatment activity decile. The model already tells us the user is
  *at risk*; the segment-level lift tells us *who is worth treating*.

**Statistical techniques.**

- **Two-proportion z-test** for the primary binary outcome.
- **CUPED** (controlled experiment using pre-experiment data) to reduce
  variance using each user's pre-period activity as a covariate - a
  particularly good fit here because we already engineered those features.
- **Bayesian estimation** of the treatment-effect distribution if the
  business prefers a posterior probability of lift to a p-value.
- **Sequential / Bonferroni correction** if multiple variants are tested
  simultaneously.
- **Segment-level inference** with hierarchical models or simple tests with
  multiple-testing correction.

A failed test is also informative: if a re-engagement push gets the same
result as control, the model is correctly identifying churners but the
intervention is not the right tool for that segment, and we should pivot
(e.g. fee waiver, in-app credit, customer-success outreach for high-value
users).

---

## 5. What I would have done differently with more time

- **Out-of-time validation.** Hold out the most recent four weeks of users
  (or use multiple rolling cutoffs) rather than a random split; that better
  approximates the deployment scenario where we score today's users to
  predict tomorrow's behaviour.
- **Gradient boosting** (XGBoost or LightGBM) with proper hyper-parameter
  search and SHAP-based feature attributions - likely a small AUC
  improvement and more nuanced explanations.
- **More expressive recency features.** Time-since-last-X for several X
  (last completed transaction, last failed transaction, last
  re-engagement notification), plus weekly transaction trajectories
  (slope of last 8 weeks).
- **Threshold optimisation against expected business value.** Combine
  per-user predicted churn probability with the user's expected lifetime
  value to pick the threshold that maximises net contribution from the
  re-engagement programme, rather than the F1-default 0.5.
- **Survival framing.** Reformulate the problem as a time-to-event /
  survival model (Cox PH or DeepSurv). Churn is not really binary; it has
  a censoring structure that survival models capture naturally.
- **Calibration.** Even with `class_weight="balanced"`, the absolute
  probabilities from the RF are not perfectly calibrated; isotonic or
  Platt calibration on a held-out fold would yield more reliable
  probabilities for downstream cost / benefit calculations.
- **Address the email deliverability finding.** 27 % of notifications are
  marked FAILED; if the same user receives only failed notifications they
  are functionally untreatable by the existing channels. This is an
  ops-level fix that would help any churn-reduction programme more than a
  better classifier would.
- **Investigate the 32 "Unknown" device brands** to ensure they are not a
  hidden segment with different behaviour.

---

### Reproducibility

The full pipeline is in `src/` and is invoked with `python -m src.main`.
Outputs are written to `outputs/figures/` (PNG plots), `outputs/tables/`
(CSV / JSON summaries) and `outputs/models/` (the pickled best model).
Random seeds are fixed in `src/config.py` (`RANDOM_STATE = 42`).
