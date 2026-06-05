# 🛡️ Insurance Risk Predictor & Explainability Dashboard

> An interactive Streamlit app that scores insurance policies for claim risk and explains **why** — every score is broken down with SHAP, for both technical and non-technical audiences.

<p align="left">
  <a href="https://insurance-risk-predictive-model-explainability-dashboard.streamlit.app/"><img alt="Live Demo" src="https://img.shields.io/badge/▶_Live_Demo-Open_App-FF4B4B?logo=streamlit&logoColor=white"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="SHAP" src="https://img.shields.io/badge/SHAP-explainability-6A5ACD">
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green">
</p>

**🔗 Live demo:** **https://insurance-risk-predictive-model-explainability-dashboard.streamlit.app/**

> First load may take ~30s to a minute while the app wakes up and warms its cache — then it's instant.

---

## Why I built this

I'm a commercial P&C underwriter moving into data & AI consulting. Underwriting already
*is* risk prediction — done by hand. This project rebuilds that judgement as a transparent
machine-learning model and, crucially, **explains every score** the way an underwriter or a
client would actually want it explained: in plain language, with the drivers ranked.

It's deliberately a single, readable file. The goal isn't a production underwriting engine —
it's to demonstrate three things end-to-end:

| Skill | How this project shows it |
|---|---|
| **Classical ML** | A real `RandomForestClassifier` trained on 100k policies, evaluated honestly with ROC-AUC on a held-out set. |
| **Explainability** | SHAP at two levels — *per-policy* (why this score?) and *global* (what drives the whole book?). |
| **Communication** | A dashboard non-technical stakeholders can use: risk tiers, plain-English drivers, and a portfolio view. |

---

## What it does

1. **Trains a RandomForest** on 100,000 rows of real French motor insurance data to predict whether a policy will generate a claim.
2. **Scores any policy** you enter in the sidebar — outputs a claim probability (0–100%) and an underwriting **risk tier** (🟢 Low → 🔴 High) relative to the book.
3. **Explains the score** with SHAP: the top 5 factors that pushed *this* policy's risk up or down, in readable labels — plus the global drivers across all 100k policies.
4. **Shows the portfolio distribution** — a histogram of every risk score with the current policy marked.

> 📸 _Add a screenshot or GIF here — it's the single biggest upgrade to this README.
> Run the app, grab a screenshot, and drop it in as `docs/screenshot.png`._

---

## How it works

```
OpenML  ──►  pandas/get_dummies  ──►  RandomForestClassifier  ──►  predict_proba
(freMTPL2freq)   (one-hot encode)        (200 trees, depth 8)      = risk score
                                              │
                                              ▼
                                       SHAP TreeExplainer
                                       ├─ per-policy drivers (local)
                                       └─ global feature importance
                                              │
                                              ▼
                                     Streamlit dashboard
```

Both the dataset and the trained model are cached (`@st.cache_data` / `@st.cache_resource`),
so the app only downloads and trains **once** — every interaction after that is instant.

---

## Model card

| | |
|---|---|
| **Task** | Binary classification — will a policy file ≥1 claim? |
| **Data** | OpenML [`freMTPL2freq`](https://www.openml.org/search?type=data&id=41214) — French motor third-party liability, 100k-row sample. |
| **Target** | `claim = 1 if ClaimNb > 0 else 0` (~5% positive — imbalanced). |
| **Features** | `VehPower`, `VehAge`, `DrivAge`, `BonusMalus`, `Area`, `VehBrand`, `VehGas`, `Density`, `Region`. |
| **Excluded** | `Exposure` — leaks the target (longer-exposed policies claim more by construction). |
| **Algorithm** | `RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced")`. |
| **Validation** | Stratified 80/20 split; class weighting to handle imbalance. |
| **Metric** | **ROC-AUC ≈ 0.63** on the held-out test set (baseline = 0.50). |
| **Explainability** | SHAP `TreeExplainer`; global values on a 2,000-row sample for speed. |

---

## Run locally

```bash
# 1. Clone
git clone https://github.com/itsakshaysb/Insurance-Risk-Predictive-Model-Explainability-Dashboard-.git
cd Insurance-Risk-Predictive-Model-Explainability-Dashboard-

# 2. Install dependencies (Python 3.11+)
pip install -r requirements.txt

# 3. Launch
streamlit run app.py
```

The app downloads the dataset from OpenML on the first run (~30 MB) and trains the model
(~1 min). Both are cached — subsequent runs start instantly.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **New app** → select this repo → set **Main file path** to `app.py` → **Deploy**.
4. Streamlit Cloud installs `requirements.txt` automatically and hosts the app for free.

> First cold start takes ~3–5 min (dataset download + training). After that the cache kicks in.

---

## Tech stack

| Layer | Tool |
|---|---|
| App / UI | Streamlit |
| Model | scikit-learn `RandomForestClassifier` |
| Explainability | SHAP `TreeExplainer` |
| Data | pandas, numpy |
| Plots | matplotlib |

---

## Honest limitations

I'd rather state these up front — knowing *why* a model can't be trusted is half of the job.

- **Imbalanced data.** Claims are rare (~5% of policies), so accuracy is misleading. The app reports **ROC-AUC**, which measures how well the model separates claimants from non-claimants regardless of threshold.
- **Frequency, not severity.** The model predicts *whether* a policy claims, not *how much* it costs. A high score ≠ a large loss.
- **French motor data only.** `freMTPL2freq` covers French third-party liability motor policies. Base rates, region codes, and feature distributions won't transfer to other markets. This is a **methodology demonstration, not a deployable underwriting model**.
- **No temporal validation.** The split is random, not time-based. A production model would be validated out-of-time to catch drift.
- **SHAP on a sample.** Global SHAP values use a 2,000-row sample for speed — rankings are stable, magnitudes approximate.

---

## Possible extensions

Things I'd build next if this became more than a portfolio piece:

- A **severity model** (claim cost) to pair with this frequency model → expected loss.
- **Out-of-time validation** and a drift monitor.
- **Calibration** (reliability curve / Platt scaling) so the probabilities are decision-grade.
- Compare the RandomForest against **gradient boosting** (XGBoost / LightGBM).

---

## Author

**Akshay S B** — commercial P&C underwriter moving into Data & AI consulting.
Built as a portfolio piece. Licensed under [MIT](LICENSE).
