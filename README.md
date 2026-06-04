# Insurance Risk Predictor & Explainability Dashboard

A Streamlit app that scores insurance policies for claim risk and explains each score using SHAP — built as a portfolio piece to demonstrate classical ML, explainability, and dashboard communication.

---

## What it does

1. **Trains a RandomForest** on 100 000 rows of real French motor insurance data to predict whether a policy will generate a claim.
2. **Scores any policy** you enter via the sidebar — outputs a claim probability (0–100%).
3. **Explains the score** using SHAP: which features pushed the risk up or down for that specific policy, and the top drivers across the whole book.
4. **Shows the portfolio distribution** — a histogram of all 100k risk scores with your policy marked in red.

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

The app downloads the dataset from OpenML on the first run (~30 MB) and trains the model (~1 min). Both are cached — subsequent runs start instantly.

---

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (already done).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select this repo → set **Main file path** to `app.py` → **Deploy**.
4. Streamlit Cloud installs `requirements.txt` automatically and hosts the app for free.

> First cold start on Streamlit Cloud takes ~3–5 min (dataset download + model training). After that, the cache kicks in and interactions are fast.

---

## Tech stack

| Layer | Tool |
|---|---|
| App | Streamlit |
| Model | scikit-learn `RandomForestClassifier` |
| Explainability | SHAP `TreeExplainer` |
| Data | pandas, numpy |
| Plots | matplotlib |

---

## Honest limitations

- **Imbalanced data.** Claims are rare (~5% of policies). Accuracy is a misleading metric here — the app reports ROC-AUC instead, which measures how well the model separates claimants from non-claimants regardless of threshold.
- **Frequency, not severity.** The model predicts *whether* a policy will claim, not *how much* it will cost. A high-risk score does not mean a large loss.
- **French motor data only.** The dataset (`freMTPL2freq`) covers French third-party liability motor policies. Feature distributions, region codes, and base rates will differ from any other market. This is a methodology demonstration, not a deployable underwriting model.
- **No temporal validation.** The train/test split is random, not time-based. A production model would be validated on out-of-time data to detect drift.
- **SHAP on a sample.** Global SHAP values are computed on a 2 000-row sample for performance. The rankings are stable but the magnitudes are approximate.
