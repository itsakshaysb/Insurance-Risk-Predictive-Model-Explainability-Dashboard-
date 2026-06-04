# PRD — Insurance Risk Predictor & Explainability Dashboard

> **This file is the source of truth.** Build strictly against it. If anything is ambiguous, ask before assuming.

## 1. What we are building
A single-file Streamlit web app that:
1. Trains a machine-learning model to predict whether an insurance policy will have a claim.
2. Outputs a **risk score** (claim probability, 0–1) for any policy.
3. Explains **why** each score was given, using SHAP (per-policy + global drivers).
4. Presents it all in an interactive dashboard usable by non-technical people.

This is a **proof of concept / portfolio piece**, not production software. Favour clarity and a working demo over completeness.

## 2. Why (context — do not skip)
The author is a commercial P&C underwriter applying for a Data & AI insurance consulting role. This project must demonstrate three things in interview:
- classical ML (a real predictive model),
- explainability (SHAP — the consulting-grade "why"),
- communication (a dashboard for technical AND non-technical audiences).
Every design choice should serve those three goals.

## 3. Data
- **Source:** OpenML, dataset `freMTPL2freq` (French motor third-party liability). Free, no login.
- **Load:** `from sklearn.datasets import fetch_openml` → `fetch_openml(name="freMTPL2freq", as_frame=True, parser="auto").frame`
- **Sample:** down to 100,000 rows (`random_state=42`) so it trains fast on a Codespace.
- **Target (binary):** `claim = 1 if ClaimNb > 0 else 0`.
- **Features (underwriting variables):** `VehPower`, `VehAge`, `DrivAge`, `BonusMalus`, `Area`, `VehBrand`, `VehGas`, `Density`, `Region`.
- **Exclude from features:** `IDpol`, `ClaimNb`, `Exposure` (Exposure leaks the target; do not use as a feature).

## 4. Model
- `sklearn.ensemble.RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", n_jobs=-1, random_state=42)`.
- Preprocessing: one-hot encode categoricals (`Area`, `VehBrand`, `VehGas`, `Region`) via `pd.get_dummies(drop_first=True)`; numerics pass through.
- Stratified train/test split (`test_size=0.2`, `random_state=42`, `stratify=y`).
- **Risk score = `model.predict_proba(X)[:, 1]`.**

## 5. Explainability (the star feature)
- Use `shap.TreeExplainer(model)`.
- **Global view:** SHAP summary / bar plot = top risk drivers across the whole book.
- **Per-policy view:** for the single policy the user enters, show the top factors that pushed its score up/down, in plain language (e.g. "Young driver → +0.12").
- ⚠️ **Known gotcha:** for a binary RandomForest, SHAP output shape varies by version (list of 2 arrays vs 3D array). Handle both; select the positive-class slice. The per-policy single-row input must be re-aligned to the trained columns with `reindex(columns=X.columns, fill_value=0)`. Budget time here; if blocked, ship the global plot first.

## 6. Dashboard (Streamlit `app.py`)
Layout:
- **Title + one-line description.**
- **Sidebar — "Score a policy":** inputs for the features (sliders for numerics like DrivAge/VehAge/BonusMalus; selectboxes for categoricals like Region/Area/VehGas/VehBrand). Sensible defaults.
- **Main area:**
  - Big **risk score** for the entered policy (st.metric, shown as a %).
  - **Per-policy SHAP explanation** — top 5 drivers up/down for this policy.
  - **Portfolio histogram** — distribution of risk scores across all sampled policies, with this policy's score marked.
  - **Global drivers** — SHAP summary plot (overall feature importance).
- Use `@st.cache_resource` for the trained model/explainer and `@st.cache_data` for the dataset so it doesn't retrain on every interaction.

## 7. Deliverables (files)
- `app.py` — the whole app, single file.
- `requirements.txt` — streamlit, scikit-learn, pandas, numpy, shap, matplotlib.
- `README.md` — what it is, how to run locally, how to deploy to Streamlit Cloud, screenshot placeholder, honest limitations.
- `CLAUDE.md` — short build conventions (already in repo).
- Keep `PRD.md` and `HANDOFF.md` updated.

## 8. Definition of done
- `streamlit run app.py` launches with no errors.
- Entering a policy returns a risk score AND a per-policy explanation.
- Portfolio histogram and global drivers render.
- README explains it and lists honest limitations.
- Pushed to GitHub; (optional) deployed to Streamlit Community Cloud with a public link.

## 9. Honest limitations (put these in README — they make the author look credible)
- Claims are rare → imbalanced data; accuracy is misleading, prefer ROC-AUC.
- Model predicts claim **frequency** (will it claim), not claim **cost/severity**.
- Trained on French motor data, not Indian commercial lines — a methodology demo, not a deployable underwriting model.

## 10. Out of scope (do NOT build)
- User accounts / databases / file uploads.
- Multiple models or hyperparameter tuning frameworks.
- Anything beyond the single `app.py` + the listed files.
