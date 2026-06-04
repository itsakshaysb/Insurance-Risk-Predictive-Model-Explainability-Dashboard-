# Project Reference — Insurance Risk Predictor & Explainability Dashboard

## What we are building
A single-file Streamlit app (`app.py`) that:
1. Trains a RandomForest to predict whether an insurance policy will result in a claim.
2. Outputs a **risk score** (claim probability %) for any policy entered by the user.
3. Explains the score using **SHAP** — both per-policy (top 5 drivers) and globally (feature importance across the full book).
4. Presents everything in an interactive dashboard for technical and non-technical users.

**Purpose:** Portfolio / proof-of-concept for a commercial P&C underwriter interviewing for a Data & AI insurance consulting role. Must demonstrate classical ML, explainability (SHAP), and communication via a dashboard.

---

## Current Status
- **Phases 0–7 complete.** App works end-to-end. ROC-AUC: 0.6259. Only Phase 8 (README + polish) remains.
- Next task: **Phase 8 — README + polish**.

## Build Phases
- [x] Phase 0 — Scaffold: `requirements.txt` + empty `app.py`, `pip install`
- [x] Phase 1 — Data: load `freMTPL2freq` from OpenML, sample 100k rows, build binary `claim` target
- [x] Phase 2 — Train: one-hot encode, stratified split, train RandomForest, cache with `@st.cache_resource`
- [x] Phase 3 — Sanity check: ROC-AUC = 0.6259 ✓
- [x] Phase 4 — Global SHAP: `shap.TreeExplainer`, render summary/bar plot
- [x] Phase 5 — Sidebar + single prediction: sidebar inputs, show risk score via `st.metric`
- [x] Phase 6 — Per-policy SHAP: top 5 up/down drivers for entered policy
- [x] Phase 7 — Portfolio histogram: all risk scores, mark entered policy's score
- [ ] Phase 8 — README + polish: run steps, Streamlit Cloud deploy, limitations

---

## Stack
| Layer | Tool |
|---|---|
| App | Streamlit (single file `app.py`) |
| Model | `sklearn.ensemble.RandomForestClassifier` |
| Explainability | `shap.TreeExplainer` |
| Data wrangling | pandas, numpy |
| Plots | matplotlib |
| Python | 3.11+ |

---

## Data
- **Source:** OpenML `freMTPL2freq` (French motor third-party liability), loaded via `sklearn.datasets.fetch_openml`
- **Sample:** 100,000 rows, `random_state=42`
- **Target:** `claim = 1 if ClaimNb > 0 else 0`
- **Features:** `VehPower`, `VehAge`, `DrivAge`, `BonusMalus`, `Area`, `VehBrand`, `VehGas`, `Density`, `Region`
- **Excluded:** `IDpol`, `ClaimNb`, `Exposure` — **Exposure is target leakage, never use as a feature**

## Model Config
```python
RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", n_jobs=-1, random_state=42)
```
- Categoricals (`Area`, `VehBrand`, `VehGas`, `Region`) → `pd.get_dummies(drop_first=True)`
- Split: `test_size=0.2`, `random_state=42`, `stratify=y`
- Risk score = `model.predict_proba(X)[:, 1]`

---

## Key Rules
- **PRD.md is source of truth.** Ask before deviating.
- **One phase at a time.** Finish, confirm it works, then move on.
- **Single-file app.** Everything in `app.py` unless PRD says otherwise.
- **Readable over clever.** Author must explain this code in an interview.
- **Never use `Exposure` as a feature** (target leakage).
- No databases, logins, file uploads, or extra models — out of scope.

## Known Gotcha — SHAP
For a binary RandomForest, SHAP output shape varies by version (list of 2 arrays vs 3D array). Handle both; select the positive-class slice. Per-policy single-row input must be re-aligned to trained columns with `reindex(columns=X.columns, fill_value=0)`.

---

## Deliverables
- `app.py` — full app, single file
- `requirements.txt`
- `README.md` — run steps, Streamlit Cloud deploy, screenshot placeholder, honest limitations
- `CLAUDE.md`, `PRD.md`, `HANDOFF.md` — keep updated

## Definition of Done
- `streamlit run app.py` launches with no errors
- Entering a policy returns a risk score + per-policy SHAP explanation
- Portfolio histogram and global drivers render
- README written with limitations
- Pushed to GitHub; optionally deployed to Streamlit Community Cloud

## Honest Limitations (go in README)
- Imbalanced data (rare claims) → use ROC-AUC, not accuracy
- Predicts claim frequency (will it claim), not cost/severity
- French motor data — methodology demo, not a deployable underwriting model

---

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
