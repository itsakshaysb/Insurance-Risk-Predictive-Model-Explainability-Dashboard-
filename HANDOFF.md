# HANDOFF — Insurance Risk Predictor

> **Read this first at the start of every session.** It tells you the current state and the single next task. Update it at the end of every session.

## Build conventions (follow exactly)
- **PRD.md is the source of truth.** Build against it. Ask before deviating.
- **One task at a time.** Do the next unchecked phase only. Stop and confirm before moving on.
- **`str_replace`-style edits only** when modifying existing files — no full-file rewrites once a file exists.
- **Keep it single-file** (`app.py`) unless PRD says otherwise.
- After finishing a task: run it, confirm it works, then update the "Current state" and "Next task" sections below.

## Current state
- Status: **Phases 0–7 complete.** Full app is working.
- Files present: `app.py`, `requirements.txt`, `PRD.md`, `HANDOFF.md`, `CLAUDE.md`, `main.md`.
- ROC-AUC confirmed: **0.6259** on held-out test set.
- SHAP: global bar chart + per-policy top-5 drivers both working.
- Streamlit app launches cleanly with `streamlit run app.py`.

## Phases (check off as you go)
- [x] **Phase 0 — Scaffold.** Create `requirements.txt` (streamlit, scikit-learn, pandas, numpy, shap, matplotlib) and an empty `app.py`. Run `pip install -r requirements.txt`.
- [x] **Phase 1 — Data.** In `app.py`, load `freMTPL2freq` from OpenML, sample 100k rows, build the binary `claim` target. Print shape + claim rate to confirm.
- [x] **Phase 2 — Train.** One-hot encode categoricals, stratified split, train the RandomForest (params per PRD §4). Cache with `@st.cache_resource`.
- [x] **Phase 3 — Sanity check.** Print ROC-AUC on the test set. Confirm it's clearly above 0.5 (expect ~0.6–0.7). This proves the model learned something.
- [x] **Phase 4 — Global SHAP.** Build `shap.TreeExplainer`; render the global summary/bar plot in the app.
- [x] **Phase 5 — Sidebar + single prediction.** Sidebar inputs for one policy; show its risk score as a % via `st.metric`.
- [x] **Phase 6 — Per-policy SHAP.** Show top 5 up/down drivers for the entered policy. (Handle the SHAP shape + column-reindex gotcha — PRD §5.)
- [x] **Phase 7 — Portfolio histogram.** Histogram of all risk scores, mark the entered policy's score.
- [x] **Phase 8 — README + polish.** Write README (run steps, Streamlit Cloud deploy steps, limitations per PRD §9). Final pass.

## Next task
✅ **All phases complete.** App is done. Optional: deploy to Streamlit Community Cloud and add a screenshot to README.

## Open questions / notes
- If SHAP per-policy (Phase 6) blocks for more than ~30 min, ship Phase 7 first and return to it — global SHAP (Phase 4) already covers the explainability story.
- Do not use `Exposure` as a feature (target leakage).
- Author will read the final code to be able to explain it in interview — favour readable code with brief comments over clever one-liners.

## Session log
- **2026-06-04** — Built Phases 0–7 in one session. Full app working: model trains to AUC 0.6259, global SHAP bar chart, per-policy top-5 drivers, portfolio histogram, sidebar inputs. SHAP shape gotcha handled (3D array → positive-class slice).
- **2026-06-04** — Phase 8 complete. README written with run steps, Streamlit Cloud deploy, honest limitations. Polish: stripped quotes from VehGas values, added Bonus-Malus tooltip, added delta vs portfolio avg to risk score metric. All phases done.
- **2026-06-05** — Portfolio/CV polish pass (no scope change). app.py: plain-language SHAP driver labels (`humanize`), underwriting risk-tier band (`risk_tier`, 🟢→🔴), and a "How to read this dashboard" expander. README rewritten as a portfolio README (business framing, "what this demonstrates" skill table, architecture diagram, model card, roadmap, badges, screenshot placeholder). Added MIT `LICENSE`. Still single-file, no new models/data.
