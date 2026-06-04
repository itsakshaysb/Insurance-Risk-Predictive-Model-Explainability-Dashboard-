# HANDOFF — Insurance Risk Predictor

> **Read this first at the start of every session.** It tells you the current state and the single next task. Update it at the end of every session.

## Build conventions (follow exactly)
- **PRD.md is the source of truth.** Build against it. Ask before deviating.
- **One task at a time.** Do the next unchecked phase only. Stop and confirm before moving on.
- **`str_replace`-style edits only** when modifying existing files — no full-file rewrites once a file exists.
- **Keep it single-file** (`app.py`) unless PRD says otherwise.
- After finishing a task: run it, confirm it works, then update the "Current state" and "Next task" sections below.

## Current state
- Status: **NOT STARTED.** Fresh repo. No code written yet.
- Files present: PRD.md, HANDOFF.md, CLAUDE.md (and this repo's README stub if any).
- Environment: GitHub Codespace, Python 3.11+, Claude Code installed.

## Phases (check off as you go)
- [ ] **Phase 0 — Scaffold.** Create `requirements.txt` (streamlit, scikit-learn, pandas, numpy, shap, matplotlib) and an empty `app.py`. Run `pip install -r requirements.txt`.
- [ ] **Phase 1 — Data.** In `app.py`, load `freMTPL2freq` from OpenML, sample 100k rows, build the binary `claim` target. Print shape + claim rate to confirm.
- [ ] **Phase 2 — Train.** One-hot encode categoricals, stratified split, train the RandomForest (params per PRD §4). Cache with `@st.cache_resource`.
- [ ] **Phase 3 — Sanity check.** Print ROC-AUC on the test set. Confirm it's clearly above 0.5 (expect ~0.6–0.7). This proves the model learned something.
- [ ] **Phase 4 — Global SHAP.** Build `shap.TreeExplainer`; render the global summary/bar plot in the app.
- [ ] **Phase 5 — Sidebar + single prediction.** Sidebar inputs for one policy; show its risk score as a % via `st.metric`.
- [ ] **Phase 6 — Per-policy SHAP.** Show top 5 up/down drivers for the entered policy. (Handle the SHAP shape + column-reindex gotcha — PRD §5.)
- [ ] **Phase 7 — Portfolio histogram.** Histogram of all risk scores, mark the entered policy's score.
- [ ] **Phase 8 — README + polish.** Write README (run steps, Streamlit Cloud deploy steps, limitations per PRD §9). Final pass.

## Next task
➡️ **Phase 0 — Scaffold.** Create requirements.txt and empty app.py, install deps.

## Open questions / notes
- If SHAP per-policy (Phase 6) blocks for more than ~30 min, ship Phase 7 first and return to it — global SHAP (Phase 4) already covers the explainability story.
- Do not use `Exposure` as a feature (target leakage).
- Author will read the final code to be able to explain it in interview — favour readable code with brief comments over clever one-liners.

## Session log
- _(empty — add a dated line after each session: what was done, what's next.)_
