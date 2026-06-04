# CLAUDE.md

## Project
Insurance Risk Predictor — a Streamlit app that scores insurance policies for claim risk and explains each score with SHAP. Portfolio / proof-of-concept.

## Stack
- Python 3.11+
- scikit-learn (RandomForestClassifier), pandas, numpy
- shap (explainability)
- streamlit (single-file `app.py`)
- matplotlib (plots)

## Conventions
- **PRD.md is the source of truth.** Build against it; ask before deviating.
- **HANDOFF.md tracks state.** Read it at session start; update it at session end.
- **One task at a time.** Do the next unchecked phase in HANDOFF only, then stop and confirm.
- **Edit, don't rewrite.** Once a file exists, make targeted edits, not full rewrites.
- **Single-file app** (`app.py`) — keep everything there unless PRD says otherwise.
- Readable over clever — the author needs to explain this code in an interview.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Do not
- Use `Exposure` as a model feature (target leakage).
- Add databases, logins, file uploads, or extra models — out of scope.
