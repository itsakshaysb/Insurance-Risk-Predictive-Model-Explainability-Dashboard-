
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="Insurance Risk Predictor", layout="wide")
st.title("Insurance Risk Predictor")
st.caption("Scores insurance policies for claim risk and explains each score using SHAP.")


# ── Plain-language helpers (so the explanations read in English, not code) ──────

FEATURE_LABELS = {
    "VehPower":   "Vehicle power",
    "VehAge":     "Vehicle age",
    "DrivAge":    "Driver age",
    "BonusMalus": "Bonus-Malus (claims history)",
    "Density":    "Population density",
}


def humanize(feature: str) -> str:
    """Turn an encoded column name into a readable label.
    'DrivAge' -> 'Driver age';  'Region_R24' -> 'Region = R24'."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    if "_" in feature:                       # one-hot column, e.g. 'VehBrand_B12'
        base, value = feature.split("_", 1)
        return f"{FEATURE_LABELS.get(base, base)} = {value}"
    return feature


def risk_tier(score, scores):
    """Band a score against the portfolio into an underwriting tier (label, emoji)."""
    pct = (scores < score).mean()            # share of the book this policy beats
    if pct < 0.50:
        return "Low", "🟢"
    if pct < 0.80:
        return "Moderate", "🟡"
    if pct < 0.95:
        return "Elevated", "🟠"
    return "High", "🔴"


# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    raw = fetch_openml(name="freMTPL2freq", as_frame=True, parser="auto").frame
    df = raw.sample(100_000, random_state=42).reset_index(drop=True)
    df["claim"] = (df["ClaimNb"].astype(float) > 0).astype(int)
    features = ["VehPower", "VehAge", "DrivAge", "BonusMalus",
                "Area", "VehBrand", "VehGas", "Density", "Region"]
    X = df[features].copy()
    # Strip stray quotes from VehGas values (present in raw OpenML data)
    X["VehGas"] = X["VehGas"].astype(str).str.strip("'")
    return X, df["claim"]


# ── Model + SHAP (cached so it only trains once) ──────────────────────────────

@st.cache_resource
def train_model(_X, _y):
    X_enc = pd.get_dummies(_X, columns=["Area", "VehBrand", "VehGas", "Region"],
                           drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, _y, test_size=0.2, random_state=42, stratify=_y
    )
    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced",
        n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # SHAP on a 2 000-row sample for speed; shape → (n, features, 2)
    explainer = shap.TreeExplainer(model)
    shap_sample = X_enc.sample(2_000, random_state=42)
    sv = explainer(shap_sample)
    global_shap = shap.Explanation(
        values=sv.values[:, :, 1],
        base_values=sv.base_values[:, 1],
        data=sv.data,
        feature_names=list(X_enc.columns),
    )

    all_scores = model.predict_proba(X_enc)[:, 1]
    return model, X_enc, auc, explainer, global_shap, all_scores


# ── Load + train ──────────────────────────────────────────────────────────────

X_raw, y = load_data()

with st.spinner("Training model — first run takes ~1 min, then cached..."):
    model, X_enc, auc, explainer, global_shap, all_scores = train_model(X_raw, y)

st.info(f"ROC-AUC on held-out test set: **{auc:.3f}**  (baseline = 0.50, expect ~0.60–0.70)")

with st.expander("ℹ️  How to read this dashboard"):
    st.markdown(
        """
- **Score a policy** in the left sidebar — adjust the underwriting variables and the
  app re-scores instantly.
- **Claim Risk Score** = the model's estimated probability that this policy files at
  least one claim, shown next to its **risk tier** relative to the whole book.
- **Top 5 Risk Drivers** explain *this specific policy* — which factors pushed its
  score up or down (SHAP values).
- **Portfolio Risk Distribution** shows where this policy sits among all 100,000.
- **Global Risk Drivers** show what matters most across the entire book.
        """
    )


# ── Sidebar: enter a policy ───────────────────────────────────────────────────

st.sidebar.header("Score a Policy")
st.sidebar.caption("Set the policy details, then read the score on the right. "
                   "Hover the **?** next to any field for an explanation.")

with st.sidebar.expander("📖  What do these fields mean?"):
    st.markdown(
        """
- **Vehicle Power** — engine power rating. Higher = more powerful car.
- **Vehicle Age** — how old the insured car is, in years.
- **Driver Age** — age of the policyholder. Very young and very old drivers tend to be riskier.
- **Bonus-Malus** — the driver's claims-history score. **50 = a clean record (best)**;
  **100+ = past claims (worse)**. This is the single strongest risk signal.
- **Population Density** — people per km² where the car is based. Cities (high density)
  see more accidents and theft than rural areas.
- **Area Code** — a grouped land-use band (A = most rural → F = most urban).
- **Vehicle Brand** — the car's manufacturer group (anonymised as B1, B2, …).
- **Fuel Type** — Diesel or Regular (petrol).
- **Region** — the French administrative region the policy is written in (codes like R24).
        """
    )

areas   = sorted(X_raw["Area"].astype(str).unique())
brands  = sorted(X_raw["VehBrand"].astype(str).unique())
gases   = sorted(X_raw["VehGas"].astype(str).unique())
regions = sorted(X_raw["Region"].astype(str).unique())

veh_power  = st.sidebar.slider("Vehicle Power",             int(X_raw["VehPower"].min()),   int(X_raw["VehPower"].max()),   6,
                               help="Engine power rating of the car. Higher means a more powerful vehicle.")
veh_age    = st.sidebar.slider("Vehicle Age (years)",       int(X_raw["VehAge"].min()),     int(X_raw["VehAge"].max()),     3,
                               help="How old the insured car is, in years.")
driv_age   = st.sidebar.slider("Driver Age",                int(X_raw["DrivAge"].min()),    int(X_raw["DrivAge"].max()),    35,
                               help="Age of the policyholder. Very young and very old drivers tend to be riskier.")
bonus_mal  = st.sidebar.slider("Bonus-Malus",               int(X_raw["BonusMalus"].min()), int(X_raw["BonusMalus"].max()), 50,
                               help="Claims-history score: 50 = no claims (best), 100+ = prior claims (worse). The strongest risk signal.")
density    = st.sidebar.slider("Population Density (km²)",  int(X_raw["Density"].min()),    int(X_raw["Density"].max()),    500,
                               help="People per km² where the car is based. Cities see more accidents and theft than rural areas.")
area       = st.sidebar.selectbox("Area Code",       areas,
                               help="Grouped land-use band, A (most rural) → F (most urban).")
brand      = st.sidebar.selectbox("Vehicle Brand",   brands,
                               help="The car's manufacturer group, anonymised as B1, B2, …")
gas        = st.sidebar.selectbox("Fuel Type",       gases,
                               help="Diesel or Regular (petrol).")
region     = st.sidebar.selectbox("Region",          regions,
                               help="The French administrative region the policy is written in (codes like R24).")


# ── Encode the entered policy and score it ────────────────────────────────────

policy_raw = pd.DataFrame([{
    "VehPower": veh_power, "VehAge": veh_age, "DrivAge": driv_age,
    "BonusMalus": bonus_mal, "Area": area, "VehBrand": brand,
    "VehGas": gas, "Density": density, "Region": region,
}])
policy_enc = pd.get_dummies(policy_raw,
                            columns=["Area", "VehBrand", "VehGas", "Region"],
                            drop_first=True)
policy_enc = policy_enc.reindex(columns=X_enc.columns, fill_value=0)

risk_score = model.predict_proba(policy_enc)[0, 1]


# ── Per-policy SHAP ───────────────────────────────────────────────────────────

sv_policy = explainer(policy_enc)
# shape (1, n_features, 2) → positive-class slice
policy_shap_vals = sv_policy.values[0, :, 1]

shap_df = (
    pd.DataFrame({"feature": X_enc.columns.tolist(), "shap": policy_shap_vals})
    .assign(abs_shap=lambda d: d["shap"].abs())
    .sort_values("abs_shap", ascending=False)
    .head(5)
)


# ── Main area ─────────────────────────────────────────────────────────────────

col_score, col_drivers = st.columns([1, 2])

with col_score:
    avg_score = float(all_scores.mean())
    delta = risk_score - avg_score
    st.metric(
        "Claim Risk Score",
        f"{risk_score:.1%}",
        delta=f"{delta:+.1%} vs portfolio avg",
        delta_color="inverse",  # red = above avg (bad), green = below avg (good)
    )
    st.caption("Probability this policy generates at least one claim.")

    tier, badge = risk_tier(risk_score, all_scores)
    pct_rank = (all_scores < risk_score).mean()
    st.markdown(f"### {badge} {tier} risk")
    st.caption(f"Riskier than {pct_rank:.0%} of the 100,000-policy book.")

with col_drivers:
    st.subheader("Top 5 Risk Drivers for This Policy")
    st.caption("How each factor moved this policy's score — most influential first.")
    for _, row in shap_df.iterrows():
        arrow, effect = ("🔺", "raises risk") if row["shap"] > 0 else ("🔻", "lowers risk")
        st.write(f"{arrow} **{humanize(row['feature'])}** — {effect}  "
                 f"(SHAP {row['shap']:+.4f})")

st.divider()

# Portfolio histogram
st.subheader("Portfolio Risk Distribution")
fig1, ax1 = plt.subplots(figsize=(10, 3))
ax1.hist(all_scores, bins=60, color="#4A90D9", edgecolor="white", alpha=0.85)
ax1.axvline(risk_score, color="#E74C3C", linewidth=2,
            label=f"This policy: {risk_score:.1%}")
ax1.set_xlabel("Risk Score (claim probability)")
ax1.set_ylabel("Number of Policies")
ax1.legend()
st.pyplot(fig1)
plt.close(fig1)

st.divider()

# Global SHAP bar chart
st.subheader("Global Risk Drivers (across all 100 000 policies)")
plt.figure()
shap.plots.bar(global_shap, max_display=12, show=False)
st.pyplot(plt.gcf())
plt.close("all")
