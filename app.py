from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "student_performance.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="EduPulse AI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(100, 116, 139, .10); padding: 16px; border-radius: 12px;}
.hero {padding: 1.2rem 1.4rem; border-radius: 16px; background: linear-gradient(120deg, #172554, #164e63); color: white; margin-bottom: 1rem;}
.small-note {color: #64748b; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

FEATURES = [
    "study_hours", "attendance_pct", "previous_score", "assignments_completed",
    "sleep_hours", "internet_access", "extracurricular"
]
NUMERIC = ["study_hours", "attendance_pct", "previous_score", "assignments_completed", "sleep_hours"]
CATEGORICAL = ["internet_access", "extracurricular"]

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

def build_pipeline(model, data):
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), CATEGORICAL)
    ])
    return Pipeline([("preprocessor", preprocessor), ("model", model)])

@st.cache_resource
def train_models(data):
    X = data[FEATURES]
    y_score = data["final_score"]
    y_pass = data["pass_status"]
    X_train, X_test, y_train_score, y_test_score, y_train_pass, y_test_pass = train_test_split(
        X, y_score, y_pass, test_size=0.2, random_state=42
    )
    score_model = build_pipeline(RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42), data)
    pass_model = build_pipeline(RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=42), data)
    score_model.fit(X_train, y_train_score)
    pass_model.fit(X_train, y_train_pass)
    score_pred = score_model.predict(X_test)
    pass_pred = pass_model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test_score, score_pred),
        "r2": r2_score(y_test_score, score_pred),
        "accuracy": accuracy_score(y_test_pass, pass_pred),
        "report": classification_report(y_test_pass, pass_pred, output_dict=True, zero_division=0)
    }
    return score_model, pass_model, metrics

data = load_data()
score_model, pass_model, metrics = train_models(data)

st.markdown("""
<div class="hero">
  <h1 style="margin:0">🎓 EduPulse AI</h1>
  <p style="margin:.4rem 0 0 0;font-size:1.05rem">Student Performance & Academic Risk Intelligence</p>
  <p style="margin:.4rem 0 0 0;opacity:.85">Explore learning patterns, estimate outcomes, and identify students who may benefit from extra support.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Overview", "Predict a Student", "Explore Data", "Model Insights", "About"])
    st.divider()
    st.caption("Demo project • Synthetic sample data")
    st.caption("Predictions support learning conversations; they should not be used as the sole basis for consequential decisions.")

if page == "Overview":
    st.subheader("Performance overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students in sample", f"{len(data):,}")
    c2.metric("Average final score", f"{data.final_score.mean():.1f}/100")
    c3.metric("Pass rate", f"{(data.pass_status == 'Pass').mean()*100:.1f}%")
    c4.metric("Average attendance", f"{data.attendance_pct.mean():.1f}%")
    left, right = st.columns(2)
    with left:
        fig = px.histogram(data, x="final_score", nbins=20, title="Final score distribution",
                           labels={"final_score": "Final score"}, color_discrete_sequence=["#0891b2"])
        st.plotly_chart(fig, use_container_width=True)
    with right:
        grouped = data.assign(study_band=pd.cut(data.study_hours, bins=[0, 1, 2, 3, 5, 10],
                                                labels=["<1", "1–2", "2–3", "3–5", "5+"])).groupby(
                                                    "study_band", observed=False).final_score.mean().reset_index()
        fig = px.bar(grouped, x="study_band", y="final_score", title="Average score by daily study hours",
                     labels={"study_band": "Daily study hours", "final_score": "Average score"},
                     color="final_score", color_continuous_scale="Teal")
        st.plotly_chart(fig, use_container_width=True)
    st.info("Use the sidebar to make a prediction, inspect the dataset, or review model quality.")

elif page == "Predict a Student":
    st.subheader("Estimate a student's outcome")
    st.write("Enter a learning profile below. The model estimates a score and a pass/fail category.")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            study_hours = st.slider("Study hours per day", 0.0, 10.0, 2.5, 0.5)
            attendance = st.slider("Attendance (%)", 0, 100, 80, 1)
            previous = st.slider("Previous assessment score", 0, 100, 65, 1)
        with col2:
            assignments = st.slider("Assignments completed (%)", 0, 100, 75, 1)
            sleep = st.slider("Sleep per night (hours)", 2.0, 12.0, 7.0, 0.5)
            internet = st.selectbox("Reliable internet access", ["Yes", "No"])
        with col3:
            extra = st.selectbox("Extracurricular participation", ["Yes", "No"])
            st.markdown("#### Prediction settings")
            threshold = st.slider("Pass threshold", 30, 70, 50, 5)
        submitted = st.form_submit_button("✨ Predict performance", type="primary")
    if submitted:
        row = pd.DataFrame([{
            "study_hours": study_hours, "attendance_pct": attendance,
            "previous_score": previous, "assignments_completed": assignments,
            "sleep_hours": sleep, "internet_access": internet, "extracurricular": extra
        }])
        predicted_score = float(np.clip(score_model.predict(row)[0], 0, 100))
        probabilities = pass_model.predict_proba(row)[0]
        classes = list(pass_model.named_steps["model"].classes_)
        pass_probability = float(probabilities[classes.index("Pass")]) if "Pass" in classes else 0.0
        category = "Pass" if predicted_score >= threshold else "Needs support"
        a, b, c = st.columns(3)
        a.metric("Estimated final score", f"{predicted_score:.1f}/100")
        b.metric("Estimated pass likelihood", f"{pass_probability*100:.1f}%")
        c.metric("Rule-based category", category)
        if predicted_score < threshold or pass_probability < 0.60:
            st.warning("Consider a supportive check-in: review attendance, assignment completion, study routines, and access to learning resources.")
        else:
            st.success("The estimate looks encouraging. Continue monitoring progress and maintaining healthy routines.")
        st.caption("This is an estimate from synthetic training data, not a guarantee of actual academic performance.")

elif page == "Explore Data":
    st.subheader("Explore the sample dataset")
    col1, col2 = st.columns(2)
    with col1:
        selected = st.multiselect("Filter by pass status", ["Pass", "Needs Support"], default=["Pass", "Needs Support"])
    with col2:
        min_att = st.slider("Minimum attendance (%)", 0, 100, 0)
    filtered = data[data.pass_status.isin(selected) & (data.attendance_pct >= min_att)]
    st.write(f"Showing **{len(filtered)}** of **{len(data)}** records.")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"),
                       file_name="student_performance_filtered.csv", mime="text/csv")
    corr = data[NUMERIC + ["final_score"]].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Numeric feature correlation",
                    color_continuous_scale="Teal")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Model Insights":
    st.subheader("Model quality & feature relationships")
    a, b, c = st.columns(3)
    a.metric("Score MAE", f"{metrics['mae']:.2f}", help="Mean absolute error in score points; lower is better.")
    b.metric("Score R²", f"{metrics['r2']:.3f}", help="Explained variance estimate; closer to 1 is generally better.")
    c.metric("Pass classifier accuracy", f"{metrics['accuracy']*100:.1f}%")
    st.write("**Important:** metrics are measured on a held-out sample from a synthetic dataset. They are for demonstrating a workflow, not validating a real educational model.")
    st.markdown("#### Average score by attendance")
    att = data.assign(attendance_band=pd.cut(data.attendance_pct, bins=[-1, 50, 65, 80, 90, 100],
                                              labels=["≤50", "51–65", "66–80", "81–90", "91–100"]))
    summary = att.groupby("attendance_band", observed=False).final_score.mean().reset_index()
    fig = px.line(summary, x="attendance_band", y="final_score", markers=True,
                  labels={"attendance_band": "Attendance (%)", "final_score": "Average score"},
                  title="Observed average score by attendance band")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Classification metrics")
    report = metrics["report"]
    rows = []
    for label in ["Pass", "Needs Support"]:
        if label in report:
            rows.append({"Class": label, "Precision": report[label]["precision"],
                         "Recall": report[label]["recall"], "F1-score": report[label]["f1-score"]})
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Class").style.format("{:.3f}"), use_container_width=True)

else:
    st.subheader("About EduPulse AI")
    st.markdown("""
**EduPulse AI** is a portfolio-ready data science application demonstrating a complete machine-learning workflow.

**Features**
- Exploratory data analysis and interactive visualizations
- Score prediction with a Random Forest regressor
- Pass/support classification with a Random Forest classifier
- Preprocessing for numeric and categorical features
- Model evaluation on a held-out test set
- Downloadable filtered dataset

**Technology stack:** Python, Streamlit, pandas, NumPy, scikit-learn, Plotly.

**Ethical use:** The included dataset is synthetic and does not represent real students. Do not upload identifiable student information or use predictions to deny opportunities. Real deployment would require consent, representative data, bias evaluation, privacy controls, and human oversight.
""")
    st.download_button("Download complete dataset", data.to_csv(index=False).encode("utf-8"),
                       file_name="student_performance.csv", mime="text/csv")
