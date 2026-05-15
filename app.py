import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Diabetes Prediction (KNN)", page_icon="🩺", layout="wide")

st.title("🩺 Diabetes Prediction using K-Nearest Neighbors")
st.caption("Upload diabetes.csv (or use the provided file in this folder) and predict whether a person has diabetes.")

DATA_PATH = "diabetes.csv"

def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(DATA_PATH)
    return df

@st.cache_data(show_spinner=False)
def get_dataset(uploaded_file=None):
    df = load_data(uploaded_file)

    # Basic validation
    if "Outcome" not in df.columns:
        raise ValueError("CSV must include an 'Outcome' column.")

    feature_cols = [c for c in df.columns if c != "Outcome"]
    X = df[feature_cols].copy()
    y = df["Outcome"].astype(int).copy()
    return df, feature_cols, X, y

@st.cache_resource(show_spinner=False)
def train_knn_model(X, y, test_size, random_state, n_neighbors, weights, metric, p):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                metric=metric,
                p=p if metric == "minkowski" else 2,
            )),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return model, acc, report, cm

def display_confusion_matrix_text(cm):
    # Text-only to avoid plots/diagrams
    st.write({"confusion_matrix": cm.tolist()})


uploaded = st.file_uploader("Upload diabetes.csv (optional)", type=["csv"])

df, feature_cols, X, y = get_dataset(uploaded)

with st.sidebar:
    st.header("Model Settings")

    test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    random_state = st.number_input("Random state", min_value=0, max_value=10_000, value=42, step=1)

    st.subheader("KNN Hyperparameters")
    n_neighbors = st.slider("n_neighbors", 1, 25, 5, 2)
    weights = st.selectbox("weights", options=["uniform", "distance"], index=0)

    metric = st.selectbox("metric", options=["minkowski", "euclidean", "manhattan"], index=0)
    p = 2
    if metric == "minkowski":
        p = st.slider("m (for Minkowski)", min_value=1, max_value=5, value=2, step=1)

    st.subheader("Action")
    train = st.button("Train / Update", type="primary")

if train:
    model, acc, report, cm = train_knn_model(
        X, y,
        test_size=test_size,
        random_state=int(random_state),
        n_neighbors=int(n_neighbors),
        weights=weights,
        metric=metric,
        p=int(p),
    )

    st.success(f"Model trained. Test Accuracy: {acc:.4f}")
    st.subheader("Confusion Matrix")
    display_confusion_matrix_text(cm)


    st.subheader("Classification Report")
    st.json({
        "accuracy": report.get("accuracy", None),
        "class_0": report.get("0", {}),
        "class_1": report.get("1", {}),
    })

    st.session_state["model"] = model
else:
    # Train a default model once so prediction works even before clicking.
    if "model" not in st.session_state:
        model, acc, report, cm = train_knn_model(
            X, y,
            test_size=test_size,
            random_state=int(random_state),
            n_neighbors=int(n_neighbors),
            weights=weights,
            metric=metric,
            p=int(p),
        )
        st.session_state["model"] = model

# Prediction UI
st.divider()
st.subheader("Predict")

model = st.session_state.get("model", None)
if model is None:
    st.warning("Click 'Train / Update' to train the model first.")
    st.stop()

# Heuristic ranges based on dataset percentiles (good UX)
quantiles = df[feature_cols].quantile([0.01, 0.99]).to_dict()

inputs = {}
cols = st.columns(4)

for i, col in enumerate(feature_cols):
    lo = float(quantiles[col][0.01])
    hi = float(quantiles[col][0.99])
    default = float(df[col].median())

    with cols[i % 4]:
        # Streamlit number_input uses floats fine
        inputs[col] = st.number_input(col, min_value=lo, max_value=hi, value=default, step=(hi - lo) / 100)

x_query = pd.DataFrame([inputs], columns=feature_cols)

colA, colB = st.columns([1, 1])

with colA:
    run_pred = st.button("Predict Outcome", type="primary")

if run_pred:
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_query)[0]

    pred = int(model.predict(x_query)[0])

    st.subheader("Result")
    if pred == 1:
        st.error("Prediction: Diabetes likely (Outcome = 1)")
    else:
        st.success("Prediction: Diabetes unlikely (Outcome = 0)")

    if proba is not None:
        st.write({"P(Outcome=0)": float(proba[0]), "P(Outcome=1)": float(proba[1])})

    st.caption("Note: This is a KNN model trained on your provided dataset split.")

