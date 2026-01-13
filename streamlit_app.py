import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="In-Vehicle Coupon Recommendation",
    layout="wide"
)

st.title("🚗 In-Vehicle Coupon Recommendation System")
st.write(
    "This application predicts whether a user will accept or reject a coupon "
    "using different machine learning classification models."
)

# -------------------------------
# Load preprocessor
# -------------------------------
@st.cache_resource
def load_preprocessor():
    return joblib.load("saved_model/preprocessor.pkl")

preprocessor = load_preprocessor()

# -------------------------------
# Model loader
# -------------------------------
@st.cache_resource
def load_model(model_name):
    model_paths = {
        "Logistic Regression": "saved_model/logistic.pkl",
        "Decision Tree": "saved_model/decision_tree.pkl",
        "K-Nearest Neighbors": "saved_model/knn.pkl",
        "Naive Bayes": "saved_model/naive_bayes.pkl",
        "Random Forest": "saved_model/random_forest.pkl",
        "XGBoost": "saved_model/xgboost.pkl"
    }
    return joblib.load(model_paths[model_name])

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("⚙️ Configuration")

model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

# -------------------------------
# Main logic
# -------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "Y" not in df.columns:
        st.error("Uploaded CSV must contain target column 'Y'")
    else:
        X = df.drop("Y", axis=1)
        y_true = df["Y"]

        # Preprocess input
        X_processed = preprocessor.transform(X)

        # Load selected model
        model = load_model(model_name)

        # Handle Gaussian Naive Bayes dense input
        if model_name == "Naive Bayes":
            X_processed = X_processed.toarray()

        # Predictions
        y_pred = model.predict(X_processed)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_processed)[:, 1]
        else:
            y_prob = None

        # -------------------------------
        # Metrics
        # -------------------------------
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        if y_prob is not None:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = np.nan

        st.subheader(f"📊 Model Performance: {model_name}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.4f}")
        col1.metric("AUC", f"{auc:.4f}")

        col2.metric("Precision", f"{precision:.4f}")
        col2.metric("Recall", f"{recall:.4f}")

        col3.metric("F1 Score", f"{f1:.4f}")
        col3.metric("MCC", f"{mcc:.4f}")

        # -------------------------------
        # Confusion Matrix
        # -------------------------------
        st.subheader("🔍 Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots()
        im = ax.imshow(cm)

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        st.pyplot(fig)

else:
    st.info("Upload a CSV file from the sidebar to begin prediction.")
