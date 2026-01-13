import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="In-Vehicle Coupon Recommendation",
    layout="wide"
)

st.title("🚗 In-Vehicle Coupon Recommendation System")
st.markdown(
    """
    This interactive application predicts whether a user will **accept or reject a coupon**
    based on contextual and behavioral features using multiple machine learning models.
    """
)

# --------------------------------------------------
# Load preprocessor
# --------------------------------------------------
@st.cache_resource
def load_preprocessor():
    return joblib.load("saved_model/preprocessor.pkl")

# --------------------------------------------------
# Load model
# --------------------------------------------------
@st.cache_resource
def load_model(model_name):
    model_paths = {
        "Logistic Regression": "saved_model/logistic.pkl",
        "Decision Tree": "saved_model/decision_tree.pkl",
        "K-Nearest Neighbors": "saved_model/knn.pkl",
        "Naive Bayes": "saved_model/naive_bayes.pkl",
        "Random Forest": "saved_model/random_forest.pkl",
        "XGBoost": "saved_model/XGBoost.pkl"
    }
    return joblib.load(model_paths[model_name])

preprocessor = load_preprocessor()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("⚙️ Configuration")

model_name = st.sidebar.selectbox(
    "Select Classification Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

threshold = st.sidebar.slider(
    "Prediction Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

st.sidebar.markdown(
    """
    **Threshold Tip:**  
    Lower values increase Recall,  
    Higher values increase Precision.
    """
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

# --------------------------------------------------
# Stop if no file uploaded
# --------------------------------------------------
if uploaded_file is None:
    st.info("👈 Upload a CSV file from the sidebar to begin.")
    st.stop()

# --------------------------------------------------
# Safe CSV loading
# --------------------------------------------------
try:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error reading CSV file: {e}")
    st.stop()

if df.empty:
    st.error("Uploaded CSV file is empty.")
    st.stop()

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------
st.subheader("📄 Uploaded Dataset Preview")
with st.expander("Click to view dataset"):
    st.write("Shape:", df.shape)
    st.dataframe(df.head(10))

# --------------------------------------------------
# Target validation
# --------------------------------------------------
if "Y" not in df.columns:
    st.error("Uploaded CSV must contain target column **'Y'**")
    st.stop()

X = df.drop("Y", axis=1)
y_true = df["Y"]


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
X_processed = preprocessor.transform(X)

# Load model
model = load_model(model_name)

# Gaussian Naive Bayes needs dense input
if model_name == "Naive Bayes":
    X_processed = X_processed.toarray()

# --------------------------------------------------
# Predictions
# --------------------------------------------------
y_prob = model.predict_proba(X_processed)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

# --------------------------------------------------
# Metrics
# --------------------------------------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)

st.subheader(f"📊 Model Performance: {model_name}")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy:.4f}")
col1.metric("AUC", f"{auc:.4f}")

col2.metric("Precision", f"{precision:.4f}")
col2.metric("Recall", f"{recall:.4f}")

col3.metric("F1 Score", f"{f1:.4f}")
col3.metric("MCC", f"{mcc:.4f}")

# --------------------------------------------------
# Classification Report
# --------------------------------------------------
st.subheader("📑 Classification Report")

report = classification_report(
    y_true,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.style.format("{:.4f}")
)

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------
st.subheader("🔍 Confusion Matrix")

cm = confusion_matrix(y_true, y_pred)
fig_cm, ax_cm = plt.subplots()
ax_cm.imshow(cm)
ax_cm.set_xlabel("Predicted Label")
ax_cm.set_ylabel("True Label")
ax_cm.set_title("Confusion Matrix")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax_cm.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig_cm)

# --------------------------------------------------
# Download Predictions
# --------------------------------------------------
st.subheader("⬇️ Download Predictions")

output_df = df.copy()
output_df["Predicted_Y"] = y_pred
output_df["Prediction_Probability"] = y_prob

st.download_button(
    label="Download Predictions as CSV",
    data=output_df.to_csv(index=False),
    file_name="coupon_predictions.csv",
    mime="text/csv"
)

# --------------------------------------------------
# Metric Explanation
# --------------------------------------------------
with st.expander("ℹ️ Metric Explanation"):
    st.markdown(
        """
        - **Accuracy**: Overall correctness  
        - **AUC**: Ability to distinguish accept vs reject  
        - **Precision**: Correct accept predictions  
        - **Recall**: Captured actual accepts  
        - **F1 Score**: Balance of precision & recall  
        - **MCC**: Robust metric for imbalanced data
        """
    )
