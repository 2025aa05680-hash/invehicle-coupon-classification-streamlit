import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model
    """
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_logistic_regression(model, X_test, y_test):
    """
    Evaluate Logistic Regression model using required metrics
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    return metrics


def save_model(model, path="saved_models/logistic.pkl"):
    """
    Save trained Logistic Regression model
    """
    joblib.dump(model, path)


def load_model(path="saved_models/logistic.pkl"):
    """
    Load trained Logistic Regression model
    """
    return joblib.load(path)
