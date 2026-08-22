"""Baseline C: Traditional ML (Random Forest, XGBoost, Logistic Regression)."""

from __future__ import annotations

import numpy as np
from typing import Optional


def train_ml_classifier(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Train and evaluate multiple ML classifiers."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    try:
        from xgboost import XGBClassifier
        has_xgb = True
    except ImportError:
        has_xgb = False

    results = {}

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    results["random_forest"] = {
        "accuracy": rf_acc,
        "proba": rf_proba,
        "feature_importance": rf.feature_importances_.tolist(),
    }

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_acc = lr.score(X_test, y_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    results["logistic_regression"] = {
        "accuracy": lr_acc,
        "proba": lr_proba,
        "coefficients": lr.coef_[0].tolist(),
    }

    # XGBoost
    if has_xgb:
        xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss")
        xgb.fit(X_train, y_train)
        xgb_acc = xgb.score(X_test, y_test)
        xgb_proba = xgb.predict_proba(X_test)[:, 1]
        results["xgboost"] = {
            "accuracy": xgb_acc,
            "proba": xgb_proba,
        }

    return results


def train_ml_regressor(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Train and evaluate ML regressors for remaining useful life."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    results = {}

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_mae = np.mean(np.abs(rf_pred - y_test))
    rf_rmse = np.sqrt(np.mean((rf_pred - y_test) ** 2))
    results["random_forest"] = {
        "mae": rf_mae,
        "rmse": rf_rmse,
        "predictions": rf_pred.tolist(),
    }

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_mae = np.mean(np.abs(lr_pred - y_test))
    lr_rmse = np.sqrt(np.mean((lr_pred - y_test) ** 2))
    results["linear_regression"] = {
        "mae": lr_mae,
        "rmse": lr_rmse,
        "predictions": lr_pred.tolist(),
    }

    return results
