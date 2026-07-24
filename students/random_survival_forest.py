"""
random_survival_forest.py

Students implement Random Survival Forest using scikit-survival.
"""

from fileinput import filename
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
import matplotlib.pyplot as plt


def fit_random_survival_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
) -> Any:
    """Train a Random Survival Forest model.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix (covariates) for training.
    y_train : np.ndarray
        Structured array with dtype=[('event', bool), ('time', float)].
        This is the scikit-survival format for survival outcomes.
    n_estimators : int
        Number of trees in the forest.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    RandomSurvivalForest
        Fitted RSF model.

    Example
    -------
    >>> from sksurv.util import Surv
    >>> y_train = Surv.from_dataframe('event', 'time', data_train)
    >>> rsf = fit_random_survival_forest(X_train, y_train)
    """
    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )

    rsf.fit(X_train, y_train)

    rsf._X_train = X_train.copy()
    rsf._y_train = y_train.copy()

    return rsf


def compute_concordance_index(
    rsf_model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> float:
    """Compute concordance index (C-index) for RSF predictions.

    Parameters
    ----------
    rsf_model : RandomSurvivalForest
        Fitted RSF model.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : np.ndarray
        Test survival outcomes in structured array format.

    Returns
    -------
    float
        Concordance index (Harrell's C-index).
        Range: [0, 1], where 0.5 = random, 1.0 = perfect.

    Example
    -------
    >>> c_index = compute_concordance_index(rsf, X_test, y_test)
    >>> print(f"C-index: {c_index:.3f}")
    """
    risk_scores = rsf_model.predict(X_test)
    c_index = concordance_index_censored(
        y_test["event"],
        y_test["time"],
    risk_scores
    )[0]

    return float(c_index)

from sklearn.inspection import permutation_importance

def get_feature_importance(
    rsf_model: Any,
    feature_names: List[str],
) -> pd.DataFrame:
    """Extract feature importance scores from RSF model.

    Parameters
    ----------
    rsf_model : RandomSurvivalForest
        Fitted RSF model.
    feature_names : list of str
        Names of features (column names from X_train).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature', 'importance'],
        sorted by importance in descending order.

    Example
    -------
    >>> importance = get_feature_importance(rsf, X_train.columns.tolist())
    >>> print(importance.head())
    """
    if not hasattr(rsf_model, "_X_train") or not hasattr(rsf_model, "_y_train"):
        raise ValueError(
            "Training data are unavailable. Store X_train and y_train on the model "
            "during fitting before calculating permutation importance."
        )

    result = permutation_importance(
        rsf_model,
        rsf_model._X_train,
        rsf_model._y_train,
        n_repeats=5,
        random_state=42,
        n_jobs=1
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": result.importances_mean
    })

    return importance_df.sort_values(
        by="importance",
        ascending=False
    ).reset_index(drop=True)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    filename: str = "rsf_importance.png",
    top_n: int = 10,
) -> None:
    """Create horizontal bar chart of feature importance.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Output from get_feature_importance().
    filename : str
        Output filename for plot.
    top_n : int
        Number of top features to display.

    Notes
    -----
    Plot should include:
    - Horizontal bars showing importance scores
    - Feature names on y-axis
    - Sorted by importance (most important at top)
    - Clear title and axis labels
    """
    top_features = importance_df.head(top_n).sort_values(
        by="importance",
        ascending=True
    )

    plt.figure(figsize=(9, 6))
    plt.barh(top_features["feature"], top_features["importance"])

    plt.title(f"Top {top_n} Random Survival Forest Predictors")
    plt.xlabel("Permutation Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
