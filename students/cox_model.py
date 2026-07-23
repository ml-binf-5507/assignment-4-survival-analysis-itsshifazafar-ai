"""
cox_model.py

Students implement Cox Proportional Hazards regression.
"""

from typing import Any, Dict, List

import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test


def fit_cox_model(
    data: pd.DataFrame,
    time_col: str,
    event_col: str,
    covariates: List[str],
) -> Any:
    """Fit Cox Proportional Hazards model.

    Parameters
    ----------
    data : pd.DataFrame
        Survival dataset with time, event, and covariate columns.
    time_col : str
        Name of survival time column.
    event_col : str
        Name of event indicator column (1=event, 0=censored).
    covariates : list of str
        List of covariate column names to include in model.
        Must contain at least 3 covariates.

    Returns
    -------
    CoxPHFitter
        Fitted Cox model object.
    """
    if len(covariates) < 3:
        raise ValueError("Cox model must include at least 3 covariates")

    required_columns = [time_col, event_col] + covariates
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}"
        )

    model_data = data[required_columns].copy()

    # Remove rows with missing survival or predictor values
    model_data = model_data.dropna()

    if model_data.empty:
        raise ValueError(
            "No complete observations remain after removing missing values."
        )

    # Convert event indicator to integer
    model_data[event_col] = model_data[event_col].astype(int)

    # One-hot encode categorical covariates
    model_data = pd.get_dummies(
        model_data,
        columns=[
            column
            for column in covariates
            if (
                pd.api.types.is_object_dtype(model_data[column])
                or isinstance(model_data[column].dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(model_data[column])
            )
        ],
        drop_first=True,
        dtype=float,
    )

    # Ensure all columns used by lifelines are numeric
    for column in model_data.columns:
        model_data[column] = pd.to_numeric(
            model_data[column],
            errors="raise",
        )

    cox = CoxPHFitter()

    cox.fit(
        model_data,
        duration_col=time_col,
        event_col=event_col,
    )

    # Save the exact encoded data used to fit the model.
    # This is needed for the proportional hazards test.
    cox._training_data = model_data.copy()

    return cox


def extract_cox_summary(cox_model: Any) -> pd.DataFrame:
    """Extract summary statistics from fitted Cox model.

    Parameters
    ----------
    cox_model : CoxPHFitter
        Fitted Cox model.

    Returns
    -------
    pd.DataFrame
        Summary table with columns:
        - covariate
        - coef
        - exp(coef)
        - se(coef)
        - z
        - p
        - lower 95%
        - upper 95%
    """
    if not hasattr(cox_model, "summary"):
        raise TypeError(
            "cox_model must be a fitted CoxPHFitter object."
        )

    summary = cox_model.summary.copy().reset_index()

    # Lifelines normally names the reset index column "covariate".
    # This fallback handles version differences.
    if "covariate" not in summary.columns:
        summary = summary.rename(
            columns={summary.columns[0]: "covariate"}
        )

    required_columns = [
        "covariate",
        "coef",
        "exp(coef)",
        "se(coef)",
        "z",
        "p",
        "exp(coef) lower 95%",
        "exp(coef) upper 95%",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in summary.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Expected Cox summary columns are missing: {missing_columns}"
        )

    summary = summary[required_columns].rename(
        columns={
            "exp(coef) lower 95%": "lower 95%",
            "exp(coef) upper 95%": "upper 95%",
        }
    )

    return summary


def test_proportional_hazards(
    cox_model: Any,
    data: pd.DataFrame,
    time_col: str,
    event_col: str,
) -> Dict[str, Dict[str, float]]:
    """Test proportional hazards assumption using Schoenfeld residuals.

    Parameters
    ----------
    cox_model : CoxPHFitter
        Fitted Cox model.
    data : pd.DataFrame
        Original survival dataset. This argument is retained to match
        the assignment function signature.
    time_col : str
        Name of time column.
    event_col : str
        Name of event column.

    Returns
    -------
    dict
        Nested dictionary containing test statistics and p-values for
        each encoded covariate.
    """
    if not hasattr(cox_model, "_training_data"):
        raise ValueError(
            "Training data were not stored on the Cox model. "
            "Refit the model using fit_cox_model()."
        )

    model_data = cox_model._training_data.copy()

    results = proportional_hazard_test(
        cox_model,
        model_data,
        time_transform="rank",
    )

    output: Dict[str, Dict[str, float]] = {}

    for covariate, row in results.summary.iterrows():
        output[str(covariate)] = {
            "test_statistic": float(row["test_statistic"]),
            "p_value": float(row["p"]),
        }

    return output