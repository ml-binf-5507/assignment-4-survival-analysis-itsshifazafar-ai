"""
kaplan_meier.py

Students implement Kaplan-Meier survival analysis and log-rank test.
"""

from fileinput import filename
from typing import Dict, Tuple, Any
from unittest import result
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
from lifelines.plotting import add_at_risk_counts


def fit_kaplan_meier(
    data: pd.DataFrame,
    time_col: str,
    event_col: str,
    group_col: str,
) -> Dict[str, Any]:
    """Fit Kaplan-Meier survival curves for different groups.

    Parameters
    ----------
    data : pd.DataFrame
        Survival dataset containing time, event, and grouping columns.
    time_col : str
        Name of column containing survival/censoring times.
    event_col : str
        Name of column containing event indicator (1=event, 0=censored).
    group_col : str
        Name of column containing group labels for comparison.

    Returns
    -------
    dict
        Dictionary containing fitted KM models for each group.
        Keys: group labels, Values: fitted KaplanMeierFitter objects.

    Example
    -------
    >>> km_models = fit_kaplan_meier(data, 'time', 'event', 'treatment')
    >>> # km_models = {'chemo': <KMF>, 'radiation': <KMF>}
    """
    km_models = {}
    for group in data[group_col].dropna().unique():
        subset = data[data[group_col] == group]

        kmf = KaplanMeierFitter()

        kmf.fit(
        durations=subset[time_col],
        event_observed=subset[event_col],
        label=str(group)
        )

        km_models[str(group)] = kmf

    return km_models


def compute_logrank_test(
    data: pd.DataFrame,
    time_col: str,
    event_col: str,
    group_col: str,
) -> Dict[str, float]:
    """Compute log-rank test comparing survival curves between groups.

    Parameters
    ----------
    data : pd.DataFrame
        Survival dataset.
    time_col : str
        Name of time column.
    event_col : str
        Name of event indicator column.
    group_col : str
        Name of grouping column.

    Returns
    -------
    dict
        Dictionary with 'test_statistic' and 'p_value' keys.

    Example
    -------
    >>> result = compute_logrank_test(data, 'time', 'event', 'stage')
    >>> # result = {'test_statistic': 12.34, 'p_value': 0.0004}
    """
    groups = data[group_col].dropna().unique()

    if len(groups) != 2:
        raise ValueError("Log-rank test requires exactly two groups.")

    group1 = data[data[group_col] == groups[0]]
    group2 = data[data[group_col] == groups[1]]
    
    result = logrank_test(
        group1[time_col],
        group2[time_col],
        event_observed_A=group1[event_col],
        event_observed_B=group2[event_col]
    )
    return {
        "test_statistic": float(result.test_statistic),
        "p_value": float(result.p_value)
    }


def plot_km_curves(
    km_models: Dict[str, Any],
    filename: str = "km_survival_plot.png",
    title: str = "Kaplan-Meier Survival Curves",
) -> None:
    """Create publication-quality Kaplan-Meier survival plot.

    Parameters
    ----------
    km_models : dict
        Dictionary of fitted KM models (output from fit_kaplan_meier).
    filename : str
        Output filename for plot.
    title : str
        Plot title.

    Notes
    -----
    Plot should include:
    - Survival curves for each group
    - Confidence intervals (shaded regions)
    - Risk table showing number at risk over time
    - Legend identifying groups
    - Proper axis labels
    """
    plt.figure(figsize=(8, 6))
    km_list = []

    for label, kmf in km_models.items():
        kmf.plot_survival_function(ci_show=True)
        km_list.append(kmf)

    if len(km_list) > 0:
        add_at_risk_counts(*km_list)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.legend(title="Group")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
