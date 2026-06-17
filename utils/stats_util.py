import math
import os
import uuid
from datetime import date
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import zipfile
from scipy.stats import linregress
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit as st
import pandas as pd
from scipy.stats import (
    shapiro,
    kruskal,
    f_oneway,
    ttest_ind,
    mannwhitneyu
)
from statsmodels.stats.multitest import multipletests
from itertools import combinations

from lifelines import (
    CoxPHFitter,
    CoxTimeVaryingFitter,
    KaplanMeierFitter
)
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import multivariate_logrank_test



def univariate_cox(df, time_col, event_col, variables, adjust_var=None):

    results = []

    for var in variables:

        base_cols = [time_col, event_col, var]

        if adjust_var is None:
            adjust_var = []
        elif isinstance(adjust_var, str):
            adjust_var = [adjust_var]

        temp = df[base_cols + adjust_var].dropna()

        # categorical detection
        cat_cols = [
            c for c in temp.columns
            if c not in [time_col, event_col]
            and (
                temp[c].dtype == "object"
                or temp[c].dtype.name == "category"
            )
        ]

        temp = pd.get_dummies(
            temp,
            columns=cat_cols,
            drop_first=True,
            dtype=float
        )

        temp[event_col] = temp[event_col].astype(int)

        # ---------- Original model ----------
        cph = CoxPHFitter()
        cph.fit(
            temp,
            duration_col=time_col,
            event_col=event_col
        )

        s = cph.summary.iloc[0]

        hr = round(s["exp(coef)"], 3)
        ci_low = round(s["exp(coef) lower 95%"], 3)
        ci_high = round(s["exp(coef) upper 95%"], 3)
        p = round(s["p"], 4)

        hr_z = None

        # ---------- z-score model ----------
        if pd.api.types.is_numeric_dtype(df[var]):

            temp_z = temp.copy()

            sd = temp_z[var].std()

            if sd > 0:
                temp_z[var] = (
                    temp_z[var] - temp_z[var].mean()
                ) / sd

                cph_z = CoxPHFitter()
                cph_z.fit(
                    temp_z,
                    duration_col=time_col,
                    event_col=event_col
                )

                hr_z = round(
                    cph_z.summary.iloc[0]["exp(coef)"],
                    3
                )

        results.append({
            "Variable": var,
            "HR": hr,
            "HR per 1 SD": hr_z,
            "CI lower": ci_low,
            "CI upper": ci_high,
            "p": p
        })

    return pd.DataFrame(results)

def multivariate_cox(df, time_col, event_col, variables):

    results = []

    cph = CoxPHFitter()
    variables = [v for v in variables if v != "None"]
    cols = [time_col, event_col] + variables
    temp = df[cols].dropna()

    # Detect categorical columns, excluding time/event
    cat_cols = [
        c for c in temp.columns
        if c not in [time_col, event_col]
        and (
            temp[c].dtype == "object"
            or temp[c].dtype.name == "category"
        )
    ]

    # One-hot encode categorical variables
    temp = pd.get_dummies(
        temp,
        columns=cat_cols,
        drop_first=True,
        dtype=float
    )

    # Ensure event is numeric
    temp[event_col] = temp[event_col].astype(int)

    cph = CoxPHFitter()
    cph.fit(
        temp,
        duration_col=time_col,
        event_col=event_col
    )

    
    summary = cph.summary.reset_index()

    results = pd.DataFrame({
        "Variable": summary["covariate"],
        "HR": summary["exp(coef)"].round(3),
        "CI lower": summary["exp(coef) lower 95%"].round(3),
        "CI upper": summary["exp(coef) upper 95%"].round(3),
        "p": summary["p"].round(4),
    })
       
    return pd.DataFrame(results)

def maxstat_cutpoint(
    df,
    value_col,
    time_col,
    event_col,
    min_prop=0.2
):

    values = np.sort(df[value_col].unique())

    # avoid tiny groups
    lower = int(len(values) * min_prop)
    upper = int(len(values) * (1 - min_prop))
    candidate_thresholds = values[lower:upper]
    best_stat = -np.inf
    best_p = None
    best_threshold = None

    for t in candidate_thresholds:

        high = df[value_col] >= t
        low = ~high
        if high.sum() == 0 or low.sum() == 0:
            continue
        result = logrank_test(
            df.loc[high, time_col],
            df.loc[low, time_col],
            event_observed_A=df.loc[high, event_col],
            event_observed_B=df.loc[low, event_col]
        )

        stat = abs(result.test_statistic)

        if stat > best_stat:
            best_stat = stat
            best_p = result.p_value
            best_threshold = t

    return best_threshold, best_stat, best_p

def plot_km(
    df,
    time_col,
    event_col,
    value_col,
    threshold_type,
    figsize=(6, 6),
    linewidth=1.2,
    ax=None,
):
    sns.set_style("white")

    data = df.copy()
    is_categorical = (
        data[value_col].dtype == "object"
        or data[value_col].dtype.name == "category"
    )

    if is_categorical:
        data = data.dropna(
            subset=[time_col, event_col, value_col]
        )

        data["group"] = data[value_col].astype(str)
        val_labels = sorted(data["group"].unique())

    else:
        if threshold_type == "Median":
            val_threshold = [data[value_col].median()]

        elif threshold_type == "Maxstat":
            threshold, _, _ = maxstat_cutpoint(
                data,
                value_col=value_col,
                time_col=time_col,
                event_col=event_col
            )
            val_threshold = [threshold]

        elif threshold_type == "Tertile":
            val_threshold = [
                data[value_col].quantile(1 / 3),
                data[value_col].quantile(2 / 3)
            ]

        elif threshold_type == "Quartile":
            val_threshold = [
                data[value_col].quantile(0.25),
                data[value_col].quantile(0.50),
                data[value_col].quantile(0.75)
            ]

        else:
            raise ValueError(f"Unknown threshold_type: {threshold_type}")

        val_threshold = sorted(val_threshold)

        val_labels = (
            [f"< {val_threshold[0]:.2f}"] +
            [
                f"{val_threshold[i]:.2f}–{val_threshold[i + 1]:.2f}"
                for i in range(len(val_threshold) - 1)
            ] +
            [f"≥ {val_threshold[-1]:.2f}"]
        )

        data["group"] = pd.cut(
            data[value_col],
            bins=[-np.inf] + val_threshold + [np.inf],
            labels=val_labels,
            include_lowest=True
        )

        data = data.dropna(
            subset=[time_col, event_col, value_col, "group"]
        )


    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    colors = [
        "#222D6C",
        "#5d2866",
        "#2B356F",
        '#317c63',     
    ]

    kmfs = []

    for i, label in enumerate(val_labels):
        mask = data["group"] == label

        if mask.sum() == 0:
            continue

        kmf = KaplanMeierFitter()

        kmf.fit(
            data.loc[mask, time_col],
            data.loc[mask, event_col],
            label=label
        )

        kmf.plot_survival_function(
            ax=ax,
            color=colors[i % len(colors)],
            linewidth=linewidth,
            ci_show=True
        )

        kmfs.append(kmf)

    result = multivariate_logrank_test(
        data[time_col],
        data["group"],
        data[event_col]
    )

    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    ax.set_title(value_col)
    ax.set_ylabel("Survival Probability")
    ax.set_xlabel("Time")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, np.max(data[time_col]) * 1.05)

    ax.legend(
        title=f"Log-rank p = {result.p_value:.3g}",
        frameon=False,
    )

    add_at_risk_counts(
        *kmfs,
        ax=ax,
        rows_to_show=["At risk"],
    )

    sns.despine(ax=ax)

    return fig, ax, result.p_value


def plot_regression(
    df,
    x_col,
    y_col,
    figsize=(6, 6),
    linewidth=1.2,
    ax=None,
):
    sns.set_style("white")

    data = df[[x_col, y_col]].dropna().copy()

    slope, intercept, r, p, se = linregress(
        data[x_col],
        data[y_col]
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.regplot(
        data=data,
        x=x_col,
        y=y_col,
        ax=ax,
        color="#317c63",
        scatter_kws={"alpha":0.7},
        line_kws={"linewidth":linewidth},
        ci=None
    )

    # # Same background as KM
    # bg = "#f6f3ea"

    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    ax.set_title(x_col)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    ax.text(
        0.05,
        0.95,
        f"β={slope:.3f}\n"
        f"p={p:.3g}\n"
        f"R²={r**2:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(
            facecolor="none",
            edgecolor="none"
        )
    )

    sns.despine(ax=ax)

    return fig, ax, p





def desc_by_grp(var_list, grp_name, dec_pl, df):
    rows = []

    grp_var = df[grp_name]
    levels = sorted(grp_var.dropna().unique())

    for var_name in var_list:
        test_var = df[var_name]

        temp = df[[var_name, grp_name]].dropna()
        x = temp[var_name]
        g = temp[grp_name]

        shapiro_p = shapiro(x).pvalue if len(x) >= 3 else 0

        row = {
            "Variable": var_name,
        }

        if shapiro_p < 0.05:
            row["Variable"] = f"{var_name}*"

            for lev in levels:
                vals = temp.loc[temp[grp_name] == lev, var_name]

                med = vals.median()
                q1 = vals.quantile(0.25)
                q3 = vals.quantile(0.75)

                row[str(lev)] = (
                    f"{med:.{dec_pl}f} "
                    f"({q1:.{dec_pl}f}-{q3:.{dec_pl}f})"
                )

            groups = [
                temp.loc[temp[grp_name] == lev, var_name]
                for lev in levels
            ]

            row["Overall p"] = round(
                kruskal(*groups).pvalue,
                4
            )

            # Pairwise Mann–Whitney with BH correction
            p_vals = []
            pair_names = []

            for a, b in combinations(levels, 2):
                vals_a = temp.loc[temp[grp_name] == a, var_name]
                vals_b = temp.loc[temp[grp_name] == b, var_name]

                p = mannwhitneyu(
                    vals_a,
                    vals_b,
                    alternative="two-sided"
                ).pvalue

                p_vals.append(p)
                pair_names.append(f"{a} vs {b}")

            p_adj = multipletests(
                p_vals,
                method="fdr_bh"
            )[1]

            for name, p in zip(pair_names, p_adj):
                row[name] = round(p, 4)

        else:
            for lev in levels:
                vals = temp.loc[temp[grp_name] == lev, var_name]

                row[str(lev)] = (
                    f"{vals.mean():.{dec_pl}f} "
                    f"+/- {vals.std():.{dec_pl}f}"
                )

            groups = [
                temp.loc[temp[grp_name] == lev, var_name]
                for lev in levels
            ]

            row["Overall p"] = round(
                f_oneway(*groups).pvalue,
                4
            )

            # Pairwise t-tests with BH correction
            p_vals = []
            pair_names = []

            for a, b in combinations(levels, 2):
                vals_a = temp.loc[temp[grp_name] == a, var_name]
                vals_b = temp.loc[temp[grp_name] == b, var_name]

                p = ttest_ind(
                    vals_a,
                    vals_b,
                    equal_var=False,
                    nan_policy="omit"
                ).pvalue

                p_vals.append(p)
                pair_names.append(f"{a} vs {b}")

            p_adj = multipletests(
                p_vals,
                method="fdr_bh"
            )[1]

            for name, p in zip(pair_names, p_adj):
                row[name] = round(p, 4)

        rows.append(row)

    return pd.DataFrame(rows)