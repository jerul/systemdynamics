# Copyright (c) 2025 Jeroen F. Uleman. Licensed under CC BY-NC 4.0.
# Non-commercial use only. See LICENSE for details.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import pandas as pd
import ipywidgets as widgets
sns.set_theme()


def plot_simulated_interventions_compare(s, df_sol_per_sample):
    """Interactive plot comparing simulated trajectories across interventions."""
    default_outcome = s.variable_of_interest
    if isinstance(default_outcome, list):
        default_outcome = default_outcome[0]

    confidence_bounds_slider = widgets.FloatSlider(value=0.95, min=0.01, max=0.99, step=0.01, description='Interval bounds:')
    variable_selector = widgets.SelectMultiple(options=s.intervention_variables, value=s.intervention_variables[:2], description='Interventions:')
    outcome_selector = widgets.Dropdown(options=s.stocks_and_auxiliaries, value=default_outcome, description='Outcome:')

    def update_plot(confidence_bounds, compare_int_vars, outcome_var):
        plt.figure(figsize=(10, 5))

        if not compare_int_vars:
            plt.text(0.5, 0.5, "No interventions selected", ha='center')
            plt.show()
            return

        interval_type = getattr(s, 'interval_type', 'percentile')
        time_unit = getattr(s, 'time_unit', 'Time')

        for var in compare_int_vars:
            avg_at_time_t = []
            lb_confs_at_time_t = []
            ub_confs_at_time_t = []

            try:
                int_idx = s.intervention_variables.index(var)
            except ValueError:
                continue

            if outcome_var not in df_sol_per_sample[0][int_idx].columns:
                continue

            t_eval = df_sol_per_sample[0][int_idx].index.values

            if interval_type != "spaghetti":
                for t in t_eval:
                    samples = np.array(
                        [df_sol_per_sample[n][int_idx].loc[t, outcome_var] for n in range(s.N)],
                        dtype=float,
                    )

                    if interval_type == "confidence":
                        mean = np.nanmean(samples)
                        standard_error = scipy.stats.sem(samples, nan_policy='omit')
                        h = standard_error * scipy.stats.t.ppf((1 + confidence_bounds) / 2., s.N - 1)
                        avg_at_time_t.append(mean)
                        lb_confs_at_time_t.append(mean - h)
                        ub_confs_at_time_t.append(mean + h)
                    else:  # percentile
                        lower_percentile = (1 - confidence_bounds) / 2 * 100
                        upper_percentile = (1 + confidence_bounds) / 2 * 100
                        avg_at_time_t.append(np.nanmedian(samples))
                        lb_confs_at_time_t.append(np.nanpercentile(samples, lower_percentile))
                        ub_confs_at_time_t.append(np.nanpercentile(samples, upper_percentile))

                pct = int(confidence_bounds * 100)
                label = " ".join(var.split("_"))
                plt.plot(t_eval, avg_at_time_t, label=label)
                plt.fill_between(t_eval, lb_confs_at_time_t, ub_confs_at_time_t,
                                 alpha=0.3, label=f"{pct}% {interval_type} interval")
            else:
                for n in range(s.N):
                    df_run = df_sol_per_sample[n][int_idx]
                    lbl = " ".join(var.split("_")) if n == 0 else ""
                    plt.plot(df_run.index, df_run[outcome_var], alpha=0.3, label=lbl)

        plt.xlabel(time_unit)
        plt.ylabel(" ".join(outcome_var.split("_")))
        plt.legend()
        plt.show()

    return widgets.interactive(
        update_plot,
        confidence_bounds=confidence_bounds_slider,
        compare_int_vars=variable_selector,
        outcome_var=outcome_selector,
    )


def plot_simulated_intervention_ranking(s, intervention_effects, voi, top_plot=None, order=None):
    """Horizontal boxplot of intervention effects, ranked by median absolute effect."""
    df_SA = pd.DataFrame(intervention_effects)
    df_SA = df_SA.reindex(columns=list(
        df_SA.abs().median().sort_values(ascending=False).index))

    if top_plot is not None:
        df_SA = df_SA[list(df_SA.columns)[:top_plot]]

    if order is not None:
        df_SA = df_SA[order]

    if voi in df_SA.columns:
        df_SA = df_SA.drop(voi, axis=1)

    name_with_intervention = []
    for name in list(df_SA.columns):
        if "+" in name:
            name_1, name_2 = name.split("+")
            effect_1 = s.intervention_strengths[name_1]
            formatted_1 = " ".join(name_1.split("_")) + f" ({'+' if effect_1 > 0 else ''}{effect_1})"
            effect_2 = s.intervention_strengths[name_2]
            formatted_2 = " ".join(name_2.split("_")) + f" ({'+' if effect_2 > 0 else ''}{effect_2})"
            name_with_intervention += [formatted_1 + ' & ' + formatted_2]
        else:
            effect = s.intervention_strengths[name]
            formatted = " ".join(name.split("_")) + f" ({'+' if effect > 0 else ''}{effect})"
            name_with_intervention.append(formatted)

    df_SA = df_SA.rename(mapper=dict(zip(df_SA.columns, name_with_intervention)), axis=1)

    unique_vars = df_SA.columns
    palette = [sns.color_palette("husl", len(unique_vars))[i] for i, _ in enumerate(unique_vars)]

    fig = plt.figure(figsize=(5, 8))
    ax = fig.add_subplot(111)
    sns.boxplot(data=df_SA, showfliers=False, whis=True, orient='h', palette=palette)
    plt.vlines(x=0, ymin=-0.5, ymax=len(df_SA.columns) - 0.6, colors='black', linestyles='dashed')
    plt.title("Effect on " + " ".join(voi.split("_")))
    plt.xlabel("Standardized effect after " + str(s.t_end) + " " + s.time_unit)
    plt.ylabel("")
    return fig
