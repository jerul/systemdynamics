import io
import os
import sys
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import streamlit as st

from funcs.cld import Extract
from funcs.d2d import D2D
from funcs.plots import plot_simulated_intervention_ranking

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Diagrams-to-Dynamics (D2D)",
    layout="wide",
)
st.title("Diagrams-to-Dynamics (D2D)")
st.caption("Exploring Causal Loop Diagram leverage points under uncertainty")

# ---------------------------------------------------------------------------
# Sidebar: simulation settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Simulation settings")
    N = st.number_input("Number of simulations", value=100, min_value=10, step=10)
    t_end = st.number_input("Final time point", value=20, min_value=1, step=1)
    time_unit = st.text_input("Time unit", value="Months")
    parameter_value_aux = st.number_input(
        "Max parameter value – auxiliaries", value=0.3, min_value=0.0, step=0.05, format="%.3f"
    )
    parameter_value_stocks = st.number_input(
        "Max parameter value – stocks", value=0.1, min_value=0.0, step=0.05, format="%.3f"
    )
    seed = st.number_input("Random seed", value=1912884, step=1)
    double_factor = st.checkbox(
        "Simulate double-factor interventions (requires interaction terms)", value=False
    )
    cut_off_SA = st.number_input(
        "Sensitivity analysis |ρ| cutoff", value=0.1, min_value=0.0, max_value=1.0,
        step=0.01, format="%.2f"
    )

# ---------------------------------------------------------------------------
# File upload + run
# ---------------------------------------------------------------------------
uploaded = st.file_uploader("Upload your Kumu Excel file (.xlsx)", type="xlsx")

run_clicked = st.button("Run simulation", type="primary", disabled=uploaded is None)

if run_clicked and uploaded is not None:
    with TemporaryDirectory() as tmp:
        file_path = os.path.join(tmp, uploaded.name)
        with open(file_path, "wb") as f:
            f.write(uploaded.getvalue())

        with st.spinner("Extracting CLD…"):
            extract = Extract(file_path)
            s = extract.extract_settings(int(double_factor))

        s.N = int(N)
        s.t_end = int(t_end)
        s.time_unit = time_unit
        s.parameter_value_aux = float(parameter_value_aux)
        s.parameter_value_stocks = float(parameter_value_stocks)
        s.seed = int(seed)
        s.interval_type = "percentile"

        sdm = D2D(s)

        with st.spinner(f"Running {int(N)} simulations…"):
            df_sol, param_samples = sdm.run_simulations()

        intervention_effects_per_voi = sdm.get_intervention_effects()

        # Cache everything; from here on only widgets trigger reruns
        st.session_state.s = s
        st.session_state.sdm = sdm
        st.session_state.df_sol = df_sol
        st.session_state.param_samples = param_samples
        st.session_state.intervention_effects = intervention_effects_per_voi

# ---------------------------------------------------------------------------
# Results (shown whenever session_state has data, survives widget reruns)
# ---------------------------------------------------------------------------
if "df_sol" not in st.session_state:
    st.info("Upload a Kumu Excel file and click **Run simulation** to begin.")
    st.stop()

s = st.session_state.s
sdm = st.session_state.sdm
df_sol = st.session_state.df_sol
intervention_effects_per_voi = st.session_state.intervention_effects

# ---------------------------------------------------------------------------
# 1. Intervention ranking
# ---------------------------------------------------------------------------
st.subheader("Intervention rankings")
for voi in s.variable_of_interest:
    fig = plot_simulated_intervention_ranking(s, intervention_effects_per_voi[voi], voi)
    st.pyplot(fig)
    plt.close(fig)

st.divider()

# ---------------------------------------------------------------------------
# 2. Interactive compare plot (ipywidgets → Streamlit native controls)
#
# Streamlit reruns the whole script whenever a widget changes.  The simulation
# results stay in st.session_state, so only the figure is re-drawn — cheap.
# ---------------------------------------------------------------------------
st.subheader("Compare interventions over time")

col_left, col_right = st.columns([1, 2])
with col_left:
    default_outcome = (
        s.variable_of_interest[0]
        if s.variable_of_interest and s.variable_of_interest[0] in s.stocks_and_auxiliaries
        else s.stocks_and_auxiliaries[0]
    )
    outcome_var = st.selectbox(
        "Outcome variable",
        options=s.stocks_and_auxiliaries,
        index=s.stocks_and_auxiliaries.index(default_outcome),
    )
    selected_interventions = st.multiselect(
        "Interventions to compare",
        options=s.intervention_variables,
        default=s.intervention_variables[:min(2, len(s.intervention_variables))],
    )
    interval_type = st.radio(
        "Interval type",
        options=["percentile", "spaghetti"],
        horizontal=True,
    )
    confidence_bounds = st.slider(
        "Interval width",
        min_value=0.50, max_value=0.99, value=0.95, step=0.01,
        disabled=(interval_type == "spaghetti"),
    )

with col_right:
    if not selected_interventions:
        st.info("Select at least one intervention.")
    else:
        fig, ax = plt.subplots(figsize=(9, 4))

        for k, var in enumerate(selected_interventions):
            try:
                int_idx = s.intervention_variables.index(var)
            except ValueError:
                continue

            if outcome_var not in df_sol[0][int_idx].columns:
                continue

            t_eval = df_sol[0][int_idx].index.values
            label = " ".join(var.split("_"))
            color = f"C{k}"

            if interval_type == "spaghetti":
                for n in range(s.N):
                    ax.plot(
                        df_sol[n][int_idx].index,
                        df_sol[n][int_idx][outcome_var],
                        alpha=0.15, color=color,
                        label=label if n == 0 else "",
                    )
            else:
                avg, lb, ub = [], [], []
                for t in t_eval:
                    samples = np.array(
                        [df_sol[n][int_idx].loc[t, outcome_var] for n in range(s.N)],
                        dtype=float,
                    )
                    if interval_type == "confidence":
                        mean = np.nanmean(samples)
                        h = (scipy.stats.sem(samples, nan_policy="omit")
                             * scipy.stats.t.ppf((1 + confidence_bounds) / 2.0, s.N - 1))
                        avg.append(mean)
                        lb.append(mean - h)
                        ub.append(mean + h)
                    else:  # percentile
                        lo_pct = (1 - confidence_bounds) / 2 * 100
                        hi_pct = (1 + confidence_bounds) / 2 * 100
                        avg.append(np.nanmedian(samples))
                        lb.append(np.nanpercentile(samples, lo_pct))
                        ub.append(np.nanpercentile(samples, hi_pct))

                pct = int(confidence_bounds * 100)
                ax.plot(t_eval, avg, label=label, color=color)
                ax.fill_between(
                    t_eval, lb, ub, alpha=0.25, color=color,
                    label=f"{pct}% {interval_type} interval" if k == 0 else "",
                )

        ax.set_xlabel(getattr(s, "time_unit", "Time"))
        ax.set_ylabel(" ".join(outcome_var.split("_")))
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

st.divider()

# ---------------------------------------------------------------------------
# 3. Pairwise comparison table
# ---------------------------------------------------------------------------
st.subheader("Pairwise intervention comparison")
for voi in s.variable_of_interest:
    st.write(f"**Variable of interest: {voi}**")
    buf = io.StringIO()
    sys.stdout = buf
    sdm.compare_interventions_table(intervention_effects_per_voi[voi])
    sys.stdout = sys.__stdout__
    st.code(buf.getvalue(), language="")

st.divider()

# ---------------------------------------------------------------------------
# 4. Sensitivity analysis
# ---------------------------------------------------------------------------
st.subheader("Sensitivity analysis")
for voi in s.variable_of_interest:
    st.write(f"**Variable of interest: {voi}**")
    buf = io.StringIO()
    sys.stdout = buf
    SA_results, df_SA = sdm.run_SA(voi, None, float(cut_off_SA))
    sys.stdout = sys.__stdout__
    st.code(buf.getvalue(), language="")
