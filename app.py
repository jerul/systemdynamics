import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from types import SimpleNamespace
import json
from tqdm import tqdm 
import sys
import os
import datetime
import scipy
import subprocess
from copy import deepcopy
from systemdynamics.cld import Extract
from systemdynamics.sdm import SDM
from systemdynamics.plots import plot_simulated_interventions, plot_simulated_intervention_ranking, plot_simulated_data
from systemdynamics.plots import plot_simulated_interventions_compare, plot_feedback_loops_ranking, plot_feedback_loops_over_time, plot_feedback_loops_over_time_bounds

st.title('Diagrams to Dynamics: A System Dynamics Analysis of a Causal Loop Diagram')

# Upload files
uploaded_kumu_excel = st.file_uploader("Upload an Excel file", type="xlsx")
uploaded_json = st.file_uploader("Upload a settings file", type="json")

if uploaded_kumu_excel is not None and uploaded_json is not None:
    # Check if we've already processed the input files
    if 'processed' not in st.session_state:
        # Process files
        uploaded_settings = json.load(uploaded_json)
        json_file_name = uploaded_json.name
        s = SimpleNamespace(**dict(uploaded_settings))
        s.setting_name = json_file_name

        extract = Extract(s, uploaded_kumu_excel)  # Load the relevant Kumu file extraction module
        s = extract.extract_settings()  # Extract the settings using the json file and the Kumu table
        sdm = SDM(s.df_adj, s.interactions_matrix, s)  # Load the module for formulating and simulating the SDM

        st.session_state.s = s
        st.session_state.sdm = sdm

        st.write("The following settings are used:")
        for sett in ["N", "time_unit", "t_end", "dt", "variable_of_interest", "max_parameter_value",
                     "double_factor_interventions", "solve_analytically"]:
            st.write(sett, "=", vars(s)[sett])

        # Run simulations and store the results in session state
        df_sol, param_samples = sdm.run_simulations()  # Run the simulated interventions
        intervention_effects = sdm.get_intervention_effects()  # Get the intervention effects

        st.session_state.df_sol = df_sol
        st.session_state.param_samples = param_samples
        st.session_state.intervention_effects = intervention_effects

        top_plot = None  # Number of top interventions to plot
        fig_var_rank = plot_simulated_intervention_ranking(s, intervention_effects, top_plot=top_plot)  # Plot the simulated interventions ranking
        st.write("\n -------- Simulated intervention results -------- ")
        st.pyplot(fig_var_rank)

        fig_var_traj = plot_simulated_interventions(s, df_sol, intervention_effects, interval_type="percentile", confidence_bounds=.95, top_plot=top_plot)  # Plot the simulated interventions
        st.pyplot(fig_var_traj)

        # Mark processing as done
        st.session_state.processed = True
    else:
        # If already processed, load from session state
        s = st.session_state.s
        sdm = st.session_state.sdm
        df_sol = st.session_state.df_sol
        param_samples = st.session_state.param_samples
        intervention_effects = st.session_state.intervention_effects

        st.write("The following settings are used:")
        for sett in ["N", "time_unit", "t_end", "dt", "variable_of_interest", "max_parameter_value",
                     "double_factor_interventions", "solve_analytically"]:
            st.write(sett, "=", vars(s)[sett])
        print("\n -------- Now select the 'what-if' scenario to analyze -------- \n")

    # User input for intervention variable of interest
    intervention_options = list(intervention_effects.keys())
    int_var = st.selectbox("Select the intervention variable of interest:", intervention_options)  # User selects the intervention variable
    
    # Feedback loop analysis
    df_loops, loopscores_per_sample = sdm.run_loops_that_matter(int_var)  # Run the feedback loop analysis

    cut_off_loop_importance = 0.05
    fig_fb_rank = plot_feedback_loops_ranking(s, df_loops, intervention_effects, int_var, cut_off_loop_importance)
    st.write("\n -------- Feedback loop analysis result for the intervention on " + str(int_var) +
             " with cut-off off >" + str(cut_off_loop_importance), "--------")
    st.pyplot(fig_fb_rank)

    fig_fb_traj = plot_feedback_loops_over_time(s, df_loops, intervention_effects, loopscores_per_sample, int_var, cut_off_loop_importance)
    st.pyplot(fig_fb_traj)

    cut_off_SA_importance = 0.05
    input_var = True  # If True, the top-ranked intervention variable will be used
    outcome_var = False  # If True, the variable of interest will be used
    SA_results, df_SA = sdm.run_SA(outcome_var, input_var, int_var, cut_off_SA_importance)
    SA_results = SA_results.loc[SA_results > cut_off_SA_importance]
    st.write("\n -------- Sensitivity analysis result for the intervention on " + str(int_var) + " with respect to " + 
             s.variable_of_interest, " with cut-off of rho>" + str(cut_off_SA_importance), "--------")
    for par in SA_results.index:
        st.write(par + ": " + str(SA_results.loc[par]))
