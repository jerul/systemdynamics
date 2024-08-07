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
from copy import deepcopy
from systemdynamics.cld import Extract
from systemdynamics.sdm import SDM
from systemdynamics.plots import plot_simulated_interventions, plot_simulated_intervention_ranking, plot_simulated_data
from systemdynamics.plots import plot_simulated_interventions_compare, plot_feedback_loops_ranking, plot_feedback_loops_over_time, plot_feedback_loops_over_time_bounds
#sns.set_theme()

st.title('Diagrams to Dynamics: A System Dynamics Analysis of a Causal Loop Diagram')
uploaded_kumu_excel = st.file_uploader("Upload an Excel file", type="xlsx")
uploaded_json = st.file_uploader("Upload a settings file", type="json")

if uploaded_kumu_excel is not None and uploaded_json is not None:
    extract = Extract(uploaded_kumu_excel)  # Load the relevant Kumu file extraction module
    uploaded_settings = json.load(uploaded_json)
    json_file_name = uploaded_json.name
    s = SimpleNamespace(**dict(uploaded_settings))
    s.setting_name = json_file_name
    
    # Load the adjacency matrix from the KUmu file
    variable_names, var_to_type_init, adjacency_matrix, interactions_matrix = extract.adjacency_matrix_from_kumu()  

    if s.interaction_terms:
        if np.abs(interactions_matrix).sum() > 0:
            #s.interaction_terms = True
            print("Solving an SDM with interaction terms.")
            if s.solve_analytically and s.interaction_terms:
                print("Cannot solve analytically with interaction terms so will proceed with numerical solution.")
        else:
            print("No interaction terms specified so will solve linear SDM.")
            s.interaction_terms = False
   
    # Load variable names and automatically fill any spaces with underscores
    s.stocks = [var.replace(" ", "_") for var in variable_names if var_to_type_init[var] == 'stock']
    s.auxiliaries = [var.replace(" ", "_") for var in variable_names if var_to_type_init[var] == 'auxiliary']
    s.constants = [var.replace(" ", "_") for var in variable_names if var_to_type_init[var] == 'constant']
    s.variables = [var.replace(" ", "_") for var in variable_names]  # s.auxiliaries + s.stocks + s.constants
    s.stocks_and_constants = [var.replace(" ", "_") for var in variable_names if var_to_type_init[var] in ['stock', 'constant']]
    s.stocks_and_auxiliaries = [var.replace(" ", "_") for var in variable_names if var_to_type_init[var] in ['stock', 'auxiliary']]
    s.var_to_type = {var.replace(" ", "_") : var_to_type_init[var] for var in variable_names}
    s.variable_of_interest = "_".join(s.variable_of_interest.split(" "))  # Ensure the variable of interest is formulated with underscores
    s.simulate_interventions = True  # Always simulate interventions
    
    # Create dataframe with adjacency matrix
    df_adj = pd.DataFrame(adjacency_matrix,
                          columns=s.variables, index=s.variables) 

    np.random.seed(s.seed)  # Set seed for reproducibility

    ### Check if any constants have incoming links
    for const in s.constants:
        num_links = np.sum(np.abs(df_adj.loc[const, :]))
        if num_links != 0:
            #if s.remove_incoming_links_constants:
            print(f'Removed {num_links} incoming links for constant {const}')
            df_adj.loc[const, :] = 0
            #else:
            #    raise(Exception(f'Number of incoming links for constant {const} is {num_links}, should be zero.'))

    # Set the SDM simulation timesteps to store 
    s.t_eval = np.array(np.array([0.0] + list(np.linspace(0, s.t_end,
                                                        int(s.t_end/s.dt) + 1)[1:])))

    # If solving the system numerically, set the solver
    s.solver = 'LSODA'  # 'LSODA' automatically switches between stiff and non-stiff methods since stiffness is not always known.

    # Select variables to simulated interventions for; all variables except the var of interest by default
    s.intervention_variables = [var for var in s.variables if var != s.variable_of_interest]  

    # If double factor interventions selected, add double factor interventions 
    if s.double_factor_interventions and s.interaction_terms == False:
        print("Without interaction terms, double factor interventions are not meaningful. Setting double_factor_interventions to False.")
        s.double_factor_interventions = 0

    if s.double_factor_interventions:
        double_intervention_variables = []
        for i, var in enumerate(s.intervention_variables):
            for j in range(i + 1, len(s.intervention_variables)):
                var_2 = s.intervention_variables[j]
                double_intervention_variables += [var + '+' + var_2]
        
        s.intervention_variables += double_intervention_variables

    # Load the module for formulating and simulating the SDM
    sdm = SDM(df_adj, interactions_matrix, s)  

    ### Run simulations
    df_sol_per_sample = []  # List for storing the solution dataframes
    df_sol_per_sample_no_int = []   # List for storuing the solution dataframes without interventions
    param_samples = {var : {} for var in s.intervention_variables}  # Dictionary for storing the parameters across samples

    for num in tqdm(range(s.N)):  # Iterate over the number of samples
        df_sol = []

        params_i = sdm.sample_model_parameters() #s.intervention_auxiliaries)  # Sample model parameters

        for i, var in enumerate(s.intervention_variables):
            # Set the initial condition for the stocks to zero
            x0 = np.zeros(len(s.stocks_and_constants), order='F')  # By default no intervention on a stock or constant (initialized in equilibrium)
            intervention_auxiliaries = {}  # By default no intervention on an auxiliary
            params = deepcopy(params_i)  # Copy the parameters to avoid overwriting the original parameters

            if '+' in var:  # Double factor intervention
                var_1, var_2 = var.split('+')
                if var_1 in s.stocks_and_constants:  # Intervention on a stock or constant (first intervention variable)
                    x0[s.stocks_and_constants.index(var_1)] += 1/2  # Increase the (baseline) value of the stock/constant by 1/2
                else:  # Intervention on an auxiliary
                    params[var_1]["Intercept"] = 1/2
                    #intervention_auxiliaries[var_1] = 1/2 # Select the auxiliary to get an intercept of 1/2 in sample_model_parameters function
                if var_2 in s.stocks_and_constants:
                    x0[s.stocks_and_constants.index(var_2)] += 1/2  
                else:
                    params[var_2]["Intercept"] = 1/2
                    #intervention_auxiliaries[var_2] = 1/2
            else:  # Single factor intervention
                if var in s.stocks_and_constants:  # Intervention on a stock or constant (only variable)
                    x0[s.stocks_and_constants.index(var)] += 1  # Increase the (baseline) value of the stock/constant by 1
                else:  # Intervention on an auxiliary (only variable)
                    params[var]["Intercept"] = 1
                    #intervention_auxiliaries[var] = 1  # Select the auxiliary to get an intercept of 1/2 in sample_model_parameters function
        
            # print("Intervention on", var)
            # print(new_params) #params)
            # print(x0)
            # print("----")

            new_params = sdm.make_equations_auxiliary_independent(params)  # Remove auxiliaries from the equations
            if np.sum([[1 for par in new_params[st] if par in s.auxiliaries] for st in new_params]) > 0:
                raise(Exception('Some parameters are defined for auxiliaries. This means the process of making equations auxiliary independent failed.',
                                'Likely because of a feedback loop with only auxiliaries. Please ensure that all feedback loops contain at least one stock.'))
            A, K, b = sdm.get_A_and_K_matrices()  # Get A and K matrices and intercept vector from the parameter dictionary without auxiliaries
            df_sol_per_intervention = sdm.run_SDM(x0, A, K, b)
            df_sol += [df_sol_per_intervention]

            # Store the model parameters
            if num == 0: 
                param_samples[var] = {target : {source : [params[target][source]] for source in params[target]} for target in params}
            else:
                for target in params:
                    for source in params[target]:
                        param_samples[var][target][source] += [params[target][source]]

        df_sol_per_sample += [df_sol]

    intervention_effects = sdm.get_intervention_effects(df_sol_per_sample)
    plot_simulated_intervention_ranking(intervention_effects, s)

    ## Create plot
    df_SA = pd.DataFrame(intervention_effects)
   
    # Order by median
    df_SA = df_SA.reindex(columns=list(
                        df_SA.abs().median().sort_values(ascending=False).index))
    df_SA = df_SA.rename(mapper=dict(
                        zip(df_SA.columns,
                           [" ".join(var.split("_")) for var in df_SA.columns])), axis=1)

    palette_dict = {var : "#4682B4" for var in df_SA.columns}  # Blue for positive effects

    medians = df_SA.median()
    lower_than_zero_vars = medians.loc[medians < 0].index
    for var in lower_than_zero_vars:
       palette_dict[var] = "#FF6347"  # Red for negative effects
    palette = list(palette_dict.values())  # Convert to list

    df_SA = df_SA.abs()
    top_plot = None
    if top_plot != None:
        df_SA = df_SA[list(df_SA.columns)[:top_plot]]  #  Take only the top X interventions to plot

    fig = plt.figure(figsize=(5, 8))
    ax = fig.add_subplot(111)
    sns.boxplot(data=df_SA, showfliers=False, whis=True, orient='h', palette=palette)
    plt.vlines(x=0, ymin=-0.5, ymax=len(df_SA.columns) -
               0.6, colors='black', linestyles='dashed')
    plt.title("Effect on " + " ".join(s.variable_of_interest.split("_")))
    plt.xlabel("Standardized effect after " + str(s.t_end) + " " + s.time_unit)
    plt.ylabel("")
    st.pyplot(fig)


    # df = pd.read_excel(uploaded_file)
    # result = system_dynamics_analysis(df)  # Replace with your function
    # st.write("Analysis Results:")
    # st.write(result)
