import sys
import os
import datetime
import scipy
from tqdm import tqdm 
import json
from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from systemdynamics.cld import Extract
from systemdynamics.sdm import SDM
from systemdynamics.plots import plot_simulated_interventions, plot_simulated_intervention_ranking, plot_simulated_data, plot_simulated_interventions_compare
sns.set_theme()

if __name__ == "__main__":
    # Assing setting has input to load the setting (sys.argv[1])
    if len(sys.argv) > 1:  # Settings passed from Jupyter
        system_argument = sys.argv[1]

    if "Results" in system_argument: 
        # Set folder path to current folder
        curr_time = "_".join(os.path.basename(system_argument).split("_")[:3])
        setting_name = "_".join(os.path.basename(system_argument).split("_")[3:])

        with open(os.path.join(system_argument, f"used_settings_{setting_name}.json")) as f:
            settings = json.load(f)

        s = SimpleNamespace(**settings)
        s.setting_name = setting_name

    else:  # Load settings from the Settings folder
        setting_name = system_argument

        # Construct the file path
        settings_path = os.path.join(os.path.dirname(__file__), 'Examples','Settings', f'{setting_name}.json')


        with open(settings_path) as f:
            settings = json.load(f)
        s = SimpleNamespace(**settings)
        s.setting_name = setting_name
        curr_time = (str(datetime.datetime.now())[0:10] + "_" +
                     str(datetime.datetime.now())[11:13] + "_" +
                     str(datetime.datetime.now())[14:16])

        if s.save_results:  # Create a directory to store results
            folder_path = folder_path = os.path.join(os.getcwd(), 'Examples',"Results", f"{curr_time}_{setting_name}")
            
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)

            #with open("Results/" + curr_time + '_' + setting_name + "/used_settings_" + setting_name +
            #          '.json', 'w+') as f:
            #    json.dump(settings, f, indent=2)  # Store current settings


            with open(os.path.join(folder_path, f"used_settings_{setting_name}.json"), 'w+') as f:
                json.dump(settings, f, indent=2)  # Store current settings

    # Get the SDM structure from the CLD

    # Load the adjacency matrix based on the Kumu table
    file = os.path.join(os.path.dirname(__file__), 'Examples',"Kumu", f"{setting_name}.xlsx")
    extract = Extract(file)  # Load the relevant Kumu file extraction module

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

    # Set the SDM simulation timesteps to store 
    s.t_eval = np.array(np.array([0.0] + list(np.linspace(0, s.t_end,
                                                        int(s.t_end/s.dt) + 1)[1:])))

    # If solving the system numerically, set the solver
    s.solver = 'LSODA'  # 'LSODA' automatically switches between stiff and non-stiff methods since stiffness is not always known.

    # Select variables to simulated interventions for; all variables except the var of interest by default
    s.intervention_variables = [var for var in s.variables if var != s.variable_of_interest]  

    sdm = SDM(df_adj, interactions_matrix, s)  # Load the module for formulating and simulating the SDM

    if s.simulate_interventions == False:
        # Set all initial conditions to a small value to get dynamics from the initial conditions
        x0 = np.ones(len(s.stocks_and_constants), order='F') * 0.01  
