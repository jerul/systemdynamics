import sys
import os
import datetime
import scipy
from tqdm import tqdm 
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from systemdynamics.cld import Extract
from systemdynamics.sdm import SDM
#from systemdynamics.plots import Plots
sns.set()

if __name__ == "__main__":
    # Assing setting has input to load the setting (sys.argv[1])
    if len(sys.argv) > 1:  # Settings passed from Jupyter
        system_argument = sys.argv[1]

    if "Results/" in system_argument:  # When loading posterior samples
        # Set folder path to current folder
        curr_time = "_".join(system_argument.split("/")[-1].split("_")[:3])
        setting_name = "_".join(
            system_argument.split("/")[1:][0].split("_")[3:])

        with open(system_argument + "/used_settings_" + setting_name + '.json') as f:
            settings = json.load(f)
        s = SimpleNamespace(**settings)

    else:  # When running new estimation
        setting_name = system_argument
        with open('Settings/'+setting_name+'.json') as f:
            settings = json.load(f)
        s = SimpleNamespace(**settings)

        curr_time = (str(datetime.datetime.now())[0:10] + "_" +
                     str(datetime.datetime.now())[11:13] + "_" +
                     str(datetime.datetime.now())[14:16])

        if s.save_results:  # Create a directory to store results
            folder_path = os.path.join(
                os.getcwd(), "Results/" + curr_time + '_' + setting_name)
            os.mkdir(folder_path)

            with open("Results/" + curr_time + '_' + setting_name + "/used_settings_" + setting_name +
                      '.json', 'w+') as f:
                json.dump(settings, f, indent=2)  # Store current settings

    # Get the SDM structure from the CLD

    # Load the adjacency matrix based on the Kumu table
    file = "Kumu/" + setting_name + ".xlsx"
    extract = Extract(file)  # Load the relevant Kumu file extraction module

    # Load the adjacency matrix from the KUmu file
    variable_names, var_to_type_init, adjacency_matrix, interactions_matrix = extract.adjacency_matrix_from_kumu()  

    if np.abs(interactions_matrix).sum() > 0:
        s.interaction_terms = True
        if s.solve_analytically and s.interaction_terms:
            print("Cannot solve analytically with interaction terms so will proceed with numerical solution.")

    else:
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

    # Create dataframe with adjacency matrix
    df_adj = pd.DataFrame(adjacency_matrix,
                          columns=s.variables, index=s.variables) 

    np.random.seed(s.seed)  # Set seed for reproducibility

    # Set the SDM simulation timesteps to store 
    t_eval = np.array(np.array([0.0] + list(np.linspace(0, s.t_end,
                                                        int(s.t_end/s.dt) + 1)[1:])))

    # If solving the system numerically, set the solver
    s.solver = 'LSODA'  # 'LSODA' automatically switches between stiff and non-stiff methods since stiffness is not always known.

    # Select variables to simulated interventions for; all variables except the var of interest by default
    s.intervention_variables = [var for var in s.variables if var != s.variable_of_interest]  

    sdm = SDM(df_adj, interactions_matrix, t_eval, s)  # Load the module for formulating and simulating the SDM

    if s.simulate_interventions == False:
        # Set all initial conditions to a small value to get dynamics from the initial conditions
        x0 = np.ones(len(s.stocks_and_constants), order='F') * 0.01  
