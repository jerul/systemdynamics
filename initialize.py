import sys
import os
import datetime
import scipy
import json
from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from systemdynamics.cld import Extract
from systemdynamics.sdm import SDM
from systemdynamics.plots import plot_simulated_interventions, plot_simulated_intervention_ranking, plot_simulated_data
from systemdynamics.plots import plot_simulated_interventions_compare, plot_feedback_loops_ranking, plot_feedback_loops_over_time, plot_feedback_loops_over_time_bounds
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
            curr_time = (str(datetime.datetime.now())[0:10])  # Create a new folder for each date
                        # + "_" #+
                        #str(datetime.datetime.now())[11:13]) # + "_" +
                        # str(datetime.datetime.now())[14:16])

            if s.save_results:  # Create a directory to store results
                folder_path = os.path.join(os.getcwd(),"Results", f"{curr_time}_{setting_name}")
                
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)

                with open(os.path.join(folder_path, f"used_settings_{setting_name}.json"), 'w+') as f:
                    json.dump(settings, f, indent=2)  # Store current settings

        s.save_path = os.path.join("Results", curr_time + '_' + setting_name + "/")  # Path for saving the results

    # Get the SDM structure from the CLD

    # Load the adjacency matrix based on the Kumu table
    file_name = os.path.join(os.path.dirname(__file__), 'Examples',"Kumu", f"{setting_name}.xlsx")
    
    extract = Extract(s, file_name)  # Load the relevant Kumu file extraction module
    s = extract.extract_settings()  # Extract the settings using the json file and the Kumu table

    # Load the module for formulating and simulating the SDM
    sdm = SDM(s.df_adj, s.interactions_matrix, s)  

