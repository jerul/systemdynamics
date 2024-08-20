import sys
import os
import datetime
import json
from types import SimpleNamespace
from copy import deepcopy
import seaborn as sns
from systemdynamics.cld import Extract
from systemdynamics.sdm import SDM
from systemdynamics.plots import plot_simulated_interventions, plot_simulated_intervention_ranking, plot_simulated_data
from systemdynamics.plots import plot_simulated_interventions_compare, plot_feedback_loops_ranking, plot_feedback_loops_over_time, plot_feedback_loops_over_time_bounds
sns.set_theme()

if __name__ == "__main__":
    if len(sys.argv) > 1:  # Settings passed from Jupyter
        setting_name = sys.argv[1]
        

        # Load the adjacency matrix based on the Kumu table
        extract = Extract(setting_name)  # Load the relevant Kumu file extraction module
        s = extract.extract_settings()  # Extract the settings using the json file and the Kumu table
        # Load the module for formulating and simulating the SDM
        sdm = SDM(s.df_adj, s.interactions_matrix, s)  
    else:
        assert False, 'No setting name provided'

