import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.stats
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
sns.set_theme()

def plot_simulated_data(df_pred, title, s):
    """
    Plot the simulated data.

    Parameters:
    - df_pred: DataFrame containing the simulated data.
    - title: Title of the plot.
    - s: Object containing system information.

    Returns:
    - None (Displays the plot).
    """

    num_plots = len(s.stocks_and_auxiliaries)
    num_rows = int(np.ceil(num_plots / 3))

    fig, axs = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))
    fig.suptitle(title)
    color_map = plt.cm.get_cmap('Paired', len(s.stocks_and_auxiliaries))

    ax = axs.flatten()
    N = len(df_pred)

    for k, var in enumerate(s.stocks_and_auxiliaries):
        if s.interval_type != "spaghetti":
            avg_at_time_t = []
            lb_confs_at_time_t = []
            ub_confs_at_time_t = []

            for t in s.t_eval:
                label_avg = "Mean"
                samples_at_time_t = [df_pred[i].loc[t, var] for i in range(s.N)]

                if s.interval_type == "confidence":
                    mean = np.mean(samples_at_time_t)
                    standard_error = scipy.stats.sem(samples_at_time_t)
                    h = standard_error * scipy.stats.t.ppf((1 + s.confidence_bounds) / 2., N-1)
                    avg_at_time_t.append(mean)
                    lb_confs_at_time_t.append(mean-h)
                    ub_confs_at_time_t.append(mean+h)
                elif s.interval_type == "percentile":
                    mean = np.median(samples_at_time_t)
                    lower_percentile = (1 - s.confidence_bounds) / 2 * 100
                    upper_percentile = (1 + s.confidence_bounds) / 2 * 100
                    avg_at_time_t.append(mean)
                    lb_confs_at_time_t.append(np.percentile(samples_at_time_t, lower_percentile))
                    ub_confs_at_time_t.append(np.percentile(samples_at_time_t, upper_percentile))

            ax[k].plot(s.t_eval, avg_at_time_t, label=label_avg)
            ax[k].fill_between(s.t_eval, lb_confs_at_time_t, ub_confs_at_time_t,
                               alpha=.3, label=str(int(s.confidence_bounds*100)) + "% CI") #Confidence interval")
        else:
            for i, data_i, in enumerate(df_pred):
                ax[k].plot(data_i.Time, data_i[var], alpha=.3, color=color_map(k)) 
        
        label = " ".join(var.split("_"))
        ax[k].set_ylabel(label)

        if k >= num_plots - 3:  # Last row of plots
            ax[k].set_xlabel(s.time_unit)

        if k == 0:
            ax[k].legend()

    # Hide unused subplot space
    for b in range(num_rows * 3 - num_plots):
        ax[num_plots + b].axis('off')

    plt.tight_layout()
    plt.show()

def plot_simulated_interventions_compare(df_sol_per_sample, s):
    """ Plot the simulated interventions in the same plot.
    """
    # Create widgets
    confidence_bounds_slider = widgets.FloatSlider(value=0.95, min=0.01, max=0.99, step=0.01, description='Interval bounds:')
    variable_selector = widgets.SelectMultiple(options=s.intervention_variables, value=['Body_fat', 'Sleep_problems'], description='Variables:')
        #options=["Sleep_problems", "Body_fat", "Other_var1", "Other_var2"], value=["Sleep_problems", "Body_fat"], description='Variables:')

    # Update plot function
    def update_plot(confidence_bounds, compare_int_vars):
        plt.figure(figsize=(10, 5))
        color_map = plt.cm.get_cmap('Paired', len(compare_int_vars))
        for k, var in enumerate(compare_int_vars):
            avg_at_time_t = []
            lb_confs_at_time_t = []
            ub_confs_at_time_t = []

            if s.interval_type != "spaghetti":
                for t in s.t_eval:
                    samples_at_time_t = [df_sol_per_sample[n][s.intervention_variables.index(var)].loc[t, s.variable_of_interest] for n in range(s.N)]

                    if s.interval_type ==  "confidence":
                        mean = np.mean(samples_at_time_t)
                        standard_error = scipy.stats.sem(samples_at_time_t)
                        h = standard_error * scipy.stats.t.ppf((1 + confidence_bounds) / 2., s.N-1)
                        avg_at_time_t.append(mean)
                        lb_confs_at_time_t.append(mean-h)
                        ub_confs_at_time_t.append(mean+h)

                    elif s.interval_type ==  "percentile":
                        avg_at_time_t.append(np.median(samples_at_time_t))
                        lower_percentile = (1 - confidence_bounds) / 2 * 100
                        upper_percentile = (1 + confidence_bounds) / 2 * 100
                        lb_confs_at_time_t.append(np.percentile(samples_at_time_t, lower_percentile))
                        ub_confs_at_time_t.append(np.percentile(samples_at_time_t, upper_percentile))
                    
                plt.plot(s.t_eval, avg_at_time_t, label=" ".join(var.split("_"))) 
                plt.fill_between(s.t_eval, lb_confs_at_time_t, ub_confs_at_time_t, alpha=.3)
            else:
                for i, data_i, in enumerate(df_sol_per_sample):
                    plt.plot(data_i[0].Time, data_i[k][s.variable_of_interest], alpha=.3, color=color_map(k))              

        plt.xlabel(s.time_unit)
        plt.ylabel(" ".join(s.variable_of_interest.split("_")))
        plt.legend()
        plt.show()

    # Display widgets and plot
    return widgets.interactive(update_plot, confidence_bounds=confidence_bounds_slider, compare_int_vars=variable_selector)
    
def plot_simulated_interventions(df_sol_per_sample, title, s):
    """
    Plot the simulated interventions.

    Parameters:
    - df_sol_per_sample: DataFrame containing the simulated data.
    - title: Title of the plot.
    - s: Object containing system information.

    Returns:
    - None (Displays the plot).
    """

    num_plots = len(s.intervention_variables)
    num_rows = int(np.ceil(num_plots / 3))

    fig, axs = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))
    fig.suptitle(title)
    color_map = plt.cm.get_cmap('Paired', len(s.stocks_and_auxiliaries))

    ax = axs.flatten()

    for k, var in enumerate(s.intervention_variables):
        if s.interval_type != "spaghetti":
            avg_at_time_t = []
            lb_confs_at_time_t = []
            ub_confs_at_time_t = []

            for t in s.t_eval:
                samples_at_time_t = [df_sol_per_sample[n][k].loc[t, s.variable_of_interest] for n in range(s.N)]

                if s.interval_type == "confidence":
                    label_avg = "Mean"
                    mean = np.mean(samples_at_time_t)
                    standard_error = scipy.stats.sem(samples_at_time_t)
                    h = standard_error * scipy.stats.t.ppf((1 + s.confidence_bounds) / 2., s.N-1)
                    avg_at_time_t.append(mean)
                    lb_confs_at_time_t.append(mean-h)
                    ub_confs_at_time_t.append(mean+h)

                elif s.interval_type == "percentile":
                    label_avg = "Median"
                    avg_at_time_t.append(np.median(samples_at_time_t))
                    lower_percentile = (1 - s.confidence_bounds) / 2 * 100
                    upper_percentile = (1 + s.confidence_bounds) / 2 * 100
                    lb_confs_at_time_t.append(np.percentile(samples_at_time_t, lower_percentile))
                    ub_confs_at_time_t.append(np.percentile(samples_at_time_t, upper_percentile))
                    
            ax[k].plot(s.t_eval, avg_at_time_t, label=label_avg)
            ax[k].fill_between(s.t_eval, lb_confs_at_time_t, ub_confs_at_time_t,
                                alpha=.3, label=str(int(s.confidence_bounds*100)) + "% " + s.interval_type + " interval")
        
        else:
            for i, data_i, in enumerate(df_sol_per_sample):
                ax[k].plot(data_i[0].Time, data_i[k][s.variable_of_interest], alpha=.3, color=color_map(k))

        label = " ".join(s.variable_of_interest.split("_"))
        ax[k].set_ylabel(label)
        title = "Intervention on " + " ".join(var.split("_"))
        ax[k].set_title(title)

        if k >= num_plots - 3:  # Last row of plots
            ax[k].set_xlabel(s.time_unit)

        if k == 0:
            ax[k].legend()

    # Hide unused subplot space
    for b in range(num_rows * 3 - num_plots):
        ax[num_plots + b].axis('off')

    plt.tight_layout()
    plt.show()

def plot_simulated_intervention_ranking(intervention_effects, s):
   """ Plot simulated intervention effects in a horizontal boxplot, ranked by median.
   """
   fig = plt.figure(figsize=(5, 8))  # 7,15
   ax = fig.add_subplot(111)
   df_SA = pd.DataFrame(intervention_effects)

   # Order by median
   df_SA = df_SA.reindex(columns=list(
                        df_SA.median().sort_values(ascending=False).index))
   df_SA = df_SA.rename(mapper=dict(
                        zip(s.intervention_variables,
                           [" ".join(var.split("_")) for var in s.intervention_variables ])), axis=1)

   sns.boxplot(data=df_SA, showfliers=False, whis=True, orient='h')
   plt.vlines(x=0, ymin=-0.5, ymax=len(s.intervention_variables) -
               0.6, colors='black', linestyles='dashed')
   plt.title("Effect on " + " ".join(s.variable_of_interest.split("_")))
   plt.xlabel("Standardized effect after " + str(s.t_end) + " " + s.time_unit)
   plt.ylabel("")
   # ax.invert_xaxis()


# plt.figure()
# confidence_bounds = .95
# compare_int_vars = ["Sleep_problems", "Body_fat"]

# for k, var in enumerate(compare_int_vars):
#     means_at_time_t = []
#     lb_confs_at_time_t = []
#     ub_confs_at_time_t = []

#     for t in t_eval:
#         samples_at_time_t = [df_sol_per_sample[n][s.intervention_variables.index(var)].loc[t, s.variable_of_interest] for n in range(s.N)]
#         mean = np.mean(samples_at_time_t)
#         standard_error = scipy.stats.sem(samples_at_time_t)
#         h = standard_error * scipy.stats.t.ppf((1 + confidence_bounds) / 2., s.N-1)
#         means_at_time_t.append(mean)
#         lb_confs_at_time_t.append(mean-h)
#         ub_confs_at_time_t.append(mean+h)

#     plt.plot(t_eval, means_at_time_t, label=" ".join(var.split("_")))
#     plt.fill_between(t_eval, lb_confs_at_time_t, ub_confs_at_time_t,
#                         alpha=.3) #, label=str(int(confidence_bounds*100)) + "% CI") #Confidence interval")
#     plt.xlabel(s.time_unit)
#     plt.ylabel(" ".join(s.variable_of_interest.split("_")))
#     plt.legend()