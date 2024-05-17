import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import pandas as pd
sns.set()

def plot_simulated_data(df_pred, title, s, confidence_bounds=.95):
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

    ax = axs.flatten()
    N = len(df_pred)

    for k, var in enumerate(s.stocks_and_auxiliaries):
        if confidence_bounds != False:
            means_at_time_t = []
            lb_confs_at_time_t = []
            ub_confs_at_time_t = []

            for t in t_eval:
                samples_at_time_t = [df_pred[i].loc[t, var] for i in range(N)]
                mean = np.mean(samples_at_time_t)
                standard_error = scipy.stats.sem(samples_at_time_t)
                h = standard_error * scipy.stats.t.ppf((1 + confidence_bounds) / 2., N-1)
                means_at_time_t.append(mean)
                lb_confs_at_time_t.append(mean-h)
                ub_confs_at_time_t.append(mean+h)

            ax[k].plot(t_eval, means_at_time_t, label="Mean")
            ax[k].fill_between(t_eval, lb_confs_at_time_t, ub_confs_at_time_t,
                               alpha=.3, label=str(int(confidence_bounds*100)) + "% CI") #Confidence interval")
        else:
            for i, data_i, in enumerate(df_pred):
                ax[k].plot(data_i.Time, data_i[var], alpha=.3) 
        
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

def plot_simulated_interventions(df_pred, title, s, confidence_bounds=.95):
    """
    Plot the simulated interventions.

    Parameters:
    - df_pred: DataFrame containing the simulated data.
    - title: Title of the plot.
    - s: Object containing system information.

    Returns:
    - None (Displays the plot).
    """

    num_plots = len(s.intervention_variables)
    num_rows = int(np.ceil(num_plots / 3))

    fig, axs = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))
    fig.suptitle(title)

    ax = axs.flatten()
    N = len(df_pred)

    for k, var in enumerate(s.intervention_variables):
       # if confidence_bounds != False:
        means_at_time_t = []
        lb_confs_at_time_t = []
        ub_confs_at_time_t = []

        for t in t_eval:
            samples_at_time_t = [df_sol_per_sample[n][k].loc[t, s.variable_of_interest] for n in range(N)]
            #[df_pred[i].loc[t, var] for i in range(N)]
            mean = np.mean(samples_at_time_t)
            standard_error = scipy.stats.sem(samples_at_time_t)
            h = standard_error * scipy.stats.t.ppf((1 + confidence_bounds) / 2., N-1)
            means_at_time_t.append(mean)
            lb_confs_at_time_t.append(mean-h)
            ub_confs_at_time_t.append(mean+h)

        ax[k].plot(t_eval, means_at_time_t, label="Mean")
        ax[k].fill_between(t_eval, lb_confs_at_time_t, ub_confs_at_time_t,
                            alpha=.3, label=str(int(confidence_bounds*100)) + "% CI") #Confidence interval")
        #else:
        #    for i, data_i, in enumerate(df_pred):
        #        ax[k].plot(data_i.Time, data_i[var], alpha=.3) 
        

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
   plt.title("Effect on " + " ".join(s.var_of_interest.split("_")))
   plt.xlabel("Standardized effect after " + str(s.t_end) + " " + s.time_unit)
   plt.ylabel("")
   ax.invert_xaxis()
