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

