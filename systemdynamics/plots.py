import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.stats
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
sns.set_theme()

def plot_simulated_data(s, df_pred, title):
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

def plot_simulated_interventions_compare(s, df_sol_per_sample):
    """ Plot the simulated interventions in the same plot.
    """
    # Create widgets
    confidence_bounds_slider = widgets.FloatSlider(value=0.95, min=0.01, max=0.99, step=0.01, description='Interval bounds:')
    variable_selector = widgets.SelectMultiple(options=s.intervention_variables, value=s.intervention_variables[:2], description='Variables:')

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

    return widgets.interactive(update_plot, confidence_bounds=confidence_bounds_slider, compare_int_vars=variable_selector, top_plot=None)
    
def plot_simulated_interventions(s, df_sol_per_sample, intervention_effects, interval_type="percentile", confidence_bounds=.95, top_plot=None):
    """
    Plot the simulated interventions over time
    """
    df_sol_per_sample_reordered = df_sol_per_sample.copy()
    top_vars_plot = list(intervention_effects.keys())[:top_plot]

    for i in range(s.N):
        df_sol_per_sample_dict_i = dict(zip(s.intervention_variables, df_sol_per_sample[i]))
        df_sol_per_sample_reordered[i] = [df_sol_per_sample_dict_i[var] for var in top_vars_plot] 

    df_SA = pd.DataFrame(intervention_effects)
    df_SA = df_SA.reindex(columns=list(
                            df_SA.abs().median().sort_values(ascending=False).index))
    palette_dict = {var : "#4682B4" for var in df_SA.columns}  # Blue for positive effects
    medians = df_SA.median()
    lower_than_zero_vars = medians.loc[medians < 0].index
    for var in lower_than_zero_vars:
        palette_dict[var] = "#FF6347"  # Red for negative effects

    num_plots = len(top_vars_plot)
    num_rows = int(np.ceil(num_plots / 3))

    fig, axs = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))
    fig.suptitle("Simulated interventions with N="+ str(s.N) + " samples")
    ax = axs.flatten()

    #max_value = max([max([df_sol_per_sample[i][j][s.variable_of_interest].max() for i in range(s.N)]) for j in range(len(s.intervention_variables))])
    #min_value = min([min([df_sol_per_sample[i][j][s.variable_of_interest].min() for i in range(s.N)]) for j in range(len(s.intervention_variables))])

    for k, var in enumerate(s.intervention_variables):
        if interval_type != "spaghetti":
            avg_at_time_t = []
            lb_confs_at_time_t = []
            ub_confs_at_time_t = []

            for t in s.t_eval:
                samples_at_time_t = [df_sol_per_sample[n][k].loc[t, s.variable_of_interest] for n in range(s.N)]

                if interval_type == "confidence":
                    label_avg = "Mean"
                    mean = np.mean(samples_at_time_t)
                    standard_error = scipy.stats.sem(samples_at_time_t)
                    h = standard_error * scipy.stats.t.ppf((1 + confidence_bounds) / 2., s.N-1)
                    avg_at_time_t.append(mean)
                    lb_confs_at_time_t.append(mean-h)
                    ub_confs_at_time_t.append(mean+h)

                elif interval_type == "percentile":
                    label_avg = "Median"
                    avg_at_time_t.append(np.median(samples_at_time_t))
                    lower_percentile = (1 - confidence_bounds) / 2 * 100
                    upper_percentile = (1 + confidence_bounds) / 2 * 100
                    lb_confs_at_time_t.append(np.percentile(samples_at_time_t, lower_percentile))
                    ub_confs_at_time_t.append(np.percentile(samples_at_time_t, upper_percentile))
    
            ax[k].plot(s.t_eval, avg_at_time_t, label=label_avg, color=palette_dict[var])
            ax[k].fill_between(s.t_eval, lb_confs_at_time_t, ub_confs_at_time_t,
                                alpha=.3, label=str(int(confidence_bounds*100)) + "% " + interval_type + " interval", color=palette_dict[var])
        
        else:
            for i, data_i, in enumerate(df_sol_per_sample):
                ax[k].plot(data_i[0].Time, data_i[k][s.variable_of_interest], alpha=.3, color=palette_dict[var])

        label = " ".join(s.variable_of_interest.split("_"))
        ax[k].set_ylabel(label)
        title = " ".join(var.split("_")) #"Intervention on " + " ".join(var.split("_"))
        ax[k].set_title(title)
        #ax[k].set_ylim([min_value, max_value])

        if k >= num_plots - 3:  # Last row of plots
            ax[k].set_xlabel(s.time_unit)

        if k == 0:
            ax[k].legend()

    # Hide unused subplot space
    for b in range(num_rows * 3 - num_plots):
        ax[num_plots + b].axis('off')

    plt.tight_layout()

    if s.save_results:
        title = 'simulated_interventions_plots_per_N' + str(s.N) + '.jpg'
        plt.savefig(s.save_path + title, format='jpg', dpi=300, bbox_inches='tight')

    return fig

def plot_simulated_intervention_ranking(s, intervention_effects, top_plot=None):
   """ Plot simulated intervention effects in a horizontal boxplot, ranked by median.
   """
   df_SA = pd.DataFrame(intervention_effects)
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
   
   if s.save_results:
      title = 'simulated_interventions_ranking_plots_per_individual_N' + str(s.N) + '.jpg'
      plt.savefig(s.save_path + title, format='jpg', dpi=300, bbox_inches='tight')

   return fig

def plot_feedback_loops_ranking(s, df_loops, intervention_effects, int_var=None, cut_off_importance=0.1):
    """ Create a ranked box plot of average feedback loop scores.
    """
    df_loops = df_loops.loc[:, df_loops.mean().abs() > cut_off_importance]  # Plot only the most relevant loops
    if int_var == None:
        int_var = list(intervention_effects.keys())[0]  # Select an intervention for which we will assess the feedback loop dominance
    fig = plt.figure(figsize=(5, 8))
    ax = fig.add_subplot(111)

    palette_dict = {var : "#4682B4" for var in df_loops.columns}  # Blue for positive effects
    medians = df_loops.median()
    lower_than_zero_vars = medians.loc[medians < 0].index
    for var in lower_than_zero_vars:
        palette_dict[var] = "#FF6347"  # Red for negative effects
    palette = list(palette_dict.values())  # Convert to list

    sns.boxplot(data=df_loops.abs(), showfliers=False, whis=True, orient='h', palette=palette);
    plt.vlines(x=0, ymin=-0.5, ymax=len(df_loops.columns) - 0.6, colors='black', linestyles='dashed');
    plt.xlabel("Contribution to average dynamics over time");  #(" + combine_loop_type + " over time)");
    plt.title("Top feedback loops for intervention on " + " ".join(int_var.split("_")));

    if s.save_results:
        title = 'feedback_loop_ranking_intervention_on_' + int_var +'_N_'+ str(s.N) + '.jpg'
        plt.savefig(s.save_path + title, format='jpg', dpi=300, bbox_inches='tight')

    return fig

def plot_feedback_loops_over_time(s, df_loops, intervention_effects, loopscores_per_sample, int_var=None, cut_off_importance=0.1):
    """ Create plots of feedback loop contributions over time.
    """
    df_loops = df_loops.loc[:, df_loops.mean().abs() > cut_off_importance]  # Plot only the most relevant loops
    if int_var == None:
        int_var = list(intervention_effects.keys())[0]  # Select an intervention for which we will assess the feedback loop dominance
    num_loops = len(df_loops.columns)
    num_rows = int(np.ceil(num_loops/2))
    fig, axs = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows))
    fig.suptitle("Top feedback loops for intervention on " + " ".join(int_var.split("_")))

    # palette_dict = {var : "#4682B4" for var in df_loops.columns}  # Blue for positive effects
    # medians = df_loops.median()
    # lower_than_zero_vars = medians.loc[medians < 0].index
    # for var in lower_than_zero_vars:
    #     palette_dict[var] = "#FF6347"  # Red for negative effects

    ax = axs.flatten()
    for i, loop in enumerate(df_loops.columns):
        for k in range(len(df_loops)):
            df_loops_t = pd.DataFrame(loopscores_per_sample[k])[df_loops.columns] #.abs()
            ax[i].plot(df_loops_t[loop], alpha=.3) #, color=palette_dict[loop])  
        ax[i].set_ylabel("Contribution to dynamics")
        ax[i].set_title(loop)

        if i > num_loops - 3:
            ax[i].set_xlabel(s.time_unit)

    if num_rows * 2 != num_loops:
        ax[-1].set_visible(False)

    plt.tight_layout()

    if s.save_results:
        title = 'feedback_loops_importance_over_time_intervention_on_' + int_var +'_N_'+ str(s.N) + '.jpg'
        plt.savefig(s.save_path + title, format='jpg', dpi=300, bbox_inches='tight')

    return fig

def plot_feedback_loops_over_time_bounds(s, df_loops, intervention_effects, loopscores_per_sample, int_var=None,  cut_off_importance=0.1):
    """ Create plots of feedback loop contributions over time with confidence bounds
    """
    df_loops = df_loops.loc[:, df_loops.mean().abs() > cut_off_importance]  # Plot only the most relevant loops
    if int_var == None:
        int_var = list(intervention_effects.keys())[0]  # Select an intervention for which we will assess the feedback loop dominance
    confidence_bounds = .50
    num_loops = len(df_loops.columns)
    num_rows = int(np.ceil(num_loops/2))
    fig, axs = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows))
    ax = axs.flatten()
    fig.suptitle("Top contributing feedback loops for intervention on " + " ".join(int_var.split("_")))

    palette_dict = {var : "#4682B4" for var in df_loops.columns}  # Blue for positive effects
    medians = df_loops.median()
    lower_than_zero_vars = medians.loc[medians < 0].index
    for var in lower_than_zero_vars:
        palette_dict[var] = "#FF6347"  # Red for negative effects
    
    for j, loop in enumerate(df_loops.columns):
        avg_at_time_t = []
        lb_confs_at_time_t = []
        ub_confs_at_time_t = []

        for i, t in enumerate(s.t_eval[1:]):
            samples_at_time_t = [loopscores_per_sample[n][loop][i] for n in range(s.N)]

            avg_at_time_t.append(np.median(samples_at_time_t))
            lower_percentile = (1 - confidence_bounds) / 2 * 100
            upper_percentile = (1 + confidence_bounds) / 2 * 100
            lb_confs_at_time_t.append(np.percentile(samples_at_time_t, lower_percentile))
            ub_confs_at_time_t.append(np.percentile(samples_at_time_t, upper_percentile))
                
        ax[j].plot(s.t_eval[1:], avg_at_time_t, color=palette_dict[loop]) 
        ax[j].fill_between(s.t_eval[1:], lb_confs_at_time_t, ub_confs_at_time_t, alpha=.3, color=palette_dict[loop])   
        ax[j].set_ylabel("Contribution to dynamics")
        ax[j].set_title(loop)
        if j > num_loops - 3: 
            ax[j].set_xlabel(s.time_unit)

    if s.save_results:
        title = 'feedback_loops_importance_over_time_w_bounds_intervention_on_' + int_var +'_N_'+ str(s.N) + '.jpg'
        plt.savefig(s.save_path + title, format='jpg', dpi=300, bbox_inches='tight')

    return fig
