# Copyright (c) 2025 Jeroen F. Uleman. Licensed under CC BY-NC 4.0.
# Non-commercial use only. See LICENSE for details.
import jax
import jax.numpy as jnp
import diffrax
import pandas as pd
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
import scipy.stats
from .sdm import SDM

class D2D(SDM):
    """
    Diagrams to Dynamics (D2D) Extension
    Handles Monte Carlo simulations based on Causal Loop Diagrams (CLD)
    parameter sampling.
    """
    def __init__(self, s):
        super().__init__(s)
        self.N = getattr(s, "N", 100)
        
    def _solve_single_sample(self, key):
        W_lin, W_quad, Bias = self.sample_parameters(key)
        
        # D2D assumes 0 initial conditions usually, modified by interventions
        x0_base = jnp.zeros(self.n_vars)
        consts_base = jnp.zeros(self.n_vars)
        
        n_inv = len(self.intervention_specs)
        max_mods = 2 
        
        spec_types = np.full((n_inv, max_mods), -1, dtype=np.int32)
        spec_indices = np.zeros((n_inv, max_mods), dtype=np.int32)
        spec_vals = np.zeros((n_inv, max_mods), dtype=np.float32)
        
        for i, mods in enumerate(self.intervention_specs):
            for j, mod in enumerate(mods):
                if j >= max_mods: break
                spec_types[i, j] = mod['type']
                spec_indices[i, j] = mod['idx']
                spec_vals[i, j] = mod['val']
                
        j_spec_types = jnp.array(spec_types)
        j_spec_idxs = jnp.array(spec_indices)
        j_spec_vals = jnp.array(spec_vals)
        
        solver = diffrax.Dopri5()
        stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-6)
        
        def run_one_intervention(i):
            s_types = j_spec_types[i]
            s_idxs = j_spec_idxs[i]
            s_vals = j_spec_vals[i]
            
            curr_x0 = x0_base
            curr_consts = consts_base
            curr_bias = Bias
            
            curr_x0, curr_consts, curr_bias = self._apply_intervention(
                curr_x0, curr_consts, curr_bias,
                s_idxs, s_types, s_vals
            )
            
            y0 = curr_x0[self.stock_idxs]
            
            args = (W_lin, W_quad, curr_bias, curr_consts, 
                    self.stock_idxs, self.aux_idxs, self.sorted_aux_idxs)
            
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(self.ode_term),
                solver,
                t0=self.t_span[0],
                t1=self.t_span[1],
                dt0=0.1,
                y0=y0,
                args=args,
                saveat=self.saveat,
                stepsize_controller=stepsize_controller,
                max_steps=4000
            ) 
            
            def get_full_state(y):
                z = jnp.zeros(self.n_vars)
                z = z.at[self.stock_idxs].set(y)
                z = z + curr_consts
                
                for idx in self.sorted_aux_idxs:
                    val = curr_bias[idx] + jnp.dot(W_lin[idx], z) + jnp.dot(z, jnp.dot(W_quad[idx], z))
                    z = z.at[idx].set(val)
                return z

            full_traj = jax.vmap(get_full_state)(sol.ys)
            return full_traj

        results = jax.vmap(run_one_intervention)(jnp.arange(n_inv))
        
        return results, (W_lin, W_quad, Bias)

    def run_simulations(self):
        """Run D2D Monte Carlo simulations with randomly sampled parameters.

        For each of the N samples a new parameter set is drawn via
        sample_parameters() and the ODE is solved for every intervention
        variable.  Results are stored in self.df_sol_per_sample and
        self.param_samples, and also returned for convenience.

        Returns:
            df_sol_per_sample: list of length N; each element is a list of
                DataFrames (one per intervention variable) with variable
                trajectories indexed by time.
            param_samples: nested dict {intervention_var: {target: {source: [values]}}}
                containing the sampled parameter values for sensitivity analysis.
        """
        print(f"Running {self.N} D2D simulations with JAX...")
        
        keys = jax.random.split(self.key, self.N)
        sim_results, params_raw = jax.vmap(self._solve_single_sample)(keys)
        
        self.df_sol_per_sample = [] 
        np_results = np.array(sim_results)
        
        self.param_samples = {var: {} for var in self.intervention_variables}
        self._reconstruct_param_samples(params_raw)
        
        time_index = np.array(self.t_eval)
        cols = self.variables
        
        for n in tqdm(range(self.N), desc="Processing Results"):
            sample_dfs = []
            for i, var in enumerate(self.intervention_variables):
                data = np_results[n, i, :, :]
                df = pd.DataFrame(data, columns=cols, index=time_index)
                df['Time'] = df.index
                sample_dfs.append(df)
            self.df_sol_per_sample.append(sample_dfs)
            
        return self.df_sol_per_sample, self.param_samples

    def get_intervention_effects(self):
        """Extract the outcome value at the final time point for every sample and intervention.

        Must be called after run_simulations().

        Returns:
            dict keyed by variable-of-interest name; each value is a dict keyed by
            intervention variable name containing a list of N outcome values (one per
            Monte Carlo sample), sorted descending by median absolute effect.
        """
        intervention_effects_per_voi = {voi : {} for voi in self.variable_of_interest}

        for voi in self.variable_of_interest:
            intervention_effects = {i_v : [self.df_sol_per_sample[n][i].loc[float(self.t_eval[-1]), voi] 
                                    for n in range(self.N)] for i, i_v in enumerate(self.intervention_variables)}
            intervention_effects = dict(sorted(intervention_effects.items(),
                                            key=lambda item: np.median(np.abs(item[1])), reverse=True))

            self.intervention_effects = intervention_effects
            intervention_effects_per_voi[voi] = intervention_effects
        
        return intervention_effects_per_voi
    
    def _reconstruct_param_samples(self, params_raw):
        W_lin_all, W_quad_all, Bias_all = params_raw
        W_lin_all = np.array(W_lin_all)
        
        param_map = []
        
        for v in self.stocks_and_auxiliaries:
            idx = self.var_to_idx[v]
            param_map.append((v, 'Intercept', idx, -1, 'bias'))
            
        for r in range(self.n_vars):
            for c in range(self.n_vars):
                if self.df_adj.values[r, c] != 0:
                    target = self.variables[r]
                    source = self.variables[c]
                    param_map.append((target, source, r, c, 'lin'))
        
        for int_var in self.intervention_variables:
            self.param_samples[int_var] = {}
            for target in self.stocks_and_auxiliaries:
                 self.param_samples[int_var][target] = {}
                 
        for (target, source, r, c, ptype) in param_map:
            if ptype == 'bias':
                vals = Bias_all[:, r]
            elif ptype == 'lin':
                vals = W_lin_all[:, r, c]
            
            for int_var in self.intervention_variables:
                if target not in self.param_samples[int_var]:
                     self.param_samples[int_var][target] = {}
                
                self.param_samples[int_var][target][source] = vals.tolist()

        for idx, int_var in enumerate(self.intervention_variables):
            spec = self.intervention_specs[idx]
            for mod in spec:
                if mod['type'] == 2: # Aux Intercept
                    v_name = self.variables[mod['idx']]
                    strength = mod['val']
                    if strength != 0:
                        self.param_samples[int_var][v_name]['Intercept'] = [strength] * self.N

    def compare_interventions_table(self, intervention_effects, n_bootstraps=200):
        """Print a pairwise comparison table of all intervention variables.

        For each pair (A, B) reports the percentage of Monte Carlo samples in
        which |effect of A| > |effect of B|, a bootstrapped 95 % CI on that
        percentage, and Cliff's delta as an effect-size measure.

        Args:
            intervention_effects: dict as returned by get_intervention_effects()
                for a single variable of interest.
            n_bootstraps: number of bootstrap resamples for the CI (default 200).
        """
        temp = []
        comparison_results = []

        for i in intervention_effects:
            for j in [i_e for i_e in intervention_effects if i_e not in temp]:
                if i != j:
                    samples_i = np.abs(intervention_effects[i])
                    samples_j = np.abs(intervention_effects[j])
                    differences = np.subtract(samples_i, samples_j)

                    greater_i = np.sum(differences > 0)
                    greater_j = np.sum(differences < 0)
                    cliff = (greater_i - greater_j) / len(differences)
                    percent_greater = round(greater_i * 100 / len(differences), 1)

                    bootstrapped_percents = []
                    for _ in range(n_bootstraps):
                        idx = np.random.choice(len(differences), size=len(differences), replace=True)
                        diff_sample = differences[idx]
                        greater_i_sample = np.sum(diff_sample > 0)
                        bootstrapped_percents.append(greater_i_sample * 100 / len(diff_sample))
                    lower = round(np.percentile(bootstrapped_percents, 2.5), 1)
                    upper = round(np.percentile(bootstrapped_percents, 97.5), 1)
                    ci_str = f"[{lower}, {upper}]"

                    comparison_results.append([i, j, percent_greater, ci_str, round(cliff, 2)])

            temp.append(i)

        print("\nComparison Table (Percentage Greater, 95% CI, Cliff’s Delta):")
        print(tabulate(
            comparison_results,
            headers=["Intervention A", "Intervention B", "% Greater", "95% CI (% Greater)", "Cliff's Delta"],
            tablefmt="grid"
        ))

    def run_SA(self, outcome_var, int_var, cut_off_SA_importance=0.1, n_bootstraps=200):
        """Run a Spearman-rank sensitivity analysis of model parameters on outcomes.

        Computes the Spearman correlation between each sampled parameter and the
        outcome value at the final time point.  Only parameters with |rho| above
        cut_off_SA_importance are reported, with bootstrapped 95 % CIs.

        Must be called after run_simulations().

        Args:
            outcome_var: name of the variable to use as the outcome (string), or
                None to use the mean absolute value across all variables.
            int_var: intervention variable to analyse (string), or None to pool
                across all intervention variables.
            cut_off_SA_importance: minimum |rho| for a parameter to be included
                in the output table (default 0.1).
            n_bootstraps: number of bootstrap resamples for CIs (default 200).

        Returns:
            sorted_p_values: dict {link_name: [rho, CI_string]} sorted by |rho|.
            df_SA: DataFrame with one row per Monte Carlo sample containing all
                parameter values and the corresponding outcome.
        """
        if int_var is None:
            loop_over = self.intervention_variables
        else:
            loop_over = [int_var]

        dfs_to_concat = []

        for i_v in loop_over:
            i_idx = self.intervention_variables.index(i_v)
            
            data_dict = {}
            for target, sources in self.param_samples[i_v].items():
                for source, values in sources.items():
                    col_name = f"{source}->{target}"
                    data_dict[col_name] = values 

            effects = []
            final_time = float(self.t_eval[-1])
            
            for n in range(self.N):
                df_run = self.df_sol_per_sample[n][i_idx]
                row = df_run.loc[final_time]
                
                if outcome_var is None:
                    val = row.abs().mean()
                else:
                    try:
                        val = abs(row[outcome_var])
                    except KeyError:
                         val = abs(df_run[outcome_var].iloc[-1])
                         
                effects.append(float(val))

            data_dict["Effect"] = effects
            data_dict["intervention_variable"] = [i_v] * self.N

            dfs_to_concat.append(pd.DataFrame(data_dict))

        df_SA = pd.concat(dfs_to_concat, ignore_index=True)

        results = []
        param_cols = [c for c in df_SA.columns if "->" in c and "Intercept" not in c]

        for col in param_cols:
            rho, pval = scipy.stats.spearmanr(df_SA[col], df_SA["Effect"])

            if abs(rho) > cut_off_SA_importance:
                n_samples = len(df_SA)
                idx = np.random.randint(0, n_samples, (n_bootstraps, n_samples))
                
                bootstrapped_corrs = []
                x_data = df_SA[col].values
                y_data = df_SA["Effect"].values
                
                for k in range(n_bootstraps):
                    indices = idx[k]
                    x_sample = x_data[indices]
                    y_sample = y_data[indices]
                    r, _ = scipy.stats.spearmanr(x_sample, y_sample)
                    bootstrapped_corrs.append(r)

                lower = np.percentile(bootstrapped_corrs, 2.5)
                upper = np.percentile(bootstrapped_corrs, 97.5)

                mean_per_int = []
                for i_v in loop_over:
                    subset = df_SA[df_SA.intervention_variable == i_v]
                    if len(subset) > 1:
                        r_sub, _ = scipy.stats.spearmanr(subset[col], subset["Effect"])
                        mean_per_int.append(r_sub)
                    else:
                        mean_per_int.append(np.nan)

                results.append([col, round(rho, 2), 
                                f"[{round(lower, 2)}, {round(upper, 2)}]",
                                round(np.nanmean(mean_per_int), 2),
                                round(np.nanstd(mean_per_int), 2)])

        results.sort(key=lambda x: abs(x[1]), reverse=True)
        headers = ["Link", "Rho", "95% CI (bootstrap)", "Mean Rho per Int", "SD Rho per Int"] 
        
        print(tabulate(results, headers=headers, tablefmt="pretty"))
        sorted_p_values = {row[0]: [row[1], row[2]] for row in results}
        return sorted_p_values, df_SA
