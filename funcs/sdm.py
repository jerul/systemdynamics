# Copyright (c) 2025 Jeroen F. Uleman. Licensed under CC BY-NC 4.0.
# Non-commercial use only. See LICENSE for details.
import jax
import jax.numpy as jnp
import diffrax
import pandas as pd
import numpy as np
from types import SimpleNamespace

# Core SDM Engine
class SDM:
    def __init__(self, s):
        self.s = s
        self.df_adj = s.df_adj
        self.interactions_matrix = s.interactions_matrix
        self.lb_matrix = getattr(s, 'lb_matrix', None)
        self.ub_matrix = getattr(s, 'ub_matrix', None)
        self.interaction_terms = getattr(s, "interaction_terms", 0)
        self.variables = s.variables
        self.stocks = s.stocks
        self.auxiliaries = s.auxiliaries
        self.constants = s.constants
        self.stocks_and_auxiliaries = getattr(s, "stocks_and_auxiliaries", s.stocks + s.auxiliaries)
        self.parameter_value_stocks = getattr(s, "parameter_value_stocks", 1.0) # Used for sampling
        self.parameter_value_aux = getattr(s, "parameter_value_aux", 1.0) # Used for sampling
        
        # Interventions (Optional for Core, used by D2D)
        self.intervention_variables = getattr(s, "intervention_variables", [])
        self.intervention_strengths = getattr(s, "intervention_strengths", {})
        self.variable_of_interest = getattr(s, "variable_of_interest", [])
        
        self.prior_type = getattr(s, "prior_type", "uniform") # "uniform" or "halfnormal"
        self.N = getattr(s, "N", 100)
        self.t_end = getattr(s, "t_end", 20.0)

        self.t_span = (0.0, float(self.t_end))
        self.t_eval = jnp.linspace(0.0, float(self.t_end), int(self.t_end) + 1)
        self.saveat = diffrax.SaveAt(ts=self.t_eval)
        
        # JAX Setup
        self._setup_indices()
        self._setup_adjacency()
        self._setup_interventions()
        
        self.seed = getattr(s, 'seed', 42)
        self.key = jax.random.PRNGKey(self.seed)

    def _setup_indices(self):
        self.var_to_idx = {var: i for i, var in enumerate(self.variables)}
        self.n_vars = len(self.variables)

        self.stock_idxs = jnp.array([self.var_to_idx[v] for v in self.stocks], dtype=jnp.int32)
        self.aux_idxs = jnp.array([self.var_to_idx[v] for v in self.auxiliaries], dtype=jnp.int32)
        self.const_idxs = jnp.array([self.var_to_idx[v] for v in self.constants], dtype=jnp.int32)
        
        self.n_stocks = len(self.stocks)
        self.n_aux = len(self.auxiliaries)
        self.n_const = len(self.constants)
        
        self.sorted_aux_idxs = self._sort_auxiliaries()

    def _sort_auxiliaries(self):
        adj = self.df_adj.loc[self.auxiliaries, self.auxiliaries].values
        n = len(self.auxiliaries)
        deps = {i: set(np.where(adj[i] != 0)[0]) for i in range(n)}
        
        sorted_local = []
        while deps:
            ready = [i for i, d in deps.items() if not d]
            if not ready:
                print("Warning: Circular dependency in auxiliaries detected.")
                break 
                
            sorted_local.extend(ready)
            for r in ready:
                del deps[r]
            for d in deps.values():
                d.difference_update(ready)
        
        return jnp.array([self.var_to_idx[self.auxiliaries[i]] for i in sorted_local], dtype=jnp.int32)

    def _setup_adjacency(self):
        self.adj_flat = self.df_adj.values.flatten()
        self.adj_is_missing = jnp.array(self.adj_flat == -999, dtype=bool)
        self.adj_polarity = jnp.array(self.adj_flat, dtype=jnp.float32)
        self.adj_nonzero = jnp.array(self.adj_flat != 0, dtype=bool)

        if self.interactions_matrix is None:
            self.int_flat = jnp.zeros(self.n_vars**3)
        else:
            self.int_flat = self.interactions_matrix.flatten()

        self.int_is_missing = jnp.array(self.int_flat == -999, dtype=bool)
        self.int_polarity = jnp.array(self.int_flat, dtype=jnp.float32)
        self.int_nonzero = jnp.array(self.int_flat != 0, dtype=bool)

        # Per-connection custom bounds: Uniform(LB, UB) overrides default scale-based sampling.
        # has_custom_bounds is True only where both LB and UB are non-NaN.
        if self.lb_matrix is not None and self.ub_matrix is not None:
            lb_np = self.lb_matrix.flatten()
            ub_np = self.ub_matrix.flatten()
            custom_mask = ~np.isnan(lb_np) & ~np.isnan(ub_np)
            self.has_custom_bounds = jnp.array(custom_mask, dtype=bool)
            self.lb_flat = jnp.array(np.where(np.isnan(lb_np), 0.0, lb_np), dtype=jnp.float32)
            self.ub_flat = jnp.array(np.where(np.isnan(ub_np), 1.0, ub_np), dtype=jnp.float32)
        else:
            n_sq = self.n_vars ** 2
            self.has_custom_bounds = jnp.zeros(n_sq, dtype=bool)
            self.lb_flat = jnp.zeros(n_sq, dtype=jnp.float32)
            self.ub_flat = jnp.ones(n_sq, dtype=jnp.float32)

    def _setup_interventions(self):
        self.intervention_specs = []
        
        for var in self.intervention_variables:
            mods = []
            if '+' in var:
                vars_in = var.split('+')
                factor = 0.5
            else:
                vars_in = [var]
                factor = 1.0
                
            for v_name in vars_in:
                strength = self.intervention_strengths.get(v_name, 0.0) * factor
                idx = self.var_to_idx[v_name]
                
                if v_name in self.stocks:
                    mods.append({'type': 0, 'idx': idx, 'val': strength})
                elif v_name in self.constants:
                    mods.append({'type': 1, 'idx': idx, 'val': strength})
                elif v_name in self.auxiliaries:
                    mods.append({'type': 2, 'idx': idx, 'val': strength})
            
            self.intervention_specs.append(mods)
            
    def sample_parameters(self, key):
        """Sample one set of model parameters.

        For each connection in the adjacency matrix the weight is drawn from a
        distribution that respects the known polarity:
          - Known polarity (+1 / -1): Uniform(0, scale) × polarity  [uniform prior]
                                      or |Normal(0, scale)| × polarity  [halfnormal prior]
          - Unknown polarity (-999):  Uniform(-scale, scale)  /  Normal(0, scale)
          - Custom LB/UB specified:   Uniform(LB, UB)  — overrides the above regardless of prior

        The scale per connection is determined by s.parameter_value_stocks (for stock
        destination variables) or s.parameter_value_aux (for auxiliaries).

        Args:
            key: JAX PRNG key.

        Returns:
            W_lin  (n_vars × n_vars):          linear weights
            W_quad (n_vars × n_vars × n_vars): interaction weights (zero if no interaction terms)
            Bias   (n_vars,):                  bias terms (currently zero)
        """
        key_lin, key_quad, key_custom = jax.random.split(key, 3)
        
        # Determine scales for each variable
        scales = jnp.zeros(self.n_vars)
        scales = scales.at[self.stock_idxs].set(self.parameter_value_stocks)
        scales = scales.at[self.aux_idxs].set(self.parameter_value_aux)
        
        # Broadcast scales to matrices/tensors
        scale_mat_lin = jnp.repeat(scales[:, None], self.n_vars, axis=1).flatten()
        scale_mat_quad = jnp.repeat(scales[:, None, None], self.n_vars * self.n_vars, axis=1).flatten() / 2.0
        
        if self.prior_type == "uniform":
            # Linear terms
            u_lin = jax.random.uniform(key_lin, shape=(self.n_vars * self.n_vars,))
            s_lin = u_lin * scale_mat_lin
            # missing links: uniform(-scale, scale)
            w_lin_raw = jnp.where(self.adj_is_missing, s_lin * 2.0 - scale_mat_lin, self.adj_polarity * s_lin)
            
            # Interaction terms
            u_quad = jax.random.uniform(key_quad, shape=(self.n_vars * self.n_vars * self.n_vars,))
            s_quad = u_quad * scale_mat_quad
            # missing links: uniform(-scale, scale)
            w_quad_raw = jnp.where(self.int_is_missing, s_quad * 2.0 - scale_mat_quad, self.int_polarity * s_quad)
            
        elif self.prior_type == "halfnormal":
            # Linear terms
            n_lin = jax.random.normal(key_lin, shape=(self.n_vars * self.n_vars,))
            # Polar links: use abs(N) * polarity. Missing links: use full N.
            w_lin_raw = jnp.where(self.adj_is_missing, n_lin * scale_mat_lin, jnp.abs(n_lin) * self.adj_polarity * scale_mat_lin)
            
            # Interaction terms
            n_quad = jax.random.normal(key_quad, shape=(self.n_vars * self.n_vars * self.n_vars,))
            w_quad_raw = jnp.where(self.int_is_missing, n_quad * scale_mat_quad, jnp.abs(n_quad) * self.int_polarity * scale_mat_quad)
        else:
            raise ValueError(f"Unknown prior_type: {self.prior_type}")

        # Per-connection LB/UB override: Uniform(LB, UB) regardless of prior type.
        # jnp.where is fully vectorised — when has_custom_bounds is all-False this is a no-op.
        u_custom = jax.random.uniform(key_custom, shape=(self.n_vars * self.n_vars,))
        w_lin_custom = self.lb_flat + u_custom * (self.ub_flat - self.lb_flat)
        w_lin_raw = jnp.where(self.has_custom_bounds, w_lin_custom, w_lin_raw)

        # Construct W_lin (Linear weights)
        w_lin_flat = jnp.where(self.adj_nonzero, w_lin_raw, 0.0)
        W_lin = w_lin_flat.reshape(self.n_vars, self.n_vars)
        
        # Construct W_quad (Quadratic/Interaction weights)
        if self.interaction_terms:
            w_quad_flat = jnp.where(self.int_nonzero, w_quad_raw, 0.0)
            W_quad = w_quad_flat.reshape(self.n_vars, self.n_vars, self.n_vars)
        else:
            W_quad = jnp.zeros((self.n_vars, self.n_vars, self.n_vars))
            
        Bias = jnp.zeros(self.n_vars)
        
        return W_lin, W_quad, Bias


    @staticmethod
    def ode_term(t, y, args):
        w_lin, w_quad, bias, constants, stock_idxs, aux_idxs, sorted_aux_idxs = args
        n_vars = w_lin.shape[0]
        
        z = jnp.zeros(n_vars)
        z = z.at[stock_idxs].set(y)
        z = z + constants 
        
        for idx in sorted_aux_idxs:
            val = bias[idx] + jnp.dot(w_lin[idx], z)
            quad_term = jnp.dot(z, jnp.dot(w_quad[idx], z))
            val = val + quad_term
            z = z.at[idx].set(val)
            
        dz_stocks = bias[stock_idxs] + jnp.dot(w_lin[stock_idxs], z)
        dz_stocks_quad = jnp.einsum('ijk,j,k->i', w_quad[stock_idxs], z, z)
        dz_stocks = dz_stocks + dz_stocks_quad
        
        return dz_stocks

    def _apply_intervention(self, x0, consts, bias, spec_indices, spec_types, spec_vals):
        def body(i, state_tuple):
            c_x0, c_consts, c_bias = state_tuple
            idx = spec_indices[i]
            val = spec_vals[i]
            type_ = spec_types[i]
            
            is_stock = (type_ == 0)
            is_const = (type_ == 1)
            is_aux = (type_ == 2)
            
            c_x0 = jnp.where(is_stock, c_x0.at[idx].add(val), c_x0)
            c_consts = jnp.where(is_const, c_consts.at[idx].add(val), c_consts)
            c_bias = jnp.where(is_aux, c_bias.at[idx].set(val), c_bias)
            
            return (c_x0, c_consts, c_bias)

        return jax.lax.fori_loop(0, len(spec_indices), body, (x0, consts, bias))

    def _prepare_initial_conditions(self, df_initial):
        N_ind = len(df_initial)
        x0_all = jnp.zeros((N_ind, self.n_vars))
        for stock in self.stocks:
            if stock in df_initial.columns:
                idx = self.var_to_idx[stock]
                x0_all = x0_all.at[:, idx].set(jnp.array(df_initial[stock].values))
        return x0_all

    def _prepare_constants(self, df_initial):
        N_ind = len(df_initial)
        consts_all = jnp.zeros((N_ind, self.n_vars))
        for const in self.constants:
            if const in df_initial.columns:
                idx = self.var_to_idx[const]
                consts_all = consts_all.at[:, idx].set(jnp.array(df_initial[const].values))
        return consts_all

    def simulate_panel_fixed(self, x0_all, consts_all, params):
        """Simulate a panel of individuals with pre-computed initial conditions.

        Args:
            x0_all   (N_ind × n_vars): initial state vectors for each individual.
            consts_all (N_ind × n_vars): constant values for each individual.
            params: tuple of (W_lin, W_quad, Bias) as returned by sample_parameters().

        Returns:
            jnp.ndarray of shape (N_ind, T, n_vars) with full-state trajectories.
        """
        W_lin, W_quad, Bias = params
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-6)
        
        def solve_one(x0, consts):
            y0 = x0[self.stock_idxs]
            args = (W_lin, W_quad, Bias, consts, self.stock_idxs, self.aux_idxs, self.sorted_aux_idxs)
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
                max_steps=10000,
                throw=False
            ) 
            
            def get_full_state(y):
                z = jnp.zeros(self.n_vars)
                z = z.at[self.stock_idxs].set(y)
                z = z + consts
                for idx in self.sorted_aux_idxs:
                    val = Bias[idx] + jnp.dot(W_lin[idx], z) + jnp.dot(z, jnp.dot(W_quad[idx], z))
                    z = z.at[idx].set(val)
                return z

            return jax.vmap(get_full_state)(sol.ys)

        return jax.vmap(solve_one)(x0_all, consts_all)

    def simulate_panel(self, df_initial, params):
        """
        Simulate a cohort of individuals over time using shared parameters.
        """
        x0_all = self._prepare_initial_conditions(df_initial)
        consts_all = self._prepare_constants(df_initial)
        return self.simulate_panel_fixed(x0_all, consts_all, params)

    def panel_to_df(self, results, as_list=False):
        """
        Converts JAX panel results into pandas DataFrames.
        
        Args:
            results (jnp.ndarray): Array of shape (N, T, n_vars)
            as_list (bool): If True, returns a list of DataFrames (one per individual).
                           If False, returns a single long-form DataFrame.
        """
        N, T, n_vars = results.shape
        cols = self.variables
        time_vals = np.array(self.t_eval)

        if as_list:
            np_results = np.array(results)
            df_list = []
            for i in range(N):
                df = pd.DataFrame(np_results[i], columns=cols)
                df['Time'] = time_vals
                df_list.append(df)
            return df_list
        else:
            # Flatten to long form
            flat_results = np.array(results).reshape(N * T, n_vars)
            df = pd.DataFrame(flat_results, columns=cols)
            df['ID'] = np.repeat(np.arange(N), T)
            df['Time'] = np.tile(time_vals, N)
            return df
