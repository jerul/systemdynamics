"""
Tests for funcs/sdm.py and funcs/d2d.py.

Covers:
  - SDM initializes correctly from Extract settings
  - Index setup matches expected variable structure
  - Parameter sampling returns correct shapes and respects polarities
  - Simulation (simulate_panel_fixed) runs and returns sensible output shape
  - D2D.run_simulations() completes and returns one DataFrame per intervention per sample
  - D2D.get_intervention_effects() returns a dict keyed by VOI and intervention variable
  - Interaction-terms model runs end-to-end in D2D
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Skip the entire module if JAX is absent or incompatible with the current NumPy.
try:
    import jax
    import jax.numpy as jnp
    jnp.linspace(0.0, 1.0, 2)   # triggers the numpy version check early
    _JAX_OK = True
except Exception as _jax_err:
    _JAX_OK = False
    _JAX_ERR = str(_jax_err)

pytestmark = pytest.mark.skipif(
    not _JAX_OK,
    reason=f"JAX not available or incompatible with installed NumPy: {'' if _JAX_OK else _JAX_ERR}",
)

from funcs.cld import Extract
from funcs.sdm import SDM
from funcs.d2d import D2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_settings(xlsx_path, N=8, t_end=5.0):
    e = Extract(xlsx_path)
    s = e.extract_settings()
    s.N = N
    s.t_end = t_end
    return s


# ---------------------------------------------------------------------------
# SDM initialisation
# ---------------------------------------------------------------------------

class TestSDMInit:
    def test_n_vars(self, simple_cld_xlsx):
        s = _get_settings(simple_cld_xlsx)
        sdm = SDM(s)
        assert sdm.n_vars == 3

    def test_n_stocks_aux_const(self, simple_cld_xlsx):
        s = _get_settings(simple_cld_xlsx)
        sdm = SDM(s)
        assert sdm.n_stocks == 1
        assert sdm.n_aux == 1
        assert sdm.n_const == 1

    def test_variable_index_mapping(self, simple_cld_xlsx):
        s = _get_settings(simple_cld_xlsx)
        sdm = SDM(s)
        assert sdm.var_to_idx["A"] == 0
        assert sdm.var_to_idx["B"] == 1
        assert sdm.var_to_idx["C"] == 2

    def test_adjacency_flat_nonzero_mask(self, simple_cld_xlsx):
        """adj_nonzero should be True only for the four known connections."""
        s = _get_settings(simple_cld_xlsx)
        sdm = SDM(s)
        # 3×3 flat → indices: (row*3+col)
        # A←B: [0,1]=1, A←C: [0,2]=2, B←A: [1,0]=3, B←C: [1,2]=5
        expected_nonzero = {1, 2, 3, 5}
        actual_nonzero = set(np.where(np.array(sdm.adj_nonzero))[0].tolist())
        assert actual_nonzero == expected_nonzero

    def test_intervention_specs_count(self, simple_cld_xlsx):
        s = _get_settings(simple_cld_xlsx)
        sdm = SDM(s)
        # B and C are intervention variables → 2 specs
        assert len(sdm.intervention_specs) == 2


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------

class TestParameterSampling:
    def test_shapes(self, simple_cld_xlsx):
        s = _get_settings(simple_cld_xlsx)
        sdm = SDM(s)
        W_lin, W_quad, Bias = sdm.sample_parameters(jax.random.PRNGKey(0))
        assert W_lin.shape == (3, 3)
        assert W_quad.shape == (3, 3, 3)
        assert Bias.shape == (3,)

    def test_zero_where_no_connection(self, simple_cld_xlsx):
        """Parameters must be zero where the adjacency matrix is zero."""
        s = _get_settings(simple_cld_xlsx)
        sdm = SDM(s)
        # Average over many keys to ensure it's not just one lucky draw
        for seed in range(20):
            W_lin, _, _ = sdm.sample_parameters(jax.random.PRNGKey(seed))
            W_np = np.array(W_lin)
            # A←A, B←B, C←* and C←C are all zero in adj
            assert W_np[0, 0] == 0.0, "A←A should be zero"
            assert W_np[1, 1] == 0.0, "B←B should be zero"
            assert W_np[2, 0] == 0.0, "C←A should be zero"
            assert W_np[2, 1] == 0.0, "C←B should be zero"
            assert W_np[2, 2] == 0.0, "C←C should be zero"

    def test_polarity_positive_links(self, simple_cld_xlsx):
        """Positive-polarity links should always sample non-negative weights."""
        s = _get_settings(simple_cld_xlsx)
        sdm = SDM(s)
        for seed in range(30):
            W_lin, _, _ = sdm.sample_parameters(jax.random.PRNGKey(seed))
            W_np = np.array(W_lin)
            assert W_np[0, 2] >= 0, f"C→A (+) sampled negative weight at seed {seed}"
            assert W_np[1, 0] >= 0, f"A→B (+) sampled negative weight at seed {seed}"
            assert W_np[1, 2] >= 0, f"C→B (+) sampled negative weight at seed {seed}"

    def test_polarity_negative_links(self, simple_cld_xlsx):
        """Negative-polarity links should always sample non-positive weights."""
        s = _get_settings(simple_cld_xlsx)
        sdm = SDM(s)
        for seed in range(30):
            W_lin, _, _ = sdm.sample_parameters(jax.random.PRNGKey(seed))
            W_np = np.array(W_lin)
            assert W_np[0, 1] <= 0, f"B→A (-) sampled positive weight at seed {seed}"

    def test_halfnormal_prior(self, simple_cld_xlsx):
        s = _get_settings(simple_cld_xlsx)
        s.prior_type = "halfnormal"
        sdm = SDM(s)
        W_lin, W_quad, Bias = sdm.sample_parameters(jax.random.PRNGKey(7))
        assert W_lin.shape == (3, 3)

    def test_unknown_prior_raises(self, simple_cld_xlsx):
        s = _get_settings(simple_cld_xlsx)
        s.prior_type = "invalid_prior"
        sdm = SDM(s)
        with pytest.raises(ValueError, match="Unknown prior_type"):
            sdm.sample_parameters(jax.random.PRNGKey(0))


# ---------------------------------------------------------------------------
# Simulation (SDM)
# ---------------------------------------------------------------------------

class TestSimulation:
    def test_simulate_panel_fixed_output_shape(self, simple_cld_xlsx):
        s = _get_settings(simple_cld_xlsx, t_end=5.0)
        sdm = SDM(s)
        N_ind = 4
        x0_all = jnp.zeros((N_ind, sdm.n_vars))
        consts_all = jnp.zeros((N_ind, sdm.n_vars))
        params = sdm.sample_parameters(jax.random.PRNGKey(0))
        results = sdm.simulate_panel_fixed(x0_all, consts_all, params)
        # Shape: (N_individuals, T_steps, n_vars)
        n_timesteps = int(s.t_end) + 1
        assert results.shape == (N_ind, n_timesteps, sdm.n_vars)

    def test_simulate_panel_fixed_finite_values(self, simple_cld_xlsx):
        s = _get_settings(simple_cld_xlsx, t_end=5.0)
        sdm = SDM(s)
        x0_all = jnp.zeros((2, sdm.n_vars))
        consts_all = jnp.zeros((2, sdm.n_vars))
        params = sdm.sample_parameters(jax.random.PRNGKey(1))
        results = sdm.simulate_panel_fixed(x0_all, consts_all, params)
        assert np.all(np.isfinite(np.array(results)))

    def test_panel_to_df_long_form(self, simple_cld_xlsx):
        s = _get_settings(simple_cld_xlsx, t_end=5.0)
        sdm = SDM(s)
        N_ind = 3
        x0_all = jnp.zeros((N_ind, sdm.n_vars))
        consts_all = jnp.zeros((N_ind, sdm.n_vars))
        params = sdm.sample_parameters(jax.random.PRNGKey(2))
        results = sdm.simulate_panel_fixed(x0_all, consts_all, params)
        df = sdm.panel_to_df(results)
        n_t = int(s.t_end) + 1
        assert len(df) == N_ind * n_t
        assert "ID" in df.columns
        assert "Time" in df.columns

    def test_panel_to_df_list_form(self, simple_cld_xlsx):
        s = _get_settings(simple_cld_xlsx, t_end=5.0)
        sdm = SDM(s)
        N_ind = 2
        x0_all = jnp.zeros((N_ind, sdm.n_vars))
        consts_all = jnp.zeros((N_ind, sdm.n_vars))
        params = sdm.sample_parameters(jax.random.PRNGKey(3))
        results = sdm.simulate_panel_fixed(x0_all, consts_all, params)
        df_list = sdm.panel_to_df(results, as_list=True)
        assert len(df_list) == N_ind
        assert "Time" in df_list[0].columns


# ---------------------------------------------------------------------------
# D2D Monte Carlo
# ---------------------------------------------------------------------------

class TestD2D:
    def test_run_simulations_returns_n_samples(self, simple_cld_xlsx):
        s = _get_settings(simple_cld_xlsx, N=6)
        d2d = D2D(s)
        df_sol, param_samples = d2d.run_simulations()
        assert len(df_sol) == 6

    def test_run_simulations_one_df_per_intervention(self, simple_cld_xlsx):
        s = _get_settings(simple_cld_xlsx, N=5)
        d2d = D2D(s)
        df_sol, _ = d2d.run_simulations()
        n_interventions = len(s.intervention_variables)
        for sample in df_sol:
            assert len(sample) == n_interventions

    def test_run_simulations_df_columns(self, simple_cld_xlsx):
        s = _get_settings(simple_cld_xlsx, N=4)
        d2d = D2D(s)
        df_sol, _ = d2d.run_simulations()
        # Each df has variable columns + 'Time'
        df = df_sol[0][0]
        for var in s.variables:
            assert var in df.columns
        assert "Time" in df.columns

    def test_run_simulations_df_row_count(self, simple_cld_xlsx):
        s = _get_settings(simple_cld_xlsx, N=4, t_end=5.0)
        d2d = D2D(s)
        df_sol, _ = d2d.run_simulations()
        expected_rows = int(s.t_end) + 1
        assert len(df_sol[0][0]) == expected_rows

    def test_param_samples_keyed_by_intervention(self, simple_cld_xlsx):
        s = _get_settings(simple_cld_xlsx, N=4)
        d2d = D2D(s)
        _, param_samples = d2d.run_simulations()
        for iv in s.intervention_variables:
            assert iv in param_samples

    def test_get_intervention_effects_structure(self, simple_cld_xlsx):
        s = _get_settings(simple_cld_xlsx, N=6)
        d2d = D2D(s)
        d2d.run_simulations()
        effects = d2d.get_intervention_effects()
        assert isinstance(effects, dict)
        for voi in s.variable_of_interest:
            assert voi in effects
            for iv in s.intervention_variables:
                assert iv in effects[voi]
                assert len(effects[voi][iv]) == s.N

    def test_get_intervention_effects_are_finite(self, simple_cld_xlsx):
        s = _get_settings(simple_cld_xlsx, N=6)
        d2d = D2D(s)
        d2d.run_simulations()
        effects = d2d.get_intervention_effects()
        for voi in effects:
            for iv in effects[voi]:
                assert all(np.isfinite(v) for v in effects[voi][iv])

    def test_d2d_with_interaction_terms_runs(self, interaction_terms_cld_xlsx):
        """End-to-end D2D run with interaction terms enabled."""
        s = _get_settings(interaction_terms_cld_xlsx, N=5)
        assert s.interaction_terms == 1
        d2d = D2D(s)
        df_sol, _ = d2d.run_simulations()
        assert len(df_sol) == 5

    def test_d2d_with_custom_bounds_runs(self, lb_ub_cld_xlsx):
        s = _get_settings(lb_ub_cld_xlsx, N=5)
        d2d = D2D(s)
        df_sol, _ = d2d.run_simulations()
        assert len(df_sol) == 5

    def test_d2d_interaction_terms_w_quad_nonzero(self, interaction_terms_cld_xlsx):
        """When interaction terms are present, W_quad should have nonzero entries."""
        s = _get_settings(interaction_terms_cld_xlsx, N=4)
        d2d = D2D(s)
        for seed in range(10):
            W_lin, W_quad, _ = d2d.sample_parameters(jax.random.PRNGKey(seed))
            if np.any(np.array(W_quad) != 0):
                return  # At least one sample has nonzero W_quad → test passes
        pytest.fail("W_quad was zero in all 10 samples despite interaction terms being set")


# ---------------------------------------------------------------------------
# Per-connection custom bounds (LB / UB)
# ---------------------------------------------------------------------------

class TestCustomBoundsSampling:
    """
    Model (same as lb_ub_cld_xlsx fixture):
      Variables: A (stock/VOI, idx=0), B (auxiliary/intervention, idx=1),
                 C (constant/intervention, idx=2)
      Custom-bound connections:
        C→A (+): W_lin[0,2] ∈ [0.10, 0.30]
        B→A (-): W_lin[0,1] ∈ [-0.80, -0.20]
      Default connections:
        A→B (+): W_lin[1,0] ≥ 0  (default uniform polarity-scaled)
        C→B (+): W_lin[1,2] ≥ 0  (default uniform polarity-scaled)
    """

    N_SEEDS = 50  # enough draws to be statistically confident

    def test_has_custom_bounds_mask(self, lb_ub_cld_xlsx):
        s = _get_settings(lb_ub_cld_xlsx)
        sdm = SDM(s)
        mask = np.array(sdm.has_custom_bounds).reshape(3, 3)
        # C→A (row=0, col=2) and B→A (row=0, col=1) are custom
        assert mask[0, 2], "C→A should be flagged as custom bounds"
        assert mask[0, 1], "B→A should be flagged as custom bounds"
        # All others must NOT be custom
        assert not mask[1, 0], "A→B should NOT be custom bounds"
        assert not mask[1, 2], "C→B should NOT be custom bounds"
        assert not mask[0, 0]
        assert not mask[1, 1]
        assert not mask[2, 2]

    def test_custom_bounded_connections_stay_within_lb_ub(self, lb_ub_cld_xlsx):
        s = _get_settings(lb_ub_cld_xlsx)
        sdm = SDM(s)
        for seed in range(self.N_SEEDS):
            W_lin, _, _ = sdm.sample_parameters(jax.random.PRNGKey(seed))
            W = np.array(W_lin)
            # C→A must be in [0.10, 0.30]
            assert 0.10 <= W[0, 2] <= 0.30, (
                f"seed={seed}: W_lin[A,C]={W[0,2]:.4f} outside [0.10, 0.30]"
            )
            # B→A must be in [-0.80, -0.20]
            assert -0.80 <= W[0, 1] <= -0.20, (
                f"seed={seed}: W_lin[A,B]={W[0,1]:.4f} outside [-0.80, -0.20]"
            )

    def test_unconstrained_connections_keep_default_polarity(self, lb_ub_cld_xlsx):
        """Connections without LB/UB still respect their polarity."""
        s = _get_settings(lb_ub_cld_xlsx)
        sdm = SDM(s)
        for seed in range(self.N_SEEDS):
            W_lin, _, _ = sdm.sample_parameters(jax.random.PRNGKey(seed))
            W = np.array(W_lin)
            assert W[1, 0] >= 0, f"seed={seed}: A→B (+) sampled negative {W[1,0]:.4f}"
            assert W[1, 2] >= 0, f"seed={seed}: C→B (+) sampled negative {W[1,2]:.4f}"

    def test_no_custom_bounds_model_unaffected(self, simple_cld_xlsx):
        """A model without LB/UB should behave identically to the original logic."""
        s = _get_settings(simple_cld_xlsx)
        sdm = SDM(s)
        assert not np.any(np.array(sdm.has_custom_bounds)), \
            "Simple model should have no custom bounds"
        # Polarity constraints still hold
        for seed in range(self.N_SEEDS):
            W_lin, _, _ = sdm.sample_parameters(jax.random.PRNGKey(seed))
            W = np.array(W_lin)
            assert W[0, 2] >= 0   # C→A (+)
            assert W[0, 1] <= 0   # B→A (-)
