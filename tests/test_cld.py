"""
Tests for funcs/cld.py – CLD extraction from xlsx files.

Covers:
  - Variable extraction (names, types, VOI, interventions)
  - Adjacency matrix correctness
  - Interaction terms matrix with new 'Interaction terms' tab name
  - Old 'Interactions' tab is ignored (backward incompatibility is intentional)
  - extract_settings() returns the right SimpleNamespace fields
  - Error cases: disallowed characters, unknown variables in Connections
  - The built-in test_extraction() helper still passes
"""

import sys
import os
import numpy as np
import pytest
import pandas as pd

# Make the package importable without installing it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from funcs.cld import Extract


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_ADJ = np.array(
    [
        [ 0, -1,  1],  # A ← B(-), A ← C(+)
        [ 1,  0,  1],  # B ← A(+), B ← C(+)
        [ 0,  0,  0],  # C is constant
    ],
    dtype=float,
)


def extract_from(path):
    e = Extract(path)
    e.adjacency_matrix_from_kumu()
    return e


# ---------------------------------------------------------------------------
# Variable extraction
# ---------------------------------------------------------------------------

class TestVariableExtraction:
    def test_variable_names(self, simple_cld_xlsx):
        e = extract_from(simple_cld_xlsx)
        assert set(e.variables) == {"A", "B", "C"}

    def test_variable_types(self, simple_cld_xlsx):
        e = extract_from(simple_cld_xlsx)
        assert e.var_to_type["A"] == "stock"
        assert e.var_to_type["B"] == "auxiliary"
        assert e.var_to_type["C"] == "constant"

    def test_variable_of_interest(self, simple_cld_xlsx):
        e = extract_from(simple_cld_xlsx)
        assert "A" in e.variable_of_interest

    def test_intervention_variables(self, simple_cld_xlsx):
        e = extract_from(simple_cld_xlsx)
        assert "B" in e.intervention_variables
        assert "C" in e.intervention_variables
        assert "A" not in e.intervention_variables

    def test_intervention_strengths(self, simple_cld_xlsx):
        e = extract_from(simple_cld_xlsx)
        assert e.intervention_strengths["B"] == 1
        assert e.intervention_strengths["C"] == 2


# ---------------------------------------------------------------------------
# Adjacency matrix
# ---------------------------------------------------------------------------

class TestAdjacencyMatrix:
    def test_shape(self, simple_cld_xlsx):
        e = extract_from(simple_cld_xlsx)
        assert e.adjacency_matrix.shape == (3, 3)

    def test_values(self, simple_cld_xlsx):
        e = extract_from(simple_cld_xlsx)
        assert np.array_equal(e.adjacency_matrix, EXPECTED_ADJ)

    def test_df_adj_index_and_columns(self, simple_cld_xlsx):
        e = extract_from(simple_cld_xlsx)
        assert list(e.df_adj.index) == ["A", "B", "C"]
        assert list(e.df_adj.columns) == ["A", "B", "C"]

    def test_df_adj_values(self, simple_cld_xlsx):
        e = extract_from(simple_cld_xlsx)
        assert np.array_equal(e.df_adj.values, EXPECTED_ADJ)


# ---------------------------------------------------------------------------
# Interaction terms matrix
# ---------------------------------------------------------------------------

class TestInteractionTerms:
    def test_no_interaction_terms_gives_zero_matrix(self, simple_cld_xlsx):
        e = extract_from(simple_cld_xlsx)
        assert np.all(e.interactions_matrix == 0)

    def test_interaction_terms_shape(self, interaction_terms_cld_xlsx):
        e = extract_from(interaction_terms_cld_xlsx)
        n = len(e.variables)
        assert e.interactions_matrix.shape == (n, n, n)

    def test_interaction_terms_value(self, interaction_terms_cld_xlsx):
        """From1=C(2), From2=A(0), To=B(1), Type=+ → matrix[1,0,2] == 1."""
        e = extract_from(interaction_terms_cld_xlsx)
        assert e.interactions_matrix[1, 0, 2] == 1.0

    def test_interaction_terms_total_nonzero(self, interaction_terms_cld_xlsx):
        e = extract_from(interaction_terms_cld_xlsx)
        assert np.sum(np.abs(e.interactions_matrix)) == 1.0

    def test_old_tab_name_ignored(self, old_interactions_cld_xlsx):
        """A tab named 'Interactions' (old name) should be silently ignored."""
        e = extract_from(old_interactions_cld_xlsx)
        assert np.all(e.interactions_matrix == 0)


# ---------------------------------------------------------------------------
# extract_settings()
# ---------------------------------------------------------------------------

class TestExtractSettings:
    def test_returns_namespace_with_required_fields(self, simple_cld_xlsx):
        e = Extract(simple_cld_xlsx)
        s = e.extract_settings()
        for attr in ("stocks", "auxiliaries", "constants", "variables",
                     "df_adj", "interactions_matrix", "intervention_variables",
                     "variable_of_interest", "interaction_terms", "solve_analytically"):
            assert hasattr(s, attr), f"Missing attribute: {attr}"

    def test_variable_lists(self, simple_cld_xlsx):
        e = Extract(simple_cld_xlsx)
        s = e.extract_settings()
        assert s.stocks == ["A"]
        assert s.auxiliaries == ["B"]
        assert s.constants == ["C"]

    def test_solve_mode_without_interaction_terms(self, simple_cld_xlsx):
        e = Extract(simple_cld_xlsx)
        s = e.extract_settings()
        assert s.interaction_terms == 0
        assert s.solve_analytically == 1

    def test_solve_mode_with_interaction_terms(self, interaction_terms_cld_xlsx):
        e = Extract(interaction_terms_cld_xlsx)
        s = e.extract_settings()
        assert s.interaction_terms == 1
        assert s.solve_analytically == 0

    def test_df_adj_in_settings(self, simple_cld_xlsx):
        e = Extract(simple_cld_xlsx)
        s = e.extract_settings()
        assert np.array_equal(s.df_adj.values, EXPECTED_ADJ)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestErrorCases:
    def test_plus_in_variable_name_raises(self, tmp_path):
        elements = pd.DataFrame({
            "Label": ["A+B", "C"],
            "Type": ["stock", "constant"],
            "Tags": [0, 1],
            "Description": ["VOI", None],
        })
        connections = pd.DataFrame({
            "From": ["C"], "Type": ["+"], "To": ["A+B"],
        })
        path = tmp_path / "bad_var.xlsx"
        with pd.ExcelWriter(path) as w:
            elements.to_excel(w, sheet_name="Elements", index=False)
            connections.to_excel(w, sheet_name="Connections", index=False)
        e = Extract(str(path))
        with pytest.raises(Exception, match="disallowed special character"):
            e.extract_adjacency_matrix()

    def test_star_in_variable_name_raises(self, tmp_path):
        elements = pd.DataFrame({
            "Label": ["A*B", "C"],
            "Type": ["stock", "constant"],
            "Tags": [0, 1],
            "Description": ["VOI", None],
        })
        connections = pd.DataFrame({
            "From": ["C"], "Type": ["+"], "To": ["A*B"],
        })
        path = tmp_path / "star_var.xlsx"
        with pd.ExcelWriter(path) as w:
            elements.to_excel(w, sheet_name="Elements", index=False)
            connections.to_excel(w, sheet_name="Connections", index=False)
        e = Extract(str(path))
        with pytest.raises(Exception, match="disallowed special character"):
            e.extract_adjacency_matrix()

    def test_unknown_from_variable_raises(self, tmp_path):
        elements = pd.DataFrame({
            "Label": ["A", "C"],
            "Type": ["stock", "constant"],
            "Tags": [0, 1],
            "Description": ["VOI", None],
        })
        connections = pd.DataFrame({
            "From": ["C", "X"],   # X is not in Elements
            "Type": ["+", "+"],
            "To": ["A", "A"],
        })
        path = tmp_path / "bad_from.xlsx"
        with pd.ExcelWriter(path) as w:
            elements.to_excel(w, sheet_name="Elements", index=False)
            connections.to_excel(w, sheet_name="Connections", index=False)
        e = Extract(str(path))
        with pytest.raises(Exception, match="not found in Elements"):
            e.extract_adjacency_matrix()

    def test_unknown_to_variable_raises(self, tmp_path):
        elements = pd.DataFrame({
            "Label": ["A", "C"],
            "Type": ["stock", "constant"],
            "Tags": [0, 1],
            "Description": ["VOI", None],
        })
        connections = pd.DataFrame({
            "From": ["C"],
            "Type": ["+"],
            "To": ["Z"],   # Z is not in Elements
        })
        path = tmp_path / "bad_to.xlsx"
        with pd.ExcelWriter(path) as w:
            elements.to_excel(w, sheet_name="Elements", index=False)
            connections.to_excel(w, sheet_name="Connections", index=False)
        e = Extract(str(path))
        with pytest.raises(Exception, match="not found in Elements"):
            e.extract_adjacency_matrix()

    def test_non_constant_without_incoming_link_raises(self, tmp_path):
        elements = pd.DataFrame({
            "Label": ["A", "B", "C"],
            "Type": ["stock", "auxiliary", "constant"],
            "Tags": [0, 1, 2],
            "Description": ["VOI", None, None],
        })
        # B has no incoming connections → should raise
        connections = pd.DataFrame({
            "From": ["C"],
            "Type": ["+"],
            "To": ["A"],
        })
        path = tmp_path / "no_incoming.xlsx"
        with pd.ExcelWriter(path) as w:
            elements.to_excel(w, sheet_name="Elements", index=False)
            connections.to_excel(w, sheet_name="Connections", index=False)
        e = Extract(str(path))
        with pytest.raises(Exception, match="incoming links"):
            e.extract_adjacency_matrix()


# ---------------------------------------------------------------------------
# Per-connection LB / UB custom bounds
# ---------------------------------------------------------------------------

class TestCustomBounds:
    def test_no_lb_ub_columns_gives_all_nan_matrices(self, simple_cld_xlsx):
        e = extract_from(simple_cld_xlsx)
        assert np.all(np.isnan(e.lb_matrix))
        assert np.all(np.isnan(e.ub_matrix))

    def test_lb_ub_matrices_shape(self, lb_ub_cld_xlsx):
        e = extract_from(lb_ub_cld_xlsx)
        assert e.lb_matrix.shape == (3, 3)
        assert e.ub_matrix.shape == (3, 3)

    def test_lb_ub_values_for_constrained_connections(self, lb_ub_cld_xlsx):
        """C→A (adj[0,2]) and B→A (adj[0,1]) carry explicit bounds."""
        e = extract_from(lb_ub_cld_xlsx)
        # C→A (+): LB=0.10, UB=0.30
        assert e.lb_matrix[0, 2] == pytest.approx(0.10)
        assert e.ub_matrix[0, 2] == pytest.approx(0.30)
        # B→A (-): LB=-0.80, UB=-0.20
        assert e.lb_matrix[0, 1] == pytest.approx(-0.80)
        assert e.ub_matrix[0, 1] == pytest.approx(-0.20)

    def test_unconstrained_connections_remain_nan(self, lb_ub_cld_xlsx):
        """A→B and C→B have no LB/UB → their entries stay NaN."""
        e = extract_from(lb_ub_cld_xlsx)
        assert np.isnan(e.lb_matrix[1, 0])  # A→B (row B=1, col A=0)
        assert np.isnan(e.ub_matrix[1, 0])
        assert np.isnan(e.lb_matrix[1, 2])  # C→B (row B=1, col C=2)
        assert np.isnan(e.ub_matrix[1, 2])

    def test_lb_ub_propagated_to_settings(self, lb_ub_cld_xlsx):
        e = Extract(lb_ub_cld_xlsx)
        s = e.extract_settings()
        assert hasattr(s, 'lb_matrix')
        assert hasattr(s, 'ub_matrix')
        assert s.lb_matrix[0, 2] == pytest.approx(0.10)
        assert s.ub_matrix[0, 2] == pytest.approx(0.30)

    def test_lb_greater_than_ub_raises(self, tmp_path):
        elements = pd.DataFrame({
            "Label": ["A", "C"],
            "Type": ["stock", "constant"],
            "Tags": [0, 1],
            "Description": ["VOI", None],
        })
        connections = pd.DataFrame({
            "From": ["C"],
            "Type": ["+"],
            "To": ["A"],
            "LB": [0.8],
            "UB": [0.2],   # LB > UB — must raise
        })
        path = tmp_path / "bad_bounds.xlsx"
        with pd.ExcelWriter(path) as w:
            elements.to_excel(w, sheet_name="Elements", index=False)
            connections.to_excel(w, sheet_name="Connections", index=False)
        e = Extract(str(path))
        with pytest.raises(ValueError, match="LB.*must be strictly less than UB"):
            e.extract_adjacency_matrix()

    def test_lb_equal_to_ub_raises(self, tmp_path):
        elements = pd.DataFrame({
            "Label": ["A", "C"],
            "Type": ["stock", "constant"],
            "Tags": [0, 1],
            "Description": ["VOI", None],
        })
        connections = pd.DataFrame({
            "From": ["C"], "Type": ["+"], "To": ["A"],
            "LB": [0.5], "UB": [0.5],  # equal — degenerate distribution, also raises
        })
        path = tmp_path / "equal_bounds.xlsx"
        with pd.ExcelWriter(path) as w:
            elements.to_excel(w, sheet_name="Elements", index=False)
            connections.to_excel(w, sheet_name="Connections", index=False)
        e = Extract(str(path))
        with pytest.raises(ValueError, match="LB.*must be strictly less than UB"):
            e.extract_adjacency_matrix()

    def test_partial_lb_ub_ignored(self, tmp_path):
        """If only LB column is present (no UB), custom bounds are not applied."""
        elements = pd.DataFrame({
            "Label": ["A", "B", "C"],
            "Type": ["stock", "auxiliary", "constant"],
            "Tags": [0, 1, 2],
            "Description": ["VOI", None, None],
        })
        connections = pd.DataFrame({
            "From": ["C", "B", "A", "C"],
            "Type": ["+", "-", "+", "+"],
            "To":   ["A", "A", "B", "B"],
            "LB":   [0.1, -0.8, float("nan"), float("nan")],
            # No UB column at all
        })
        path = tmp_path / "only_lb.xlsx"
        with pd.ExcelWriter(path) as w:
            elements.to_excel(w, sheet_name="Elements", index=False)
            connections.to_excel(w, sheet_name="Connections", index=False)
        e = Extract(str(path))
        e.extract_adjacency_matrix()
        assert np.all(np.isnan(e.lb_matrix))  # ignored — no UB column


# ---------------------------------------------------------------------------
# Built-in test helper
# ---------------------------------------------------------------------------

class TestBuiltinTestExtraction:
    def test_test_extraction_passes(self, tmp_path, monkeypatch):
        """The built-in test_extraction() method should pass with the new tab name."""
        e = Extract(str(tmp_path / "dummy.xlsx"))   # path not used until test_extraction runs
        e.test_extraction()   # creates its own file in test_files/ and checks assertions
