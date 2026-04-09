"""
Shared pytest fixtures for the D2D test suite.

Model used across tests
-----------------------
Variables: A (stock, VOI), B (auxiliary, intervention), C (constant, intervention)
Connections:
  C → A  (+)
  B → A  (-)
  A → B  (+)
  C → B  (+)

Expected adjacency matrix (row=to, col=from, order A/B/C):
         A    B    C
    A  [ 0,  -1,   1]
    B  [ 1,   0,   1]
    C  [ 0,   0,   0]

For the interaction-terms fixture an extra sheet 'Interaction terms' is added:
  From1=C, From2=A, Type=+, To=B
  → interactions_matrix[B=1, A=0, C=2] = 1
"""

import pytest
import pandas as pd


def _make_elements():
    return pd.DataFrame({
        "Label": ["A", "B", "C"],
        "Type": ["stock", "auxiliary", "constant"],
        "Tags": [0, 1, 2],           # B and C are intervention variables
        "Description": ["VOI", None, None],
    })


def _make_connections():
    return pd.DataFrame({
        "From": ["C", "B", "A", "C"],
        "Type": ["+", "-", "+", "+"],
        "To":   ["A", "A", "B", "B"],
    })


def _make_interaction_terms():
    return pd.DataFrame({
        "From1": ["C"],
        "From2": ["A"],
        "Type": ["+"],
        "To": ["B"],
    })


@pytest.fixture
def simple_cld_xlsx(tmp_path):
    """Minimal CLD xlsx *without* an 'Interaction terms' sheet."""
    path = tmp_path / "simple_cld.xlsx"
    with pd.ExcelWriter(path) as writer:
        _make_elements().to_excel(writer, sheet_name="Elements", index=False)
        _make_connections().to_excel(writer, sheet_name="Connections", index=False)
    return str(path)


@pytest.fixture
def interaction_terms_cld_xlsx(tmp_path):
    """Minimal CLD xlsx *with* an 'Interaction terms' sheet."""
    path = tmp_path / "interaction_terms_cld.xlsx"
    with pd.ExcelWriter(path) as writer:
        _make_elements().to_excel(writer, sheet_name="Elements", index=False)
        _make_connections().to_excel(writer, sheet_name="Connections", index=False)
        _make_interaction_terms().to_excel(writer, sheet_name="Interaction terms", index=False)
    return str(path)


@pytest.fixture
def lb_ub_cld_xlsx(tmp_path):
    """CLD xlsx where two connections carry explicit LB/UB bounds.

    Custom-bound connections:
      C → A (+):  LB=0.10, UB=0.30   (positive bounds)
      B → A (-):  LB=-0.80, UB=-0.20  (negative bounds)
    Unconstrained connections (no LB/UB):
      A → B (+)
      C → B (+)
    """
    elements = _make_elements()
    connections = pd.DataFrame({
        "From": ["C",   "B",    "A",          "C"],
        "Type": ["+",   "-",    "+",           "+"],
        "To":   ["A",   "A",    "B",           "B"],
        "LB":   [0.10,  -0.80,  float("nan"),  float("nan")],
        "UB":   [0.30,  -0.20,  float("nan"),  float("nan")],
    })
    path = tmp_path / "lb_ub_cld.xlsx"
    with pd.ExcelWriter(path) as writer:
        elements.to_excel(writer, sheet_name="Elements", index=False)
        connections.to_excel(writer, sheet_name="Connections", index=False)
    return str(path)


@pytest.fixture
def old_interactions_cld_xlsx(tmp_path):
    """xlsx with the *old* 'Interactions' tab name (should produce zero interaction matrix)."""
    path = tmp_path / "old_interactions.xlsx"
    with pd.ExcelWriter(path) as writer:
        _make_elements().to_excel(writer, sheet_name="Elements", index=False)
        _make_connections().to_excel(writer, sheet_name="Connections", index=False)
        _make_interaction_terms().to_excel(writer, sheet_name="Interactions", index=False)
    return str(path)
