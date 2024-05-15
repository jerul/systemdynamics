import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from copy import deepcopy
from sympy.parsing.sympy_parser import parse_expr
import sympy as sym

class SDM:
    def __init__(self, df_adj, interactions_matrix, t_eval, s):
        self.df_adj = df_adj
        self.interactions_matrix = interactions_matrix
        self.interaction_terms = s.interaction_terms
        self.solve_analytically = s.solve_analytically
        self.solver = s.solver
        self.t_eval = t_eval
        self.t_span = [t_eval[0], t_eval[-1]]
        self.stocks_and_auxiliaries = s.stocks_and_auxiliaries
        self.stocks_and_constants = s.stocks_and_constants
        self.constants = s.constants
        self.auxiliaries = s.auxiliaries
        self.simulate_interventions = s.simulate_interventions
        self.variables = s.variables
        self.stocks = s.stocks
        self.max_parameter_value = s.max_parameter_value
        self.max_parameter_value_int = s.max_parameter_value_int
        self.test_vectorized_eqs()  # Call the test_vectorized_eqs function when the class is loaded

    def sample_model_parameters(self, intervention_auxiliaries):
        """ Sample from the model parameters using a bounded uniform distribution. 
            The possible parameters are given by the adjacency and interactions matrices.
        """
        params = {var : {} for var in self.stocks_and_auxiliaries}
        num_pars = int(self.df_adj.abs().sum().sum())
        num_pars_int = int(np.abs(self.interactions_matrix).sum().sum())
        sample_pars = np.random.uniform(0, self.max_parameter_value, size=(num_pars))
        sample_pars_int = np.random.uniform(0, self.max_parameter_value_int, size=(num_pars_int))

        par_int_count = 0
        par_count = 0
    
        for i, var in enumerate(self.variables):
            # Intercept
            if var in self.stocks_and_auxiliaries:
                params[var]["Intercept"] = 0
            if self.simulate_interventions:
                if var in intervention_auxiliaries:
                    if var in self.auxiliaries:
                        params[var]["Intercept"] = 1
                    else:
                        raise Exception("Intervention auxiliary is not an auxiliary variable.")

            # Pairwise interactions
            for j, var_2 in enumerate(self.variables):
                if self.df_adj.loc[var_2, var] != 0:
                    params[var_2][var] = self.df_adj.loc[var_2, var] * sample_pars[par_count]
                    par_count += 1

                # 2nd-order interaction terms
                if self.interaction_terms:
                    for k, var_3 in enumerate(self.variables):
                        if self.interactions_matrix[k, j, i] != 0:
                            params[var_3] = {}
                            params[var_3][var_2 + " * " + var] = self.interactions_matrix[k, j, i] * sample_pars_int[par_int_count]
                            par_int_count += 1
        self.params = params
        return params

    def make_equations_auxiliary_independent(self): #, params):
        """" Create independent equations without auxiliaries.
        Input: parameter dictionary with auxiliary terms
        Output: parameter dictionary without auxiliary terms (i.e., only in terms of stocks and constants)
        """
        new_params = deepcopy(self.params)

        original_equations = {var : " + ".join([pred + " * " + str(new_params[var][pred]) if
                                                pred != "Intercept" else
                                                str(new_params[var][pred])
                                                for pred in new_params[var]]) for var in self.stocks_and_auxiliaries}

        new_equations = original_equations
        for k, var in enumerate(original_equations):
            eq = original_equations[var]

            if np.any([aux in eq for aux in self.auxiliaries]):  # If the equation contains auxiliaries
                while np.any([aux in eq for aux in self.auxiliaries]):  # Iterate until all auxiliaries are removed from the equations
                    for aux in self.auxiliaries:
                        eq = eq.replace(aux, "( " + aux + " )")
                        eq = eq.replace(aux, new_equations[aux])
                    eq_sym = sym.simplify(parse_expr(eq))
                    eq = str(eq_sym)

                eq_sym_exp = sym.expand(eq_sym)  # Expand the equation to get rid of parentheses

                new_params[var] = {str(key) : float(eq_sym_exp.as_coefficients_dict()[key]) for key in eq_sym_exp.as_coefficients_dict()}

                if "1" in new_params[var]:  # If an intercept term is present
                    new_params[var]["Intercept"] = new_params[var]["1"]  # Rename the "1" key to "Intercept"
                    new_params[var] = {key : new_params[var][key] for key in new_params[var] if key != "1"}  # Remove the "1" key
        self.new_params = new_params
        return new_params

    def get_A_and_K_matrices(self): #, new_params):
        """ Create matrices A and K from the parameter dictionary without auxiliary terms.
            Also returns the intercept terms as a vector.
        """
        params_wo_auxiliaries = {var : self.new_params[var] for var in self.stocks}  # Remove auxiliaries from the parameter dictionary

        A = np.zeros((len(self.stocks_and_constants), len(self.stocks_and_constants)), order='F')
        K = np.zeros((len(self.stocks_and_constants), len(self.stocks_and_constants), len(self.stocks_and_constants)), order='F')
        b = np.zeros(len(self.stocks_and_constants), order='F')

        for i, var in enumerate(self.stocks_and_constants):
            if var in self.stocks and "Intercept" in params_wo_auxiliaries[var]:
                b[i] = params_wo_auxiliaries[var]["Intercept"]
            else:  # Constants; no intercept term
                b[i] = 0

        for destination in params_wo_auxiliaries:
            for origin in params_wo_auxiliaries[destination]:
                if origin != "Intercept":
                    if "*" in origin:  # Interaction term
                        destination_index = self.stocks_and_constants.index(destination)
                        if "**" in origin:
                            origin_1_index = self.stocks_and_constants.index(origin.split("**")[0])
                            origin_2_index = self.stocks_and_constants.index(origin.split("**")[0])
                        else:
                            origin_split = origin.split("*")
                            origin_1_index = self.stocks_and_constants.index(origin_split[0])
                            origin_2_index = self.stocks_and_constants.index(origin_split[1])
                        K[destination_index, origin_2_index, origin_1_index] = params_wo_auxiliaries[destination][origin]

                    else:  # Not an interaction term
                        destination_index = self.stocks_and_constants.index(destination)
                        origin_index = self.stocks_and_constants.index(origin)
                        A[destination_index, origin_index] = params_wo_auxiliaries[destination][origin]
        return A, K, b

    def run_SDM(self, x0, A, K, b):
        """ Run the SDM and return a dataframe with all the variables at every time step, including auxiliaries.
        """
        if self.interaction_terms:
            solution = solve_ivp(self.solve_sdm, self.t_span, x0, args=(A, K, b),
                                t_eval=self.t_eval, jac=self.jac,
                                method=self.solver, rtol=1e-6, atol=1e-6)
        else:  # Linear system
            if self.solve_analytically: 
                A_inv = np.linalg.pinv(A)  # Pseudo-inverse for singular matrices
                I = np.identity(A.shape[0])
                solution = self.analytical_solution(self.t_eval[:, None], x0, A, b, A_inv, I).T
            else:
                solution = solve_ivp(self.solve_sdm_linear, self.t_span, x0, args=(A, b),
                                   t_eval=self.t_eval, jac=self.jac_linear,
                                   method=self.solver, rtol=1e-6, atol=1e-6)

        df_sol = pd.DataFrame(solution.y.T, columns=self.stocks_and_constants, index=self.t_eval)
        df_sol["Time"] = df_sol.index

        params_wo_stocks = {var : self.new_params[var] for var in self.auxiliaries}
        df_sol_with_aux = self.evaluate_auxiliaries(params_wo_stocks, df_sol, self.t_eval)
        return df_sol_with_aux

    def analytical_solution(self, t, x0, A, b, A_inv, I):
        """ Analytical solution for a linear system of ODEs.
        Note this solution only works for non-singular matrices.
        That is, it only works without constant variables because these introduce rows/columns of zero in matrix A.
        """
        exp_At = np.linalg.expm(A * t)
        return np.matmul((exp_At - I), np.matmul(A_inv, b)) + np.matmul(exp_At, x0)

    def solve_sdm(self, t, x, A, K, b):
        """ Solve the system of differential equations representing the SDM.
        x: vector containing the stock and constant variables
        A: matrix of coefficients for the linear terms of len(x) in both dimensions.
        K: 3rd order tensor of coefficients for the interaction terms of len(x) in all three dimensions.
        Outputs the derivative of x.
        """
        Kx = np.matmul(K, x) 
        dx_dt = np.matmul(A, x) + np.matmul(Kx, x) + b
        return dx_dt

    def solve_sdm_linear(self, t, x, A, b):
        """ Solve the linear system of differential equations representing the SDM.
        x: vector containing the stock and constant variables
        A: matrix of coefficients for the linear terms of len(x) in both dimensions.
        Outputs the derivative of x.
        """
        dx_dt = np.matmul(A, x) + b
        return dx_dt

    def evaluate_auxiliaries(self, params, df_sol, t_eval):
        """ Evaluate the auxiliary variables at each time step.
        Input: parameter dictionary with auxiliary terms, and the solution dataframe
        Output: list of auxiliary values at each time step
        """
        df_sol_with_auxiliaries = df_sol.copy()
        for aux in self.auxiliaries:
            aux_values = []
            for t in t_eval:
                aux_value = 0
                for origin in params[aux]:
                    if origin == "Intercept":
                        aux_value += params[aux][origin]
                    else:
                        if "*" in origin:  # Interaction term
                            origin_1 = origin.split("*")[0]
                            origin_2 = origin.split("*")[1]
                            aux_value += params[aux][origin] * df_sol[origin_1][int(t)] * df_sol[origin_2][int(t)]
                        else:  # Not an interaction term
                            aux_value += params[aux][origin] * df_sol[origin][int(t)]
                aux_values.append(aux_value)
            df_sol_with_auxiliaries[aux] = aux_values
        return df_sol_with_auxiliaries

    def jac(self, t, x, A, K, b):
        """ Jacobian matrix, depends on A and K, not b (which is a constant vector)
        """
        return A + 2 * np.matmul(K, x)

    def jac_linear(self, t, x, A, b):
        """ Jacobian matrix is equal to A for linear systems
        """
        return A



### TESTING ###
    def f_no_aux(self, time, x, params_wo_auxiliaries):
        """ Test the vectorized equation x' = Ax + Kxx.
        """
        eqs = []
        for var in self.stocks_and_constants:
            eq_i = 0

            if var not in self.constants:
                for pred in params_wo_auxiliaries[var]:
                    if pred == "Intercept":
                        eq_i += params_wo_auxiliaries[var][pred]
                    else:
                        if "**" in pred:  # Quadratic term
                            eq_i += x[self.stocks_and_constants.index(pred.split("*")[0])]**2 * params_wo_auxiliaries[var][pred]
                        elif "*" in pred:  # Interaction term
                            eq_i += (x[self.stocks_and_constants.index(pred.split("*")[0])] * 
                                     x[self.stocks_and_constants.index(pred.split("*")[1])] * 
                                     params_wo_auxiliaries[var][pred])
                        else:  # Linear term
                            eq_i += x[self.stocks_and_constants.index(pred)] * params_wo_auxiliaries[var][pred]
            eqs += [eq_i]
        
        return np.array(eqs)

    def test_vectorized_eqs(self):
        """ Test whether the vectorized equations are the same as the non-vectorized equations.
        """
        self.params = self.sample_model_parameters([])  # Sample model parameters
        self.new_params = self.make_equations_auxiliary_independent()  # Remove auxiliaries from the equations
        x0 = np.ones(len(self.stocks_and_constants), order='F') * 0.01  
        A, K, b = self.get_A_and_K_matrices()  # Get A and K matrices and intercept vector from the parameter dictionary without auxiliaries

        # Obtain the vectorized solution
        solution = solve_ivp(self.solve_sdm, self.t_span, x0, args=(A, K, b),
                            t_eval=self.t_eval, method=self.solver, rtol=1e-6, atol=1e-6)
        
        # Obtain the test solution
        params_wo_auxiliaries = {var : self.new_params[var] for var in self.stocks}  # Remove auxiliaries from the parameter dictionary
        sol_test = solve_ivp(self.f_no_aux, self.t_span, x0, args=(params_wo_auxiliaries,), 
                             t_eval=self.t_eval, method=self.solver, rtol=1e-6, atol=1e-6)
        #df_sol_test = pd.DataFrame(sol_test.y.T, columns=s.stocks_and_constants, index=t_eval)

        #assert ((df_sol_per_sample[-1][-1]-df_sol_test)**2).sum().sum() < 1e-15 # Check if the solutions are the same
        assert np.allclose(sol_test.y, solution.y)
