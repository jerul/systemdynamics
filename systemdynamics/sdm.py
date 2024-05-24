import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import scipy
from copy import deepcopy
from sympy.parsing.sympy_parser import parse_expr
import sympy as sym

class SDM:
    def __init__(self, df_adj, interactions_matrix, s):
        self.df_adj = df_adj
        self.N = s.N
        self.interactions_matrix = interactions_matrix
        self.interaction_terms = s.interaction_terms
        self.solve_analytically = s.solve_analytically
        self.solver = s.solver
        self.t_eval = s.t_eval
        self.t_span = [s.t_eval[0], s.t_eval[-1]]
        self.stocks_and_auxiliaries = s.stocks_and_auxiliaries
        self.stocks_and_constants = s.stocks_and_constants
        self.constants = s.constants
        self.auxiliaries = s.auxiliaries
        self.simulate_interventions = s.simulate_interventions
        self.variables = s.variables
        self.stocks = s.stocks
        self.max_parameter_value = s.max_parameter_value
        self.max_parameter_value_int = s.max_parameter_value_int
        self.variable_of_interest = s.variable_of_interest
        self.intervention_variables = s.intervention_variables

        # Run tests
        self.test_vectorized_eqs()  # Call the test_vectorized_eqs function when the class is loaded

        if s.interaction_terms == 0:
            self.test_with_linear_model()  # Test whether analytical solution and numerical solution match

        if s.setting_name == "Sleep" and s.variable_of_interest == "Depressive_symptoms":
            self.test_with_sleep_depression_model() # Call the test_with_sleep_depression_model function when the class is loaded

    def get_intervention_effects(self, df_sol_per_sample, print_effects=True):
        """ Obtain intervention effects from a dataframe with model simulation results.
        """
        # Create a dictionary with intervention effects on the variable of interest
        intervention_effects = {i_v : [(df_sol_per_sample[n][i].loc[self.t_eval[-1], self.variable_of_interest] -
                                df_sol_per_sample[n][i].loc[0, self.variable_of_interest]) 
                                for n in range(self.N)] for i, i_v in enumerate(self.intervention_variables)}

        # Sort the dictionary by the mean intervention effect
        intervention_effects = dict(sorted(intervention_effects.items(),
                                        key=lambda item: np.median(item[1]), reverse=True))

        if print_effects:
            print("Intervention effect on var of interest", self.variable_of_interest, "by:")
            for i, i_v in enumerate(intervention_effects.keys()):
                print("-", i_v, ":", round(np.mean(intervention_effects[i_v]), 2),
                      "+- SD:", np.round(np.std(intervention_effects[i_v]), 2))
    
        return intervention_effects

    def sample_model_parameters(self, intervention_auxiliaries=None):
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
                solution = self.analytical_solution(self.t_eval[:, None], x0, A, b).T
            else:
                solution = solve_ivp(self.solve_sdm_linear, self.t_span, x0, args=(A, b),
                                   t_eval=self.t_eval, jac=self.jac_linear,
                                   method=self.solver, rtol=1e-6, atol=1e-6).y

        df_sol = pd.DataFrame(solution.T, columns=self.stocks_and_constants, index=self.t_eval)
        df_sol["Time"] = df_sol.index

        params_wo_stocks = {var : self.new_params[var] for var in self.auxiliaries}
        df_sol_with_aux = self.evaluate_auxiliaries(params_wo_stocks, df_sol, self.t_eval)
        return df_sol_with_aux

    def analytical_solution(self, t, x0, A, b):
        """ Analytical solution for a linear system of ODEs.
        Note this solution only works for non-singular matrices.
        That is, it only works without constant variables because these introduce rows/columns of zero in matrix A.
        """
        A_inv = np.linalg.pinv(A)  # Pseudo-inverse for singular matrices
        I = np.identity(A.shape[0])
        A_inv_b = np.matmul(A_inv, b)
        sol = np.zeros((self.t_eval.shape[0], x0.shape[0]))
        for i, t in enumerate(self.t_eval):
            exp_At = scipy.linalg.expm(A * t)
            sol[i, :] = np.matmul((exp_At - I), A_inv_b) + np.matmul(exp_At, x0)
        return sol
        #exp_At = scipy.linalg.expm(A * t)
        #return np.matmul((exp_At - I), np.matmul(A_inv, b)) + np.matmul(exp_At, x0)
    
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
        print("Test comparison with vectorized implementation passed.")

    def f_manual(self, time, x, params):
        """ Manual equations for the Sleep example for testing purposes.
        """
        # Auxiliaries
        p_a = (x[self.stocks_and_constants.index("Depressive_symptoms")] * params["Physical_activity"]["Depressive_symptoms"] + 
                params["Physical_activity"]["Intercept"])# Physical activity
        p_p = (x[self.stocks_and_constants.index("Depressive_symptoms")] * params["Proinflammatory_processes"]["Depressive_symptoms"] + 
                p_a * params["Proinflammatory_processes"]["Physical_activity"] +
                x[self.stocks_and_constants.index("Body_fat")] * params["Proinflammatory_processes"]["Body_fat"] +
                x[self.stocks_and_constants.index("Perceived_stress")] * params["Proinflammatory_processes"]["Perceived_stress"] +
                params["Proinflammatory_processes"]["Intercept"]) # Proinflammatory processes
        s_p = (x[self.stocks_and_constants.index("Body_fat")] * params["Sleep_problems"]["Body_fat"] + 
                x[self.stocks_and_constants.index("Perceived_stress")] * params["Sleep_problems"]["Perceived_stress"] +
                p_p * params["Sleep_problems"]["Proinflammatory_processes"] + params["Sleep_problems"]["Intercept"]) # Sleep problems
        
        # print("Auxiliaries: ", p_a, p_p, s_p)

        # Stocks, order: Depressive_symptoms, Childhood_adversity, Body_fat, Perceived_stress, Treatment
        d_s = (s_p * params["Depressive_symptoms"]["Sleep_problems"] + 
                x[self.stocks_and_constants.index("Childhood_adversity")] * params["Depressive_symptoms"]["Childhood_adversity"] +
                x[self.stocks_and_constants.index("Perceived_stress")] * params["Depressive_symptoms"]["Perceived_stress"] +
                x[self.stocks_and_constants.index("Treatment")] * params["Depressive_symptoms"]["Treatment"] + 
        p_p * params["Depressive_symptoms"]["Proinflammatory_processes"] + params["Depressive_symptoms"]["Intercept"])  # Depressive symptoms
        c_a = 0  # Childhod adversity
        b_f = (s_p * params["Body_fat"]["Sleep_problems"] + 
                p_a * params["Body_fat"]["Physical_activity"] + params["Body_fat"]["Intercept"])  # Body fat
        if self.interaction_terms:
            b_f += s_p * p_a * params["Body_fat"]["Physical_activity * Sleep_problems"]
        p_s = (s_p * params["Perceived_stress"]["Sleep_problems"] +
                x[self.stocks_and_constants.index("Depressive_symptoms")] * params["Perceived_stress"]["Depressive_symptoms"] +
                x[self.stocks_and_constants.index("Childhood_adversity")] * params["Perceived_stress"]["Childhood_adversity"] + 
                params["Perceived_stress"]["Intercept"])  # Perceived stress'
        t_m = (x[self.stocks_and_constants.index("Depressive_symptoms")] * params["Treatment"]["Depressive_symptoms"] + 
                params["Treatment"]["Intercept"]) # Treatment
        return np.array([d_s, c_a, b_f, p_s, t_m])

    def test_with_sleep_depression_model(self):
        """ Test whether the vectorized equations are the same as the manually implemented equations.
        """
        self.params = self.sample_model_parameters([])  # Sample model parameters
        self.new_params = self.make_equations_auxiliary_independent()  # Remove auxiliaries from the equations
        A, K, b = self.get_A_and_K_matrices()  # Get A and K matrices and intercept vector from the parameter dictionary without auxiliaries

        x0 = np.ones(len(self.stocks_and_constants), order='F') * 0.01  
        solution = solve_ivp(self.solve_sdm, self.t_span, x0, args=(A, K, b),
                                t_eval=self.t_eval, method=self.solver, rtol=1e-6, atol=1e-6)
        
        sol_test = solve_ivp(self.f_manual, self.t_span, x0, args=(self.params,), 
                                t_eval=self.t_eval, method=self.solver, rtol=1e-6, atol=1e-6)
        assert np.allclose(sol_test.y, solution.y)
        print("Test comparison with manual implementation for Sleep example passed.")

    #### Compare the results to straightforward implementation of equations
    def test_with_linear_model(self):
        """ Test whether the algebraic solution is similar to the numerical solution.
        """
        self.params = self.sample_model_parameters([])  # Sample model parameters
        self.new_params = self.make_equations_auxiliary_independent()  # Remove auxiliaries from the equations
        A, K, b = self.get_A_and_K_matrices()  # Get A and K matrices and intercept vector from the parameter dictionary without auxiliaries

        x0 = np.ones(len(self.stocks_and_constants), order='F') * 0.01  
        solution = solve_ivp(self.solve_sdm, self.t_span, x0, args=(A, K, b),
                                t_eval=self.t_eval, method=self.solver, rtol=1e-12, atol=1e-12)
        analytical_solution = self.analytical_solution(self.t_eval[:, None], x0, A, b).T
        assert np.allclose(analytical_solution, solution.y)
        print("Test comparison analytic and numerical solution for linear model passed.")
