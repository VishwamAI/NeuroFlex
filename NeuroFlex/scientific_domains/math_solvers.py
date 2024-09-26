# MIT License
# 
# Copyright (c) 2024 VishwamAI
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import sympy as sp
from scipy import optimize, integrate, linalg
import warnings

class MathSolver:
    def __init__(self):
        pass

    def numerical_root_finding(self, func, initial_guess):
        """
        Find the root of a function using numerical methods.
        """
        return optimize.root(func, initial_guess)

    def symbolic_equation_solving(self, equation, variable):
        """
        Solve an equation symbolically.
        """
        return sp.solve(equation, variable)

    def numerical_optimization(self, func, initial_guess, method='BFGS'):
        """
        Find the minimum of a function using numerical optimization.
        """
        return self._optimize_with_fallback(func, initial_guess, method)

    def _optimize_with_fallback(self, func, initial_guess, method='BFGS'):
        """
        Perform optimization with fallback methods and custom error handling.
        """
        methods = [method, 'L-BFGS-B', 'TNC', 'SLSQP', 'Nelder-Mead', 'Powell', 'CG', 'trust-constr', 'dogleg', 'trust-ncg', 'COBYLA']
        max_iterations = 1000000
        best_result = None
        best_fun = float('inf')

        def log_optimization_details(m, result, w, context):
            print(f"Method: {m}")
            print(f"Function: {func.__name__}")
            print(f"Initial guess: {initial_guess}")
            print(f"Result: success={result.success if result else 'N/A'}, message={result.message if result else 'N/A'}")
            print(f"Function value at result: {result.fun if result else 'N/A'}")
            print(f"Number of iterations: {result.nit if result else 'N/A'}")
            if w:
                print(f"Warning: {w[-1].message}")
            print(f"Additional context: {context}")
            print("--------------------")

        def adjust_initial_guess(guess, scale=0.01):
            return guess + np.random.normal(0, scale, size=guess.shape)

        for m in methods:
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    options = {
                        'maxiter': max_iterations,
                        'ftol': 1e-10,
                        'gtol': 1e-10,
                        'maxls': 100,
                        'maxcor': 100
                    }

                    if m in ['trust-constr', 'dogleg', 'trust-ncg']:
                        options['gtol'] = 1e-8
                        options['xtol'] = 1e-8
                    elif m == 'COBYLA':
                        options = {'maxiter': max_iterations, 'tol': 1e-8}
                    elif m == 'Nelder-Mead':
                        options = {'maxiter': max_iterations, 'xatol': 1e-8, 'fatol': 1e-8}

                    current_guess = initial_guess
                    for attempt in range(10):  # Increased attempts to 10
                        result = optimize.minimize(func, current_guess, method=m, options=options)

                        if result.success or (result.fun < best_fun and np.isfinite(result.fun)):
                            if result.fun < best_fun:
                                best_result = result
                                best_fun = result.fun
                            log_optimization_details(m, result, w, f"Optimization successful (attempt {attempt + 1})")
                            return result

                        if w and "line search cannot locate an adequate point after maxls" in str(w[-1].message).lower():
                            log_optimization_details(m, result, w, f"Line search warning (attempt {attempt + 1})")
                            current_guess = adjust_initial_guess(current_guess)
                            options['maxls'] *= 2  # Increase maxls for the next attempt
                        elif w:
                            log_optimization_details(m, result, w, f"Unexpected warning (attempt {attempt + 1})")
                            current_guess = adjust_initial_guess(current_guess)
                        else:
                            log_optimization_details(m, result, None, f"Optimization failed without warning (attempt {attempt + 1})")
                            current_guess = adjust_initial_guess(current_guess)

                    print(f"Method {m} failed after 10 attempts. Trying next method.")

            except Exception as e:
                log_optimization_details(m, None, None, f"Error: {str(e)}")
                print("Trying next method.")

        if best_result is not None:
            print("Returning best result found.")
            return best_result

        # If all methods fail, use differential evolution as a last resort
        print("All methods failed. Using differential evolution as a last resort.")
        bounds = [(x - abs(x), x + abs(x)) for x in initial_guess]  # Create bounds based on initial guess
        result = optimize.differential_evolution(func, bounds, maxiter=max_iterations, tol=1e-10, strategy='best1bin', popsize=20)
        log_optimization_details('Differential Evolution', result, None, "Final fallback method")
        return result

    def linear_algebra_operations(self, matrix_a, matrix_b, operation='multiply'):
        """
        Perform various linear algebra operations.
        """
        if operation == 'multiply':
            return np.dot(matrix_a, matrix_b)
        elif operation == 'inverse':
            return linalg.inv(matrix_a)
        elif operation == 'eigenvalues':
            return linalg.eigvals(matrix_a)
        else:
            raise ValueError("Unsupported linear algebra operation")

    def numerical_integration(self, func, lower_bound, upper_bound):
        """
        Perform numerical integration of a function.
        """
        return integrate.quad(func, lower_bound, upper_bound)

    def symbolic_differentiation(self, expr, variable):
        """
        Perform symbolic differentiation.
        """
        return sp.diff(expr, variable)

    def solve(self, problem):
        """
        Solve a mathematical problem given as a string.

        Args:
            problem (str): A string representing the mathematical problem.

        Returns:
            str: The solution to the problem.
        """
        try:
            # Extract the equation from the problem string
            equation_str = problem.split(":")[1].strip()

            # Parse the equation
            left, right = equation_str.split("=")
            equation = sp.Eq(sp.sympify(left), sp.sympify(right))

            # Identify the variable to solve for
            variables = list(equation.free_symbols)
            if len(variables) != 1:
                raise ValueError("The equation should contain exactly one variable")
            variable = variables[0]

            # Solve the equation
            solution = sp.solve(equation, variable)

            # Format the solution
            if len(solution) == 1:
                return f"The solution is: {variable} = {solution[0]}"
            else:
                return f"The solutions are: {', '.join([f'{variable} = {sol}' for sol in solution])}"
        except Exception as e:
            return f"Error solving the problem: {str(e)}"

# Additional methods can be added here as needed
