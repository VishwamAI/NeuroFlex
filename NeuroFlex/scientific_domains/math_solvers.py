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
        max_iterations = 100000000  # Further increased from 50000000
        for m in methods:
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    if m in ['trust-constr', 'dogleg', 'trust-ncg']:
                        result = optimize.minimize(func, initial_guess, method=m, options={'maxiter': max_iterations, 'gtol': 1e-26, 'xtol': 1e-26})
                    elif m == 'COBYLA':
                        result = optimize.minimize(func, initial_guess, method=m, options={'maxiter': max_iterations, 'tol': 1e-26})
                    else:
                        result = optimize.minimize(func, initial_guess, method=m, options={'maxiter': max_iterations, 'ftol': 1e-30, 'gtol': 1e-30, 'maxls': 10000000})
                    if len(w) == 0:  # No warnings
                        print(f"Optimization successful with method {m}")
                        print(f"Result: success={result.success}, message={result.message}")
                        return result
                    elif "line search cannot locate an adequate point after maxls" in str(w[-1].message).lower():
                        print(f"Warning in {m}: {w[-1].message}")
                        print(f"Context: func={func.__name__}, initial_guess={initial_guess}")
                        print(f"Result: success={result.success}, message={result.message}")
                        print(f"Function value at result: {result.fun}")
                        print(f"Number of iterations: {result.nit}")
                        print("Adjusting parameters and trying again.")
                        result = optimize.minimize(func, initial_guess, method=m, options={'maxiter': max_iterations * 2, 'maxls': 20000000, 'ftol': 1e-32, 'gtol': 1e-32})
                        print(f"Retry result: success={result.success}, message={result.message}")
                        print(f"Retry function value at result: {result.fun}")
                        print(f"Retry number of iterations: {result.nit}")
                        if result.success:
                            return result
                        print("Trying next method.")
                    else:
                        print(f"Unexpected warning in {m}: {w[-1].message}")
                        print(f"Context: func={func.__name__}, initial_guess={initial_guess}")
                        print(f"Result: success={result.success}, message={result.message}")
                        print(f"Function value at result: {result.fun}")
                        print(f"Number of iterations: {result.nit}")
                        print("Trying next method.")
            except Exception as e:
                if "ABNORMAL_TERMINATION_IN_LNSRCH" in str(e):
                    print(f"LNSRCH termination in {m}.")
                    print(f"Context: func={func.__name__}, initial_guess={initial_guess}")
                    print(f"Error details: {str(e)}")
                    print("Trying next method.")
                else:
                    print(f"Unexpected error in {m}: {str(e)}")
                    print(f"Context: func={func.__name__}, initial_guess={initial_guess}")
                    print(f"Error details: {str(e)}")
                    print("Trying next method.")

        # If all methods fail, return the best result so far using a robust method
        print("All methods failed. Using Nelder-Mead as a last resort.")
        result = optimize.minimize(func, initial_guess, method='Nelder-Mead', options={'maxiter': max_iterations * 10000, 'ftol': 1e-32, 'adaptive': True})
        print(f"Final result: success={result.success}, message={result.message}")
        print(f"Final function value at result: {result.fun}")
        print(f"Final number of iterations: {result.nit}")
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
