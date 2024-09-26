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
        methods = [method, 'Nelder-Mead', 'Powell', 'CG', 'L-BFGS-B']
        for m in methods:
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    result = optimize.minimize(func, initial_guess, method=m)
                    if len(w) == 0:  # No warnings
                        return result
                    elif "line search cannot locate an adequate point after maxls" in str(w[-1].message).lower():
                        print(f"Warning in {m}: {w[-1].message}. Trying next method.")
                    else:
                        print(f"Unexpected warning in {m}: {w[-1].message}. Trying next method.")
            except Exception as e:
                if "ABNORMAL_TERMINATION_IN_LNSRCH" in str(e):
                    print(f"LNSRCH termination in {m}. Trying next method.")
                else:
                    print(f"Unexpected error in {m}: {str(e)}. Trying next method.")

        # If all methods fail, return the best result so far
        return optimize.minimize(func, initial_guess, method='Nelder-Mead')

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
