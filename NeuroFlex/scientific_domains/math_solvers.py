import numpy as np
import sympy as sp
from scipy import optimize, integrate, linalg

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
        return optimize.minimize(func, initial_guess, method=method)

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

# Additional methods can be added here as needed
