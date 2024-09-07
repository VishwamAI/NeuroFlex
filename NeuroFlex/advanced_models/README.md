# NeuroFlex Advanced Models

This directory contains advanced mathematical models and solvers for the NeuroFlex project.

## Advanced Math Solver

The `AdvancedMathSolver` class in `advanced_math_solving.py` provides solutions for various advanced mathematical problems, including:

1. Linear Algebra
2. Calculus
3. Optimization
4. Differential Equations

### Differential Equation Solver

The differential equation solver has been recently updated with the following improvements:

- Uses the RK45 method (Runge-Kutta 4(5)) for enhanced accuracy
- Increased number of time points (1000) for a denser solution
- Improved precision settings with relative and absolute tolerances set to 1e-8

These enhancements result in more accurate and stable solutions for differential equations, addressing previous numerical discrepancies.

## Usage

To use the `AdvancedMathSolver`, import it from `advanced_math_solving.py` and create an instance:

```python
from advanced_math_solving import AdvancedMathSolver

solver = AdvancedMathSolver()
```

Then, you can solve problems by calling the `solve` method with the appropriate problem type and data:

```python
problem_type = 'differential_equations'
problem_data = {
    'function': '-2*y',
    'initial_conditions': [1],
    't_span': (0, 1)
}

solution = solver.solve(problem_type, problem_data)
```

For more detailed examples and usage instructions, please refer to the docstrings in `advanced_math_solving.py`.
