# NeuroFlex Advanced Models Tests

This directory contains test cases for the advanced mathematical models and solvers in the NeuroFlex project.

## Test Structure

The test suite is implemented in `test_advanced_math_solving.py` and uses the `unittest` framework. It covers various aspects of the `AdvancedMathSolver` class, including:

1. Linear Algebra
2. Calculus
3. Optimization
4. Differential Equations

## Test Cases

### Linear Algebra
- `test_solve_linear_algebra_system`: Tests solving a system of linear equations
- `test_solve_linear_algebra_eigenvalues`: Tests eigenvalue computation

### Calculus
- `test_solve_calculus`: Tests definite integral calculation

### Optimization
- `test_solve_optimization`: Tests finding the minimum of a function

### Differential Equations
- `test_solve_differential_equations`: Tests solving an ordinary differential equation

The differential equation solver test has been recently updated to account for improvements in the solver's accuracy. It now uses the RK45 method with increased precision settings.

### Error Handling
- Tests for unsupported problem types and invalid input data

## Running the Tests

To run the tests, execute the following command from the project root directory:

```
python -m unittest tests.advanced_models.test_advanced_math_solving
```

## Notes for Developers

- The differential equation solver uses the RK45 method with 1000 time points.
- Relative and absolute tolerances are set to 1e-8 for improved precision.
- The test case for differential equations uses a relative tolerance of 1e-3 for comparison to account for small numerical discrepancies.

When modifying the `AdvancedMathSolver` class, ensure that all tests pass and consider adding new tests for any additional functionality.
