"""
NeuroFlex Advanced Models Example

This script demonstrates the usage of advanced models in NeuroFlex.
It includes initialization, configuration, and basic operations.
"""

from NeuroFlex.advanced_models import AdvancedMathSolver, AdvancedTimeSeriesAnalysis, MultiModalLearning
from NeuroFlex.utils import data_loader

def demonstrate_advanced_math_solver():
    print("Demonstrating Advanced Math Solver:")
    solver = AdvancedMathSolver()
    problem = "Solve the differential equation: dy/dx = 2x + y"
    solution = solver.solve(problem)
    print(f"Problem: {problem}")
    print(f"Solution: {solution}")
    print()

def demonstrate_time_series_analysis():
    print("Demonstrating Advanced Time Series Analysis:")
    analyzer = AdvancedTimeSeriesAnalysis()
    time_series_data = data_loader.load_example_time_series()
    forecast = analyzer.forecast(time_series_data, steps=5)
    print(f"Time Series Data: {time_series_data[:5]}...")
    print(f"Forecast (next 5 steps): {forecast}")
    print()

def demonstrate_multi_modal_learning():
    print("Demonstrating Multi-Modal Learning:")
    multi_modal = MultiModalLearning()
    text_data = "The cat sat on the mat"
    image_data = data_loader.load_example_image()
    combined_prediction = multi_modal.predict(text=text_data, image=image_data)
    print(f"Text input: {text_data}")
    print(f"Image input: (shape: {image_data.shape})")
    print(f"Combined prediction: {combined_prediction}")
    print()

def main():
    demonstrate_advanced_math_solver()
    demonstrate_time_series_analysis()
    demonstrate_multi_modal_learning()

if __name__ == "__main__":
    main()
