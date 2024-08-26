import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Dict, Any, Tuple, Optional
import sympy
import logging
from .generative_ai import GenerativeAIFramework, create_generative_ai_framework
from .consciousness_simulation import ConsciousnessSimulation, create_consciousness_simulation
from .sentence_piece_integration import SentencePieceIntegration

class AdvancedMathSolving(nn.Module):
    features: List[int]
    output_dim: int
    sentence_piece_model_path: Optional[str] = None

    def setup(self):
        try:
            self.generative_ai = self.create_generative_ai()
            self.consciousness = self.create_consciousness()
            self.tokenizer = self.create_tokenizer()
            logging.info("All components initialized successfully")
        except Exception as e:
            logging.error(f"Error during setup: {str(e)}")
            raise ValueError(f"Failed to initialize components: {str(e)}")

    def __call__(self, x):
        if not self.is_initialized():
            logging.error("AdvancedMathSolving components not properly initialized")
            return {"error": "Components not properly initialized"}
        try:
            return self.solve_problem(x)
        except Exception as e:
            logging.error(f"Unexpected error in __call__: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}"}

    def is_initialized(self):
        return hasattr(self, 'generative_ai') and hasattr(self, 'consciousness')

    def create_generative_ai(self):
        logging.info(f"Attempting to create GenerativeAIFramework with features={self.features} and output_dim={self.output_dim}")
        try:
            logging.debug("Calling create_generative_ai_framework")
            generative_ai = create_generative_ai_framework(tuple(self.features), self.output_dim)
            logging.debug("create_generative_ai_framework call completed")

            if generative_ai is None:
                raise ValueError("create_generative_ai_framework returned None")

            # Initialize the generative AI model
            rng = jax.random.PRNGKey(0)
            dummy_input = jnp.ones((1, self.features[0]))
            generative_ai.init_model(rng, dummy_input.shape)

            logging.info("GenerativeAIFramework initialized successfully")
            return generative_ai
        except Exception as e:
            logging.error(f"Error initializing GenerativeAIFramework: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to initialize GenerativeAIFramework: {str(e)}") from e

    def create_consciousness(self):
        try:
            consciousness = ConsciousnessSimulation(
                features=self.features,
                output_dim=self.output_dim,
                working_memory_size=64,  # Default value, can be made configurable
                attention_heads=4  # Default value, can be made configurable
            )
            rng = jax.random.PRNGKey(0)
            dummy_input = jnp.ones((1, self.features[0]))  # Use the first feature dimension
            params = consciousness.init(rng, dummy_input)
            consciousness = consciousness.bind(params)

            logging.info("ConsciousnessSimulation initialized successfully")
            return consciousness
        except Exception as e:
            logging.error(f"Error initializing ConsciousnessSimulation: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to initialize ConsciousnessSimulation: {str(e)}")

    def create_tokenizer(self):
        if self.sentence_piece_model_path:
            try:
                tokenizer = SentencePieceIntegration(self.sentence_piece_model_path)
                logging.info("SentencePieceIntegration initialized successfully")
                return tokenizer
            except Exception as e:
                logging.error(f"Error initializing SentencePiece: {str(e)}")
                logging.warning("Continuing without tokenizer")
        return None

    def solve_problem(self, problem: str) -> Dict[str, Any]:
        logging.info(f"Solving problem: {problem}")
        try:
            problem_embedding = self.tokenize_problem(problem, self.tokenizer)

            # Use the ConsciousnessSimulation object directly
            consciousness_state, working_memory = self.consciousness.simulate_consciousness(problem_embedding)

            solution_steps = self.generative_ai.generate_step_by_step_solution(problem)
            solution = self.generative_ai.solve_math_problem(problem)
            thought = self.consciousness.generate_thought(consciousness_state)

            return {
                "problem": problem,
                "solution": solution,
                "steps": solution_steps,
                "consciousness_state": consciousness_state,
                "working_memory": working_memory,
                "thought": thought
            }
        except Exception as e:
            logging.error(f"Error solving problem: {str(e)}")
            return {"problem": problem, "error": str(e)}

    def tokenize_problem(self, problem: str, tokenizer) -> jnp.ndarray:
        if tokenizer is not None:
            tokenized_problem = tokenizer.tokenize(problem)
            return jnp.array([tokenizer.piece_to_id(token) for token in tokenized_problem])
        return jnp.array([ord(c) for c in problem])

    def generate_problem(self, difficulty: int) -> Tuple[str, Any, Any]:
        logging.info(f"Generating problem with difficulty: {difficulty}")
        try:
            problem, solution = self.generative_ai.generate_math_problem(difficulty)
            problem_embedding = self.tokenize_problem(problem, self.tokenizer)
            consciousness_state, _ = self.consciousness.simulate_consciousness(problem_embedding)
            thought = self.consciousness.generate_thought(consciousness_state)
            return problem, solution, thought
        except Exception as e:
            logging.error(f"Error generating problem: {str(e)}")
            return f"Error: {str(e)}", None, None

    def evaluate_solution(self, problem: str, user_solution: Any, correct_solution: Any) -> Dict[str, Any]:
        logging.info(f"Evaluating solution for problem: {problem}")
        try:
            is_correct = self.generative_ai.evaluate_math_solution(problem, user_solution, correct_solution)
            user_solution_embedding = self.tokenize_problem(str(user_solution), self.tokenizer)
            consciousness_state, _ = self.consciousness.simulate_consciousness(user_solution_embedding)
            thought = self.consciousness.generate_thought(consciousness_state)
            return {
                "is_correct": is_correct,
                "thought": thought
            }
        except Exception as e:
            logging.error(f"Error evaluating solution: {str(e)}")
            return {"error": str(e)}

def create_advanced_math_solving(features: List[int], output_dim: int, sentence_piece_model_path: Optional[str] = None) -> AdvancedMathSolving:
    logging.info("Creating AdvancedMathSolving instance")
    try:
        solver = AdvancedMathSolving(features=features, output_dim=output_dim, sentence_piece_model_path=sentence_piece_model_path)
        # Initialize the solver
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, features[0]))
        params = solver.init(rng, dummy_input)
        solver = solver.bind(params)

        # Test initialization by solving a simple problem
        dummy_problem = "2 + 2"
        result = solver(dummy_problem)
        if 'error' in result:
            raise ValueError(f"Failed to initialize AdvancedMathSolving: {result['error']}")
        logging.info("AdvancedMathSolving instance created and initialized successfully")
        return solver
    except Exception as e:
        logging.error(f"Failed to create AdvancedMathSolving instance: {str(e)}")
        raise ValueError(f"AdvancedMathSolving initialization failed: {str(e)}")

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    math_solver = create_advanced_math_solving([64, 32], 10, sentence_piece_model_path="path/to/sentencepiece.model")

    # Generate a problem
    problem, solution, thought = math_solver.generate_problem(difficulty=3)
    print(f"Generated problem: {problem}")
    print(f"Correct solution: {solution}")
    print(f"Thought about the problem: {thought}")

    # Solve the problem
    result = math_solver.solve_problem(problem)
    print(f"Solved problem: {result['problem']}")
    print(f"Solution: {result['solution']}")
    print("Solution steps:")
    for step in result['steps']:
        print(f"- {step}")
    print(f"Thought about the solution: {result['thought']}")

    # Evaluate a user solution
    user_solution = solution[0] if isinstance(solution, list) else solution
    evaluation = math_solver.evaluate_solution(problem, user_solution, solution)
    print(f"User's solution is correct: {evaluation['is_correct']}")
    print(f"Thought about the evaluation: {evaluation['thought']}")
