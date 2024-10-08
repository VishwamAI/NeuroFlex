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

import jax
import jax.numpy as jnp
import numpy as np
from jax import random, grad, jit, vmap
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Tuple, List, Dict, Any, Union, Optional
from ..core_neural_networks.advanced_thinking import CDSTDP
from .cognitive_architecture import CognitiveArchitecture
import sympy
import logging

class TransformerModel(nn.Module):
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    vocab_size: int

    @nn.compact
    def __call__(self, x):
        embed = self.param('embed', nn.initializers.normal(stddev=0.02), (self.vocab_size, self.d_model))
        x = jnp.take(embed, x, axis=0)
        for i in range(self.num_layers):
            attention_output = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.d_model,
                name=f'attention_{i}'
            )(x, x)
            x = nn.LayerNorm(name=f'ln1_{i}')(x + attention_output)
            ff_output = nn.Dense(self.d_ff, name=f'ff1_{i}')(x)
            ff_output = nn.relu(ff_output)
            ff_output = nn.Dense(self.d_model, name=f'ff2_{i}')(ff_output)
            x = nn.LayerNorm(name=f'ln2_{i}')(x + ff_output)
        return nn.Dense(self.vocab_size, name='output')(x)

class GenerativeAIModel(nn.Module):
    features: Tuple[int, ...]
    output_dim: int
    use_transformer: bool = False
    transformer_config: Dict[str, Any] = None
    max_length: int = 512  # Maximum sequence length for text-to-text generation

    @nn.compact
    def __call__(self, x, targets=None):
        if self.use_transformer:
            transformer = TransformerModel(**self.transformer_config)
            # Cast input to integer type for the Embed layer
            x = jnp.asarray(x, dtype=jnp.int32)
            x = transformer(x)

            if targets is not None:
                # Training mode: return logits for all tokens
                return x
            else:
                # Inference mode: return logits for the last token
                return x[:, -1, :]
        else:
            for feat in self.features:
                x = nn.Dense(feat)(x)
                x = nn.relu(x)
            x = nn.Dense(self.output_dim)(x)
        return x.reshape(-1, self.output_dim)  # Ensure consistent output shape

    def generate(self, input_ids, max_new_tokens, temperature=1.0):
        # Text-to-text generation method
        for _ in range(max_new_tokens):
            # Get the predictions for the last token
            outputs = self(input_ids)
            next_token_logits = outputs

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Sample from the distribution
            next_token = jax.random.categorical(jax.random.PRNGKey(0), next_token_logits)

            # Append the new token to the input
            input_ids = jnp.concatenate([input_ids, next_token[:, None]], axis=-1)

            # Break if we exceed max_length
            if input_ids.shape[1] >= self.max_length:
                break

        return input_ids

    def simulate_consciousness(self, x):
        consciousness = jax.nn.sigmoid(x)
        threshold = 0.5
        conscious_state = jnp.where(consciousness > threshold, 1.0, 0.0)
        return conscious_state

    def generate_math_problem(self, difficulty: int, problem_rng: jax.random.PRNGKey):
        if difficulty == 1:
            a, b = jax.random.randint(problem_rng, (2,), 1, 10)
            problem = f"{a} + {b}"
            solution = a + b
        elif difficulty == 2:
            a, b = jax.random.randint(problem_rng, (2,), 1, 20)
            problem = f"{a} * {b}"
            solution = a * b
        elif difficulty == 3:
            x = sympy.Symbol('x')
            a, b, c = jax.random.randint(problem_rng, (3,), 1, 10)
            problem = f"{a}x^2 + {b}x + {c} = 0"
            solution = sympy.solve(a*x**2 + b*x + c, x)
        elif difficulty == 4:
            x, y = sympy.symbols('x y')
            a, b, c = jax.random.randint(problem_rng, (3,), 1, 5)
            problem = f"{a}x + {b}y = {c}, {b}x - {a}y = {c-a}"
            solution = sympy.solve((a*x + b*y - c, b*x - a*y - (c-a)), (x, y))
        else:
            raise ValueError("Difficulty level not supported")
        return problem, solution

class GenerativeAIFramework:
    def __init__(self, features: Tuple[int, ...], output_dim: int, input_shape: Tuple[int, ...], hidden_layers: Tuple[int, ...], learning_rate: float = 1e-3, use_transformer: bool = False, transformer_config: Optional[Dict[str, Any]] = None):
        self.model = GenerativeAIModel(features=features, output_dim=output_dim, use_transformer=use_transformer, transformer_config=transformer_config)
        self.learning_rate = learning_rate
        self.cdstdp = CDSTDP(input_shape=input_shape, output_dim=output_dim, hidden_layers=list(hidden_layers), learning_rate=learning_rate)
        self.cognitive_arch = CognitiveArchitecture({"learning_rate": learning_rate})
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate, weight_decay=1e-4)
        )
        self.input_shape = input_shape
        self.hidden_layers = hidden_layers
        self.use_transformer = use_transformer
        self.transformer_config = transformer_config

    def init_model(self, rng: Any, input_shape: Tuple[int, ...]):
        params = self.model.init(rng, jnp.ones(input_shape))
        if self.use_transformer:
            return train_state.TrainState.create(
                apply_fn=self.model.apply,
                params=params,
                tx=self.optimizer
            )
        else:
            renamed_params = {'params': {f'Dense_{i}': {'kernel': layer['kernel'], 'bias': layer['bias']}
                                         for i, layer in enumerate(params['params'].values())}}
            return train_state.TrainState.create(
                apply_fn=self.model.apply,
                params=renamed_params,
                tx=self.optimizer
            )

    def train_step(self, state: train_state.TrainState, batch: Dict[str, jnp.ndarray]):
        def loss_fn(params):
            logits = self.model.apply(params, batch['input'])
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['target']).mean()
            return loss, logits

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params)

        grads = jax.tree_map(lambda g: g / 4, grads)
        state = state.apply_gradients(grads=grads)

        conscious_state = self.model.simulate_consciousness(logits)
        print(f"conscious_state shape: {conscious_state.shape}")
        print(f"loss shape: {loss.shape}")
        feedback = self.cognitive_arch.apply_feedback(conscious_state, loss)
        print(f"feedback shape: {feedback.shape}")
        updated_params = self.cdstdp.update_weights(
            batch['input'], conscious_state, feedback
        )

        # Ensure updated_params have the same structure as state.params
        if not self.use_transformer:
            updated_params = {
                'params': {
                    layer_name: {
                        'kernel': updated_params.get(layer_name, {}).get('kernel', jnp.zeros_like(state.params['params'][layer_name]['kernel'])),
                        'bias': updated_params.get(layer_name, {}).get('bias', jnp.zeros_like(state.params['params'][layer_name]['bias']))
                    }
                    for layer_name in state.params['params'].keys()
                }
            }

        state = state.replace(params=jax.tree_map(lambda p, u: p + u, state.params, updated_params))

        return state, loss, logits

    def generate(self, state: train_state.TrainState, input_data: jnp.ndarray):
        logits = self.model.apply(state.params, input_data)
        return jax.nn.softmax(logits)

    def generate_math_problem(self, difficulty: int):
        return self.model.generate_math_problem(difficulty)

    def solve_math_problem(self, problem: str):
        try:
            import re

            def preprocess_equation(eq):
                # Replace ^ with ** for exponentiation
                eq = eq.replace('^', '**')
                # Add * before x if it's preceded by a number or closing parenthesis
                eq = re.sub(r'(\d|\))\s*x', r'\1*x', eq)
                # Add * after x if it's followed by a number or opening parenthesis
                eq = re.sub(r'x\s*(\d|\()', r'x*\1', eq)
                return eq

            # Convert equation to standard form (everything on one side, = 0)
            if '=' in problem:
                left, right = problem.split('=')
                left = preprocess_equation(left.strip())
                right = preprocess_equation(right.strip())
                expr = sympy.sympify(f"({left})-({right})", locals={'x': sympy.Symbol('x')})
            else:
                expr = sympy.sympify(preprocess_equation(problem), locals={'x': sympy.Symbol('x')})

            print(f"Parsed expression: {expr}")  # Debug print

            # Ensure the expression is in the form ax + b = 0
            x = sympy.Symbol('x')
            expr = sympy.expand(expr)

            print(f"Expanded expression: {expr}")  # Debug print

            # Check for unsupported equation types (e.g., transcendental equations)
            if any(func in str(expr) for func in ['sin', 'cos', 'tan', 'exp', 'log']):
                return jnp.zeros(2, dtype=jnp.complex64)

            # Use sympy's solve function for all equation types
            solution = sympy.solve(expr, x)

            print(f"Raw solution: {solution}")  # Debug print

            # Handle different solution types
            if isinstance(solution, dict):
                solution = list(solution.values())
            elif not isinstance(solution, list):
                solution = [solution]

            # Check for infinite solutions
            if solution == [sympy.S.Reals]:
                return jnp.array([jnp.inf, jnp.inf], dtype=jnp.complex64)

            # Convert solutions to complex numbers and handle special cases
            complex_solutions = []
            for sol in solution:
                if isinstance(sol, sympy.Expr):
                    complex_sol = complex(sol.evalf())
                elif isinstance(sol, (int, float, sympy.Rational)):
                    complex_sol = complex(float(sol))
                else:
                    complex_sol = sol
                complex_solutions.append(complex_sol)

            print(f"Complex solutions: {complex_solutions}")  # Debug print

            # Sort solutions: real solutions first (negative, then zero, then positive), then complex solutions
            def sort_key(x):
                if abs(x.imag) < 1e-10:  # Real solution
                    return (0, x.real < 0, abs(x.real))
                else:  # Complex solution
                    return (1, abs(x.imag), abs(x.real))

            complex_solutions.sort(key=sort_key)

            # Handle linear equations specifically
            if len(complex_solutions) == 1 and abs(complex_solutions[0].imag) < 1e-10:
                return jnp.array([complex_solutions[0], 0], dtype=jnp.complex64)

            # Ensure we always return exactly two solutions
            if len(complex_solutions) == 0:
                jax_solution = jnp.zeros(2, dtype=jnp.complex64)
            elif len(complex_solutions) == 1:
                jax_solution = jnp.array([complex_solutions[0], 0], dtype=jnp.complex64)
            else:
                jax_solution = jnp.array(complex_solutions[:2], dtype=jnp.complex64)

            # Ensure consistent ordering for complex conjugate pairs
            if jnp.isclose(jax_solution[0], jax_solution[1].conj()):
                if jax_solution[0].imag > 0:
                    jax_solution = jnp.array([jax_solution[0], jax_solution[0].conj()], dtype=jnp.complex64)
                else:
                    jax_solution = jnp.array([jax_solution[1], jax_solution[1].conj()], dtype=jnp.complex64)

            print(f"Final solution: {jax_solution}")  # Debug print

            return jax_solution

        except sympy.SympifyError as e:
            logging.error(f"Invalid equation format: {problem}. Error: {str(e)}")
            return jnp.zeros(2, dtype=jnp.complex64)
        except Exception as e:
            logging.error(f"Error solving math problem '{problem}': {str(e)}")
            return jnp.zeros(2, dtype=jnp.complex64)

    def evaluate_math_solution(self, problem: str, user_solution: Any, correct_solution: jnp.ndarray):
        try:
            # Convert user_solution to a JAX array of complex numbers
            user_solution = jnp.atleast_1d(jnp.array(user_solution, dtype=jnp.complex64))
            correct_solution = jnp.atleast_1d(correct_solution)

            # Ensure user_solution has the same shape as correct_solution
            if user_solution.size < correct_solution.size:
                user_solution = jnp.pad(user_solution, (0, correct_solution.size - user_solution.size), constant_values=jnp.nan)
            elif user_solution.size > correct_solution.size:
                user_solution = user_solution[:correct_solution.size]

            # Function to check if two complex numbers are close
            def is_close(a, b, rtol=1e-5, atol=1e-8):
                return jnp.abs(a - b) <= atol + rtol * jnp.abs(b)

            # Check if solutions are close, considering both real and imaginary parts
            is_correct = jnp.any(jnp.array([
                is_close(user_sol, correct_sol)
                for user_sol in user_solution
                for correct_sol in correct_solution
            ]))

            # If not correct, perform additional checks
            if not is_correct:
                # Handle special cases
                if jnp.all(jnp.isinf(correct_solution)):  # Infinite solutions
                    is_correct = jnp.any(jnp.isinf(user_solution))
                elif jnp.all(jnp.isnan(correct_solution)):  # No solutions
                    is_correct = jnp.any(jnp.isnan(user_solution))
                else:
                    # Additional check for equations
                    if '=' in problem:
                        left, right = problem.split('=')
                        expr = sympy.sympify(f"({left})-({right})")
                    else:
                        expr = sympy.sympify(problem)

                    symbols = list(expr.free_symbols)

                    # Check each solution
                    for sol in user_solution:
                        if jnp.isfinite(sol):
                            sol_complex = complex(sol)
                            if len(symbols) == 1:
                                substituted = expr.subs(symbols[0], sol_complex)
                            else:
                                substituted = expr.subs(dict(zip(symbols, [sol_complex for _ in symbols])))
                            if abs(complex(substituted)) < 1e-8:
                                is_correct = True
                                break

            # Additional check for linear equations
            if not is_correct and len(symbols) == 1:
                x = symbols[0]
                coeff = expr.coeff(x)
                constant = expr.subs(x, 0)
                if coeff != 0:
                    exact_solution = -constant / coeff
                    is_correct = jnp.any(is_close(user_solution, exact_solution))

            return bool(is_correct)
        except Exception as e:
            logging.error(f"Error in evaluate_math_solution: {str(e)}")
            return False

    def _compare_solutions(self, user_solution, correct_solution):
        if isinstance(user_solution, (int, float)) and isinstance(correct_solution, (int, float)):
            return jnp.isclose(user_solution, correct_solution, rtol=1e-5)
        elif isinstance(user_solution, sympy.Expr) and isinstance(correct_solution, sympy.Expr):
            return sympy.simplify(user_solution - correct_solution) == 0
        else:
            return user_solution == correct_solution

    def generate_step_by_step_solution(self, problem: str):
        try:
            expr = sympy.sympify(problem)
            steps = []

            if isinstance(expr, sympy.Equality):
                steps.append("Move all terms to one side of the equation")
                expr = expr.lhs - expr.rhs

            steps.append("Factor the expression")
            factored = sympy.factor(expr)
            steps.append(f"Factored form: {factored}")

            steps.append("Find the roots")
            solutions = sympy.solve(factored)
            steps.append(f"Solutions: {solutions}")

            return steps
        except Exception as e:
            return [f"Unable to generate step-by-step solution: {str(e)}"]

    def evaluate(self, state: train_state.TrainState, batch: Dict[str, jnp.ndarray]):
        logits = self.model.apply(state.params, batch['input'])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['target']).mean()
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch['target'])
        return loss, accuracy

def create_generative_ai_framework(features: Tuple[int, ...], output_dim: int) -> GenerativeAIFramework:
    return GenerativeAIFramework(features, output_dim)

if __name__ == "__main__":
    rng = random.PRNGKey(0)
    framework = create_generative_ai_framework((64, 32), 10)
    state = framework.init_model(rng, (1, 28, 28))

    dummy_input = random.normal(rng, (32, 28, 28))
    dummy_target = random.randint(rng, (32,), 0, 10)

    for _ in range(10):
        state, loss, _ = framework.train_step(state, {'input': dummy_input, 'target': dummy_target})
        print(f"Loss: {loss}")

    generated = framework.generate(state, dummy_input[:1])
    print(f"Generated output shape: {generated.shape}")

    conscious_state = framework.model.simulate_consciousness(generated)
    print(f"Conscious state: {conscious_state}")

    problem, solution = framework.generate_math_problem(3)
    print(f"Generated problem: {problem}")
    print(f"Solution: {solution}")

    steps = framework.generate_step_by_step_solution(problem)
    print("Step-by-step solution:")
    for step in steps:
        print(step)

    user_solution = solution[0] if isinstance(solution, list) else solution
    is_correct = framework.evaluate_math_solution(problem, user_solution, solution)
    print(f"User's solution is correct: {is_correct}")

class GAN(nn.Module):
    latent_dim: int
    generator_features: Tuple[int, ...]
    discriminator_features: Tuple[int, ...]
    output_shape: Tuple[int, ...]

    def setup(self):
        self.generator = self.Generator(self.generator_features, self.output_shape)
        self.discriminator = self.Discriminator(self.discriminator_features)

    def __call__(self, x, z, train=True):
        generated_images = self.generator(z)
        real_output = self.discriminator(x)
        fake_output = self.discriminator(generated_images)
        return generated_images, real_output, fake_output

    class Generator(nn.Module):
        features: Tuple[int, ...]
        output_shape: Tuple[int, ...]

        @nn.compact
        def __call__(self, z):
            x = z
            for feat in self.features:
                x = nn.Dense(feat)(x)
                x = nn.relu(x)
            x = nn.Dense(np.prod(self.output_shape))(x)
            x = nn.sigmoid(x)
            return x.reshape((-1,) + self.output_shape)

    @nn.compact
    class Discriminator(nn.Module):
        features: Tuple[int, ...]

        @nn.compact
        def __call__(self, x):
            x = x.reshape((x.shape[0], -1))
            for feat in self.features:
                x = nn.Dense(feat)(x)
                x = nn.leaky_relu(x, 0.2)
            x = nn.Dense(1)(x)
            return x

    def generator_loss(self, fake_output):
        return optax.sigmoid_binary_cross_entropy(fake_output, jnp.ones_like(fake_output)).mean()

    def discriminator_loss(self, real_output, fake_output):
        real_loss = optax.sigmoid_binary_cross_entropy(real_output, jnp.ones_like(real_output))
        fake_loss = optax.sigmoid_binary_cross_entropy(fake_output, jnp.zeros_like(fake_output))
        return (real_loss + fake_loss).mean()

    def sample(self, rng, num_samples):
        z = jax.random.normal(rng, (num_samples, self.latent_dim))
        return self.generator(z)
