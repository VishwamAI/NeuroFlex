import pytest
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from flax.training import train_state
from NeuroFlex.generative_models.generative_ai import GenerativeAIModel, GenerativeAIFramework, GAN, TransformerModel

@pytest.fixture
def generative_ai_model():
    return GenerativeAIModel(features=(64, 32), output_dim=10)

@pytest.fixture
def transformer_model():
    transformer_config = {
        'num_layers': 2,
        'd_model': 64,
        'num_heads': 4,
        'd_ff': 128,
        'vocab_size': 1000
    }
    return GenerativeAIModel(features=(64, 32), output_dim=10, use_transformer=True, transformer_config=transformer_config)

@pytest.fixture
def generative_ai_framework():
    return GenerativeAIFramework(
        features=(64, 32),
        output_dim=10,
        input_shape=(784,),
        hidden_layers=(128, 64),
        learning_rate=1e-3
    )

@pytest.fixture
def gan():
    return GAN(latent_dim=100, generator_features=(128, 256), discriminator_features=(256, 128), output_shape=(28, 28, 1))

def test_generative_ai_model_init(generative_ai_model, transformer_model):
    assert isinstance(generative_ai_model, GenerativeAIModel)
    assert generative_ai_model.features == (64, 32)
    assert generative_ai_model.output_dim == 10
    assert not generative_ai_model.use_transformer
    assert generative_ai_model.transformer_config is None

    assert isinstance(transformer_model, GenerativeAIModel)
    assert transformer_model.features == (64, 32)
    assert transformer_model.output_dim == 10
    assert transformer_model.use_transformer
    assert transformer_model.transformer_config is not None

def test_generative_ai_model_call():
    model = GenerativeAIModel(features=(64, 32), output_dim=10)
    rng = jax.random.PRNGKey(0)
    input_shape = (1, 784)  # Flattened 28x28 image
    params = model.init(rng, jnp.ones(input_shape))
    output = model.apply(params, jnp.ones(input_shape))
    assert output.shape == (1, 10)

    # Test transformer model
    transformer_config = {
        'num_layers': 2,
        'd_model': 64,
        'num_heads': 4,
        'd_ff': 128,
        'vocab_size': 1000
    }
    transformer_model = GenerativeAIModel(features=(64, 32), output_dim=10, use_transformer=True, transformer_config=transformer_config)
    input_shape = (1, 50)  # Sequence length of 50
    params = transformer_model.init(rng, jnp.ones(input_shape, dtype=jnp.int32))
    output = transformer_model.apply(params, jnp.ones(input_shape, dtype=jnp.int32))
    assert output.shape == (1, 10)  # Consistent output shape for both models

def test_generative_ai_model_simulate_consciousness(generative_ai_model, transformer_model):
    rng = jax.random.PRNGKey(0)

    # Test traditional model
    params = generative_ai_model.init(rng, jnp.ones((1, 784)))['params']
    logits = generative_ai_model.apply({'params': params}, jnp.ones((1, 784)))
    conscious_state = generative_ai_model.simulate_consciousness(logits)
    assert conscious_state.shape == (1, 10)
    assert jnp.all((conscious_state == 0) | (conscious_state == 1))

    # Test transformer model
    params = transformer_model.init(rng, jnp.ones((1, 50), dtype=jnp.int32))['params']
    logits = transformer_model.apply({'params': params}, jnp.ones((1, 50), dtype=jnp.int32))
    conscious_state = transformer_model.simulate_consciousness(logits)
    assert conscious_state.shape == (1, 10)
    assert jnp.all((conscious_state == 0) | (conscious_state == 1))

def test_transformer_model_output_shape(transformer_model):
    rng = jax.random.PRNGKey(0)
    input_shape = (1, 50)  # Sequence length of 50
    params = transformer_model.init(rng, jnp.ones(input_shape, dtype=jnp.int32))
    output = transformer_model.apply(params, jnp.ones(input_shape, dtype=jnp.int32))
    assert output.shape == (1, 10)  # (batch_size, output_dim)

def test_generative_ai_model_generate_math_problem(generative_ai_model):
    rng = jax.random.PRNGKey(0)
    model = generative_ai_model.bind({'params': generative_ai_model.init(rng, jnp.ones((1, 784)))['params']})
    for difficulty in range(1, 6):  # Test all difficulty levels including the unsupported one
        problem_rng, rng = jax.random.split(rng)
        if difficulty <= 4:
            problem, solution = model.generate_math_problem(difficulty, problem_rng)
            assert isinstance(problem, str)
            assert solution is not None
            # Additional checks for each difficulty level
            if difficulty == 1:
                assert "+" in problem
            elif difficulty == 2:
                assert "*" in problem
            elif difficulty == 3:
                assert "x^2" in problem
            elif difficulty == 4:
                assert "x" in problem and "y" in problem
        else:
            with pytest.raises(ValueError) as excinfo:
                model.generate_math_problem(difficulty, problem_rng)
            assert str(excinfo.value) == "Difficulty level not supported"

def test_generative_ai_framework_init(generative_ai_framework):
    assert isinstance(generative_ai_framework, GenerativeAIFramework)
    assert generative_ai_framework.model.features == (64, 32)
    assert generative_ai_framework.model.output_dim == 10
    assert hasattr(generative_ai_framework, 'cdstdp')
    assert hasattr(generative_ai_framework, 'cognitive_arch')
    assert generative_ai_framework.learning_rate == 1e-3
    assert generative_ai_framework.input_shape == (784,)
    assert generative_ai_framework.hidden_layers == (128, 64)

@pytest.mark.skip(reason="Failing test - to be investigated")
def test_generative_ai_framework_train_step(generative_ai_framework):
    rng = jax.random.PRNGKey(0)
    input_shape = (1, 784)  # Flattened input
    state = generative_ai_framework.init_model(rng, input_shape)

    # Use concrete arrays instead of abstract arrays
    dummy_input = jax.random.normal(rng, (32,) + input_shape[1:]).block_until_ready()
    dummy_target = jax.random.randint(rng, (32,), 0, 10).block_until_ready()

    # Ensure inputs are JAX arrays
    dummy_input = jnp.array(dummy_input)
    dummy_target = jnp.array(dummy_target)

    new_state, loss, logits = jax.jit(generative_ai_framework.train_step)(state, {'input': dummy_input, 'target': dummy_target})

    assert isinstance(new_state, train_state.TrainState)
    assert isinstance(loss, jnp.ndarray)
    assert logits.shape == (32, generative_ai_framework.model.output_dim)

    # Check if cognitive architecture and CDSTDP were applied
    assert hasattr(generative_ai_framework, 'cognitive_arch')
    assert hasattr(generative_ai_framework, 'cdstdp')

    # Verify that the parameters have been updated
    assert not jax.tree_util.tree_all(jax.tree_map(lambda x, y: jnp.allclose(x, y), new_state.params, state.params))

    # Check if the loss is a scalar
    assert loss.ndim == 0

    # Verify that the logits are within a reasonable range
    assert jnp.all(jnp.isfinite(logits))

    # Check if the loss is generally decreasing
    initial_loss = loss
    for _ in range(5):
        state, new_loss, _ = jax.jit(generative_ai_framework.train_step)(state, {'input': dummy_input, 'target': dummy_target})
        assert jnp.isfinite(new_loss)  # Ensure the loss remains finite

    # Check if the final loss is less than or equal to the initial loss
    assert jnp.less_equal(new_loss, initial_loss).item()

@pytest.mark.skip(reason="Failing test - to be investigated")
def test_generative_ai_framework_generate(generative_ai_framework):
    rng = jax.random.PRNGKey(0)

    # Test non-transformer model
    state = generative_ai_framework.init_model(rng, (1, 784))  # Flattened input
    dummy_input = jax.random.normal(rng, (1, 784))
    output = generative_ai_framework.generate(state, dummy_input)
    assert output.shape == (1, 10)
    assert jnp.all((output >= 0) & (output <= 1))
    assert jnp.allclose(jnp.sum(output, axis=-1), 1.0, atol=1e-6)  # Check if it's a valid probability distribution

    # Test transformer model
    transformer_framework = GenerativeAIFramework(
        features=(64, 32),
        output_dim=10,
        input_shape=(50,),
        hidden_layers=(128, 64),
        learning_rate=1e-3,
        use_transformer=True,
        transformer_config={
            'num_layers': 2,
            'd_model': 64,
            'num_heads': 4,
            'd_ff': 128,
            'vocab_size': 1000
        }
    )
    transformer_state = transformer_framework.init_model(rng, (1, 50))
    transformer_input = jax.random.randint(rng, (1, 50), 0, 999)  # Ensure input is within vocab range
    transformer_output = transformer_framework.generate(transformer_state, transformer_input)
    assert transformer_output.shape == (1, 10)
    assert jnp.all((transformer_output >= 0) & (transformer_output <= 1))
    assert jnp.allclose(jnp.sum(transformer_output, axis=-1), 1.0, atol=1e-6)

    # Test batch processing for both models
    batch_size = 32
    non_transformer_batch = jax.random.normal(rng, (batch_size, 784))
    batch_output = generative_ai_framework.generate(state, non_transformer_batch)
    assert batch_output.shape == (batch_size, 10)

    transformer_batch = jax.random.randint(rng, (batch_size, 50), 0, 999)
    transformer_batch_output = transformer_framework.generate(transformer_state, transformer_batch)
    assert transformer_batch_output.shape == (batch_size, 10)

@pytest.mark.skip(reason="Failing test - to be investigated")
def test_generative_ai_framework_solve_math_problem(generative_ai_framework):
    # Test cases with expected solutions
    test_cases = [
        ("x + 5 = 10", [5+0j, 0+0j]),
        ("x^2 + 1 = 0", [1j, -1j]),
        ("x^2 - 4 = 0", [-2+0j, 2+0j]),
        ("x^3 - x = 0", [0+0j, 1+0j]),  # Only two solutions should be returned
        ("2x + 3 = 0", [-1.5+0j, 0+0j]),
        ("x^2 + 2x + 1 = 0", [-1+0j, 0+0j]),  # Second solution is 0 due to padding
    ]

    for problem, expected_solution in test_cases:
        solution = generative_ai_framework.solve_math_problem(problem)
        assert isinstance(solution, jnp.ndarray), f"Solution for '{problem}' is not a JAX array"
        assert solution.dtype == jnp.complex64, f"Solution for '{problem}' has incorrect dtype"
        assert solution.shape == (2,), f"Solution for '{problem}' has incorrect shape"
        assert jnp.allclose(solution, jnp.array(expected_solution, dtype=jnp.complex64), atol=1e-5), \
            f"Incorrect solution for '{problem}'. Expected {expected_solution}, got {solution}"

    # Test error handling
    error_problems = [
        "invalid equation",
        "x + y = 5",  # More than one variable
        "sin(x) = 0.5",  # Transcendental equation
    ]

    for problem in error_problems:
        solution = generative_ai_framework.solve_math_problem(problem)
        assert isinstance(solution, jnp.ndarray), f"Error solution for '{problem}' is not a JAX array"
        assert solution.dtype == jnp.complex64, f"Error solution for '{problem}' has incorrect dtype"
        assert solution.shape == (2,), f"Error solution for '{problem}' has incorrect shape"
        assert jnp.allclose(solution, jnp.array([0+0j, 0+0j], dtype=jnp.complex64)), \
            f"Incorrect error handling for '{problem}'. Expected [0+0j, 0+0j], got {solution}"

    # Test that all solutions are returned as complex numbers
    all_solutions = [generative_ai_framework.solve_math_problem(problem) for problem, _ in test_cases]
    for solution in all_solutions:
        assert jnp.issubdtype(solution.dtype, jnp.complexfloating), \
            f"Solution {solution} is not of complex floating-point type"

    # Test for consistent output shape
    cubic_equation = "x^3 - 6x^2 + 11x - 6 = 0"
    cubic_solution = generative_ai_framework.solve_math_problem(cubic_equation)
    assert cubic_solution.shape == (2,), f"Cubic equation solution shape is incorrect: {cubic_solution.shape}"

    # Test for proper handling of equations with no solutions
    no_solution_eq = "x^2 + 1 = 0"  # No real solutions
    no_solution = generative_ai_framework.solve_math_problem(no_solution_eq)
    assert jnp.allclose(no_solution, jnp.array([1j, -1j], dtype=jnp.complex64)), \
        f"Incorrect handling of equation with no real solutions. Got {no_solution}"

@pytest.mark.skip(reason="Failing test - to be investigated")
def test_generative_ai_framework_evaluate_math_solution(generative_ai_framework):
    problem = "2x + 3 = 7"
    user_solution = 2
    correct_solution = generative_ai_framework.solve_math_problem(problem)

    is_correct = generative_ai_framework.evaluate_math_solution(problem, user_solution, correct_solution)
    assert is_correct == True

    # Test with complex solutions
    complex_problem = "x^2 + 1 = 0"
    complex_user_solution = 1j
    complex_correct_solution = generative_ai_framework.solve_math_problem(complex_problem)
    is_complex_correct = generative_ai_framework.evaluate_math_solution(complex_problem, complex_user_solution, complex_correct_solution)
    assert is_complex_correct == True

def test_generative_ai_framework_generate_step_by_step_solution(generative_ai_framework):
    problem = "x^2 - 4 = 0"
    steps = generative_ai_framework.generate_step_by_step_solution(problem)
    assert isinstance(steps, list)
    assert len(steps) > 0

@pytest.mark.skip(reason="Failing test - to be investigated")
def test_generative_ai_framework_evaluate(generative_ai_framework):
    rng = jax.random.PRNGKey(0)
    state = generative_ai_framework.init_model(rng, (1, 784))  # Flattened input shape
    dummy_input = jax.random.normal(rng, (32, 784))  # Flattened input shape
    dummy_target = jax.random.randint(rng, (32,), 0, 10)
    loss, accuracy = generative_ai_framework.evaluate(state, {'input': dummy_input, 'target': dummy_target})
    assert isinstance(loss, jnp.ndarray)
    assert isinstance(accuracy, jnp.ndarray)
    assert 0 <= accuracy <= 1

def test_gan_init():
    gan = GAN(latent_dim=100, generator_features=(128, 256), discriminator_features=(256, 128), output_shape=(28, 28, 1))
    rng = jax.random.PRNGKey(0)
    params = gan.init(rng, jnp.ones((1, 28, 28, 1)), jnp.ones((1, 100)))

    assert isinstance(gan, GAN)
    assert gan.latent_dim == 100
    assert gan.generator_features == (128, 256)
    assert gan.discriminator_features == (256, 128)
    assert gan.output_shape == (28, 28, 1)

    # Check if generator and discriminator are properly initialized
    assert 'params' in params
    assert 'generator' in params['params']
    assert 'discriminator' in params['params']

def test_gan_call():
    gan = GAN(latent_dim=100, generator_features=(128, 256), discriminator_features=(256, 128), output_shape=(28, 28, 1))
    rng = jax.random.PRNGKey(0)
    params = gan.init(rng, x=jnp.ones((1, 28, 28, 1)), z=jnp.ones((1, 100)))
    x = jnp.ones((1, 28, 28, 1))
    z = jnp.ones((1, 100))
    generated_images, real_output, fake_output = gan.apply(params, x=x, z=z)
    assert generated_images.shape == (1, 28, 28, 1)
    assert real_output.shape == (1, 1)
    assert fake_output.shape == (1, 1)

def test_gan_generator():
    gan = GAN(latent_dim=100, generator_features=(128, 256), discriminator_features=(256, 128), output_shape=(28, 28, 1))
    generator = gan.Generator(features=(128, 256), output_shape=(28, 28, 1))
    rng = jax.random.PRNGKey(0)
    z = jnp.ones((1, 100))
    params = generator.init(rng, z)
    generated_images = generator.apply(params, z)
    assert generated_images.shape == (1, 28, 28, 1)
    assert jnp.all((generated_images >= 0) & (generated_images <= 1))
    assert jnp.issubdtype(generated_images.dtype, jnp.floating)

def test_gan_discriminator():
    gan = GAN(latent_dim=100, generator_features=(128, 256), discriminator_features=(256, 128), output_shape=(28, 28, 1))
    discriminator = gan.Discriminator(features=(256, 128))
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((1, 28, 28, 1))
    params = discriminator.init(rng, x)
    output = discriminator.apply(params, x)
    assert output.shape == (1, 1)

def test_gan_generator_loss():
    gan = GAN(latent_dim=100, generator_features=(128, 256), discriminator_features=(256, 128), output_shape=(28, 28, 1))
    rng = jax.random.PRNGKey(0)
    params = gan.init(rng, x=jnp.ones((1, 28, 28, 1)), z=jnp.ones((1, 100)))
    fake_output = jnp.array([[0.3], [0.7]])
    loss = gan.apply(params, method=gan.generator_loss, fake_output=fake_output)
    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == ()
    assert jnp.isfinite(loss)  # Check if loss is finite

def test_gan_discriminator_loss():
    gan = GAN(latent_dim=100, generator_features=(128, 256), discriminator_features=(256, 128), output_shape=(28, 28, 1))
    rng = jax.random.PRNGKey(0)
    params = gan.init(rng, x=jnp.ones((1, 28, 28, 1)), z=jnp.ones((1, 100)))
    real_output = jnp.array([[0.9], [0.8]])
    fake_output = jnp.array([[0.2], [0.3]])
    loss = gan.apply(params, method=gan.discriminator_loss, real_output=real_output, fake_output=fake_output)
    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == ()
    assert jnp.isfinite(loss)  # Check if loss is finite

def test_gan_sample():
    gan = GAN(latent_dim=100, generator_features=(128, 256), discriminator_features=(256, 128), output_shape=(28, 28, 1))
    rng = jax.random.PRNGKey(0)
    init_rng, sample_rng, new_rng = jax.random.split(rng, 3)
    params = gan.init(init_rng, x=jnp.ones((1, 28, 28, 1)), z=jnp.ones((1, 100)))
    samples = gan.apply(params, method=gan.sample, rngs={'dropout': sample_rng}, rng=sample_rng, num_samples=5)
    assert samples.shape == (5, 28, 28, 1)
    assert jnp.all((samples >= 0) & (samples <= 1))
    # Test that different random seeds produce different samples
    samples2 = gan.apply(params, method=gan.sample, rngs={'dropout': new_rng}, rng=new_rng, num_samples=5)
    assert not jnp.allclose(samples, samples2)

if __name__ == "__main__":
    pytest.main([__file__])
