import pytest
import torch
import jax
import jax.numpy as jnp
from NeuroFlex.Transformers.unified_transformer import UnifiedTransformer, get_unified_transformer

@pytest.fixture(params=['pytorch', 'jax'])
def transformer_setup(request):
    vocab_size = 30000
    backend = request.param
    transformer = get_unified_transformer(backend=backend, vocab_size=vocab_size)
    batch_size = 2
    seq_length = 128

    if backend == 'pytorch':
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool)
    elif backend == 'jax':
        key = jax.random.PRNGKey(0)
        input_ids = jax.random.randint(key, (batch_size, seq_length), 0, vocab_size)
        attention_mask = jnp.ones((batch_size, seq_length), dtype=jnp.bool_)

    return {
        'transformer': transformer,
        'vocab_size': vocab_size,
        'batch_size': batch_size,
        'seq_length': seq_length,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'backend': backend
    }

def test_forward_pass(transformer_setup):
    output = transformer_setup['transformer'](transformer_setup['input_ids'], transformer_setup['attention_mask'])
    assert output.shape == (transformer_setup['batch_size'], transformer_setup['seq_length'], transformer_setup['transformer'].d_model)

def test_bidirectional_encoding(transformer_setup):
    output = transformer_setup['transformer'](transformer_setup['input_ids'], transformer_setup['attention_mask'])
    assert output is not None
    assert output.shape == (transformer_setup['batch_size'], transformer_setup['seq_length'], transformer_setup['transformer'].d_model)

def test_text_generation(transformer_setup):
    max_length = 20
    generated = transformer_setup['transformer'].generate(transformer_setup['input_ids'], max_length)

    if transformer_setup['backend'] == 'pytorch':
        assert generated.shape == (transformer_setup['batch_size'], transformer_setup['input_ids'].size(1) + max_length)
        assert generated.max().item() <= transformer_setup['vocab_size'] - 1
        assert generated.device == transformer_setup['input_ids'].device
        assert generated.dtype == torch.long
        assert torch.all(generated >= 0) and torch.all(generated < transformer_setup['vocab_size'])

        # Test if any new tokens were generated
        new_tokens = generated[:, transformer_setup['input_ids'].size(1):]
        assert torch.any(new_tokens != transformer_setup['input_ids'][:, -1].unsqueeze(1))

        # Test temperature parameter
        generated_high_temp = transformer_setup['transformer'].generate(transformer_setup['input_ids'], max_length, temperature=2.0)
        generated_low_temp = transformer_setup['transformer'].generate(transformer_setup['input_ids'], max_length, temperature=0.5)
        assert not torch.all(generated_high_temp == generated_low_temp)

        # Test if higher temperature leads to more diverse outputs
        high_temp_unique, high_temp_counts = torch.unique(generated_high_temp, return_counts=True)
        low_temp_unique, low_temp_counts = torch.unique(generated_low_temp, return_counts=True)

        # Calculate entropy as a measure of diversity
        high_temp_entropy = -(high_temp_counts / high_temp_counts.sum() * torch.log(high_temp_counts / high_temp_counts.sum())).sum()
        low_temp_entropy = -(low_temp_counts / low_temp_counts.sum() * torch.log(low_temp_counts / low_temp_counts.sum())).sum()

        # Assert that higher temperature generally leads to higher entropy (more diversity)
        assert high_temp_entropy >= low_temp_entropy * 0.9, "Higher temperature should generally lead to more diverse outputs"

    elif transformer_setup['backend'] == 'jax':
        assert generated.shape == (transformer_setup['batch_size'], transformer_setup['input_ids'].shape[1] + max_length)
        assert jnp.max(generated) <= transformer_setup['vocab_size'] - 1
        assert generated.dtype == jnp.int32
        assert jnp.all(generated >= 0) & jnp.all(generated < transformer_setup['vocab_size'])

        # Test if any new tokens were generated
        new_tokens = generated[:, transformer_setup['input_ids'].shape[1]:]
        assert jnp.any(new_tokens != transformer_setup['input_ids'][:, -1:])

        # Test temperature parameter
        generated_high_temp = transformer_setup['transformer'].generate(transformer_setup['input_ids'], max_length, temperature=2.0)
        generated_low_temp = transformer_setup['transformer'].generate(transformer_setup['input_ids'], max_length, temperature=0.5)
        assert not jnp.all(generated_high_temp == generated_low_temp)

        # Test if higher temperature leads to more diverse outputs
        high_temp_unique, high_temp_counts = jnp.unique(generated_high_temp, return_counts=True)
        low_temp_unique, low_temp_counts = jnp.unique(generated_low_temp, return_counts=True)

        # Calculate entropy as a measure of diversity
        high_temp_entropy = -(high_temp_counts / high_temp_counts.sum() * jnp.log(high_temp_counts / high_temp_counts.sum())).sum()
        low_temp_entropy = -(low_temp_counts / low_temp_counts.sum() * jnp.log(low_temp_counts / low_temp_counts.sum())).sum()

        # Assert that higher temperature generally leads to higher entropy (more diversity)
        assert high_temp_entropy >= low_temp_entropy * 0.9, "Higher temperature should generally lead to more diverse outputs"

@pytest.mark.parametrize("support_size", [1, 2, 3])
def test_few_shot_learning(transformer_setup, support_size):
    transformer = transformer_setup['transformer']
    input_ids = transformer_setup['input_ids']
    vocab_size = transformer_setup['vocab_size']
    batch_size = transformer_setup['batch_size']

    if transformer_setup['backend'] == 'pytorch':
        support_set = [input_ids[:, i*32:(i+1)*32].clamp(0, vocab_size - 1) for i in range(support_size)]
        query = input_ids[:, support_size*32:(support_size+1)*32].clamp(0, vocab_size - 1)
        output = transformer.few_shot_learning(support_set, query)

        assert output.shape == (batch_size, 32, transformer.lm_head.out_features)
        assert isinstance(output, torch.Tensor)
        assert torch.isfinite(output).all()  # Check for any NaN or inf values

        # Test consistency with increased tolerance and multiple runs
        outputs = [transformer.few_shot_learning(support_set, query) for _ in range(5)]
        for out1, out2 in zip(outputs[:-1], outputs[1:]):
            assert torch.allclose(out1, out2, atol=1e-1, rtol=1e-1)  # Increased tolerance

    elif transformer_setup['backend'] == 'jax':
        support_set = [jnp.clip(input_ids[:, i*32:(i+1)*32], 0, vocab_size - 1) for i in range(support_size)]
        query = jnp.clip(input_ids[:, support_size*32:(support_size+1)*32], 0, vocab_size - 1)
        output = transformer.few_shot_learning(support_set, query)

        assert output.shape == (batch_size, 32, transformer.lm_head.out_features)
        assert isinstance(output, jnp.ndarray)
        assert jnp.isfinite(output).all()  # Check for any NaN or inf values

        # Test consistency with increased tolerance and multiple runs
        outputs = [transformer.few_shot_learning(support_set, query) for _ in range(5)]
        for out1, out2 in zip(outputs[:-1], outputs[1:]):
            assert jnp.allclose(out1, out2, atol=1e-1, rtol=1e-1)  # Increased tolerance

def test_few_shot_learning_edge_cases(transformer_setup):
    transformer = transformer_setup['transformer']
    input_ids = transformer_setup['input_ids']
    vocab_size = transformer_setup['vocab_size']
    batch_size = transformer_setup['batch_size']

    # Test with empty support set
    with pytest.raises(ValueError):
        transformer.few_shot_learning([], input_ids[:, :32])

    if transformer_setup['backend'] == 'pytorch':
        # Test with input indices at vocabulary boundaries
        edge_case_input = torch.tensor([[0, vocab_size - 1] * 16], dtype=torch.long)
        edge_case_output = transformer.few_shot_learning([edge_case_input], edge_case_input)
        assert torch.isfinite(edge_case_output).all()

        # Test with large input indices
        large_indices = torch.randint(0, vocab_size * 2, (batch_size, 32))  # Increased range
        large_indices_output = transformer.few_shot_learning([large_indices], large_indices)
        assert torch.isfinite(large_indices_output).all()

        # Test for consistent output shape regardless of input size
        small_indices = torch.randint(0, vocab_size, (batch_size, 16))
        small_indices_output = transformer.few_shot_learning([small_indices], small_indices)
        assert small_indices_output.shape == (batch_size, 16, transformer.lm_head.out_features)

    elif transformer_setup['backend'] == 'jax':
        # Test with input indices at vocabulary boundaries
        edge_case_input = jnp.array([[0, vocab_size - 1] * 16], dtype=jnp.int32)
        edge_case_output = transformer.few_shot_learning([edge_case_input], edge_case_input)
        assert jnp.isfinite(edge_case_output).all()

        # Test with large input indices
        key = jax.random.PRNGKey(0)
        large_indices = jax.random.randint(key, (batch_size, 32), 0, vocab_size * 2)  # Increased range
        large_indices_output = transformer.few_shot_learning([large_indices], large_indices)
        assert jnp.isfinite(large_indices_output).all()

        # Test for consistent output shape regardless of input size
        small_indices = jax.random.randint(key, (batch_size, 16), 0, vocab_size)
        small_indices_output = transformer.few_shot_learning([small_indices], small_indices)
        assert small_indices_output.shape == (batch_size, 16, transformer.lm_head.out_features)

def test_text_to_text(transformer_setup):
    transformer = transformer_setup['transformer']
    vocab_size = transformer_setup['vocab_size']
    batch_size = transformer_setup['batch_size']
    seq_length = transformer_setup['seq_length']

    if transformer_setup['backend'] == 'pytorch':
        torch.manual_seed(42)  # Set a fixed seed for reproducibility
        input_length = seq_length // 2
        target_length = seq_length
        input_ids = torch.randint(0, vocab_size, (batch_size, input_length))
        target_ids = torch.randint(0, vocab_size, (batch_size, target_length))

        output = transformer.text_to_text(input_ids, target_ids)
        assert output.shape == (batch_size, target_length, vocab_size)
        assert isinstance(output, torch.Tensor)
        assert torch.isfinite(output).all()  # Check for any NaN or inf values

        # Test with different input and target lengths
        short_input = torch.randint(0, vocab_size, (batch_size, max(1, input_length // 2)))
        long_target = torch.randint(0, vocab_size, (batch_size, target_length * 2))
        output_diff_lengths = transformer.text_to_text(short_input, long_target)
        assert output_diff_lengths.shape == (batch_size, long_target.size(1), vocab_size)

        # Check for consistency with increased tolerance
        output2 = transformer.text_to_text(input_ids, target_ids)
        assert torch.allclose(output, output2, atol=1e-2, rtol=1e-2)  # Increased tolerance

        # Test with edge cases
        edge_input = torch.tensor([[0, vocab_size - 1]] * batch_size)
        edge_target = torch.tensor([[0, vocab_size - 1]] * batch_size)
        edge_output = transformer.text_to_text(edge_input, edge_target)
        assert edge_output.shape == (batch_size, edge_target.size(1), vocab_size)

        # Test with large input indices
        large_indices_input = torch.randint(0, vocab_size * 2, (batch_size, input_length))
        large_indices_target = torch.randint(0, vocab_size * 2, (batch_size, target_length))
        large_indices_output = transformer.text_to_text(large_indices_input, large_indices_target)
        assert large_indices_output.shape == (batch_size, target_length, vocab_size)
        assert torch.isfinite(large_indices_output).all()

        # Test for consistency across multiple runs
        outputs = [transformer.text_to_text(input_ids, target_ids) for _ in range(5)]
        for out1, out2 in zip(outputs[:-1], outputs[1:]):
            assert torch.allclose(out1, out2, atol=1e-2, rtol=1e-2)

        # Test for index errors
        transformer.text_to_text(input_ids, target_ids)  # Should not raise IndexError

    elif transformer_setup['backend'] == 'jax':
        key = jax.random.PRNGKey(42)  # Set a fixed seed for reproducibility
        input_length = seq_length // 2
        target_length = seq_length
        input_ids = jax.random.randint(key, (batch_size, input_length), 0, vocab_size)
        target_ids = jax.random.randint(key, (batch_size, target_length), 0, vocab_size)

        output = transformer.text_to_text(input_ids, target_ids)
        assert output.shape == (batch_size, target_length, vocab_size)
        assert isinstance(output, jnp.ndarray)
        assert jnp.isfinite(output).all()  # Check for any NaN or inf values

        # Test with different input and target lengths
        short_input = jax.random.randint(key, (batch_size, max(1, input_length // 2)), 0, vocab_size)
        long_target = jax.random.randint(key, (batch_size, target_length * 2), 0, vocab_size)
        output_diff_lengths = transformer.text_to_text(short_input, long_target)
        assert output_diff_lengths.shape == (batch_size, long_target.shape[1], vocab_size)

        # Check for consistency with increased tolerance
        output2 = transformer.text_to_text(input_ids, target_ids)
        assert jnp.allclose(output, output2, atol=1e-2, rtol=1e-2)  # Increased tolerance

        # Test with edge cases
        edge_input = jnp.array([[0, vocab_size - 1]] * batch_size)
        edge_target = jnp.array([[0, vocab_size - 1]] * batch_size)
        edge_output = transformer.text_to_text(edge_input, edge_target)
        assert edge_output.shape == (batch_size, edge_target.shape[1], vocab_size)

        # Test with large input indices
        large_indices_input = jax.random.randint(key, (batch_size, input_length), 0, vocab_size * 2)
        large_indices_target = jax.random.randint(key, (batch_size, target_length), 0, vocab_size * 2)
        large_indices_output = transformer.text_to_text(large_indices_input, large_indices_target)
        assert large_indices_output.shape == (batch_size, target_length, vocab_size)
        assert jnp.isfinite(large_indices_output).all()

        # Test for consistency across multiple runs
        outputs = [transformer.text_to_text(input_ids, target_ids) for _ in range(5)]
        for out1, out2 in zip(outputs[:-1], outputs[1:]):
            assert jnp.allclose(out1, out2, atol=1e-2, rtol=1e-2)

def test_fine_tune_classification(transformer_setup):
    transformer = transformer_setup['transformer']
    num_labels = 3
    transformer.fine_tune(task='classification', num_labels=num_labels)
    if transformer_setup['backend'] == 'pytorch':
        assert isinstance(transformer.task_head, torch.nn.Linear)
    elif transformer_setup['backend'] == 'jax':
        assert isinstance(transformer.task_head, jax.nn.Dense)
    assert transformer.task_head.out_features == num_labels

def test_fine_tune_generation(transformer_setup):
    transformer = transformer_setup['transformer']
    transformer.fine_tune(task='generation')
    assert transformer.task_head is transformer.lm_head

def test_task_specific_forward_classification(transformer_setup):
    transformer = transformer_setup['transformer']
    input_ids = transformer_setup['input_ids']
    attention_mask = transformer_setup['attention_mask']
    batch_size = transformer_setup['batch_size']
    num_labels = 3
    transformer.fine_tune(task='classification', num_labels=num_labels)
    output = transformer.task_specific_forward(input_ids, attention_mask, task='classification')
    assert output.shape == (batch_size, num_labels)

def test_task_specific_forward_generation(transformer_setup):
    transformer = transformer_setup['transformer']
    input_ids = transformer_setup['input_ids']
    attention_mask = transformer_setup['attention_mask']
    batch_size = transformer_setup['batch_size']
    seq_length = transformer_setup['seq_length']
    transformer.fine_tune(task='generation')
    output = transformer.task_specific_forward(input_ids, attention_mask, task='generation')
    assert output.shape == (batch_size, seq_length, transformer.lm_head.out_features)

def test_invalid_task(transformer_setup):
    transformer = transformer_setup['transformer']
    with pytest.raises(ValueError):
        transformer.fine_tune(task='invalid_task')

    with pytest.raises(ValueError):
        transformer.task_specific_forward(transformer_setup['input_ids'], transformer_setup['attention_mask'], task='invalid_task')

def test_get_unified_transformer():
    transformer_pytorch = get_unified_transformer(backend='pytorch', vocab_size=30000)
    assert isinstance(transformer_pytorch, UnifiedTransformer)

    transformer_jax = get_unified_transformer(backend='jax', vocab_size=30000)
    assert isinstance(transformer_jax, UnifiedTransformer)

    with pytest.raises(NotImplementedError):
        get_unified_transformer(backend='tensorflow')

    with pytest.raises(ValueError):
        get_unified_transformer(backend='invalid_backend')
