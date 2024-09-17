import unittest
from unittest.mock import Mock, patch
import jax.numpy as jnp
from NeuroFlex.Prompt_Agent.agentic_behavior import ZeroShotAgent, FewShotAgent, ChainOfThoughtAgent, MetaPromptingAgent, BaseAgent
import flax.linen as nn
from NeuroFlex.utils.utils import tokenize_text

class TestZeroShotAgent(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock(spec=nn.Module)
        self.mock_model.params = {'mock_params': 'value'}
        self.mock_model.apply = Mock()
        self.agent = ZeroShotAgent(self.mock_model)

    def test_self_correct(self):
        output = "The capital of France is London."
        expected_correction_prompt = f"The previous output was:\n{output}\n\nPlease review and correct any errors in the above output:"
        expected_tokens = tokenize_text(expected_correction_prompt)
        expected_encoded_input = jnp.array([hash(token) % 10000 for token in expected_tokens])

        with patch.object(self.agent, '_encode_input', return_value=expected_encoded_input) as mock_encode:
            with patch.object(self.agent.model, 'apply', return_value=jnp.array([16.0, 17.0, 18.0])) as mock_apply:
                with patch.object(self.agent, '_decode_output', return_value="The capital of France is Paris.") as mock_decode:
                    result = self.agent.self_correct(output)

                    mock_encode.assert_called_once_with(expected_tokens)
                    mock_apply.assert_called_once_with({'params': self.mock_model.params}, expected_encoded_input)
                    self.assertTrue(jnp.allclose(mock_decode.call_args[0][0], jnp.array([16.0, 17.0, 18.0])))
                    self.assertEqual(result, "The capital of France is Paris.")

    def test_self_update(self):
        feedback = "The model's performance on math problems needs improvement."
        with patch('builtins.print') as mock_print:
            self.agent.self_update(feedback)
            mock_print.assert_called_once_with(f"Received feedback for self-update: {feedback}")

    def test_zero_shot(self):
        prompt = "Translate 'Hello' to French"
        expected_tokens = tokenize_text(prompt)
        expected_encoded_input = jnp.array([hash(token) % 10000 for token in expected_tokens])

        with patch.object(self.agent, '_encode_input', return_value=expected_encoded_input) as mock_encode:
            with patch.object(self.agent.model, 'apply', return_value=jnp.array([1.0, 2.0, 3.0])) as mock_apply:
                with patch.object(self.agent, '_decode_output', return_value="Bonjour") as mock_decode:
                    result = self.agent.zero_shot(prompt)

                    mock_encode.assert_called_once_with(expected_tokens)
                    mock_apply.assert_called_once_with({'params': self.mock_model.params}, expected_encoded_input)
                    self.assertTrue(jnp.allclose(mock_decode.call_args[0][0], jnp.array([1.0, 2.0, 3.0])))
                    self.assertEqual(result, "Bonjour")



class TestFewShotAgent(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock(spec=nn.Module)
        self.mock_model.params = {'mock_params': 'value'}
        self.mock_model.apply = Mock()
        self.agent = FewShotAgent(self.mock_model)

    def test_few_shot(self):
        prompt = "Translate 'Good morning' to Spanish"
        examples = [
            {"input": "Translate 'Hello' to Spanish", "output": "Hola"},
            {"input": "Translate 'Goodbye' to Spanish", "output": "Adiós"}
        ]
        expected_context = "Input: Translate 'Hello' to Spanish\nOutput: Hola\n\nInput: Translate 'Goodbye' to Spanish\nOutput: Adiós\n\nTranslate 'Good morning' to Spanish"
        expected_tokens = tokenize_text(expected_context)
        expected_encoded_input = jnp.array([hash(token) % 10000 for token in expected_tokens])

        with patch.object(self.agent, '_encode_input', return_value=expected_encoded_input) as mock_encode:
            with patch.object(self.agent.model, 'apply', return_value=jnp.array([4.0, 5.0, 6.0])) as mock_apply:
                with patch.object(self.agent, '_decode_output', return_value="Buenos días") as mock_decode:
                    result = self.agent.few_shot(prompt, examples)

                    mock_encode.assert_called_once_with(expected_tokens)
                    mock_apply.assert_called_once_with({'params': self.mock_model.params}, expected_encoded_input)
                    self.assertTrue(jnp.allclose(mock_decode.call_args[0][0], jnp.array([4.0, 5.0, 6.0])))
                    self.assertEqual(result, "Buenos días")

class TestChainOfThoughtAgent(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock(spec=nn.Module)
        self.mock_model.params = {'mock_params': 'value'}
        self.mock_model.apply = Mock()
        self.agent = ChainOfThoughtAgent(self.mock_model)

    def test_chain_of_thought(self):
        prompt = "Solve 2 + 2"
        expected_cot_prompt = f"Let's approach this step-by-step:\n1) {prompt}\n2) "
        initial_tokens = tokenize_text(expected_cot_prompt)
        initial_encoded_input = jnp.array([hash(token) % 10000 for token in initial_tokens])

        apply_side_effects = [
            jnp.array([7.0, 8.0, 9.0]),
            jnp.array([10.0, 11.0, 12.0]),
            jnp.array([13.0, 14.0, 15.0]),
            jnp.array([16.0, 17.0, 18.0]),
            jnp.array([19.0, 20.0, 21.0])
        ]
        decode_side_effects = [
            "First, we identify the numbers: 2 and 2",
            "Next, we add them: 2 + 2 = 4",
            "Then, we verify the result",
            "We can also represent it as: 4 = 2 + 2",
            "Finally, we conclude that 2 + 2 = 4"
        ]

        encode_side_effects = [initial_encoded_input] + [
            jnp.array([hash(token) % 10000 for token in tokenize_text(step)])
            for step in decode_side_effects
        ]

        with patch.object(self.agent, '_encode_input', side_effect=encode_side_effects) as mock_encode:
            with patch.object(self.agent.model, 'apply', side_effect=apply_side_effects) as mock_apply:
                with patch.object(self.agent, '_decode_output', side_effect=decode_side_effects) as mock_decode:
                    result = self.agent.chain_of_thought(prompt)

                    self.assertEqual(mock_encode.call_count, 6)  # Initial + 5 steps
                    mock_encode.assert_has_calls([
                        unittest.mock.call(initial_tokens),
                        *[unittest.mock.call(tokenize_text(step)) for step in decode_side_effects]
                    ])

                    self.assertEqual(mock_apply.call_count, 5)
                    mock_apply.assert_has_calls([
                        unittest.mock.call({'params': self.mock_model.params}, unittest.mock.ANY)
                        for _ in range(5)
                    ])

                    self.assertEqual(mock_decode.call_count, 5)
                    for i, array in enumerate(apply_side_effects):
                        self.assertTrue(jnp.allclose(mock_decode.call_args_list[i][0][0], array))

                    expected_result = "\n".join(decode_side_effects)
                    self.assertEqual(result, expected_result)

class TestMetaPromptingAgent(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock(spec=nn.Module)
        self.mock_model.params = {'mock_params': 'value'}
        self.mock_model.apply = Mock()
        self.agent = MetaPromptingAgent(self.mock_model)

    def test_meta_prompting(self):
        prompt = "Translate 'Hello' to Japanese"
        meta_prompt = "You are a helpful AI assistant specializing in language translation."
        expected_full_prompt = f"{meta_prompt}\n\nTask: {prompt}"
        expected_tokens = tokenize_text(expected_full_prompt)
        expected_encoded_input = jnp.array([hash(token) % 10000 for token in expected_tokens])

        with patch.object(self.agent, '_encode_input', return_value=expected_encoded_input) as mock_encode:
            with patch.object(self.agent.model, 'apply', return_value=jnp.array([13.0, 14.0, 15.0])) as mock_apply:
                with patch.object(self.agent, '_decode_output', return_value="こんにちは") as mock_decode:
                    result = self.agent.meta_prompting(prompt, meta_prompt)

                    mock_encode.assert_called_once_with(expected_tokens)
                    mock_apply.assert_called_once_with({'params': self.mock_model.params}, expected_encoded_input)
                    self.assertTrue(jnp.allclose(mock_decode.call_args[0][0], jnp.array([13.0, 14.0, 15.0])))
                    self.assertEqual(result, "こんにちは")

if __name__ == '__main__':
    unittest.main()
