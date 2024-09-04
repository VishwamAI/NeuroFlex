import unittest
import jax
import jax.numpy as jnp
from NeuroFlex.rl_module import RLAgent, RLEnvironment, train_rl_agent, create_train_state, select_action

class TestReinforcementLearning(unittest.TestCase):
    def setUp(self):
        self.input_dim = 4  # Example input dimension
        self.action_dim = 2  # Example action dimension
        self.learning_rate = 0.001
        self.agent = RLAgent(features=[64, 32], action_dim=self.action_dim)
        self.env = RLEnvironment("CartPole-v1")  # Using CartPole as an example

    def test_rl_agent_initialization(self):
        self.assertIsInstance(self.agent, RLAgent)
        self.assertEqual(self.agent.action_dim, self.action_dim)

    def test_rl_environment_initialization(self):
        self.assertIsInstance(self.env, RLEnvironment)
        self.assertEqual(self.env.env.action_space.n, self.action_dim)

    def test_create_train_state(self):
        rng = jax.random.PRNGKey(0)
        state, _, _ = create_train_state(rng, self.agent, (self.input_dim,), self.learning_rate)
        self.assertIsNotNone(state)
        self.assertIsNotNone(state.params)

    def test_select_action(self):
        rng = jax.random.PRNGKey(0)
        state, _, _ = create_train_state(rng, self.agent, (self.input_dim,), self.learning_rate)
        observation = jnp.zeros(self.input_dim)
        action = select_action(observation, self.agent, state.params)
        self.assertIsInstance(action, jnp.ndarray)
        self.assertEqual(action.shape, ())
        self.assertTrue(0 <= action < self.action_dim)

    def test_train_rl_agent(self):
        rng = jax.random.PRNGKey(0)
        state, _, _ = create_train_state(rng, self.agent, (self.input_dim,), self.learning_rate)
        num_episodes = 5
        max_steps = 100
        final_state, rewards, _ = train_rl_agent(
            self.agent, self.env,
            num_episodes=num_episodes,
            max_steps=max_steps,
            learning_rate=self.learning_rate
        )
        self.assertIsNotNone(final_state)
        self.assertEqual(len(rewards), num_episodes)
        self.assertTrue(all(isinstance(r, float) for r in rewards))

if __name__ == '__main__':
    unittest.main()
