import unittest
import jax
import jax.numpy as jnp
import numpy as np
import gym
from modules.rl_module import RLAgent, RLEnvironment, create_train_state, select_action, train_rl_agent
from typing import Tuple

class TestRLModule(unittest.TestCase):
    def setUp(self):
        self.env_name = "CartPole-v1"
        self.env = RLEnvironment(self.env_name)
        self.agent = RLAgent(features=[64, 64], action_dim=self.env.action_space.n)
        self.rng = jax.random.PRNGKey(0)

    def test_rl_agent_initialization(self):
        dummy_input = jnp.ones((1, self.env.observation_space.shape[0]))
        params = self.agent.init(self.rng, dummy_input)['params']
        self.assertIsNotNone(params)
        self.assertIn('Dense_0', params)
        self.assertIn('Dense_1', params)
        self.assertIn('Dense_2', params)

    def test_create_train_state(self):
        dummy_input = jnp.ones((1, self.env.observation_space.shape[0]))
        state = create_train_state(self.rng, self.agent, dummy_input)
        self.assertIsNotNone(state)
        self.assertIsNotNone(state.params)
        self.assertIsNotNone(state.apply_fn)
        self.assertIsNotNone(state.tx)

    def test_select_action(self):
        dummy_input = jnp.ones((1, self.env.observation_space.shape[0]))
        state = create_train_state(self.rng, self.agent, dummy_input)
        action = select_action(state, dummy_input)
        self.assertIsInstance(action, jnp.ndarray)
        self.assertEqual(action.shape, ())
        self.assertTrue(0 <= action < self.env.action_space.n)

    def test_train_rl_agent(self):
        import optax  # Add missing import
        num_episodes = 2000
        max_steps = 1000
        early_stop_threshold = 195.0
        early_stop_episodes = 100
        validation_episodes = 10
        learning_rate = 1e-3
        seed = 42

        try:
            trained_state, rewards, training_info = train_rl_agent(
                self.agent, self.env, num_episodes=num_episodes, max_steps=max_steps,
                early_stop_threshold=early_stop_threshold, early_stop_episodes=early_stop_episodes,
                validation_episodes=validation_episodes, learning_rate=learning_rate,
                seed=seed
            )

            self.assertIsNotNone(trained_state, "Trained state should not be None")
            self.assertLessEqual(len(rewards), num_episodes, f"Expected at most {num_episodes} rewards")
            self.assertTrue(all(isinstance(r, float) for r in rewards), "All rewards should be floats")

            # Check if the agent is learning
            self.assertGreater(np.mean(rewards[-100:]), np.mean(rewards[:100]), "Agent should show significant improvement over time")

            # Check if the final rewards are better than random policy
            random_policy_reward = 20  # Approximate value for CartPole-v1
            self.assertGreater(np.mean(rewards[-100:]), random_policy_reward * 3, "Agent should perform significantly better than random policy")

            # Check if the model parameters have changed
            initial_params = self.agent.init(jax.random.PRNGKey(0), jnp.ones((1, self.env.observation_space.shape[0])))['params']
            param_diff = jax.tree_util.tree_map(lambda x, y: jnp.sum(jnp.abs(x - y)), initial_params, trained_state.params)
            total_diff = sum(jax.tree_util.tree_leaves(param_diff))
            self.assertGreater(total_diff, 0, "Model parameters should have changed during training")

            # Check if the agent can solve the environment
            self.assertGreaterEqual(np.mean(rewards[-100:]), early_stop_threshold, "Agent should solve the environment")

            # Check if early stopping worked
            self.assertLess(len(rewards), num_episodes, "Early stopping should have terminated training before max episodes")

            # Check for learning stability
            last_100_rewards = rewards[-100:]
            self.assertLess(np.std(last_100_rewards), 30, "Agent should show stable performance in the last 100 episodes")

            # Check for consistent performance
            self.assertGreater(np.min(last_100_rewards), 150, "Agent should consistently perform well in the last 100 episodes")

            # Check if learning rate scheduling is working
            self.assertIsInstance(trained_state.tx, optax.GradientTransformation, "Learning rate scheduler should be applied")
            self.assertLess(training_info['final_lr'], learning_rate, "Learning rate should decrease over time")

            # Check if validation was performed
            self.assertIn('validation_rewards', training_info, "Validation rewards should be present in training info")
            self.assertGreaterEqual(np.mean(training_info['validation_rewards']), early_stop_threshold,
                                    "Agent should pass validation before stopping")

            # Check for error handling
            self.assertIn('errors', training_info, "Error information should be present in training info")
            self.assertEqual(len(training_info['errors']), 0, "There should be no errors during successful training")

            # Check for early stopping reason
            self.assertIn('early_stop_reason', training_info, "Early stop reason should be provided")
            self.assertIn(training_info['early_stop_reason'], ['solved', 'max_episodes_reached', 'no_improvement'],
                          "Early stop reason should be valid")

            # Check for learning rate decay
            self.assertIn('lr_history', training_info, "Learning rate history should be present in training info")
            self.assertTrue(training_info['lr_history'][-1] < training_info['lr_history'][0],
                            "Learning rate should decay over time")

            # Check for improved early stopping
            if training_info['early_stop_reason'] == 'solved':
                self.assertGreaterEqual(training_info['best_average_reward'], early_stop_threshold,
                                        "Best average reward should meet or exceed early stopping threshold")

            # Check for detailed logging
            self.assertIn('episode_lengths', training_info, "Episode lengths should be logged")
            self.assertIn('epsilon_history', training_info, "Epsilon history should be logged")
            self.assertEqual(len(training_info['episode_lengths']), len(rewards),
                             "Episode lengths should match the number of episodes")

            # Check for exploration strategy
            self.assertIn('epsilon_history', training_info, "Epsilon history should be logged")
            self.assertTrue(training_info['epsilon_history'][0] > training_info['epsilon_history'][-1],
                            "Epsilon should decrease over time")

            # Check for reward shaping
            self.assertIn('shaped_rewards', training_info, "Shaped rewards should be logged")
            self.assertGreater(np.mean(training_info['shaped_rewards'][-100:]), np.mean(training_info['shaped_rewards'][:100]),
                               "Shaped rewards should show improvement over time")
            self.assertGreater(np.mean(training_info['shaped_rewards']), np.mean(rewards),
                               "Shaped rewards should be higher on average than raw rewards")
            self.assertLess(np.std(training_info['shaped_rewards']), np.std(rewards),
                            "Shaped rewards should have lower variance than raw rewards")

            # Check for training stability
            self.assertIn('loss_history', training_info, "Loss history should be logged")
            self.assertLess(np.mean(training_info['loss_history'][-100:]), np.mean(training_info['loss_history'][:100]),
                            "Loss should decrease over time")

            # Check for proper handling of NaN values
            self.assertFalse(np.isnan(np.array(training_info['loss_history'])).any(), "Loss history should not contain NaN values")

            # Check for improvement in shaped rewards
            self.assertGreater(np.mean(training_info['shaped_rewards'][-100:]), np.mean(training_info['shaped_rewards'][:100]),
                               "Shaped rewards should show improvement over time")

            # Check for correlation between shaped rewards and actual rewards
            shaped_rewards = np.array(training_info['shaped_rewards'])
            correlation = np.corrcoef(shaped_rewards, rewards)[0, 1]
            self.assertGreater(correlation, 0.5, "Shaped rewards should be positively correlated with actual rewards")

            # Check for exploration strategy effectiveness
            unique_actions = len(set(training_info['actions']))
            self.assertEqual(unique_actions, self.env.action_space.n, "Agent should explore all possible actions")

            # Check for learning rate adaptation
            lr_changes = np.diff(training_info['lr_history'])
            self.assertTrue(np.any(lr_changes != 0), "Learning rate should adapt during training")

            # Check for proper handling of edge cases
            self.assertIn('edge_case_handling', training_info, "Edge case handling should be logged")
            self.assertTrue(training_info['edge_case_handling'], "Agent should properly handle edge cases")

        except Exception as e:
            self.fail(f"train_rl_agent raised an unexpected exception: {str(e)}")

        # Test with impossibly high early stopping threshold
        try:
            _, early_stop_rewards, early_stop_info = train_rl_agent(
                self.agent, self.env, num_episodes=num_episodes, max_steps=max_steps,
                early_stop_threshold=1000.0, early_stop_episodes=early_stop_episodes
            )
            self.assertEqual(len(early_stop_rewards), num_episodes, "Training should run for full number of episodes with impossible threshold")
            self.assertIn('early_stop_reason', early_stop_info, "Early stop reason should be provided")
            self.assertEqual(early_stop_info['early_stop_reason'], 'max_episodes_reached', "Early stop reason should be max episodes reached")
        except Exception as e:
            self.fail(f"Impossible threshold test failed: {str(e)}")

        # Test for reproducibility
        try:
            _, rewards1, info1 = train_rl_agent(self.agent, self.env, num_episodes=100, max_steps=max_steps, seed=42)
            _, rewards2, info2 = train_rl_agent(self.agent, self.env, num_episodes=100, max_steps=max_steps, seed=42)
            self.assertAlmostEqual(np.mean(rewards1), np.mean(rewards2), delta=1,
                                   msg="Training results should be reproducible with the same seed")
            self.assertEqual(info1['total_episodes'], info2['total_episodes'],
                             "Number of episodes should be the same for reproducible runs")
            self.assertEqual(info1['total_steps'], info2['total_steps'],
                             "Number of steps should be the same for reproducible runs")
        except Exception as e:
            self.fail(f"Reproducibility test failed: {str(e)}")

        # Test for handling of unstable training
        try:
            with self.assertLogs(level='WARNING') as cm:
                _, unstable_rewards, unstable_info = train_rl_agent(
                    self.agent, self.env, num_episodes=num_episodes, max_steps=max_steps,
                    learning_rate=1e2,  # Unreasonably high learning rate to induce instability
                    seed=42
                )
            self.assertTrue(any("Detected training instability" in msg for msg in cm.output),
                            "Warning about training instability should be logged")
            self.assertIn('training_stopped_early', unstable_info, "Training info should indicate early stopping due to instability")
            self.assertTrue(unstable_info['training_stopped_early'], "Training should stop early due to instability")
            self.assertLess(len(unstable_rewards), num_episodes, "Training should stop before reaching max episodes")
        except Exception as e:
            self.fail(f"Unstable training test failed: {str(e)}")

    def test_rl_environment(self):
        obs, _ = self.env.reset()
        self.assertIsInstance(obs, jnp.ndarray)
        self.assertEqual(obs.shape, (self.env.observation_space.shape[0],))

        action = self.env.action_space.sample()
        next_obs, reward, done, truncated, info = self.env.step(action)
        self.assertIsInstance(next_obs, jnp.ndarray)
        self.assertEqual(next_obs.shape, (self.env.observation_space.shape[0],))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

if __name__ == '__main__':
    unittest.main()
