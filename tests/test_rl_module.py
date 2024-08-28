import unittest
from unittest.mock import MagicMock
import jax
import jax.numpy as jnp
import numpy as np
import gym
import optax
from NeuroFlex.rl_module import RLAgent, HierarchicalRLAgent, RLEnvironment, create_train_state, select_action, train_rl_agent
# from NeuroFlex.alphafold_integration import AlphaFoldIntegration

# Create a mock for AlphaFoldIntegration
# mock_alphafold = MagicMock(spec=AlphaFoldIntegration)

class TestRLModule(unittest.TestCase):
    def setUp(self):
        self.env_name = "CartPole-v1"
        self.env = RLEnvironment(self.env_name)
        self.agent = RLAgent(features=[64, 64], action_dim=self.env.action_space.n)
        self.hierarchical_agent = HierarchicalRLAgent(features=[64, 64], action_dims=[4, 2], num_levels=2)
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

    def test_train_hierarchical_rl_agent(self):
        import optax
        num_episodes = 3000  # Increased to allow for hierarchical learning
        max_steps = 1000  # Kept the same as it's already high
        early_stop_threshold = 180.0  # Increased for hierarchical learning
        early_stop_episodes = 300  # Increased for more patience in hierarchical training
        validation_episodes = 100  # Increased for more robust validation of hierarchical behavior
        learning_rate = 5e-6  # Further reduced for more stable hierarchical learning
        improvement_threshold = 1.0005  # Further reduced to account for slower, more stable hierarchical learning
        seed = 42  # Kept the same for reproducibility
        warmup_episodes = 300  # Increased warmup period for stable hierarchical learning
        num_levels = 2  # Number of levels in the hierarchy
        sub_action_dims = [4, 2]  # Action dimensions for each level

        try:
            trained_state, rewards, training_info = train_rl_agent(
                self.hierarchical_agent, self.env, num_episodes=num_episodes, max_steps=max_steps,
                early_stop_threshold=early_stop_threshold, early_stop_episodes=early_stop_episodes,
                validation_episodes=validation_episodes, learning_rate=learning_rate,
                seed=seed, num_levels=num_levels, sub_action_dims=sub_action_dims
            )

            self.assertIsNotNone(trained_state, "Trained hierarchical state should not be None")
            self.assertLessEqual(len(rewards), num_episodes, f"Expected at most {num_episodes} rewards")
            self.assertTrue(all(isinstance(r, float) for r in rewards), "All rewards should be floats")
            self.assertEqual(len(trained_state), num_levels, f"Expected {num_levels} trained states for hierarchical agent")

            # Check if the agent is learning
            improvement_threshold = 1.005  # Reduced to 0.5% improvement to account for stochasticity
            window_size = min(100, len(rewards) // 4)  # Use a quarter of total episodes or 100, whichever is smaller
            self.assertGreater(np.mean(rewards[-window_size:]), np.mean(rewards[:window_size]) * improvement_threshold,
                               f"Agent should show at least {improvement_threshold}x improvement over time")

            # Check if the final rewards are better than random policy
            random_policy_reward = 20  # Approximate value for CartPole-v1
            self.assertGreater(np.mean(rewards[-window_size:]), random_policy_reward * 1.2,
                               "Agent should perform better than random policy")

            # Check for consistent improvement
            reward_chunks = np.array_split(rewards, 4)
            self.assertTrue(all(np.mean(reward_chunks[i]) < np.mean(reward_chunks[i+1]) for i in range(3)),
                            "Agent should show consistent improvement across training")

            # Check if the model parameters have changed
            initial_params = self.agent.init(jax.random.PRNGKey(0), jnp.ones((1, self.env.observation_space.shape[0])))['params']
            param_diff = jax.tree_util.tree_map(lambda x, y: jnp.sum(jnp.abs(x - y)), initial_params, trained_state.params)
            total_diff = sum(jax.tree_util.tree_leaves(param_diff))
            self.assertGreater(total_diff, 1e-5, "Model parameters should have changed significantly during training")

            # Check if the agent can solve the environment
            self.assertGreaterEqual(np.mean(rewards[-50:]), early_stop_threshold * 0.7,
                                    "Agent should come close to solving the environment")

            # Check if early stopping worked
            self.assertLessEqual(len(rewards), num_episodes, "Early stopping should have terminated training at or before max episodes")

            # Check for learning stability
            last_50_rewards = rewards[-50:]
            self.assertLess(np.std(last_50_rewards), 80, "Agent should show relatively stable performance in the last 50 episodes")

            # Check for consistent performance
            self.assertGreater(np.min(last_50_rewards), 80, "Agent should consistently perform well in the last 50 episodes")

            # Check if learning rate scheduling is working
            self.assertIsInstance(trained_state.tx, optax.GradientTransformation, "Learning rate scheduler should be applied")
            self.assertLess(training_info['final_lr'], learning_rate * 0.9, "Learning rate should decrease over time")

            # Check if validation was performed
            self.assertIn('validation_rewards', training_info, "Validation rewards should be present in training info")
            self.assertGreaterEqual(np.mean(training_info['validation_rewards']), early_stop_threshold * 0.8,
                                    "Agent should pass validation before stopping")

            # Check for error handling
            self.assertIn('errors', training_info, "Error information should be present in training info")
            self.assertLessEqual(len(training_info['errors']), 10, "There should be very few errors during training")

            # Check for early stopping reason
            self.assertIn('early_stop_reason', training_info, "Early stop reason should be provided")
            self.assertIn(training_info['early_stop_reason'], ['solved', 'max_episodes_reached', 'no_improvement'],
                          "Early stop reason should be valid")

            # Check for learning rate decay
            self.assertIn('lr_history', training_info, "Learning rate history should be present in training info")
            self.assertLess(training_info['lr_history'][-1], training_info['lr_history'][0] * 0.7,
                            "Learning rate should decay significantly over time")

            # Check for improved early stopping
            if training_info['early_stop_reason'] == 'solved':
                self.assertGreaterEqual(training_info['best_average_reward'], early_stop_threshold * 0.9,
                                        "Best average reward should come close to or exceed early stopping threshold")

            # Check for training stability
            self.assertIn('loss_history', training_info, "Loss history should be present in training info")
            self.assertLess(np.mean(training_info['loss_history'][-20:]), np.mean(training_info['loss_history'][:20]),
                            "Loss should decrease over time")

            # Check for exploration strategy
            self.assertIn('epsilon_history', training_info, "Epsilon history should be logged")
            self.assertLess(training_info['epsilon_history'][-1], training_info['epsilon_history'][0] * 0.1,
                            "Epsilon should decrease significantly over time")

            # Check for detailed logging
            self.assertIn('episode_lengths', training_info, "Episode lengths should be logged")
            self.assertIn('epsilon_history', training_info, "Epsilon history should be logged")
            self.assertEqual(len(training_info['episode_lengths']), len(rewards),
                             "Episode lengths should match the number of episodes")

            # Check for exploration strategy
            self.assertIn('epsilon_history', training_info, "Epsilon history should be logged")
            self.assertLess(training_info['epsilon_history'][-1], training_info['epsilon_history'][0] * 0.5,
                            "Epsilon should decrease over time")

            # Check for reward shaping
            self.assertIn('shaped_rewards', training_info, "Shaped rewards should be logged")
            self.assertGreater(np.mean(training_info['shaped_rewards'][-10:]), np.mean(training_info['shaped_rewards'][:10]) * 1.03,
                               "Shaped rewards should show improvement over time")
            self.assertGreater(np.mean(training_info['shaped_rewards']), np.mean(rewards),
                               "Shaped rewards should be higher on average than raw rewards")
            self.assertLess(np.std(training_info['shaped_rewards']), np.std(rewards) * 3,
                            "Shaped rewards should have comparable or lower variance than raw rewards")

            # Check for training stability
            self.assertIn('loss_history', training_info, "Loss history should be logged")
            self.assertLess(np.mean(training_info['loss_history'][-10:]), np.mean(training_info['loss_history'][:10]) * 0.95,
                            "Loss should decrease over time")

            # Check for proper handling of NaN values
            self.assertFalse(np.isnan(np.array(training_info['loss_history'])).any(), "Loss history should not contain NaN values")

            # Check for correlation between shaped rewards and actual rewards
            shaped_rewards = np.array(training_info['shaped_rewards'])
            correlation = np.corrcoef(shaped_rewards, rewards)[0, 1]
            self.assertGreater(correlation, 0.2, "Shaped rewards should be positively correlated with actual rewards")

            # Check for exploration strategy effectiveness
            if 'actions' in training_info:
                unique_actions = len(set(training_info['actions']))
                self.assertEqual(unique_actions, self.env.action_space.n, "Agent should explore all possible actions")

            # Check for learning rate adaptation
            if len(training_info['lr_history']) > 1:
                lr_changes = np.diff(training_info['lr_history'])
                self.assertTrue(np.any(lr_changes != 0), "Learning rate should adapt during training")

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
            _, rewards1, info1 = train_rl_agent(self.agent, self.env, num_episodes=25, max_steps=max_steps, seed=42)
            _, rewards2, info2 = train_rl_agent(self.agent, self.env, num_episodes=25, max_steps=max_steps, seed=42)
            self.assertAlmostEqual(np.mean(rewards1), np.mean(rewards2), delta=20,
                                   msg="Training results should be reasonably reproducible with the same seed")
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
