import unittest
import torch
import gym
from unittest.mock import patch, MagicMock
from NeuroFlex.reinforcement_learning.reinforcement_learning_advancements import (
    AdvancedRLAgent,
    MultiAgentEnvironment,
    create_ppo_agent,
    create_sac_agent,
    train_multi_agent_rl,
    advanced_rl_training,
)


class TestReinforcementLearningAdvancements(unittest.TestCase):
    def setUp(self):
        self.env_id = "CartPole-v1"
        self.num_agents = 2
        self.action_dim = 2
        self.observation_dim = 4
        self.features = [64, 64]

    def test_advanced_rl_agent_initialization(self):
        agent = AdvancedRLAgent(self.observation_dim, self.action_dim, self.features)
        self.assertIsInstance(agent, AdvancedRLAgent)
        self.assertEqual(agent.observation_dim, self.observation_dim)
        self.assertEqual(agent.action_dim, self.action_dim)
        self.assertEqual(agent.features, self.features)
        self.assertFalse(agent.is_trained)
        self.assertIsInstance(agent.q_network, torch.nn.Sequential)
        self.assertIsInstance(agent.optimizer, torch.optim.Adam)

    def test_advanced_rl_agent_select_action(self):
        agent = AdvancedRLAgent(self.observation_dim, self.action_dim, self.features)
        state = torch.rand(self.observation_dim)
        action = agent.select_action(state)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)

    def test_multi_agent_environment(self):
        env = MultiAgentEnvironment(self.num_agents, self.env_id)
        self.assertEqual(len(env.envs), self.num_agents)
        observations = env.reset()
        self.assertEqual(len(observations), self.num_agents)
        for obs in observations:
            self.assertIsInstance(obs, torch.Tensor)
            self.assertEqual(obs.shape, (self.observation_dim,))

        actions = [0] * self.num_agents
        next_obs, rewards, dones, truncated, infos = env.step(actions)
        self.assertEqual(len(next_obs), self.num_agents)
        self.assertEqual(len(rewards), self.num_agents)
        self.assertEqual(len(dones), self.num_agents)
        self.assertEqual(len(truncated), self.num_agents)
        self.assertEqual(len(infos), self.num_agents)

    def test_create_ppo_agent(self):
        env = gym.make(self.env_id)
        agent = create_ppo_agent(env)
        self.assertIsInstance(agent, AdvancedRLAgent)
        self.assertEqual(agent.action_dim, env.action_space.n)

    def test_create_sac_agent(self):
        env = gym.make(self.env_id)
        agent = create_sac_agent(env)
        self.assertIsInstance(agent, AdvancedRLAgent)
        self.assertEqual(agent.action_dim, env.action_space.n)

    @patch(
        "NeuroFlex.reinforcement_learning.reinforcement_learning_advancements."
        "AdvancedRLAgent"
    )
    def test_train_multi_agent_rl(self, mock_agent_class):
        mock_agent = MagicMock()
        mock_agent.device = torch.device("cpu")
        mock_agent.select_action.return_value = 0
        mock_agent.replay_buffer = MagicMock()
        mock_agent.replay_buffer.batch_size = 32
        mock_agent.replay_buffer.__len__.return_value = (
            100  # Ensure buffer appears to have enough samples
        )
        mock_agent.replay_buffer.sample.return_value = {
            "observations": torch.rand(32, self.observation_dim),
            "actions": torch.randint(0, self.action_dim, (32, 1)),
            "rewards": torch.rand(32),
            "next_observations": torch.rand(32, self.observation_dim),
            "dones": torch.randint(0, 2, (32,)).bool(),
        }
        mock_agent.to.return_value = mock_agent
        mock_agent.performance = 0.0
        mock_agent.performance_threshold = 0.8
        mock_agent.epsilon = 0.5
        mock_agent_class.return_value = mock_agent

        env = MultiAgentEnvironment(self.num_agents, self.env_id)
        agents = [mock_agent_class() for _ in range(self.num_agents)]
        total_timesteps = 200

        trained_agents = train_multi_agent_rl(env, agents, total_timesteps)

        self.assertEqual(len(trained_agents), self.num_agents)
        for agent in trained_agents:
            self.assertTrue(agent.update.called)
            self.assertTrue(agent.select_action.called)
            self.assertGreater(agent.replay_buffer.__len__.call_count, 0)
            self.assertGreater(agent.update.call_count, 0)

    @patch(
        "NeuroFlex.reinforcement_learning.reinforcement_learning_advancements."
        "train_multi_agent_rl"
    )
    def test_advanced_rl_training(self, mock_train):
        mock_train.return_value = [
            MagicMock(is_trained=True) for _ in range(self.num_agents)
        ]
        trained_agents = advanced_rl_training(
            self.env_id, self.num_agents, algorithm="PPO", total_timesteps=100
        )
        self.assertEqual(len(trained_agents), self.num_agents)
        for agent in trained_agents:
            self.assertTrue(agent.is_trained)

    def test_agent_self_healing(self):
        agent = AdvancedRLAgent(self.observation_dim, self.action_dim, self.features)
        agent.is_trained = True
        agent.performance = 0.5
        agent.last_update = 0
        agent.performance_threshold = 0.8

        issues = agent.diagnose()
        self.assertIn("Model performance is below threshold", issues)
        self.assertIn("Model hasn't been updated in 24 hours", issues)

        with patch.object(agent, "train") as mock_train:
            mock_train.return_value = {
                "final_reward": 0.9,
                "episode_rewards": [0.5, 0.6, 0.7, 0.8, 0.9],
            }
            env = gym.make(self.env_id)
            agent.heal(env, num_episodes=10, max_steps=100)

        self.assertGreater(agent.performance, 0.8)
        self.assertGreater(agent.last_update, 0)
        mock_train.assert_called_once()

    def test_agent_train(self):
        agent = AdvancedRLAgent(self.observation_dim, self.action_dim, self.features)
        env = gym.make(self.env_id)

        with patch.object(agent, "select_action", return_value=0), patch.object(
            agent, "update", return_value=0.1
        ):
            result = agent.train(env, num_episodes=200, max_steps=100)

        self.assertIn("final_reward", result)
        self.assertIn("episode_rewards", result)
        self.assertLessEqual(
            len(result["episode_rewards"]), 200
        )  # May stop early due to performance threshold
        self.assertTrue(agent.is_trained)
        self.assertGreaterEqual(agent.performance, agent.performance_threshold)
        self.assertLess(agent.epsilon, agent.epsilon_start)

        # Check if moving average calculation is working
        if len(result["episode_rewards"]) >= 100:
            moving_avg = sum(result["episode_rewards"][-100:]) / 100
            self.assertAlmostEqual(agent.performance, moving_avg, places=5)
        else:
            moving_avg = sum(result["episode_rewards"]) / len(result["episode_rewards"])
            self.assertAlmostEqual(agent.performance, moving_avg, places=5)

    def test_agent_update(self):
        agent = AdvancedRLAgent(self.observation_dim, self.action_dim, self.features)
        batch = {
            "observations": torch.rand(32, self.observation_dim),
            "actions": torch.randint(0, self.action_dim, (32, 1)),
            "rewards": torch.rand(32),
            "next_observations": torch.rand(32, self.observation_dim),
            "dones": torch.randint(0, 2, (32,)).bool(),
        }
        loss = agent.update(batch)
        self.assertIsInstance(loss, float)


if __name__ == "__main__":
    unittest.main()
