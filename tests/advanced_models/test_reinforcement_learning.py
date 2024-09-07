import unittest
import torch
import numpy as np
from NeuroFlex.reinforcement_learning.self_curing_rl import SelfCuringRLAgent
from NeuroFlex.reinforcement_learning.rl_module import RLEnvironment, PrioritizedReplayBuffer

class TestReinforcementLearning(unittest.TestCase):
    def setUp(self):
        self.features = [4]  # CartPole-v1 has 4 observation dimensions
        self.action_dim = 2
        self.agent = SelfCuringRLAgent(features=self.features, action_dim=self.action_dim)
        self.env = RLEnvironment("CartPole-v1")
        self.replay_buffer = PrioritizedReplayBuffer(100000, (4,), (self.action_dim,))

    def test_agent_initialization(self):
        self.assertIsInstance(self.agent, SelfCuringRLAgent)
        self.assertEqual(self.agent.features, self.features)
        self.assertEqual(self.agent.action_dim, self.action_dim)
        self.assertFalse(self.agent.is_trained)

    def test_select_action(self):
        state = torch.rand(self.features[0])
        action = self.agent.select_action(state)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)

    def test_update(self):
        batch_size = 32
        states = torch.rand(batch_size, self.features[0])
        actions = torch.randint(0, self.action_dim, (batch_size, 1))  # Changed to (batch_size, 1)
        rewards = torch.rand(batch_size)
        next_states = torch.rand(batch_size, self.features[0])
        dones = torch.randint(0, 2, (batch_size,)).bool()

        batch = {
            'observations': states,
            'actions': actions,
            'rewards': rewards,
            'next_observations': next_states,
            'dones': dones
        }

        with self.assertLogs(level='DEBUG') as log:
            loss = self.agent.update(batch)

        self.assertIsInstance(loss, float)
        self.assertGreater(len(log.output), 0, "Debug logs should be generated")

        # Check for specific debug logs
        self.assertTrue(any("Initial shapes" in output for output in log.output))
        self.assertTrue(any("Actions shape after adjustment" in output for output in log.output))
        self.assertTrue(any("Q-values shape before gather" in output for output in log.output))
        self.assertTrue(any("Q-values shape after gather" in output for output in log.output))
        self.assertTrue(any("Next Q-values shape" in output for output in log.output))
        self.assertTrue(any("Targets shape after computation" in output for output in log.output))
        self.assertTrue(any("Final shapes" in output for output in log.output))
        self.assertTrue(any("Computed loss" in output for output in log.output))
        self.assertTrue(any("Actions sample" in output for output in log.output))
        self.assertTrue(any("Q-values sample after gather" in output for output in log.output))

        # Check shapes of q_values and targets
        q_values = self.agent(states).gather(1, actions[:, 0].unsqueeze(1))
        targets = rewards.unsqueeze(1) + self.agent.gamma * self.agent(next_states).max(1, keepdim=True)[0] * (~dones.unsqueeze(1))
        self.assertEqual(q_values.shape, targets.shape, "Shape of q_values and targets should match")

        # Additional check for actions shape
        self.assertEqual(actions.shape, (batch_size, 1), "Actions shape should be (batch_size, 1)")

    def test_train(self):
        num_episodes = 10
        max_steps = 100
        training_info = self.agent.train(self.env, num_episodes, max_steps)

        self.assertIn('final_reward', training_info)
        self.assertIn('episode_rewards', training_info)
        self.assertEqual(len(training_info['episode_rewards']), num_episodes)
        self.assertTrue(self.agent.is_trained)

    def test_diagnose(self):
        issues = self.agent.diagnose()
        self.assertIsInstance(issues, list)
        self.assertIn("Model is not trained", issues)

        self.agent.is_trained = True
        self.agent.performance = 0.7
        self.agent.last_update = 0
        issues = self.agent.diagnose()
        self.assertIn("Model performance is below threshold", issues)
        self.assertIn("Model hasn't been updated in 24 hours", issues)

    def test_heal(self):
        self.agent.is_trained = False
        self.agent.performance = 0.7
        self.agent.last_update = 0

        num_episodes = 5
        max_steps = 50
        self.agent.heal(self.env, num_episodes, max_steps)

        self.assertTrue(self.agent.is_trained)
        self.assertGreater(self.agent.performance, 0.7)
        self.assertGreater(self.agent.last_update, 0)

if __name__ == '__main__':
    unittest.main()
