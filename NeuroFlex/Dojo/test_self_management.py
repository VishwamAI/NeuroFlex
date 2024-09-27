import unittest
from NeuroFlex.Dojo.dojo_framework import DojoFramework, ResourceManager, TaskScheduler, Optimizer, configure_dojo

class TestSelfManagement(unittest.TestCase):
    def setUp(self):
        self.config = configure_dojo()
        self.dojo = DojoFramework(self.config)

    def test_resource_allocation(self):
        resource_manager = ResourceManager(self.config)
        task = {'name': 'test_task', 'complexity': 'high'}
        allocated_resources = resource_manager.allocate_resources(task)
        self.assertIsNotNone(allocated_resources)
        self.assertGreater(allocated_resources['cpu'], 0)
        self.assertGreater(allocated_resources['memory'], 0)
        self.assertGreaterEqual(allocated_resources['gpu'], 0)

    def test_task_scheduling(self):
        task_scheduler = TaskScheduler(self.config)
        tasks = [
            {'name': 'task1', 'importance': 5, 'urgency': 3},
            {'name': 'task2', 'importance': 8, 'urgency': 2},
            {'name': 'task3', 'importance': 3, 'urgency': 7}
        ]
        for task in tasks:
            task_scheduler.schedule_task(task)

        next_task = task_scheduler.get_next_task()
        self.assertEqual(next_task['name'], 'task2')  # Highest priority task

    def test_optimization(self):
        optimizer = Optimizer(self.config)
        mock_model = MockModel()
        mock_data = [1, 2, 3, 4, 5]  # Mock training data

        optimized_model = optimizer.optimize(mock_model, mock_data)
        self.assertIsNotNone(optimized_model)
        self.assertLess(optimized_model.loss, mock_model.initial_loss)

class MockModel:
    def __init__(self):
        self.initial_loss = 1.0
        self.loss = self.initial_loss

    def train(self, data, learning_rate):
        # Simulate training by reducing loss
        self.loss *= 0.9
        return self.loss

if __name__ == '__main__':
    unittest.main()
