import unittest
import numpy as np
from bayesian_inference_module import BayesianInferenceModule, configure_bayesian_inference

class TestBayesianInference(unittest.TestCase):
    def setUp(self):
        self.config = configure_bayesian_inference()
        self.bi_module = BayesianInferenceModule(self.config)

    def test_initialization(self):
        self.assertIsInstance(self.bi_module, BayesianInferenceModule)
        self.assertEqual(self.bi_module.prior.shape, (self.config['num_hypotheses'],))
        self.assertEqual(self.bi_module.likelihood.shape, (self.config['num_hypotheses'], self.config['num_observations']))

    def test_prior_initialization(self):
        np.testing.assert_almost_equal(np.sum(self.bi_module.prior), 1.0)
        np.testing.assert_array_equal(self.bi_module.prior, np.ones(self.config['num_hypotheses']) / self.config['num_hypotheses'])

    def test_update_belief(self):
        initial_prior = self.bi_module.prior.copy()
        observation = 2
        updated_belief = self.bi_module.update_belief(observation)
        self.assertEqual(updated_belief.shape, initial_prior.shape)
        np.testing.assert_almost_equal(np.sum(updated_belief), 1.0)
        self.assertFalse(np.array_equal(updated_belief, initial_prior))

    def test_predict(self):
        new_data = [1, 3, 0]
        predictions = self.bi_module.predict(new_data)
        self.assertEqual(len(predictions), len(new_data))
        self.assertTrue(np.all(predictions >= 0) and np.all(predictions <= 1))

    def test_integrate_with_standalone_model_list(self):
        input_data = [1, 2, 3]
        result = self.bi_module.integrate_with_standalone_model(input_data)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), self.config['num_hypotheses'])
        np.testing.assert_almost_equal(np.sum(result), 1.0)

    def test_integrate_with_standalone_model_dict(self):
        input_data = {'observations': [1, 2], 'predictions': [0, 1]}
        result = self.bi_module.integrate_with_standalone_model(input_data)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)  # Number of predictions
        self.assertTrue(np.all(np.array(result) >= 0) and np.all(np.array(result) <= 1))

if __name__ == '__main__':
    unittest.main()
