import unittest
from standalone_model import StandaloneCognitiveModel, configure_model, SituatedCognitionModule

class TestSituatedCognition(unittest.TestCase):
    def setUp(self):
        self.config = configure_model()
        self.model = StandaloneCognitiveModel(self.config)
        self.situated_cognition_module = self.model.modules['situated_cognition']

    def test_module_initialization(self):
        self.assertIsInstance(self.situated_cognition_module, SituatedCognitionModule)
        self.assertIn('situated_cognition', self.model.modules)

    def test_environment_integration(self):
        input_data = "Test input"
        self.situated_cognition_module.environment = {"location": "office", "time": "morning"}
        result = self.situated_cognition_module.process(input_data)
        self.assertEqual(result['environment']['location'], "office")
        self.assertEqual(result['environment']['time'], "morning")

    def test_social_context_integration(self):
        input_data = "Test input"
        self.situated_cognition_module.social_context = {"num_people": 3, "relationship": "colleagues"}
        result = self.situated_cognition_module.process(input_data)
        self.assertEqual(result['social_context']['num_people'], 3)
        self.assertEqual(result['social_context']['relationship'], "colleagues")

    def test_cultural_context_integration(self):
        input_data = "Test input"
        self.situated_cognition_module.cultural_context = {"language": "English", "customs": ["handshake"]}
        result = self.situated_cognition_module.process(input_data)
        self.assertEqual(result['cultural_context']['language'], "English")
        self.assertEqual(result['cultural_context']['customs'], ["handshake"])

    def test_physical_context_integration(self):
        input_data = "Test input"
        self.situated_cognition_module.physical_context = {"temperature": 22, "lighting": "bright"}
        result = self.situated_cognition_module.process(input_data)
        self.assertEqual(result['physical_context']['temperature'], 22)
        self.assertEqual(result['physical_context']['lighting'], "bright")

    def test_full_context_integration(self):
        input_data = "Test input"
        self.situated_cognition_module.environment = {"location": "office"}
        self.situated_cognition_module.social_context = {"num_people": 3}
        self.situated_cognition_module.cultural_context = {"language": "English"}
        self.situated_cognition_module.physical_context = {"temperature": 22}
        result = self.situated_cognition_module.process(input_data)
        self.assertEqual(result['environment']['location'], "office")
        self.assertEqual(result['social_context']['num_people'], 3)
        self.assertEqual(result['cultural_context']['language'], "English")
        self.assertEqual(result['physical_context']['temperature'], 22)

if __name__ == '__main__':
    unittest.main()
