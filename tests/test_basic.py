import unittest
from NeuroFlex.utils.tokenizer import Tokenizer
from NeuroFlex.scientific_domains.biology.synthetic_biology_insights import SyntheticBiologyInsights

class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()

    def test_tokenize(self):
        text = "Hello, world!"
        tokens = self.tokenizer.tokenize(text)
        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)

    def test_encode_decode(self):
        text = "This is a test sentence."
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(text, decoded)

    def test_multilingual_support(self):
        texts = {
            "en": "Hello, world!",
            "es": "¡Hola, mundo!",
            "fr": "Bonjour, le monde!",
            "de": "Hallo, Welt!"
        }
        for lang, text in texts.items():
            encoded = self.tokenizer.encode(text, lang=lang)
            self.assertIsInstance(encoded, list)
            self.assertTrue(len(encoded) > 0)

class TestSyntheticBiologyInsights(unittest.TestCase):
    def setUp(self):
        self.synbio = SyntheticBiologyInsights()

    def test_design_genetic_circuit(self):
        circuit_name = "test_circuit"
        components = ["pTac", "B0034", "GFP", "T1"]
        result = self.synbio.design_genetic_circuit(circuit_name, components)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["circuit_name"], circuit_name)
        self.assertEqual(result["components"], components)

    def test_simulate_metabolic_pathway(self):
        pathway_name = "test_pathway"
        reactions = ["glucose -> glucose-6-phosphate", "glucose-6-phosphate -> fructose-6-phosphate"]
        result = self.synbio.simulate_metabolic_pathway(pathway_name, reactions)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["pathway_name"], pathway_name)
        self.assertEqual(result["reactions"], reactions)

if __name__ == '__main__':
    unittest.main()
