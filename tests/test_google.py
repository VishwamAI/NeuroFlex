import unittest
from NeuroFlex.scientific_domains import GoogleIntegration

class TestGoogleIntegration(unittest.TestCase):
    def setUp(self):
        self.google_integration = GoogleIntegration()

    def test_google_search(self):
        query = "NeuroFlex AI"
        results = self.google_integration.search(query)
        self.assertIsNotNone(results)
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)

    def test_google_translate(self):
        text = "Hello, world!"
        target_language = "es"
        translated_text = self.google_integration.translate(text, target_language)
        self.assertIsNotNone(translated_text)
        self.assertIsInstance(translated_text, str)
        self.assertNotEqual(text, translated_text)

    def test_google_vision(self):
        image_path = "path/to/test/image.jpg"
        labels = self.google_integration.analyze_image(image_path)
        self.assertIsNotNone(labels)
        self.assertIsInstance(labels, list)
        self.assertTrue(len(labels) > 0)

if __name__ == '__main__':
    unittest.main()
