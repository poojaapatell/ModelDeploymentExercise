import unittest
import json
from app import app
from model_utils import load_artifacts
from model_utils import make_predictions
from preprocessing import preprocess_input

class ModelDeploymentTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up test client and load artifacts once for all tests
        cls.client = app.test_client()
        cls.model, cls.encoder, cls.mod_vars = load_artifacts()

    def test_1_health_check(self):
        # Check if Flask app is running
        response = self.client.get('/inference/')
        self.assertEqual(response.status_code, 405)  # GET method not allowed

    def test_2_invalid_content_type(self):
        # Test API with incorrect content type
        response = self.client.post('/inference/', data="invalid_data", content_type='text/plain')
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid Content-Type", response.get_json()["error"])

    def test_3_missing_json_body(self):
        #Test API with empty request body
        response = self.client.post('/inference/', data=json.dumps(None), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertIn("No data provided", response.get_json()["error"])

    def test_4_valid_inference(self):
        #Test API with valid input
        test_data = [{"feature1": "A", "feature2": 5, "feature3": "B"}]
        response = self.client.post('/inference/', json=test_data)
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.get_json(), list)

    def test_5_model_predictions(self):
        # Test make_predictions function
        test_data = [{"feature1": "A", "feature2": 5, "feature3": "B"}]
        processed_input = preprocess_input(test_data, self.mod_vars, self.encoder)
        predictions = make_predictions(processed_input, test_data, self.model)

        self.assertIsInstance(predictions, list)
        self.assertIn("business_outcome", predictions[0])
        self.assertIn("prediction", predictions[0])

if __name__ == "__main__":
    unittest.main()
