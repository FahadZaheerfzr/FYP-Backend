import unittest
from fastapi.testclient import TestClient
from main import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_predict_file(self):
        # Prepare a mock .dat file for testing
        test_file = 'test_files/test_1.dat'
        # ... Write code to create a test .dat file with the necessary data for prediction ...

        # Send a POST request to the API endpoint
        with open(test_file, 'rb') as file:
            response = self.client.post('/predict_file/', files={'file': file})

        # Assert the response status code
        self.assertEqual(response.status_code, 200)

        # Assert the response JSON content
        expected_result = {"modulation:type": "QPSK"}  # Expected result based on your example
        self.assertEqual(response.json(), expected_result)

        # Clean up the test file
        # ... Write code to delete the test .dat file ...

if __name__ == '__main__':
    unittest.main()
