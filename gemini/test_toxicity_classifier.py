import unittest
from unittest.mock import patch, MagicMock
from toxicity_classifier import analyze_text_toxicity, ToxicityAnalysis


class TestToxicityClassifier(unittest.TestCase):
    """
    Unit tests for the toxicity classifier function.
    Uses 'unittest.mock' to simulate Google API responses without making actual network calls.
    """

    def setUp(self):
        """
        Prepare reusable mock data for tests.
        """
        # A sample structured response that the API would normally return
        self.mock_analysis_result = ToxicityAnalysis(
            toxicity=0.95,
            severe_toxicity=0.1,
            obscene=0.8,
            threat=0.0,
            insult=0.9,
            identity_attack=0.0,
            sexual_explicit=0.0,
            deciding_fragments=["idioto", "prawo jazdy w chipsach"],
            ambiguous_fragments=["genialne"],
            justification="The text contains direct insults and sarcasm.",
        )

    @patch("toxicity_classifier.genai.Client")
    def test_valid_text_analysis(self, mock_client_class):
        """
        Test Case 1: Happy Path.
        Verifies that the function correctly parses a valid response from the API.
        """
        # 1. Setup the Mock
        # Create a mock instance of the API client
        mock_client_instance = mock_client_class.return_value

        # Create a mock response object that mimics the SDK structure
        # The SDK returns an object where '.parsed' contains our Pydantic model
        mock_response = MagicMock()
        mock_response.parsed = self.mock_analysis_result

        # Tell the mock client to return our mock_response when generate_content is called
        mock_client_instance.models.generate_content.return_value = mock_response

        # 2. Execute the function
        input_text = "Ty idioto."
        result = analyze_text_toxicity(input_text)

        # 3. Assertions (Validation)
        # Check if result is not None
        self.assertIsNotNone(result)
        # Check if the returned object is strictly an instance of our Pydantic class
        self.assertIsInstance(result, ToxicityAnalysis)
        # Check if values match our mock data
        self.assertEqual(result.toxicity, 0.95)
        self.assertEqual(
            result.justification, "The text contains direct insults and sarcasm."
        )

        # Verify that the API was actually called with the correct model
        mock_client_instance.models.generate_content.assert_called_once()
        call_args = mock_client_instance.models.generate_content.call_args
        self.assertEqual(call_args.kwargs["model"], "gemini-3-flash-preview")

    @patch("toxicity_classifier.genai.Client")
    def test_api_failure_handling(self, mock_client_class):
        """
        Test Case 2: API Error.
        Verifies that the function handles exceptions (e.g., network error, invalid key) gracefully
        and returns None instead of crashing.
        """
        # 1. Setup the Mock to raise an exception
        mock_client_instance = mock_client_class.return_value
        mock_client_instance.models.generate_content.side_effect = Exception(
            "API Connection Failed"
        )

        # 2. Execute the function
        result = analyze_text_toxicity("Some text")

        # 3. Assertions
        # The function should catch the exception and return None (as defined in our script)
        self.assertIsNone(result)

    @patch("toxicity_classifier.genai.Client")
    def test_empty_string_input(self, mock_client_class):
        """
        Test Case 3: Edge Case - Empty String.
        Even if we send an empty string, we want to ensure the function attempts to process it
        or handles the response correctly.
        """
        # Setup mock to return a neutral result for empty text
        neutral_result = ToxicityAnalysis(
            toxicity=0.0,
            severe_toxicity=0.0,
            obscene=0.0,
            threat=0.0,
            insult=0.0,
            identity_attack=0.0,
            sexual_explicit=0.0,
            deciding_fragments=[],
            ambiguous_fragments=[],
            justification="No text provided.",
        )

        mock_client_instance = mock_client_class.return_value
        mock_response = MagicMock()
        mock_response.parsed = neutral_result
        mock_client_instance.models.generate_content.return_value = mock_response

        # Execute with empty string
        result = analyze_text_toxicity("")

        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.toxicity, 0.0)


if __name__ == "__main__":
    unittest.main()
