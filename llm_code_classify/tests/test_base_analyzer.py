"""Tests for the base analyzer module."""

import unittest
from unittest.mock import patch, MagicMock

from llm_hallucination_detect.base_analyzer import CodeHallucinationAnalyzer


class MockAnalyzer(CodeHallucinationAnalyzer):
    """Mock implementation of analyzer for testing."""
    
    def _get_model_specific_params(self):
        return {}


class TestBaseAnalyzer(unittest.TestCase):
    """Test case for the base analyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_api_key = "test_api_key"
        self.mock_base_url = "https://api.example.com/v1"
        self.mock_model = "test-model"
        
        # Create a patcher for OpenAI client
        self.client_patcher = patch('llm_hallucination_detect.base_analyzer.OpenAI')
        self.mock_client = self.client_patcher.start()
        
        # Set up the analyzer with the mocked client
        self.analyzer = MockAnalyzer(
            api_key=self.mock_api_key,
            base_url=self.mock_base_url,
            model=self.mock_model
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.client_patcher.stop()
    
    def test_init(self):
        """Test the initialization of the analyzer."""
        self.assertEqual(self.analyzer.model, self.mock_model)
        self.assertEqual(self.analyzer.temperature, 0.3)  # Default value
        self.assertEqual(self.analyzer.max_tokens, 1024)  # Default value
        self.assertEqual(self.analyzer.top_p, 0.9)  # Default value
        self.assertEqual(self.analyzer.extra_params, {})  # Default value
        
    @patch('llm_hallucination_detect.base_analyzer.CodeHallucinationAnalyzer._get_api_response')
    def test_analyze_code_success(self, mock_get_api_response):
        """Test successful code analysis."""
        # Set up the mock response with proper formatting to match the pattern
        mock_get_api_response.return_value = """
        Analysis Results
        label: [1.1]
        description: [Data type mismatch at line 5]

        label: [2.2]
        description: [Context inconsistency at line 10]
        """
        
        result = self.analyzer.analyze_code("test code")
        
        # Verify the results
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["label"], "1.1")
        self.assertEqual(result[0]["description"], "Data type mismatch at line 5")
        self.assertEqual(result[0]["line_number"], 5)
        self.assertEqual(result[1]["label"], "2.2")
        self.assertEqual(result[1]["description"], "Context inconsistency at line 10")
        self.assertEqual(result[1]["line_number"], 10)
    
    @patch('llm_hallucination_detect.base_analyzer.CodeHallucinationAnalyzer._get_api_response')
    def test_analyze_code_error(self, mock_get_api_response):
        """Test error handling during code analysis."""
        # Set up the mock to raise an exception
        mock_get_api_response.side_effect = Exception("API error")
        
        result = self.analyzer.analyze_code("test code")
        
        # Verify the results
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["label"], "ERROR")
        self.assertEqual(result[0]["description"], "API error")


if __name__ == '__main__':
    unittest.main() 