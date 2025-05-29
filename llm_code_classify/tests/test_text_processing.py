"""Tests for text processing utilities."""

import unittest

from llm_hallucination_detect.utils import clean_description, extract_line_number


class TestTextProcessing(unittest.TestCase):
    """Test cases for text processing utilities."""
    
    def test_clean_description(self):
        """Test clean_description function."""
        # Test basic cleaning
        self.assertEqual(
            clean_description("This is a  test"),
            "This is a test"
        )
        
        # Test with newlines
        self.assertEqual(
            clean_description("This\nis\na\ntest"),
            "This is a test"
        )
        
        # Test with multiple spaces and tabs
        self.assertEqual(
            clean_description("This    is\t\ta      test"),
            "This is a test"
        )
        
        # Test with leading and trailing whitespace
        self.assertEqual(
            clean_description("  \tThis is a test\n  "),
            "This is a test"
        )
    
    def test_extract_line_number(self):
        """Test extract_line_number function."""
        # Test with line number at start
        self.assertEqual(
            extract_line_number("line 42: There is an error"),
            42
        )
        
        # Test with line number in middle
        self.assertEqual(
            extract_line_number("Error at line 123 in the function"),
            123
        )
        
        # Test with uppercase "Line"
        self.assertEqual(
            extract_line_number("Error at Line 99 in the code"),
            99
        )
        
        # Test with no line number
        self.assertIsNone(
            extract_line_number("There is an error in the code")
        )
        
        # Test with multiple line numbers (should return first match)
        self.assertEqual(
            extract_line_number("Error on line 5 and also on line 10"),
            5
        )


if __name__ == '__main__':
    unittest.main() 