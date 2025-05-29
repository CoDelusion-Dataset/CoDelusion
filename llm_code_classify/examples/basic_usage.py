#!/usr/bin/env python3
"""
Basic usage examples for the LLM Code Hallucination Detection package.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_hallucination_detect.analyzers import (
    DeepseekAnalyzer,
    GPTAnalyzer,
    ClaudeAnalyzer,
    LlamaAnalyzer,
    QwenAnalyzer,
)


def analyze_single_code_example():
    """Example of analyzing a single code snippet."""
    # Replace with your actual API key
    api_key = os.environ.get("OPENAI_API_KEY", "your_api_key_here")
    
    # Initialize analyzer
    analyzer = GPTAnalyzer(
        api_key=api_key,
        model="gpt-4o",
    )
    
    # Sample code with hallucination issues
    code_to_analyze = """
def calculate_average(numbers):
    \"\"\"
    Calculate the average of a list of numbers.
    
    Args:
        numbers: A list of numbers
    
    Returns:
        The average of the numbers
    \"\"\"
    total = 0
    count = 0
    
    # Iterate through the numbers
    for num in numbers:
        # Add the number to the total
        total += num
        count += 1
    
    # Return the average
    if count == 0:
        return "No numbers provided"  # String instead of numeric value
    else:
        return total / count
    """
    
    # Analyze code
    results = analyzer.analyze_code(code_to_analyze)
    
    # Print results
    print("Analysis Results:")
    for issue in results:
        print(f"Label: {issue['label']}")
        print(f"Description: {issue['description']}")
        print(f"Line: {issue['line_number']}")
        print("-" * 40)


def process_csv_example():
    """Example of processing a CSV file with code snippets."""
    # Replace with your actual API key
    api_key = os.environ.get("DEEPSEEK_API_KEY", "your_api_key_here")
    
    # Initialize analyzer
    analyzer = DeepseekAnalyzer(
        api_key=api_key,
        model="deepseek-chat",
    )
    
    # Process a CSV file
    # The CSV should have a 'content' column with code snippets
    input_file = "example_data.csv"
    output_file = "example_results.csv"
    
    print(f"Processing {input_file}...")
    
    # Check if file exists (this is just for the example)
    if not Path(input_file).exists():
        print(f"File {input_file} not found. This is just an example.")
        return
    
    # Process the CSV file
    analyzer.process_csv(input_file, output_file)
    print(f"Results saved to {output_file}")


def using_different_analyzers_example():
    """Example showing how to use different analyzers."""
    print("Available analyzers:")
    print("- DeepseekAnalyzer: For DeepSeek models")
    print("- GPTAnalyzer: For OpenAI GPT models")
    print("- ClaudeAnalyzer: For Anthropic Claude models")
    print("- LlamaAnalyzer: For Llama models")
    print("- QwenAnalyzer: For Qwen models")
    
    print("\nExample initialization:")
    print("""
# For DeepSeek
analyzer = DeepseekAnalyzer(
    api_key="your_deepseek_api_key",
    model="deepseek-chat"
)

# For OpenAI GPT
analyzer = GPTAnalyzer(
    api_key="your_openai_api_key",
    model="gpt-4o"
)

# For Anthropic Claude
analyzer = ClaudeAnalyzer(
    api_key="your_anthropic_api_key",
    model="claude-3.5-sonnet"
)
    """)


if __name__ == "__main__":
    print("LLM Code Hallucination Detection Examples")
    print("=" * 50)
    
    # Uncomment the examples you want to run
    # analyze_single_code_example()
    # process_csv_example()
    using_different_analyzers_example() 