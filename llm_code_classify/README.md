# LLM Code Hallucination Detection Tool

This project provides a set of tools for detecting and evaluating hallucinations produced by Large Language Models (LLMs) during code generation.

## Project Introduction

Large Language Models may produce various types of hallucinations or errors when generating code, such as data type mismatches, logical errors, API usage errors, etc. This project detects and classifies these issues in code using different LLMs (including DeepSeek, GPT-4o, Claude, Llama, and Qwen), and provides detailed analysis reports.

## Project Structure

```
llm_code_classify/
│
├── llm_hallucination_detect/             # Main package directory
│   ├── __init__.py                       # Package initialization
│   ├── base_analyzer.py                  # Base analyzer class with common functionality
│   ├── analyzers/                        # Specific analyzer implementations
│   │   ├── __init__.py
│   │   ├── deepseek_analyzer.py          # DeepSeek implementation
│   │   ├── gpt_analyzer.py               # GPT models implementation
│   │   ├── claude_analyzer.py            # Claude models implementation
│   │   ├── llama_analyzer.py             # Llama models implementation
│   │   └── qwen_analyzer.py              # Qwen models implementation
│   ├── templates/                        # Prompt templates
│   │   ├── __init__.py
│   │   └── hallucination_template.py     # Hallucination detection template
│   └── utils/                            # Utility functions
│       ├── __init__.py
│       └── text_processing.py            # Text processing utilities
│
├── scripts/                              # Executable scripts
│   ├── detect.py                         # Detection script
│   └── evaluate.py                       # Evaluation script
│
├── examples/                             # Usage examples
│   └── basic_usage.py                    # Basic usage examples
```

## Installation

Install from source:

```bash
git clone xxxxx
cd llm_code_classify
pip install -e .
```

## Usage

### Python API

```python
from llm_hallucination_detect.analyzers import GPTAnalyzer

# Initialize an analyzer for a specific model
analyzer = GPTAnalyzer(
    api_key="your_api_key",
    model="gpt-4o"
)

# Analyze a single code snippet
results = analyzer.analyze_code("your_code_here")

# Process a input
analyzer.process_csv(
    input_path="input.csv",
    output_path="results.csv"
)
```

### Command Line Interface

For detection:

```bash
# Set your API key as an environment variable
export OPENAI_API_KEY="your_api_key"

# Run detection 
hallucination-detect input.csv --model gpt --output results.csv
```

For evaluation results:

```bash
# Evaluate detection results
hallucination-evaluate results.csv --graphs
```

## Output Format

Output files will contain the following details:
- Original input 
- `label`: Detected issue type codes
- `description`: Detailed description of issues

## Notes

- Be sure to set environment variables for API keys or provide them directly
- Different LLM APIs may have slightly different interfaces; refer to the specific analyzer implementations

## License

[MIT License](LICENSE)

## Contribution

Issue reports and pull requests are welcome to help improve this project. 