# Project Structure

```
llm_code_classify/
│
├── llm_hallucination_detect/             # Main package directory
│   ├── __init__.py                       # Package initialization
│   ├── base_analyzer.py                  # Base analyzer class with common functionality
│   ├── analyzers/                        # Specific analyzer implementations
│   │   ├── __init__.py
│   │   ├── deepseek_analyzer.py          # DeepSeek specific implementation
│   │   ├── gpt_analyzer.py               # GPT models implementation
│   │   ├── claude_analyzer.py            # Claude models implementation
│   │   ├── llama_analyzer.py             # Llama models implementation
│   │   └── qwen_analyzer.py              # Qwen models implementation
│   ├── templates/                        # Prompt templates
│   │   ├── __init__.py
│   │   └── hallucination_template.py     # Hallucination detection prompt template
│   └── utils/                            # Utility functions
│       ├── __init__.py
│       └── text_processing.py            # Text processing utilities
│
├── scripts/                              # Executable scripts
│   ├── detect.py                         # Script to run hallucination detection
│   └── evaluate.py                       # Script to evaluate detection results
│
├── tests/                                # Tests directory
│   ├── __init__.py
│   ├── test_base_analyzer.py
│   └── test_text_processing.py
│
├── examples/                             # Usage examples
│   └── basic_usage.py
│
├── README.md                             # Project documentation
├── LICENSE                               # License file
├── requirements.txt                      # Project dependencies
├── setup.py                              # Package setup file
└── .gitignore                            # Git ignore file
``` 