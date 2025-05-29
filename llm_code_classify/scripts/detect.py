#!/usr/bin/env python3
"""
Command-line tool for running hallucination detection on code.
"""

import argparse
import os
import sys
import logging
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


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


ANALYZER_MAPPING = {
    "deepseek": DeepseekAnalyzer,
    "gpt": GPTAnalyzer,
    "claude": ClaudeAnalyzer,
    "llama": LlamaAnalyzer,
    "qwen": QwenAnalyzer,
}


def get_api_key_from_env(model_type):
    """Get API key from environment variables based on model type."""
    env_mappings = {
        "deepseek": "DEEPSEEK_API_KEY",
        "gpt": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "llama": "LLAMA_API_KEY",
        "qwen": "QWEN_API_KEY",
    }
    
    env_var = env_mappings.get(model_type)
    if not env_var:
        return None
        
    api_key = os.environ.get(env_var)
    return api_key


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Detect hallucinations in code using LLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_file", 
        help="Path to input CSV file containing code to analyze"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Path to output CSV file (defaults to input file path with suffix)",
        default=None
    )
    
    parser.add_argument(
        "--model", "-m",
        choices=list(ANALYZER_MAPPING.keys()),
        default="deepseek",
        help="Model type to use for detection"
    )
    
    parser.add_argument(
        "--model-name",
        help="Specific model name (if different from default)",
        default=None
    )
    
    parser.add_argument(
        "--api-key", "-k",
        help="API key (if not set, will try to load from environment variables)",
        default=None
    )
    
    parser.add_argument(
        "--base-url",
        help="Custom base URL for API",
        default=None
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Get API key
    api_key = args.api_key or get_api_key_from_env(args.model)
    if not api_key:
        logger.error(f"No API key provided for {args.model}. "
                    f"Set it via --api-key or environment variables.")
        return 1
    
    # Set default output file if not specified
    if not args.output:
        input_base = os.path.splitext(args.input_file)[0]
        args.output = f"{input_base}_{args.model}_results.csv"
    
    # Get analyzer class
    analyzer_cls = ANALYZER_MAPPING[args.model]
    
    # Initialize keyword arguments
    kwargs = {"api_key": api_key}
    if args.model_name:
        kwargs["model"] = args.model_name
    if args.base_url:
        kwargs["base_url"] = args.base_url
    
    # Create analyzer
    logger.info(f"Initializing {args.model} analyzer")
    analyzer = analyzer_cls(**kwargs)
    
    # Run analysis
    try:
        logger.info(f"Processing {args.input_file}")
        analyzer.process_csv(args.input_file, args.output)
        logger.info(f"Analysis complete. Results saved to {args.output}")
        return 0
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 