"""
Base analyzer for code hallucination detection.
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Any

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

from .templates import HALLUCINATION_DETECTION_PROMPT
from .utils import clean_description, extract_line_number


class CodeHallucinationAnalyzer(ABC):
    """
    Base class for code hallucination analyzers.
    
    This class provides the core functionality for detecting hallucinations
    in code snippets using various LLMs.
    """
    
    def __init__(
        self, 
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        extra_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the code hallucination analyzer.
        
        Args:
            api_key: API key for accessing the LLM service
            base_url: Base URL for the LLM API endpoint
            model: Name of the model to use
            temperature: Temperature parameter for generation (default: 0.3)
            max_tokens: Maximum number of tokens to generate (default: 1024)
            top_p: Top-p sampling parameter (default: 0.9)
            extra_params: Additional model-specific parameters (default: None)
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.extra_params = extra_params or {}
        
        # Regex pattern for extracting labels and descriptions
        self.result_pattern = re.compile(
            r'label:\s*\[(\d+\.\d+)\]\s*'          # Match label (like [2.3])
            r'description:\s*\[(.*?)\]'             # Match description content [description text]
            r'(?=\s*label:|\Z)',                   # Lookahead assertion to ensure match endpoint
            re.DOTALL | re.IGNORECASE
        )

    def analyze_code(self, code: str) -> List[Dict[str, str]]:
        """
        Analyze a single code snippet for hallucinations.
        
        Args:
            code: The code snippet to analyze
            
        Returns:
            A list of dictionaries containing identified hallucination issues
        """
        try:
            response = self._get_api_response(code)
            return self._parse_response(response)
        except Exception as e:
            print(f"Analysis Error: {str(e)}")
            return [{"label": "ERROR", "description": str(e)}]

    def _get_api_response(self, code: str) -> str:
        """
        Get API response with error handling.
        
        Args:
            code: The code snippet to analyze
            
        Returns:
            The raw API response text
            
        Raises:
            Various exceptions depending on API errors
        """
        messages = [
            {"role": "system", "content": "You are a meticulous code quality analyzer"},
            {"role": "user", "content": self._get_prompt_template().format(code=code)}
        ]
        
        # Build API parameters
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            **self._get_model_specific_params()
        }
        
        completion = self.client.chat.completions.create(**params)
        return completion.choices[0].message.content

    def _parse_response(self, response: str) -> List[Dict[str, str]]:
        """
        Parse API response with pattern matching.
        
        Args:
            response: The raw API response text
            
        Returns:
            A list of dictionaries containing structured hallucination data
        """
        results = []
        matches = self.result_pattern.findall(response)
        
        for label, desc in matches:
            # Clean and extract information
            clean_desc = clean_description(desc)
            line_num = extract_line_number(clean_desc)
            
            results.append({
                "label": label.strip(),
                "description": clean_desc,
                "line_number": line_num or 'N/A'
            })
        return results

    def process_csv(self, input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Process a CSV file containing code snippets.
        
        Args:
            input_path: Path to the input CSV file
            output_path: Path to save results (if None, input file will be overwritten)
            
        Returns:
            The processed DataFrame with analysis results
            
        Note:
            If output_path is None and modify_input=True, the input file will be modified.
        """
        # Determine output path
        if output_path is None:
            output_path = input_path
            in_place = True
        else:
            in_place = False
        
        # Read input file
        df = pd.read_csv(input_path, keep_default_na=False)
        
        # Ensure target columns exist
        for col in ['label', 'description']:
            if col not in df.columns:
                df[col] = ''
        
        # Create progress bar
        pbar = tqdm(total=len(df), desc=f"Analyzing with {self.model}")
        
        # Process each row
        modified = False
        for index, row in df.iterrows():
            # Check if we need to process this row
            process_row = (not row['label'] or row['label'] == 'nan') and pd.notnull(row.get('content', ''))
            
            if process_row:
                try:
                    # Get API response
                    response = self._get_api_response(row['content'])
                    
                    # Parse results
                    results = self._parse_response(response)
                    
                    # Combine labels and descriptions
                    labels = []
                    descs = []
                    for item in results:
                        if isinstance(item, dict):
                            labels.append(item.get('label', ''))
                            line = f"[Line {item['line_number']}]" if item.get('line_number') else ""
                            descs.append(f"{line} {item.get('description', '')}")
                    
                    # Update current row
                    df.at[index, 'label'] = ' '.join(labels)
                    df.at[index, 'description'] = ' | '.join(descs)
                    modified = True
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"\nRow {index} analysis failed: {error_msg[:100]}...")
                    df.at[index, 'label'] = 'ERROR'
                    df.at[index, 'description'] = error_msg[:200]
                    modified = True
            
            pbar.update(1)
        
        pbar.close()
        
        # Save results if modified
        if modified or not in_place:
            df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        else:
            print("No records needed processing")
        
        return df

    def _get_prompt_template(self) -> str:
        """
        Get the prompt template for hallucination detection.
        
        Returns:
            The prompt template string
        """
        return HALLUCINATION_DETECTION_PROMPT
    
    @abstractmethod
    def _get_model_specific_params(self) -> Dict[str, Any]:
        """
        Get model-specific parameters for API calls.
        
        Returns:
            A dictionary of model-specific parameters
        """
        pass 