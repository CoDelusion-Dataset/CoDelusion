"""
GPT/OpenAI-specific analyzer implementation.
"""

from typing import Dict, Any

from ..base_analyzer import CodeHallucinationAnalyzer


class GPTAnalyzer(CodeHallucinationAnalyzer):
    """
    Code hallucination analyzer implementation for OpenAI GPT models.
    
    This implementation provides configuration specific to OpenAI API.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1/",
        **kwargs
    ):
        """
        Initialize a GPT-specific analyzer.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model name (default: "gpt-4o")
            base_url: OpenAI API base URL (default: "https://api.openai.com/v1/")
            **kwargs: Additional parameters passed to the base class
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            **kwargs
        )

    def _get_model_specific_params(self) -> Dict[str, Any]:
        """
        Get OpenAI-specific parameters for API calls.
        
        Returns:
            Empty dictionary as standard OpenAI doesn't need special parameters
        """
        return {} 