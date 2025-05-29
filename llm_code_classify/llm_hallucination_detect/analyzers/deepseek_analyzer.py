"""
DeepSeek-specific analyzer implementation.
"""

from typing import Dict, Any

from ..base_analyzer import CodeHallucinationAnalyzer


class DeepseekAnalyzer(CodeHallucinationAnalyzer):
    """
    Code hallucination analyzer implementation for DeepSeek models.
    
    This implementation provides configuration specific to DeepSeek API.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        **kwargs
    ):
        """
        Initialize a DeepSeek-specific analyzer.
        
        Args:
            api_key: DeepSeek API key
            model: DeepSeek model name (default: "deepseek-chat")
            base_url: DeepSeek API base URL (default: "https://api.deepseek.com/v1")
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
        Get DeepSeek-specific parameters for API calls.
        
        Returns:
            Empty dictionary as DeepSeek doesn't need special parameters
        """
        return {} 