"""
Qwen-specific analyzer implementation.
"""

from typing import Dict, Any

from ..base_analyzer import CodeHallucinationAnalyzer


class QwenAnalyzer(CodeHallucinationAnalyzer):
    """
    Code hallucination analyzer implementation for Qwen models.
    
    This implementation provides configuration specific to Qwen API.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "qwen3",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        **kwargs
    ):
        """
        Initialize a Qwen-specific analyzer.
        
        Args:
            api_key: Qwen API key
            model: Qwen model name (default: "qwen3")
            base_url: Qwen API base URL (default: "https://dashscope.aliyuncs.com/compatible-mode/v1")
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
        Get Qwen-specific parameters for API calls.
        
        Returns:
            Dictionary with Qwen-specific parameters
        """
        # Qwen requires the enable_thinking parameter set to False
        return {
            "extra_body": {"enable_thinking": False}
        } 