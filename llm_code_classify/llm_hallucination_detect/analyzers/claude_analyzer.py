"""
Anthropic Claude-specific analyzer implementation.
"""

from typing import Dict, Any

from ..base_analyzer import CodeHallucinationAnalyzer


class ClaudeAnalyzer(CodeHallucinationAnalyzer):
    """
    Code hallucination analyzer implementation for Anthropic Claude models.
    
    This implementation provides configuration specific to Anthropic API.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3.5-sonnet",
        base_url: str = "https://api.anthropic.com/v1/",
        **kwargs
    ):
        """
        Initialize a Claude-specific analyzer.
        
        Args:
            api_key: Anthropic API key
            model: Claude model name (default: "claude-3.5-sonnet")
            base_url: Anthropic API base URL (default: "https://api.anthropic.com/v1/")
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
        Get Anthropic Claude-specific parameters for API calls.
        
        Returns:
            Dictionary with Claude-specific parameters
        """
        # Add any Claude-specific parameters here if needed
        return {} 