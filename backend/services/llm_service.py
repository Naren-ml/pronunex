"""
LLM Service for Pronunex.

Unified wrapper for Groq and Cerebras APIs.
Handles rate limiting, fallback, and response validation.

CRITICAL: LLM is ONLY used for generating human-readable feedback text.
LLM must NEVER:
- Score pronunciation
- Detect phonemes
- Judge correctness of speech
"""

import os
import json
import logging
import hashlib
from typing import Optional
from django.conf import settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    Unified LLM service with provider fallback.
    
    Priority order: Groq (fast) -> Cerebras (high-throughput)
    """
    
    def __init__(self):
        self.groq_client = None
        self.cerebras_client = None
        self._init_clients()
    
    def _init_clients(self):
        """Initialize LLM client connections."""
        # Groq
        groq_key = settings.GROQ_API_KEY
        if groq_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=groq_key)
                logger.info("Groq client initialized")
            except ImportError:
                logger.warning("Groq package not installed")
            except Exception as e:
                logger.error(f"Groq init failed: {str(e)}")
        
        # Cerebras
        cerebras_key = settings.CEREBRAS_API_KEY
        if cerebras_key:
            try:
                from cerebras.cloud.sdk import Cerebras
                self.cerebras_client = Cerebras(api_key=cerebras_key)
                logger.info("Cerebras client initialized")
            except ImportError:
                logger.warning("Cerebras package not installed")
            except Exception as e:
                logger.error(f"Cerebras init failed: {str(e)}")
    
    def generate(
        self, 
        prompt: str, 
        provider: str = "auto",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        response_format: str = "json"
    ) -> dict:
        """
        Generate response from LLM with automatic fallback.
        
        Args:
            prompt: The prompt to send
            provider: 'groq', 'cerebras', or 'auto'
            max_tokens: Maximum response tokens
            temperature: Creativity (0-1)
            response_format: 'json' or 'text'
        
        Returns:
            dict: {success: bool, content: str/dict, provider: str}
        """
        providers = self._get_provider_order(provider)
        
        for prov in providers:
            try:
                result = self._call_provider(prov, prompt, max_tokens, temperature)
                
                if result:
                    content = result
                    
                    # Parse JSON if requested
                    if response_format == "json":
                        content = self._parse_json_response(result)
                    
                    # Log for auditing
                    self._log_usage(prov, prompt, result)
                    
                    return {
                        'success': True,
                        'content': content,
                        'provider': prov,
                    }
                    
            except Exception as e:
                logger.warning(f"Provider {prov} failed: {str(e)}")
                continue
        
        return {
            'success': False,
            'error': 'All LLM providers failed',
            'content': None,
        }
    
    def _get_provider_order(self, provider: str) -> list:
        """Get ordered list of providers to try."""
        if provider == "auto":
            order = []
            if self.groq_client:
                order.append("groq")
            if self.cerebras_client:
                order.append("cerebras")
            return order
        return [provider]
    
    def _call_provider(
        self, 
        provider: str, 
        prompt: str, 
        max_tokens: int,
        temperature: float
    ) -> Optional[str]:
        """Call specific LLM provider."""
        
        if provider == "groq" and self.groq_client:
            return self._call_groq(prompt, max_tokens, temperature)
        elif provider == "cerebras" and self.cerebras_client:
            return self._call_cerebras(prompt, max_tokens, temperature)
        
        return None
    
    def _call_groq(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call Groq API."""
        response = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful pronunciation coach. Always respond in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    
    def _call_cerebras(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call Cerebras API."""
        response = self.cerebras_client.chat.completions.create(
            model="llama-3.3-70b",
            messages=[
                {"role": "system", "content": "You are a helpful pronunciation coach. Always respond in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    
    def _parse_json_response(self, content: str) -> dict:
        """
        Parse JSON from LLM response.
        
        Handles various response formats:
        - Pure JSON
        - Markdown code blocks (```json or ```)
        - JSON embedded in text
        """
        import re
        
        original_content = content
        
        # Try 1: Handle markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
        
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass
        
        # Try 2: Find JSON object in text using regex
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, original_content)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Try 3: Find JSON with nested objects
        start_idx = original_content.find('{')
        if start_idx != -1:
            # Find matching closing brace
            brace_count = 0
            for i, char in enumerate(original_content[start_idx:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = original_content[start_idx:start_idx + i + 1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            break
        
        # Fallback: Return structured text response
        logger.warning("Failed to parse JSON response, returning as text")
        return {"raw_text": original_content, "parse_error": True}
    
    def _log_usage(self, provider: str, prompt: str, response: str):
        """Log LLM usage for auditing."""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        logger.info(f"LLM usage: provider={provider}, prompt_hash={prompt_hash}, response_len={len(response)}")


# Singleton instance
_llm_service = None


def get_llm_service() -> LLMService:
    """Get or create singleton LLM service."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
