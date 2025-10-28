# llm/deepseek_client.py
"""
DeepSeek client with debug logging to identify JSON parsing issues.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
import os
import requests
from dotenv import load_dotenv


class DeepSeekClient:
    """
    DeepSeek client with detailed logging.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        max_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: float = 1.0,
    ):
        load_dotenv()
        self.model = model
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def __call__(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Direct callable interface"""
        return self.generate(messages, json_schema=json_schema)

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate response with detailed debug logging"""
        
        # Use provided params or fall back to defaults
        temp = temperature if temperature is not None else self.temperature
        tp = top_p if top_p is not None else self.top_p
        mt = max_tokens if max_tokens is not None else self.max_tokens
        
        # Make the API request
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "top_p": tp,
            "max_tokens": mt,
        }

        # Tell DeepSeek to return JSON when we request it
        if json_schema:
            body["response_format"] = {"type": "json_object"}

        # POST request
        response = requests.post(url, headers=headers, json=body, timeout=90.0)
        
        if response.status_code != 200:
            raise RuntimeError(f"DeepSeek API error: {response.status_code} {response.text[:200]}")

        # Extract the text response
        data = response.json()
        text = (data["choices"][0]["message"]["content"] or "").strip()

        # If no schema requested, just return text
        if not json_schema:
            return {"text": text}

        # DEBUG: Print what we're trying to parse
        # print(f"[DeepSeekClient] text length: {len(text)}")
        # print(f"[DeepSeekClient] text starts with: {text[:100]}")
        # print(f"[DeepSeekClient] text ends with: {text[-100:]}")
        # print(f"[DeepSeekClient] text type: {type(text)}")
        
        # Check if it's wrapped in quotes (string representation of JSON)
        if text.startswith('"') and text.endswith('"'):
            # print("[DeepSeekClient] WARNING: Text is quoted! Unquoting...")
            text = text[1:-1]
            # print(f"[DeepSeekClient] After unquote: {text[:100]}")

        # If schema requested, try to parse as JSON
        try:
            result = json.loads(text)
            # print(f"[DeepSeekClient] ✓ json.loads() succeeded!")
            # print(f"[DeepSeekClient] result type: {type(result)}")
            # print(f"[DeepSeekClient] result keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
            
            if isinstance(result, dict):
                return result
            else:
                # print(f"[DeepSeekClient] ✗ Result is not dict, got: {type(result)}")
                return {"_raw": text}
                
        except json.JSONDecodeError as e:
            # print(f"[DeepSeekClient] ✗ json.loads() FAILED!")
            # print(f"[DeepSeekClient] Error: {e}")
            # print(f"[DeepSeekClient] Error position: {e.pos}")
            if e.pos is not None and e.pos < len(text):
                # print(f"[DeepSeekClient] Text around error: {text[max(0,e.pos-50):e.pos+50]}")
                a = "ok"
            return {"_raw": text}


# For compatibility with code that expects DeepSeekLLM
DeepSeekLLM = DeepSeekClient