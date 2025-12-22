"""
OpenRouter API client for sentence splitting task.
Supports multiple models via OpenRouter API.
"""

import os
import json
import time
import requests
from typing import List, Dict, Optional
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenRouter API configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Available models on OpenRouter
OPENROUTER_MODELS = {
    "gpt-oss-120b": "openai/gpt-oss-120b:free",
    "gpt-oss-20b": "openai/gpt-oss-20b:free",
    "kimi-k2": "moonshotai/kimi-k2:free",
    # "qwen3-8b": "qwen/qwen3-8b:free",
    # "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct:free",
    "glm-z1-9b": "thudm/glm-z1-9b-0414:free",
}


class OpenRouterClient:
    """
    Client for making requests to OpenRouter API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key. If None, reads from model_calls.openrouter_api_KEY env var.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/sentence-splitting",
            "X-Title": "Sentence Splitting NLP HW3",
        }
    
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 512,
        enable_reasoning: bool = False,
        retry_count: int = 3,
        retry_delay: float = 2.0,
    ) -> str:
        """
        Make a chat completion request.
        
        Args:
            model: Model identifier (key from OPENROUTER_MODELS or full model string)
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            enable_reasoning: Whether to enable reasoning mode (for compatible models)
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds
            
        Returns:
            Generated response text
        """
        # Resolve model name
        model_id = OPENROUTER_MODELS.get(model, model)
        
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if enable_reasoning:
            payload["reasoning"] = {"enabled": True}
        
        last_error = None
        for attempt in range(retry_count):
            try:
                response = requests.post(
                    OPENROUTER_API_URL,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=120
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data['choices'][0]['message']['content']
                
                elif response.status_code == 429:
                    # Rate limited, wait and retry
                    wait_time = retry_delay * (attempt + 1)
                    print(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                else:
                    error_msg = response.text
                    print(f"API error ({response.status_code}): {error_msg}")
                    last_error = Exception(f"API error: {response.status_code} - {error_msg}")
                    
            except requests.exceptions.Timeout:
                print(f"Request timeout, attempt {attempt + 1}/{retry_count}")
                last_error = Exception("Request timeout")
                
            except Exception as e:
                print(f"Request error: {e}")
                last_error = e
            
            if attempt < retry_count - 1:
                time.sleep(retry_delay)
        
        raise last_error or Exception("Unknown error during API call")
    
    def generate(
        self,
        model: str,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        enable_reasoning: bool = False,
    ) -> str:
        """
        Simple generate interface.
        
        Args:
            model: Model identifier
            prompt: User prompt
            system_message: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            enable_reasoning: Enable reasoning mode
            
        Returns:
            Generated text
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        return self.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_reasoning=enable_reasoning,
        )


def run_openrouter_inference(
    model_key: str,
    prompts: List[str],
    api_key: Optional[str] = None,
    system_message: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 512,
    enable_reasoning: bool = False,
    show_progress: bool = True,
    delay_between_requests: float = 0.5,
) -> List[str]:
    """
    Run inference on a list of prompts using OpenRouter API.
    
    Args:
        model_key: Key from OPENROUTER_MODELS or full model ID
        prompts: List of prompts to process
        api_key: OpenRouter API key
        system_message: Optional system message for all prompts
        temperature: Sampling temperature
        max_tokens: Maximum tokens per response
        enable_reasoning: Enable reasoning mode
        show_progress: Whether to show progress bar
        delay_between_requests: Delay between API calls in seconds
        
    Returns:
        List of generated responses
    """
    client = OpenRouterClient(api_key=api_key)
    
    responses = []
    iterator = tqdm(prompts, desc=f"Running {model_key}") if show_progress else prompts
    
    for prompt in iterator:
        try:
            response = client.generate(
                model=model_key,
                prompt=prompt,
                system_message=system_message,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_reasoning=enable_reasoning,
            )
            responses.append(response)
        except Exception as e:
            print(f"Error processing prompt: {e}")
            responses.append("")
        
        # Rate limiting
        time.sleep(delay_between_requests)
    
    return responses


def get_available_openrouter_models() -> List[str]:
    """Return list of available OpenRouter model keys."""
    return list(OPENROUTER_MODELS.keys())


def test_openrouter_connection(api_key: Optional[str] = None) -> bool:
    """
    Test OpenRouter API connection.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        client = OpenRouterClient(api_key=api_key)
        response = client.generate(
            model="gpt-oss-20b",
            prompt="Say 'hello' and nothing else.",
            max_tokens=10,
        )
        print(f"Connection test successful. Response: {response}")
        return True
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False


if __name__ == "__main__":
    print("Available OpenRouter models:")
    for key, model_id in OPENROUTER_MODELS.items():
        print(f"  {key}: {model_id}")
    
    print("\nTesting OpenRouter connection...")
    test_openrouter_connection()
