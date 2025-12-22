"""
Local model inference for sentence splitting using HuggingFace transformers.
Supports Llama, Qwen, and other local models with CUDA acceleration.
Optimized for RTX A4500 (20GB VRAM).
"""

import os
import torch
from typing import List, Optional, Generator
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)

# Model configurations
# LOCAL_MODELS = {
#     "llama-3.1-1b": "meta-llama/Llama-3.2-1B-Instruct",
#     "llama-3.1-3b": "meta-llama/Llama-3.2-3B-Instruct",
#     "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
#     "qwen3-8b": "Qwen/Qwen2.5-7B-Instruct",
#     "glm-9b": "THUDM/glm-4-9b-chat",
# }
LOCAL_MODELS = {
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3-8b": "Qwen/Qwen3-8B",
}


class LocalModelInference:
    """
    Class for running inference with local HuggingFace models.
    Optimized for RTX A4500 with batched inference.
    """
    
    def __init__(
        self, 
        model_key: str,
        use_quantization: bool = False,
        device: str = "cuda"
    ):
        self.model_key = model_key
        self.model_name = LOCAL_MODELS.get(model_key, model_key)
        self.device = device
        
        print(f"Loading model: {self.model_name}")
        
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            torch.cuda.empty_cache()
        else:
            print("CUDA not available, using CPU")
            self.device = "cpu"
        
        # Quantization for larger models
        quantization_config = None
        if use_quantization and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"  # Important for batch generation
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model with optimizations
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto",
            "attn_implementation": "sdpa",  # Use scaled dot product attention
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        self.model.eval()
        
        # Compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("Model compiled with torch.compile")
            except Exception as e:
                print(f"torch.compile not available: {e}")
        
        print(f"Model loaded successfully")
    
    @torch.inference_mode()
    def generate_batch(
        self, 
        prompts: List[str], 
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        batch_size: int = 8,
    ) -> List[str]:
        """
        Batch generation for efficiency.
        """
        all_responses = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Apply chat template to all prompts
            formatted_prompts = []
            for prompt in batch_prompts:
                messages = [{"role": "user", "content": prompt}]
                try:
                    formatted = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    formatted_prompts.append(formatted)
                except Exception:
                    formatted_prompts.append(prompt)
            
            # Tokenize batch
            inputs = self.tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Generate
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            
            # Decode responses
            for j, output in enumerate(outputs):
                input_len = inputs['input_ids'][j].shape[0]
                generated = output[input_len:]
                response = self.tokenizer.decode(generated, skip_special_tokens=True)
                all_responses.append(response.strip())
        
        return all_responses
    
    @torch.inference_mode()
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> str:
        """Single prompt generation."""
        messages = [{"role": "user", "content": prompt}]
        
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            formatted = prompt
        
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        return response.strip()
    
    def cleanup(self):
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def run_local_inference(
    model_key: str,
    prompts: List[str],
    use_quantization: bool = False,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    show_progress: bool = True,
) -> List[str]:
    """
    Run batched inference on prompts using a local model.
    Optimized for RTX A4500.
    """
    model = LocalModelInference(
        model_key=model_key,
        use_quantization=use_quantization
    )
    
    responses = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    
    try:
        with tqdm(total=len(prompts), desc=f"Running {model_key}", disable=not show_progress) as pbar:
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                batch_responses = model.generate_batch(
                    batch,
                    max_new_tokens=max_new_tokens,
                    batch_size=len(batch)
                )
                responses.extend(batch_responses)
                pbar.update(len(batch))
    finally:
        model.cleanup()
    
    return responses


def get_available_local_models() -> List[str]:
    """Return list of available local model keys."""
    return list(LOCAL_MODELS.keys())


def check_cuda_status():
    """Print CUDA status information."""
    print("\n=== CUDA Status ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  Device {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")
        
        # Memory info
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"  Memory allocated: {allocated:.2f} GB")
        print(f"  Memory reserved: {reserved:.2f} GB")
    print("="*20 + "\n")


if __name__ == "__main__":
    check_cuda_status()
    
    print("Available local models:")
    for key, name in LOCAL_MODELS.items():
        print(f"  {key}: {name}")
