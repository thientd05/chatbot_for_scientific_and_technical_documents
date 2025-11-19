import os
from typing import List, Dict, Optional, Generator
from pathlib import Path
import torch

try:
    from llama_cpp import Llama
except ImportError:
    raise ImportError(
        "llama-cpp-python is not installed. Please install it using:\n"
        "pip install llama-cpp-python"
    )

class DeviceManager: 
    @staticmethod
    def get_optimal_device() -> tuple[str, int]:
        device = "cpu"
        n_gpu_layers = 0
        if torch.cuda.is_available():
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory
                total_memory_gb = total_memory / (1024**3)     
                device = "cuda"
                
                if total_memory_gb >= 6.0:
                    n_gpu_layers = -1
                elif total_memory_gb >= 4.0:
                    n_gpu_layers = 32
                elif total_memory_gb >= 3.0:
                    n_gpu_layers = 24
                elif total_memory_gb >= 2.0:
                    n_gpu_layers = 16
                else:
                    device = "cpu"
                    
            except Exception as e:
                device = "cpu"
        
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            n_gpu_layers = -1
        else:
            device = "cpu"
        
        return device, n_gpu_layers


class Generator:
    MODEL_REPO_ID = "lmstudio-community/Phi-3.1-mini-4k-instruct-GGUF"
    MODEL_FILENAME = "*Q4_K_M.gguf"
    def __init__(
        self,
        model_filename: Optional[str] = None,
        n_ctx: int = 2048,
        n_threads: int = -1,
        verbose: bool = False,
    ):
        self.model_filename = model_filename or self.MODEL_FILENAME
        self.n_ctx = n_ctx
        self.n_threads = n_threads if n_threads > 0 else os.cpu_count() or 8
        self.verbose = verbose
        
        self.device, self.n_gpu_layers = DeviceManager.get_optimal_device()
        
        try:
            self.llm = Llama.from_pretrained(
                repo_id=self.MODEL_REPO_ID,
                filename=self.model_filename,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=verbose,
                chat_format="phi3"
            )
            
        except Exception as e:
            raise
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
    ):
        prompt = self._format_prompt(messages)
        
        try:
            if stream:
                return self._stream_generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stop_sequences=stop_sequences,
                )
            else:
                output = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stop=stop_sequences or ["<|end|>", "<|user|>"],
                    echo=False,
                )
                return output["choices"][0]["text"].strip()
        except Exception as e:
            raise
    
    def _stream_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        stop_sequences: Optional[List[str]] = None,
    ):
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop_sequences or ["<|end|>", "<|user|>"],
            echo=False,
            stream=True,
        )
        
        for chunk in output:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("text", "")
                if delta:
                    yield delta
    
    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt = ""
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt += f"<|system|>\n{content}\n<|end|>\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n<|end|>\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n<|end|>\n"
    
        prompt += "<|assistant|>\n"
        
        return prompt
