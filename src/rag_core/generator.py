import os
import logging
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeviceManager: 
    @staticmethod
    def get_optimal_device() -> tuple[str, int]:
        device = "cpu"
        n_gpu_layers = 0
        
        if torch.cuda.is_available():
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory
                total_memory_gb = total_memory / (1024**3)
                
                logger.info(f"✅ CUDA available - GPU VRAM: {total_memory_gb:.2f}GB")
                
                device = "cuda"
                
                if total_memory_gb >= 6.0:
                    # GPU 6GB+: đẩy tất cả layers (mô hình có 32 layers)
                    n_gpu_layers = -1
                    logger.info(f"🚀 GPU {total_memory_gb:.2f}GB: Pushing ALL layers to GPU")
                elif total_memory_gb >= 4.0:
                    # GPU 4GB: cân bằng tốt nhất
                    # Đẩy 28-32 layers để tối ưu hóa GPU
                    n_gpu_layers = 32  # Push all layers, rely on KV cache management
                    logger.info(f"🚀 GPU {total_memory_gb:.2f}GB: Pushing {n_gpu_layers} layers to GPU (maximize GPU usage)")
                elif total_memory_gb >= 3.0:
                    # GPU 3GB: push as many layers as possible
                    n_gpu_layers = 24
                    logger.info(f"🚀 GPU {total_memory_gb:.2f}GB: Pushing {n_gpu_layers} layers to GPU (CPU will handle overflow)")
                elif total_memory_gb >= 2.0:
                    # GPU 2GB: minimize but still use GPU
                    n_gpu_layers = 16
                    logger.info(f"⚠️  GPU {total_memory_gb:.2f}GB: Pushing {n_gpu_layers} layers to GPU (heavy CPU fallback)")
                else:
                    # GPU < 2GB: use CPU mostly
                    device = "cpu"
                    logger.warning(f"⚠️  GPU VRAM too small ({total_memory_gb:.2f}GB), using CPU")
                    
            except Exception as e:
                logger.warning(f"⚠️  Error checking CUDA: {e}, falling back to CPU")
                device = "cpu"
        
        # Kiểm tra Metal (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            n_gpu_layers = -1  # Macs typically have good GPU, push all layers
            logger.info("✅ Apple Metal GPU detected - using all GPU layers")
        
        else:
            logger.info("ℹ️  No GPU detected, using CPU")
            device = "cpu"
        
        return device, n_gpu_layers


class Generator:
    MODEL_REPO_ID = "lmstudio-community/Phi-3.1-mini-4k-instruct-GGUF"
    MODEL_FILENAME = "*Q4_K_M.gguf"
    
    def __init__(
        self,
        model_filename: Optional[str] = None,
        n_ctx: int = 2048,
        n_threads: int = -1,  # -1 = use all available
        verbose: bool = False,
    ):
        self.model_filename = model_filename or self.MODEL_FILENAME
        self.n_ctx = n_ctx
        self.n_threads = n_threads if n_threads > 0 else os.cpu_count() or 8
        self.verbose = verbose
        
        self.device, self.n_gpu_layers = DeviceManager.get_optimal_device()
        
        logger.info(f"🔧 Loading model: {self.MODEL_REPO_ID}")
        logger.info(f"📊 Configuration:")
        logger.info(f"   - Context window: {self.n_ctx} tokens")
        logger.info(f"   - Device: {self.device}")
        logger.info(f"   - GPU layers: {self.n_gpu_layers}")
        logger.info(f"   - CPU threads: {self.n_threads}")
        
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
            logger.info("✅ Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            logger.info("💡 Tips:")
            logger.info("   1. Ensure huggingface-hub is installed: pip install huggingface-hub")
            logger.info("   2. Check internet connection")
            logger.info("   3. Check disk space for model cache")
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
        # Format messages theo Phi-3 chat template
        prompt = self._format_prompt(messages)
        
        if self.verbose:
            logger.info(f"\n🎯 Generating with Phi-3 format")
            logger.info(f"   - Messages: {len(messages)}")
            logger.info(f"   - Max tokens: {max_tokens}")
            logger.info(f"   - Temperature: {temperature}")
            logger.info(f"   - Stream: {stream}")
        
        try:
            if stream:
                # Return generator for streaming
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
                # Return complete string
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
            logger.error(f"❌ Generation failed: {e}")
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
        """Generator để stream tokens one-by-one"""
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
        
        # Thêm <|assistant|> tag để bắt đầu sinh response
        prompt += "<|assistant|>\n"
        
        return prompt


def test_generator():
    print("=" * 80)
    print("TEST: Generator")
    print("=" * 80)
    
    try:
        # Khởi tạo generator
        generator = Generator(verbose=True)
        
        # Test 1: Simple completion
        print("\n" + "=" * 80)
        print("TEST 1: Simple text generation")
        print("=" * 80)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain what is artificial intelligence in 3 sentences."}
        ]
        print(f"\nMessages: {messages}")
        response = generator.generate(messages, max_tokens=200, temperature=0.7)
        print(f"\nResponse:\n{response}")
        
        # Test 2: RAG response with streaming
        print("\n" + "=" * 80)
        print("TEST 2: RAG response generation (with streaming)")
        print("=" * 80)
        query = "What is a Transformer architecture?"
        context = [
            "The Transformer is a neural network architecture introduced in the paper 'Attention is All You Need' in 2017.",
            "Transformers use the attention mechanism to learn relationships between words in a sentence.",
            "The Transformer architecture consists of an encoder and decoder stack.",
        ]
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Based on the provided context, answer the user's question accurately and in detail. If the information is not in the context, clearly state that you don't have enough information."},
            {"role": "user", "content": f"Context:\n\n" + "\n\n".join([f"[Passage {i+1}]:\n{chunk}" for i, chunk in enumerate(context)]) + f"\n\nQuestion: {query}"}
        ]
        print(f"\nQuery: {query}")
        print("\nResponse (streaming):")
        response_generator = generator.generate(messages, max_tokens=300, temperature=0.7, stream=True)
        # Iterate through the generator and print tokens as they come
        for token in response_generator:
            print(token, end="", flush=True)
        print()  # Newline after streaming completes
        
        # Test 3: RAG response without streaming
        print("\n" + "=" * 80)
        print("TEST 3: RAG response generation (without streaming)")
        print("=" * 80)
        print(f"\nQuery: {query}")
        print("\nResponse (complete):")
        response = generator.generate(messages, max_tokens=300, temperature=0.7, stream=False)
        print(response)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    test_generator()
