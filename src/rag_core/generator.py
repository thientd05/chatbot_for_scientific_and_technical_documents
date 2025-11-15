"""
Generator class cho RAG pipeline
- Sử dụng mô hình Phi-3.1-mini-4k-instruct-GGUF (3.8B parameters)
- Tối ưu hóa cho GPU 4GB VRAM
- Tự động lựa chọn thiết bị (GPU/CPU)
- Hỗ trợ streaming generation
"""

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
    """Quản lý thiết bị cho inference tối ưu với GPU hạn chế"""
    
    @staticmethod
    def get_optimal_device() -> tuple[str, int]:
        """
        Xác định thiết bị tối ưu dựa trên GPU VRAM có sẵn
        
        Returns:
            tuple: (device_name, n_gpu_layers)
            - device_name: "cuda", "mps" hoặc "cpu"
            - n_gpu_layers: số layers để đẩy lên GPU (-1 = all, 0 = cpu-only)
        """
        device = "cpu"
        n_gpu_layers = 0
        
        # Kiểm tra CUDA
        if torch.cuda.is_available():
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory
                total_memory_gb = total_memory / (1024**3)
                
                logger.info(f"✅ CUDA available - GPU VRAM: {total_memory_gb:.2f}GB")
                
                # Tối ưu cho Phi-3.1-mini với 4GB VRAM
                # Mô hình kích thước: ~2.39GB (Q4_K_M quant)
                # KV cache + overhead: ~1.5GB
                # Total: ~3.9GB
                
                if total_memory_gb >= 4.0:
                    device = "cuda"
                    # Cho GPU 4GB, đẩy ~20-24 layers (mô hình có ~32 layers)
                    n_gpu_layers = 20
                    logger.info(f"🚀 Optimizing for 4GB GPU: using {n_gpu_layers} GPU layers")
                elif total_memory_gb >= 8.0:
                    device = "cuda"
                    n_gpu_layers = -1  # All layers
                    logger.info(f"🚀 Optimizing for 8GB+ GPU: using all GPU layers")
                else:
                    device = "cpu"
                    logger.warning(f"⚠️  GPU VRAM too small ({total_memory_gb:.2f}GB), using CPU")
                    
            except Exception as e:
                logger.warning(f"⚠️  Error checking CUDA: {e}, falling back to CPU")
                device = "cpu"
        
        # Kiểm tra Metal (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            n_gpu_layers = -1  # Macs typically have good GPU
            logger.info("✅ Apple Metal GPU detected - using all GPU layers")
        
        else:
            logger.info("ℹ️  No GPU detected, using CPU")
            device = "cpu"
        
        return device, n_gpu_layers


class Generator:
    """
    Generator class sử dụng mô hình Phi-3.1-mini-4k-instruct-GGUF
    cho RAG pipeline
    
    Attributes:
        model_repo_id: Hugging Face repo ID của mô hình GGUF
        model_filename: Tên file GGUF (pattern)
        llm: Llama model instance
        device: Device đang sử dụng (cuda/mps/cpu)
        n_gpu_layers: Số layers đẩy lên GPU
    """
    
    # Model configuration
    MODEL_REPO_ID = "lmstudio-community/Phi-3.1-mini-4k-instruct-GGUF"
    MODEL_FILENAME = "*Q4_K_M.gguf"  # Quantized 4-bit (Q4_K_M) - ~2.39GB, tối ưu cho 4GB GPU
    
    # Alternative quantizations cho GPU với VRAM khác nhau:
    # IQ3_M: ~1.86GB (3-bit, rất nhỏ, chất lượng kém)
    # IQ4_XS: ~2.06GB (4-bit, rất tốt, nhỏ)
    # Q4_K_M: ~2.39GB (4-bit, tốt nhất, cân bằng)
    # Q5_K_M: ~2.82GB (5-bit, chất lượng cao)
    # Q8_0: ~4.06GB (8-bit, chất lượng cao nhất)
    
    def __init__(
        self,
        model_filename: Optional[str] = None,
        n_ctx: int = 2048,
        n_threads: int = -1,  # -1 = use all available
        verbose: bool = False,
    ):
        """
        Khởi tạo Generator
        
        Args:
            model_filename: Tên file GGUF (default: Q4_K_M)
            n_ctx: Context window size (default: 2048)
            n_threads: Số threads CPU (default: -1 = tất cả)
            verbose: Log chi tiết (default: False)
        """
        self.model_filename = model_filename or self.MODEL_FILENAME
        self.n_ctx = n_ctx
        self.n_threads = n_threads if n_threads > 0 else os.cpu_count() or 8
        self.verbose = verbose
        
        # Xác định thiết bị tối ưu
        self.device, self.n_gpu_layers = DeviceManager.get_optimal_device()
        
        logger.info(f"🔧 Loading model: {self.MODEL_REPO_ID}")
        logger.info(f"📊 Configuration:")
        logger.info(f"   - Context window: {self.n_ctx} tokens")
        logger.info(f"   - Device: {self.device}")
        logger.info(f"   - GPU layers: {self.n_gpu_layers}")
        logger.info(f"   - CPU threads: {self.n_threads}")
        
        # Load model từ Hugging Face Hub
        try:
            self.llm = Llama.from_pretrained(
                repo_id=self.MODEL_REPO_ID,
                filename=self.model_filename,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=verbose,
                # Phi-3.1 sử dụng chat format khác
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
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
    ) -> str:
        """
        Generate text từ prompt
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens để generate (default: 512)
            temperature: Sampling temperature (default: 0.7)
            top_p: Nucleus sampling parameter (default: 0.95)
            top_k: Top-k sampling parameter (default: 40)
            repeat_penalty: Penalize repetition (default: 1.1)
            stop_sequences: List stop sequences (default: None)
            stream: Stream output token-by-token (default: False)
        
        Returns:
            str: Generated text
        """
        if self.verbose:
            logger.info(f"\n🎯 Generating with parameters:")
            logger.info(f"   - Prompt length: {len(prompt)} chars")
            logger.info(f"   - Max tokens: {max_tokens}")
            logger.info(f"   - Temperature: {temperature}")
        
        try:
            if stream:
                return self._generate_stream(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stop_sequences=stop_sequences,
                )
            else:
                return self._generate_complete(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stop_sequences=stop_sequences,
                )
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            raise
    
    def _generate_complete(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate text hoàn chỉnh (không stream)"""
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop_sequences or [],
            echo=False,
        )
        return output["choices"][0]["text"].strip()
    
    def _generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        stop_sequences: Optional[List[str]] = None,
    ):
        """Generate text với streaming"""
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop_sequences or [],
            echo=False,
            stream=True,
        )
        
        for chunk in output:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("text", "")
                if delta:
                    yield delta
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Chat completion interface (OpenAI-compatible)
        
        Args:
            messages: List of message dicts with "role" and "content"
            max_tokens: Maximum tokens để generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
        
        Returns:
            str: Generated response
        """
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return response["choices"][0]["message"]["content"].strip()
    
    def rag_response(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate response cho RAG pipeline
        
        Args:
            query: User query
            context: List of retrieved context chunks
            system_prompt: Custom system prompt (optional)
            max_tokens: Maximum tokens
            temperature: Sampling temperature
        
        Returns:
            str: Generated response
        """
        if system_prompt is None:
            system_prompt = """You are a helpful AI assistant. 
Based on the provided context, answer the user's question accurately and in detail.
If the information is not in the context, clearly state that you don't have enough information."""
        
        # Combine context
        context_text = "\n\n".join([f"[Passage {i+1}]:\n{chunk}" for i, chunk in enumerate(context)])
        
        # Build prompt
        prompt = f"""<|system|>
{system_prompt}
<|end|>

<|user|>
Context:
{context_text}

Question: {query}
<|end|>

<|assistant|>
"""
        
        return self.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )


def test_generator():
    """Test Generator class"""
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
        prompt = "Explain what is artificial intelligence: "
        print(f"\nPrompt: {prompt}")
        response = generator.generate(prompt, max_tokens=200, temperature=0.7)
        print(f"\nResponse:\n{response}")
        
        # Test 2: RAG response
        print("\n" + "=" * 80)
        print("TEST 2: RAG response generation")
        print("=" * 80)
        query = "What is a Transformer architecture?"
        context = [
            "The Transformer is a neural network architecture introduced in the paper 'Attention is All You Need' in 2017.",
            "Transformers use the attention mechanism to learn relationships between words in a sentence.",
            "The Transformer architecture consists of an encoder and decoder stack.",
        ]
        print(f"\nQuery: {query}")
        response = generator.rag_response(query, context, max_tokens=300)
        print(f"\nResponse:\n{response}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    test_generator()
