"""
RAG Chain - Retrieval Augmented Generation Pipeline
- Kết hợp Retriever và Generator
- Tìm kiếm context liên quan
- Sinh response dựa trên context
- Hỗ trợ streaming generation
"""

import os
import logging
from typing import List, Dict, Optional, Generator as GeneratorType
from dataclasses import dataclass

# Import ChunkMetadata before Retriever (needed for pickle deserialization)
@dataclass
class ChunkMetadata:
    """Lưu trữ metadata cho mỗi chunk"""
    chunk_id: int  # ID của chunk
    content: str  # Nội dung chunk
    heading: Optional[str]  # Tiêu đề cha

from .retriever import Retriever
from .generator import Generator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGChain:
    """
    RAG Chain - Retrieval Augmented Generation Pipeline
    
    Workflow:
    1. User submits query
    2. Retriever searches for relevant chunks from FAISS index
    3. Top-k context passages are combined with system prompt
    4. Generator creates response based on context
    
    Attributes:
        retriever: Retriever instance for semantic search
        generator: Generator instance for LLM inference
        top_k: Number of context chunks to retrieve (default: 3)
    """
    
    def __init__(
        self,
        embeddings_dir: Optional[str] = None,
        top_k: int = 3,
        model_filename: Optional[str] = None,
        n_ctx: int = 2048,
        verbose: bool = False,
    ):
        """
        Khởi tạo RAG Chain
        
        Args:
            embeddings_dir: Path to embeddings directory (default: auto)
            top_k: Number of context chunks to retrieve (default: 3)
            model_filename: GGUF model filename (default: *Q4_K_M.gguf)
            n_ctx: Context window size (default: 2048)
            verbose: Enable verbose logging (default: False)
        """
        logger.info("🔗 Initializing RAG Chain...")
        
        self.top_k = top_k
        self.verbose = verbose
        
        # Load Retriever
        logger.info("📚 Loading Retriever...")
        try:
            self.retriever = Retriever(embeddings_dir=embeddings_dir)
            logger.info("✅ Retriever loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Retriever: {e}")
            raise
        
        # Load Generator
        logger.info("🤖 Loading Generator...")
        try:
            self.generator = Generator(
                model_filename=model_filename,
                n_ctx=n_ctx,
                verbose=verbose,
            )
            logger.info("✅ Generator loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Generator: {e}")
            raise
        
        logger.info("✅ RAG Chain initialized!")
    
    def generate(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
        stream: bool = False,
    ):
        """
        Generate response từ query sử dụng RAG
        
        Workflow:
        1. Retrieve relevant context chunks
        2. Format messages với system prompt + context + query
        3. Generate response từ LLM
        
        Args:
            query: User query
            system_prompt: Custom system prompt (default: built-in)
            max_tokens: Maximum tokens to generate (default: 512)
            temperature: Sampling temperature (default: 0.7)
            top_p: Nucleus sampling parameter (default: 0.95)
            top_k: Override default top_k for this query (default: None)
            stream: Stream output token-by-token (default: False)
        
        Returns:
            str or Generator: Response text
                - If stream=False: str with full response
                - If stream=True: Generator yielding tokens
        
        Example:
            # Without streaming
            response = rag_chain.generate("What is a Transformer?")
            print(response)
            
            # With streaming
            for token in rag_chain.generate("What is a Transformer?", stream=True):
                print(token, end="", flush=True)
        """
        # Use custom top_k or default
        retrieve_top_k = top_k if top_k is not None else self.top_k
        
        if self.verbose:
            logger.info(f"\n🔍 RAG Query: '{query}'")
            logger.info(f"   - Retrieving top {retrieve_top_k} chunks")
        
        # Step 1: Retrieve context
        try:
            search_results = self.retriever.search(query, top_k=retrieve_top_k)
            context_chunks = [result["content"] for result in search_results]
            
            if self.verbose:
                logger.info(f"   - Retrieved {len(context_chunks)} chunks")
                for i, result in enumerate(search_results, 1):
                    logger.info(f"     [{i}] Score: {result['hybrid_score']:.4f}, Heading: {result['heading']}")
            
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            raise
        
        # Step 2: Build system prompt
        if system_prompt is None:
            system_prompt = """You are a helpful AI assistant specialized in document understanding and question answering.
Based on the provided context from the document, answer the user's question accurately and in detail.
If the information needed to answer the question is not in the context, clearly state that you don't have enough information.
Always cite the specific parts of the context that support your answer when possible."""
        
        # Step 3: Format context
        context_text = "\n\n".join([
            f"[Document Passage {i+1}]:\n{chunk}" 
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Step 4: Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Document Context:\n\n{context_text}\n\nQuestion: {query}"}
        ]
        
        if self.verbose:
            logger.info(f"\n🎯 Generating response...")
            logger.info(f"   - Max tokens: {max_tokens}")
            logger.info(f"   - Temperature: {temperature}")
            logger.info(f"   - Stream: {stream}")
        
        # Step 5: Generate response
        try:
            response = self.generator.generate(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
            )
            
            if not stream:
                if self.verbose:
                    logger.info(f"✅ Response generated successfully")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            raise
    
    def interactive(self):
        """
        Interactive RAG chat mode
        
        Allows users to ask multiple questions in sequence.
        Type 'exit' or 'quit' to end session.
        """
        print("\n" + "=" * 80)
        print("RAG Chat - Document Question Answering")
        print("=" * 80)
        print("Type 'exit' or 'quit' to end session\n")
        
        while True:
            try:
                query = input("\n🔍 Query: ").strip()
                
                if query.lower() in ["exit", "quit"]:
                    print("\n👋 Goodbye!")
                    break
                
                if not query:
                    print("⚠️  Please enter a query")
                    continue
                
                print("\n📝 Response (streaming):")
                print("-" * 80)
                
                # Stream response
                for token in self.generate(query, stream=True):
                    print(token, end="", flush=True)
                
                print("\n" + "-" * 80)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                print(f"⚠️  Error: {e}")
                continue


def test_rag_chain():
    """Test RAG Chain"""
    print("=" * 80)
    print("TEST: RAG Chain")
    print("=" * 80)
    
    try:
        # Initialize RAG Chain
        rag_chain = RAGChain(top_k=3, verbose=True)
        
        # Test queries
        test_queries = [
            "What is positional encoding?",
            "Explain the attention mechanism",
        ]
        
        for query in test_queries:
            print("\n" + "=" * 80)
            print(f"QUERY: {query}")
            print("=" * 80)
            
            print("\n📝 Response (streaming):")
            print("-" * 80)
            
            # Generate with streaming
            response_stream = rag_chain.generate(query, stream=True, max_tokens=300)
            
            # Consume the generator and print tokens
            for token in response_stream:
                print(token, end="", flush=True)
            
            print("\n" + "-" * 80)
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        # Interactive mode
        rag_chain = RAGChain(verbose=True)
        rag_chain.interactive()
    else:
        # Test mode
        test_rag_chain()
