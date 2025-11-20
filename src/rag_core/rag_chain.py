import os
from typing import List, Dict, Optional, Generator as GeneratorType
from dataclasses import dataclass

@dataclass
class ChunkMetadata:
    chunk_id: int 
    content: str 
    heading: Optional[str]

from .retriever import Retriever
from .generator import Generator


class RAGChain:
    def __init__(
        self,
        embeddings_dir: Optional[str] = None,
        top_k: int = 3,
        model_filename: Optional[str] = None,
        n_ctx: int = 2048,
        verbose: bool = False,
    ):
        self.top_k = top_k
        self.verbose = verbose
        try:
            self.retriever = Retriever(embeddings_dir=embeddings_dir)
        except Exception as e:
            raise
        
        try:
            self.generator = Generator(
                model_filename=model_filename,
                n_ctx=n_ctx,
                verbose=verbose,
            )
        except Exception as e:
            raise
        
    
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
        retrieve_top_k = top_k if top_k is not None else self.top_k

        
        try:
            search_results = self.retriever.search(query, top_k=retrieve_top_k)
            context_chunks = [result["content"] for result in search_results]
            
        except Exception as e:
            raise
        
        if system_prompt is None:
            system_prompt = """You are a helpful AI assistant specialized in document understanding and question answering.
Based on the provided context from the document, answer the user's question accurately and in detail.
If the information needed to answer the question is not in the context, clearly state that you don't have enough information.
Always cite the specific parts of the context that support your answer when possible."""
        
        context_text = "\n\n".join([
            f"[Document Passage {i+1}]:\n{chunk}" 
            for i, chunk in enumerate(context_chunks)
        ])
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Document Context:\n\n{context_text}\n\nQuestion: {query}"}
        ]

        try:
            response = self.generator.generate(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
            )
            
            return response
            
        except Exception as e:
            raise
