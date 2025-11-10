from typing import List, Dict, Optional, Any, Tuple
from .retriever import EnhancedRAGRetriever
from .generator import RAGGenerator
import logging
from dataclasses import dataclass
from langchain.schema import BaseRetriever, Document # type: ignore
from sentence_transformers import CrossEncoder
import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for reranking
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
RERANK_CUTOFF_THRESHOLD = 0.5  # Minimum relevance score to keep after reranking

@dataclass
class RAGResponse:
    """Response structure for RAG chain output"""
    answer: str
    source_documents: List[Document]
    metadata: Dict[str, Any]

class RAGChain:
    """Advanced RAG Chain combining retrieval, reranking, and generation with additional features"""
    
    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        generator: Optional[RAGGenerator] = None,
        retriever_kwargs: Dict = {},
        generator_kwargs: Dict = {},
        enable_reranking: bool = True,
        cross_encoder_model: str = CROSS_ENCODER_MODEL,
        rerank_top_k: int = 100,  # Number of documents to rerank
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the RAG chain with reranking capabilities.
        
        Args:
            retriever: Custom retriever instance
            generator: Custom generator instance
            retriever_kwargs: Arguments for default retriever
            generator_kwargs: Arguments for default generator
            enable_reranking: Whether to use cross-encoder reranking
            cross_encoder_model: Model to use for reranking
            rerank_top_k: Number of documents to consider for reranking
            device: Device to run reranking on
        """
        logger.info("Initializing RAG Chain")
        
        # Initialize retriever with higher initial top_k for reranking
        retriever_kwargs["top_k"] = max(retriever_kwargs.get("top_k", 5), rerank_top_k)
        self.retriever = retriever or EnhancedRAGRetriever(**retriever_kwargs)
        
        # Initialize generator
        self.generator = generator or RAGGenerator(**generator_kwargs)
        
        # Reranking settings
        self.enable_reranking = enable_reranking
        self.rerank_top_k = rerank_top_k
        self.device = device
        
        if enable_reranking:
            logger.info(f"Initializing cross-encoder for reranking: {cross_encoder_model}")
            self.cross_encoder = CrossEncoder(
                cross_encoder_model,
                device=device
            )

    def _rerank_documents(
        self,
        query: str,
        docs: List[Document]
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents using cross-encoder for more accurate relevance scoring.
        
        Args:
            query: User query
            docs: List of retrieved documents
            
        Returns:
            List of (document, score) tuples sorted by relevance
        """
        if not docs:
            return []
        
        # Prepare document texts
        doc_texts = [doc.page_content for doc in docs]
        
        # Create query-document pairs for cross-encoder
        pairs = [[query, text] for text in doc_texts]
        
        # Get cross-encoder scores
        with torch.no_grad():
            scores = self.cross_encoder.predict(
                pairs,
                batch_size=32,
                show_progress_bar=False
            )
        
        # Combine documents with their scores
        doc_scores = list(zip(docs, scores))
        
        # Sort by score in descending order
        reranked_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)
        
        # Filter out low-scoring documents
        filtered_docs = [
            (doc, score) for doc, score in reranked_docs 
            if score >= RERANK_CUTOFF_THRESHOLD
        ]
        
        logger.info(f"Reranked {len(docs)} documents, kept {len(filtered_docs)} after filtering")
        return filtered_docs

    def _process_retrieved_documents(
        self,
        docs: List[Document],
        query: str = "",
        max_tokens: int = 3000
    ) -> List[Dict]:
        """
        Process and filter retrieved documents to fit context window.
        Includes reranking if enabled.
        
        Args:
            docs: Retrieved documents
            query: Original query for reranking
            max_tokens: Maximum tokens to include in context
            
        Returns:
            List of processed document dictionaries
        """
        processed_docs = []
        total_tokens = 0
        
        # Rerank if enabled and we have a query
        if self.enable_reranking and query and docs:
            reranked_docs = self._rerank_documents(query, docs)
            # Use only the documents, scores are stored in metadata
            docs = [doc for doc, score in reranked_docs]
            # Update relevance scores in metadata
            for doc, score in zip(docs, [score for _, score in reranked_docs]):
                doc.metadata["reranked_score"] = float(score)
        
        for doc in docs:
            # Convert document to dict format
            doc_dict = {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            
            # Skip if document would exceed token limit
            doc_tokens = len(doc.page_content.split())
            if total_tokens + doc_tokens > max_tokens:
                continue
                
            processed_docs.append(doc_dict)
            total_tokens += doc_tokens
        
        # Sort by reranked score if available
        if self.enable_reranking:
            processed_docs.sort(
                key=lambda x: x["metadata"].get("reranked_score", 0),
                reverse=True
            )
        
        return processed_docs

    async def agenerate(
        self,
        query: str,
        **kwargs
    ) -> RAGResponse:
        """Asynchronous version of the generate method"""
        try:
            # Retrieve relevant documents
            docs = await self.retriever.aget_relevant_documents(query)
            
            # Process documents
            processed_docs = self._process_retrieved_documents(docs)
            
            # Generate response
            answer = await self.generator.agenerate(
                query=query,
                context_docs=processed_docs,
                **kwargs
            )
            
            return RAGResponse(
                answer=answer,
                source_documents=docs,
                metadata={
                    "num_source_docs": len(docs),
                    "retriever_type": type(self.retriever).__name__,
                    "generator_type": type(self.generator).__name__
                }
            )
            
        except Exception as e:
            logger.error(f"Error in async generation: {str(e)}")
            raise

    def generate(
        self,
        query: str,
        **kwargs
    ) -> RAGResponse:
        """
        Generate response using the RAG pipeline with reranking.
        
        Args:
            query: User query
            **kwargs: Additional arguments for generator
            
        Returns:
            RAGResponse containing answer and source documents
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Retrieve relevant documents
            docs = self.retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(docs)} relevant documents")
            
            # Process documents with reranking
            processed_docs = self._process_retrieved_documents(
                docs,
                query=query  # Pass query for reranking
            )
            logger.info(f"Processed and reranked {len(processed_docs)} documents for context")
            
            # Generate response
            answer = self.generator.generate(
                query=query,
                context_docs=processed_docs,
                **kwargs
            )
            
            return RAGResponse(
                answer=answer,
                source_documents=docs,
                metadata={
                    "num_source_docs": len(docs),
                    "retriever_type": type(self.retriever).__name__,
                    "generator_type": type(self.generator).__name__
                }
            )
            
        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            raise

    def stream(
        self,
        query: str,
        **kwargs
    ):
        """Stream the response token by token"""
        try:
            # Retrieve and process documents
            docs = self.retriever.get_relevant_documents(query)
            processed_docs = self._process_retrieved_documents(docs)
            
            # Stream generation
            for token in self.generator.stream(
                query=query,
                context_docs=processed_docs,
                **kwargs
            ):
                yield token
                
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            raise
