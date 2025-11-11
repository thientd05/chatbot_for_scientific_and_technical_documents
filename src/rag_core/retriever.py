from typing import List, Dict, Optional, Tuple
from langchain.schema import BaseRetriever, Document # type: ignore
from langchain.vectorstores import Milvus # type: ignore
from langchain.embeddings import HuggingFaceBgeEmbeddings # type: ignore
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metadata boosting weights for reranking
METADATA_BOOST_WEIGHTS = {
    "is_abstract": 0.3,
    "is_conclusion": 0.2,
    "importance_score": 0.25,
    "content_type": {
        "section_header": 0.2,
        "equation": 0.15,
        "text": 0.1,
        "reference": 0.05,
        "table": 0.12,
        "figure": 0.12,
    }
}

class EnhancedRAGRetriever(BaseRetriever):
    """Enhanced retriever with metadata-aware scoring for better reranking"""
    
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        milvus_connection_args: Dict = {
            "host": "localhost",
            "port": "19530",
            "collection_name": "scientific_papers"
        },
        top_k: int = 5,
        score_threshold: float = 0.6,
        model_kwargs: Optional[dict] = {"device": "cuda"},
        encode_kwargs: Optional[dict] = {"normalize_embeddings": True}
    ):
        """Initialize the retriever with advanced features"""
        super().__init__()
        
        logger.info(f"Initializing EnhancedRAGRetriever with model: {embedding_model_name}")
        
        # Initialize embedding model
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        # Initialize Milvus vector store
        self.vectorstore = Milvus(
            embedding_function=self.embeddings,
            collection_name=milvus_connection_args["collection_name"],
            connection_args=milvus_connection_args
        )
        
        self.top_k = top_k
        self.score_threshold = score_threshold

    def _enhance_retrieved_docs(
        self, 
        docs: List[Document], 
        query: str
    ) -> List[Document]:
        """
        Enhance retrieved documents with metadata-aware scoring.
        Boost scores based on document importance and relevance to academic papers.
        """
        enhanced_docs = []
        
        # Sort by similarity score
        docs = sorted(docs, key=lambda x: x.metadata.get("score", 0), reverse=True)
        
        for doc in docs:
            # Skip documents below threshold
            if doc.metadata.get("score", 0) < self.score_threshold:
                continue
            
            # Get base semantic similarity score
            base_score = doc.metadata.get("score", 0)
            boost_score = 0.0
            
            # Apply metadata-based boosting
            # Boost for abstract sections
            if doc.metadata.get("is_abstract", False):
                boost_score += METADATA_BOOST_WEIGHTS["is_abstract"]
            
            # Boost for conclusion sections
            if doc.metadata.get("is_conclusion", False):
                boost_score += METADATA_BOOST_WEIGHTS["is_conclusion"]
            
            # Boost based on importance score
            importance = doc.metadata.get("importance_score", 0.5)
            boost_score += importance * METADATA_BOOST_WEIGHTS["importance_score"]
            
            # Boost based on content type
            content_type = doc.metadata.get("content_type", "text")
            content_boost = METADATA_BOOST_WEIGHTS["content_type"].get(content_type, 0.1)
            boost_score += content_boost
            
            # Boost based on citations (more citations = more important)
            citation_count = doc.metadata.get("citation_count", 0)
            citation_boost = min(citation_count / 5, 0.15) * 0.1
            boost_score += citation_boost
            
            # Calculate final enhanced score
            final_score = min(1.0, base_score + boost_score * 0.3)  # Cap at 1.0, boost contributes 30%
            
            # Enhance metadata with computed scores
            doc.metadata.update({
                "section": doc.metadata.get("section", "unknown"),
                "subsection": doc.metadata.get("subsection", ""),
                "content_type": content_type,
                "relevance_score": final_score,
                "semantic_score": base_score,
                "metadata_boost": boost_score,
                "importance_score": importance,
                "citation_count": citation_count,
            })
            
            enhanced_docs.append(doc)
        
        # Sort by final enhanced score
        enhanced_docs = sorted(enhanced_docs, key=lambda x: x.metadata.get("relevance_score", 0), reverse=True)
        
        return enhanced_docs

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get relevant documents using semantic search with metadata enhancement.
        
        Args:
            query: The search query
            
        Returns:
            List of Document objects sorted by enhanced relevance score
        """
        logger.info(f"Retrieving documents for query: {query}")
        
        try:
            # Retrieve documents from vector store with higher k for better metadata filtering
            docs = self.vectorstore.similarity_search_with_score(
                query,
                k=self.top_k * 2  # Retrieve more to allow for filtering
            )
            
            # Convert to Document objects with scores
            scored_docs = []
            for doc, score in docs:
                doc.metadata["score"] = score
                scored_docs.append(doc)
            
            # Enhance and filter documents with metadata-aware boosting
            enhanced_docs = self._enhance_retrieved_docs(scored_docs, query)
            
            # Return only top_k after enhancement
            final_docs = enhanced_docs[:self.top_k]
            
            logger.info(
                f"Retrieved {len(final_docs)} relevant documents. "
                f"Top doc: {final_docs[0].metadata.get('section', 'unknown')} "
                f"(score: {final_docs[0].metadata.get('relevance_score', 0):.3f})"
            )
            return final_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise

    def get_relevant_documents_with_metadata_scores(
        self,
        query: str
    ) -> List[Dict]:
        """
        Get relevant documents with detailed metadata scores for analysis.
        
        Args:
            query: The search query
            
        Returns:
            List of dictionaries with document info and scores
        """
        docs = self.get_relevant_documents(query)
        return [
            {
                "text": doc.page_content,
                "section": doc.metadata.get("section", "unknown"),
                "subsection": doc.metadata.get("subsection", ""),
                "content_type": doc.metadata.get("content_type", "text"),
                "relevance_score": doc.metadata.get("relevance_score", 0),
                "semantic_score": doc.metadata.get("semantic_score", 0),
                "metadata_boost": doc.metadata.get("metadata_boost", 0),
                "importance_score": doc.metadata.get("importance_score", 0),
                "citation_count": doc.metadata.get("citation_count", 0),
                "is_abstract": doc.metadata.get("is_abstract", False),
                "is_conclusion": doc.metadata.get("is_conclusion", False),
            }
            for doc in docs
        ]
