from typing import List, Dict, Optional
from langchain.schema import BaseRetriever, Document # type: ignore
from langchain.vectorstores import Milvus # type: ignore
from langchain.embeddings import HuggingFaceBgeEmbeddings # type: ignore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedRAGRetriever(BaseRetriever):
    """Enhanced retriever for RAG with re-ranking and filtering capabilities"""
    
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
        """Enhance retrieved documents with additional metadata and scoring"""
        enhanced_docs = []
        
        # Sort by similarity score
        docs = sorted(docs, key=lambda x: x.metadata.get("score", 0), reverse=True)
        
        for doc in docs:
            # Skip documents below threshold
            if doc.metadata.get("score", 0) < self.score_threshold:
                continue
                
            # Extract section information if available
            section = eval(doc.metadata.get("metadata", "{}")).get("section", "unknown")
            position = eval(doc.metadata.get("metadata", "{}")).get("position", 0)
            
            # Enhance metadata
            doc.metadata.update({
                "section": section,
                "position": position,
                "relevance_score": doc.metadata.get("score", 0)
            })
            
            enhanced_docs.append(doc)
        
        return enhanced_docs

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using semantic search"""
        logger.info(f"Retrieving documents for query: {query}")
        
        try:
            # Retrieve documents from vector store
            docs = self.vectorstore.similarity_search_with_score(
                query,
                k=self.top_k
            )
            
            # Convert to Document objects with scores
            scored_docs = []
            for doc, score in docs:
                doc.metadata["score"] = score
                scored_docs.append(doc)
            
            # Enhance and filter documents
            enhanced_docs = self._enhance_retrieved_docs(scored_docs, query)
            
            logger.info(f"Retrieved and enhanced {len(enhanced_docs)} relevant documents")
            return enhanced_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise

    def get_relevant_documents_and_scores(
        self,
        query: str
    ) -> List[tuple[Document, float]]:
        """Get relevant documents with their scores"""
        docs = self.get_relevant_documents(query)
        return [(doc, doc.metadata.get("relevance_score", 0)) for doc in docs]
