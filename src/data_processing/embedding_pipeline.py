from typing import List, Optional, Dict
import torch
from transformers import AutoTokenizer, AutoModel
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import numpy as np
from text_splitter import BoundaryAwareTextSplitter
import os
import logging
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        collection_name: str = "scientific_papers",
        dim: int = 1024,
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        logger.info(f"Initializing DocumentEmbedder with model: {model_name}")
        
        # Initialize the embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.device = device
        
        # Initialize text splitter
        self.text_splitter = BoundaryAwareTextSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
        
        # Connect to Milvus
        connections.connect(host=milvus_host, port=milvus_port)
        
        # Create collection if it doesn't exist
        self.collection_name = collection_name
        if not utility.has_collection(collection_name):
            self._create_collection(dim)
        
        self.collection = Collection(collection_name)
        self.collection.load()

    def _create_collection(self, dim: int):
        """Create a new Milvus collection with rich metadata schema."""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim),
            # Metadata fields for better reranking
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="subsection", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="hierarchy_level", dtype=DataType.INT8),
            FieldSchema(name="importance_score", dtype=DataType.FLOAT),
            FieldSchema(name="citation_count", dtype=DataType.INT32),
            FieldSchema(name="equation_count", dtype=DataType.INT32),
            FieldSchema(name="is_abstract", dtype=DataType.BOOL),
            FieldSchema(name="is_conclusion", dtype=DataType.BOOL),
            FieldSchema(name="word_count", dtype=DataType.INT32),
            FieldSchema(name="chunk_index", dtype=DataType.INT32),
            # Source metadata
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="processing_timestamp", dtype=DataType.VARCHAR, max_length=30),
        ]
        schema = CollectionSchema(
            fields=fields,
            description="Scientific paper chunks collection with rich metadata"
        )
        Collection(self.collection_name, schema)

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of text chunks."""
        # Tokenize and encode text
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Use CLS token embedding
            embeddings = model_output.last_hidden_state[:, 0]
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()

    def read_scientific_paper(self, file_path: str) -> str:
        """Read and preprocess the scientific paper from final_text.txt"""
        logger.info(f"Reading scientific paper from: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise

    def process_document(self, text: str, metadata: Optional[Dict] = None, batch_size: int = 32):
        """Process a document by splitting it with metadata and generating embeddings."""
        logger.info("Starting document processing")
        
        # Split text into chunks with metadata
        chunks_with_metadata = self.text_splitter.split_text(text)
        logger.info(f"Split document into {len(chunks_with_metadata)} chunks with metadata")
        
        # Add document-level metadata
        doc_metadata = metadata or {}
        source_file = doc_metadata.get("source_file", "unknown")
        processing_timestamp = doc_metadata.get("processing_timestamp", datetime.datetime.now().isoformat())
        
        # Process chunks in batches
        total_processed = 0
        for i in range(0, len(chunks_with_metadata), batch_size):
            batch = chunks_with_metadata[i:i + batch_size]
            batch_chunks = [item[0] for item in batch]
            batch_chunk_metadata = [item[1] for item in batch]
            
            batch_embeddings = self._get_embeddings(batch_chunks)
            
            # Insert into Milvus with full metadata
            entities = []
            for chunk, chunk_meta, embedding in zip(
                batch_chunks,
                batch_chunk_metadata,
                batch_embeddings
            ):
                entity = {
                    "text": chunk,
                    "embeddings": embedding.tolist(),
                    # Chunk-level metadata
                    "section": chunk_meta.get("section", "unknown"),
                    "subsection": chunk_meta.get("subsection", ""),
                    "content_type": chunk_meta.get("content_type", "text"),
                    "hierarchy_level": chunk_meta.get("hierarchy_level", 0),
                    "importance_score": chunk_meta.get("importance_score", 0.5),
                    "citation_count": chunk_meta.get("citation_count", 0),
                    "equation_count": chunk_meta.get("equation_count", 0),
                    "is_abstract": chunk_meta.get("is_abstract", False),
                    "is_conclusion": chunk_meta.get("is_conclusion", False),
                    "word_count": chunk_meta.get("word_count", 0),
                    "chunk_index": chunk_meta.get("chunk_index", i),
                    # Source metadata
                    "source_file": source_file,
                    "processing_timestamp": processing_timestamp,
                }
                entities.append(entity)
            
            self.collection.insert(entities)
            
            total_processed += len(batch)
            logger.info(f"Processed {total_processed}/{len(chunks_with_metadata)} chunks")
        
        # Flush to ensure data is written
        self.collection.flush()
        logger.info("Document processing completed successfully")

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.5
    ) -> List[dict]:
        """Search for similar chunks using the query."""
        # Generate query embedding
        query_embedding = self._get_embeddings([query])[0]
        
        # Search in Milvus
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embeddings",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )
        
        # Format results
        matches = []
        for hits in results:
            for hit in hits:
                if hit.score >= score_threshold:
                    matches.append({
                        "text": hit.entity.get("text"),
                        "score": hit.score
                    })
        
        return matches

    def process_scientific_paper(self, file_path: str, paper_metadata: Optional[Dict] = None):
        """Process a scientific paper from file and store in vector database"""
        try:
            # Read and process the paper
            content = self.read_scientific_paper(file_path)
            
            # Add basic paper metadata if not provided
            if paper_metadata is None:
                paper_metadata = {
                    "source_file": file_path,
                    "paper_type": "scientific_paper",
                    "processing_timestamp": datetime.datetime.now().isoformat()
                }
            
            # Process the document with metadata
            self.process_document(content, metadata=paper_metadata)
            logger.info(f"Successfully processed paper: {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing paper {file_path}: {str(e)}")
            raise

    def close(self):
        """Clean up connections."""
        try:
            self.collection.release()
            connections.disconnect("default")
            logger.info("Successfully closed all connections")
        except Exception as e:
            logger.error(f"Error closing connections: {str(e)}")
            raise
