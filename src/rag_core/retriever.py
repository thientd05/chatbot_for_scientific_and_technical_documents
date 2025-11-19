import os
import json
import pickle
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss


@dataclass
class ChunkMetadata:
    chunk_id: int  # ID của chunk
    content: str  # Nội dung chunk
    heading: Optional[str]  # Tiêu đề cha


class Retriever:
    
    def __init__(self, embeddings_dir: str = None):
        if embeddings_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            embeddings_dir = os.path.join(current_dir, '../../data/embeddings')
        
        embeddings_dir = os.path.abspath(embeddings_dir)
        
        if not os.path.exists(embeddings_dir):
            raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")
        
        print(f"📂 Loading embeddings from: {embeddings_dir}")
        
        config_path = os.path.join(embeddings_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.model_name = config['model_name']
        self.embedding_dim = config['embedding_dim']
        
        self.model = SentenceTransformer(self.model_name)
        
        index_path = os.path.join(embeddings_dir, 'faiss_index.bin')
        self.index = faiss.read_index(index_path)
        
        metadata_path = os.path.join(embeddings_dir, 'metadata.pkl')
        with open(metadata_path, 'rb') as f:
            self.metadata_list = pickle.load(f)
        
        self.heading_embeddings_cache = {}
        
        self._build_heading_index()
    
    def _build_heading_index(self):
        """Tạo mapping từ heading đến danh sách indices"""
        self.heading_to_indices = {}
        
        for idx, metadata in enumerate(self.metadata_list):
            heading = metadata.heading or 'N/A'
            if heading == "N/A" or heading == "References  ":
                continue
            if heading not in self.heading_to_indices:
                self.heading_to_indices[heading] = []
            self.heading_to_indices[heading].append(idx)
    

    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.model.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        distances, indices = self.index.search(query_embedding, top_k)  
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.metadata_list):
                metadata = self.metadata_list[idx]
                
                vector_similarity = max(0.0, 1.0 - (distance ** 2) / 2.0)
                
                heading_similarity = 0.0
                if metadata.heading:
                    if metadata.heading not in self.heading_embeddings_cache:
                        heading_vec = self.model.encode([metadata.heading], show_progress_bar=False)
                        self.heading_embeddings_cache[metadata.heading] = np.array(heading_vec[0], dtype=np.float32)
                    
                    heading_embedding = self.heading_embeddings_cache[metadata.heading]

                    norm_query = np.linalg.norm(query_embedding[0])
                    norm_heading = np.linalg.norm(heading_embedding)
                    
                    if norm_query > 0 and norm_heading > 0:
                        heading_similarity = float(np.dot(query_embedding[0], heading_embedding) / (norm_query * norm_heading))
                
                # Hybrid score: 0.6 * vector_sim + 0.4 * heading_sim
                hybrid_score = 0.6 * vector_similarity + 0.4 * heading_similarity
                
                results.append({
                    'chunk_id': metadata.chunk_id,
                    'content': metadata.content,
                    'heading': metadata.heading,
                    'vector_similarity': float(vector_similarity),
                    'heading_similarity': float(heading_similarity),
                    'hybrid_score': float(hybrid_score)
                })
        
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return results
