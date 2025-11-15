"""
Retriever class cho RAG pipeline
- Load FAISS index và metadata
- Phân loại query để lấy metadata phù hợp
- Search vector trong từng category
"""

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
    """Lưu trữ metadata cho mỗi chunk (định nghĩa lại ở đây để dùng chung)"""
    chunk_id: int  # ID của chunk
    content: str  # Nội dung chunk
    heading: Optional[str]  # Tiêu đề cha


class Retriever:
    """
    Retriever class để tìm kiếm chunks dựa trên semantic search
    với lọc metadata dựa trên query classification
    
    Attributes:
        embeddings_dir: Đường dẫn thư mục chứa embeddings
        model_name: Tên model SentenceTransformer
        model: SentenceTransformer model
        index: FAISS index
        metadata_list: Danh sách metadata
        heading_to_indices: Map từ heading đến indices trong FAISS
    """
    
    def __init__(self, embeddings_dir: str = None):
        """
        Khởi tạo Retriever
        
        Args:
            embeddings_dir: Đường dẫn thư mục chứa embeddings
                           (mặc định: ../../data/embeddings từ vị trí file này)
        """
        if embeddings_dir is None:
            # Đường dẫn mặc định từ vị trí file này
            current_dir = os.path.dirname(os.path.abspath(__file__))
            embeddings_dir = os.path.join(current_dir, '../../data/embeddings')
        
        embeddings_dir = os.path.abspath(embeddings_dir)
        
        if not os.path.exists(embeddings_dir):
            raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")
        
        print(f"📂 Loading embeddings from: {embeddings_dir}")
        
        # Load config
        config_path = os.path.join(embeddings_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.model_name = config['model_name']
        self.embedding_dim = config['embedding_dim']
        
        print(f"📥 Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # Load FAISS index
        index_path = os.path.join(embeddings_dir, 'faiss_index.bin')
        self.index = faiss.read_index(index_path)
        print(f"✅ Loaded FAISS index (ntotal={self.index.ntotal})")
        
        # Load metadata
        metadata_path = os.path.join(embeddings_dir, 'metadata.pkl')
        with open(metadata_path, 'rb') as f:
            self.metadata_list = pickle.load(f)
        print(f"✅ Loaded {len(self.metadata_list)} metadata entries")
        
        # Cache heading embeddings để tối ưu
        self.heading_embeddings_cache = {}
        
        # Tạo mapping từ heading đến indices
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
        
        print(f"✅ Built heading index with {len(self.heading_to_indices)} unique headings")
    

    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Tìm kiếm chunks dựa trên query
        
        Công thức tính điểm tương đồng:
        hybrid_score = 0.6 * tương_đồng_cosin(query_embedding, chunk_embedding) 
                     + 0.4 * tương_đồng_cosin(query_embedding, heading_embedding)
        
        Args:
            query: Query text
            top_k: Số chunks cần lấy
        
        Returns:
            List[Dict]: Danh sách kết quả sắp xếp theo hybrid_score giảm dần
                Keys: chunk_id, content, heading, vector_similarity, heading_similarity, hybrid_score
        """
        print(f"\n🔍 Query: '{query}'")
        
        # Embed query
        query_embedding = self.model.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # Search trên toàn bộ index
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.metadata_list):
                metadata = self.metadata_list[idx]
                
                # Convert L2 distance to cosine similarity
                vector_similarity = max(0.0, 1.0 - (distance ** 2) / 2.0)
                
                # Tính heading similarity
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
        
        # Sort results by hybrid_score descending
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return results


def test_retriever():
    """Test Retriever class"""
    print("=" * 80)
    print("TEST: Retriever")
    print("=" * 80)
    
    # Khởi tạo retriever
    retriever = Retriever()
    
    # Test queries
    test_queries = [
        "explain positional encoding",
    ]
    
    print("\n" + "=" * 80)
    print("TEST: Search Queries")
    print("=" * 80)
    
    for query in test_queries:
        results = retriever.search(query, top_k=3)
        
        # Print results
        print("-" * 80)
        for i, result in enumerate(results, 1):
            print(f"\n  [{i}] Chunk ID {result['chunk_id']}")
            print(f"      Heading: {result['heading'] or 'N/A'}")
            print(f"      Vector Similarity: {result['vector_similarity']:.4f}")
            print(f"      Heading Similarity: {result['heading_similarity']:.4f}")
            print(f"      Hybrid Score: {result['hybrid_score']:.4f}")
            content_display = result['content'][:100]
            if len(result['content']) > 100:
                content_display += "..."
            print(f"      Content: {content_display}")
        print()


if __name__ == "__main__":
    test_retriever()
