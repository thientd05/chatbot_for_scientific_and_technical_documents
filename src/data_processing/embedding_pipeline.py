import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import pickle

from sentence_transformers import SentenceTransformer
import faiss


@dataclass
class ChunkMetadata:
    """Lưu trữ metadata cho mỗi chunk"""
    chunk_id: int  # ID của chunk
    content: str  # Nội dung chunk
    heading: Optional[str]  # Tiêu đề cha


class EmbeddingPipeline:
    """
    Pipeline để embedding text chunks và lưu vào FAISS
    
    Attributes:
        model_name: Tên model sử dụng (mặc định: BAAI/bge-large-en-v1.5)
        embedding_dim: Dimension của embedding (1024 cho BGE large)
        model: SentenceTransformer model
        index: FAISS index
        metadata_list: Danh sách metadata tương ứng với vectors
    """
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        """
        Khởi tạo EmbeddingPipeline
        
        Args:
            model_name: Tên model từ HuggingFace (mặc định: BAAI/bge-large-en-v1.5)
        """
        self.model_name = model_name
        self.embedding_dim = 1024  # BGE large có 1024 dimensions
        
        print(f"📥 Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # FAISS index sử dụng L2 distance
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Lưu metadata tương ứng với mỗi vector
        self.metadata_list: List[ChunkMetadata] = []
        
        print(f"✅ Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_chunks(self, chunks: List[Dict]) -> np.ndarray:
        """
        Embed danh sách chunks
        
        Args:
            chunks: Danh sách dict có keys 'content' và 'metadata'
        
        Returns:
            np.ndarray: Ma trận embedding (n_chunks, embedding_dim)
        """
        print(f"\n📊 Embedding {len(chunks)} chunks...")
        
        # Trích xuất content từ chunks
        contents = [chunk['content'] for chunk in chunks]
        
        # Embed sử dụng model
        embeddings = self.model.encode(contents, show_progress_bar=True)
        
        # Chuyển thành float32 cho FAISS
        embeddings = np.array(embeddings, dtype=np.float32)
        
        print(f"✅ Embedding completed. Shape: {embeddings.shape}")
        
        return embeddings
    
    def add_chunks(self, chunks: List[Dict]) -> None:
        """
        Thêm chunks vào FAISS index
        
        Args:
            chunks: Danh sách dict có keys 'content' và 'metadata'
        """
        # Embed chunks
        embeddings = self.embed_chunks(chunks)
        
        # Thêm vào FAISS index
        self.index.add(embeddings)
        
        # Lưu metadata
        for i, chunk in enumerate(chunks):
            metadata = ChunkMetadata(
                chunk_id=len(self.metadata_list) + i,
                content=chunk['content'],
                heading=chunk['metadata'].get('heading')
            )
            self.metadata_list.append(metadata)
        
        print(f"✅ Added {len(chunks)} chunks to index")
        print(f"   Total chunks in index: {len(self.metadata_list)}")
    
    def save(self, save_dir: str) -> None:
        """
        Lưu FAISS index và metadata
        
        Args:
            save_dir: Đường dẫn thư mục để lưu
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Lưu FAISS index
        index_path = os.path.join(save_dir, 'faiss_index.bin')
        faiss.write_index(self.index, index_path)
        print(f"✅ Saved FAISS index to: {index_path}")
        
        # Lưu metadata
        metadata_path = os.path.join(save_dir, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata_list, f)
        print(f"✅ Saved metadata to: {metadata_path}")
        
        # Lưu metadata dưới dạng JSON để dễ đọc
        metadata_json_path = os.path.join(save_dir, 'metadata.json')
        metadata_json = [asdict(m) for m in self.metadata_list]
        with open(metadata_json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_json, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved metadata (JSON) to: {metadata_json_path}")
        
        # Lưu config
        config = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'num_chunks': len(self.metadata_list)
        }
        config_path = os.path.join(save_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Saved config to: {config_path}")
    
    def get_statistics(self) -> Dict:
        """
        Lấy thống kê về index
        
        Returns:
            Dict: Thống kê bao gồm:
                - num_vectors: Số vectors trong index
                - embedding_dim: Dimension của mỗi vector
                - num_chunks: Số chunks
                - headings: Dict đếm chunks theo heading
        """
        # Đếm chunks theo heading
        heading_counts = {}
        for metadata in self.metadata_list:
            heading = metadata.heading or 'N/A'
            heading_counts[heading] = heading_counts.get(heading, 0) + 1
        
        return {
            'num_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'num_chunks': len(self.metadata_list),
            'headings': heading_counts
        }
    
    def print_statistics(self) -> None:
        """In thống kê index"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 80)
        print("📊 EMBEDDING INDEX STATISTICS")
        print("=" * 80)
        print(f"Number of vectors: {stats['num_vectors']}")
        print(f"Embedding dimension: {stats['embedding_dim']}")
        print(f"Number of chunks: {stats['num_chunks']}")
        print(f"\n📋 Chunks by heading (top 10):")
        
        for heading, count in sorted(stats['headings'].items(), 
                                     key=lambda x: x[1], 
                                     reverse=True)[:10]:
            heading_display = heading[:60] + '...' if len(heading) > 60 else heading
            print(f"  '{heading_display}': {count} chunks")
        
        print("=" * 80 + "\n")


def load_chunks_from_jsonl(jsonl_path: str) -> List[Dict]:
    """
    Load chunks từ JSONL file
    
    Args:
        jsonl_path: Đường dẫn đến file JSONL
    
    Returns:
        List[Dict]: Danh sách chunks
    """
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunk = json.loads(line)
                chunks.append(chunk)
    
    print(f"✅ Loaded {len(chunks)} chunks from {jsonl_path}")
    return chunks


def process_and_embed(
    chunks_jsonl_path: str,
    output_dir: str,
    model_name: str = "BAAI/bge-large-en-v1.5"
) -> EmbeddingPipeline:
    """
    Hàm tiện lợi: Load chunks từ JSONL và tạo embedding index
    
    Args:
        chunks_jsonl_path: Đường dẫn đến chunks.jsonl
        output_dir: Đường dẫn thư mục output
        model_name: Tên model
    
    Returns:
        EmbeddingPipeline: Pipeline đã embedding
    """
    # Load chunks
    chunks = load_chunks_from_jsonl(chunks_jsonl_path)
    
    # Tạo pipeline
    pipeline = EmbeddingPipeline(model_name=model_name)
    
    # Thêm chunks vào index
    pipeline.add_chunks(chunks)
    
    # In thống kê
    pipeline.print_statistics()
    
    # Lưu
    pipeline.save(output_dir)
    
    return pipeline

if __name__ == "__main__":
    process_and_embed(
        chunks_jsonl_path="../../data/splitted/chunks.jsonl",
        output_dir="../../data/embeddings",
        model_name="BAAI/bge-large-en-v1.5"
    )
