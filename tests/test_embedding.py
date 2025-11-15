"""
Test script cho EmbeddingPipeline
"""

import os
from embedding_pipeline import EmbeddingPipeline, load_chunks_from_jsonl, process_and_embed


def test_embedding_pipeline():
    """Test embedding pipeline"""
    print("=" * 80)
    print("TEST: Embedding Pipeline")
    print("=" * 80)
    
    # Đường dẫn
    chunks_jsonl_path = '/home/thienta/HUST_20235839/AI/rag/data/splitted/chunks.jsonl'
    output_dir = '/home/thienta/HUST_20235839/AI/rag/data/embeddings'
    
    if not os.path.exists(chunks_jsonl_path):
        print(f"❌ Không tìm thấy file chunks: {chunks_jsonl_path}")
        return
    
    print(f"\n📂 Input: {chunks_jsonl_path}")
    print(f"📂 Output: {output_dir}")
    
    # Process và embed
    pipeline = process_and_embed(chunks_jsonl_path, output_dir)
    
    # Test search
    print("\n" + "=" * 80)
    print("TEST: Search")
    print("=" * 80)
    
    test_queries = [
        "attention mechanism",
        "transformer architecture",
        "neural networks",
        "machine translation"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 80)
        
        results = pipeline.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n  [{i}] Chunk ID {result['chunk_id']}")
            print(f"      Heading: {result['heading'] or 'N/A'}")
            print(f"      Similarity: {result['similarity']:.4f}")
            print(f"      Content: {result['content'][:100]}{'...' if len(result['content']) > 100 else ''}")
    
    print("\n" + "=" * 80)
    print("✅ Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_embedding_pipeline()
