"""
Test script cho MarkdownChunker
"""

import sys
import os
from pathlib import Path
from text_splitter import MarkdownChunker, chunk_markdown_file
import json


def test_chunker_with_file():
    """Test chunker với file final_text.txt"""
    print("=" * 80)
    print("TEST: Chunking file final_text.txt")
    print("=" * 80)
    
    # Đường dẫn file tuyệt đối
    file_path = '/home/thienta/HUST_20235839/AI/rag/data/processed/final_text.txt'
    
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file: {file_path}")
        return []
    
    # Chunking
    print(f"📂 Đọc file: {file_path}")
    chunks = chunk_markdown_file(file_path)
    
    print(f"\n📊 Tổng số chunks: {len(chunks)}")
    
    # In ra 10 chunks đầu tiên
    print(f"\n📝 Chi tiết 10 chunks đầu tiên:")
    print("-" * 80)
    
    for i, chunk in enumerate(chunks[:10], 1):
        print(f"\n[Chunk {i}]")
        print(f"  Heading: {chunk.metadata.get('heading', 'N/A')}")
        print(f"  Content: {chunk.content[:100]}{'...' if len(chunk.content) > 100 else ''}")
        print(f"  Length: {len(chunk.content)} ký tự")
    
    print("\n" + "=" * 80)
    
    # Thống kê
    if chunks:
        print("\n📈 Thống kê:")
        print("-" * 80)
        
        # Đếm chunks theo heading
        heading_counts = {}
        for chunk in chunks:
            heading = chunk.metadata.get('heading', 'No heading')
            heading_counts[heading] = heading_counts.get(heading, 0) + 1
        
        print("\n📋 Số chunks theo heading (top 10):")
        for heading, count in sorted(heading_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            heading_display = heading[:60] + '...' if heading and len(heading) > 60 else (heading or 'N/A')
            print(f"  '{heading_display}': {count} chunks")
        
        # Thống kê độ dài
        lengths = [len(chunk.content) for chunk in chunks]
        print(f"\n📊 Độ dài chunks:")
        print(f"  Min: {min(lengths)} ký tự")
        print(f"  Max: {max(lengths)} ký tự")
        print(f"  Trung bình: {sum(lengths) // len(lengths):.0f} ký tự")
    
    print("\n" + "=" * 80)
    
    return chunks

def test_save_to_jsonl(chunks):
    """Test lưu chunks thành file JSONL"""
    print("\n" + "=" * 80)
    print("TEST: Lưu chunks thành file JSONL")
    print("=" * 80)
    
    if not chunks:
        print("❌ Không có chunks để lưu")
        return
    
    # Lưu thành JSONL
    output_path = '/home/thienta/HUST_20235839/AI/rag/data/processed/chunks.jsonl'
    
    chunker = MarkdownChunker()
    chunker.save_to_jsonl(chunks, output_path)
    
    print(f"\n✅ Đã lưu {len(chunks)} chunks thành file: {output_path}")
    
    # In ra 5 dòng đầu tiên của file JSONL
    print(f"\n📝 5 dòng đầu tiên của file JSONL:")
    print("-" * 80)
    
    with open(output_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if i <= 5:
                data = json.loads(line)
                print(f"\nLine {i}:")
                print(f"  Heading: {data['metadata'].get('heading', 'N/A')}")
                print(f"  Content: {data['content'][:80]}{'...' if len(data['content']) > 80 else ''}")
            else:
                break
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    chunks = test_chunker_with_file()
    
    # Test 3: Lưu thành JSONL
    test_save_to_jsonl(chunks)
    
    print("\n✅ Tất cả tests hoàn tất!")
