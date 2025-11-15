import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Chunk:
    """Đại diện cho một chunk text với metadata"""
    content: str  # Nội dung của đoạn văn
    metadata: Dict[str, Optional[str]]  # Metadata: tiêu đề cha gần nhất


class MarkdownChunker:
    """
    Class để chunking văn bản markdown thành các đoạn văn (paragraphs)
    với metadata là tiêu đề cận trên gần nhất (##, ###, hoặc ####)
    """
    
    def __init__(self):
        # Pattern để match các mức heading: ##, ###, ####
        self.heading_pattern = re.compile(r'^(#{2,4}) +(.+)$', re.MULTILINE)
    
    def chunk(self, text: str, min_chunk_length: int = 1) -> List[Chunk]:
        """
        Chia nhỏ văn bản thành các chunk (mỗi dòng là một chunk)
        Metadata là tiêu đề cận trên gần nhất (##, ###, hoặc ####)
        
        Args:
            text: Văn bản markdown cần chunking
            min_chunk_length: Độ dài tối thiểu của một chunk (mặc định 1 ký tự)
        
        Returns:
            List[Chunk]: Danh sách các chunk với metadata
        """
        chunks = []
        
        # Tách văn bản thành các dòng
        lines = text.split('\n')
        
        current_heading = None  # Tiêu đề cận trên gần nhất
        
        for line in lines:
            stripped_line = line.strip()
            
            # Kiểm tra nếu là heading (##, ###, hoặc ####)
            heading_match = self.heading_pattern.match(line)
            
            if heading_match:
                # Cập nhật tiêu đề cận trên
                heading_text = heading_match.group(2)
                current_heading = heading_text
            
            elif stripped_line:  # Nếu là dòng text không trống
                # Bỏ qua hình ảnh, HTML tags, và các ký tự đặc biệt
                if (not stripped_line.startswith('!') and 
                    not stripped_line.startswith('<') and
                    not stripped_line.startswith('|') and
                    len(stripped_line) >= min_chunk_length):
                    
                    chunk = Chunk(
                        content=stripped_line,
                        metadata={
                            'heading': current_heading
                        }
                    )
                    chunks.append(chunk)
        
        return chunks
    
    def to_dict_list(self, chunks: List[Chunk]) -> List[Dict]:
        """
        Chuyển đổi danh sách Chunk thành danh sách dictionary
        
        Args:
            chunks: Danh sách Chunk
        
        Returns:
            List[Dict]: Danh sách dictionary với keys 'content' và 'metadata'
        """
        return [
            {
                'content': chunk.content,
                'metadata': chunk.metadata
            }
            for chunk in chunks
        ]
    
    def save_to_jsonl(self, chunks: List[Chunk], file_path: str) -> None:
        """
        Lưu chunks thành file JSONL (JSON Lines)
        
        Args:
            chunks: Danh sách Chunk
            file_path: Đường dẫn file JSONL cần lưu
        """
        import json
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                line = json.dumps({
                    'content': chunk.content,
                    'metadata': chunk.metadata
                }, ensure_ascii=False)
                f.write(line + '\n')


def main(file_path: str, output_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    chunker = MarkdownChunker()
    chunks = chunker.chunk(text=text, min_chunk_length=5)
    chunker.save_to_jsonl(chunks, output_path)

if __name__ == "__main__":
    main(
        file_path="/home/thienta/HUST_20235839/AI/rag/data/processed/final_text.txt",
        output_path="/home/thienta/HUST_20235839/AI/rag/data/splitted/chunks.jsonl"
    )