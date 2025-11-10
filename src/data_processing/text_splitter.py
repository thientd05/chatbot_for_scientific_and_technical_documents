from typing import List, Optional
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore

class BoundaryAwareTextSplitter:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        boundary_patterns: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default boundary patterns for scientific papers
        self.boundary_patterns = boundary_patterns or [
            # Headers and sections
            r"\n#{1,6}\s.*\n",         # Markdown headers
            r"\n\d+\.?\d*\s.*\n",      # Numbered sections (including subsections like 3.1)
            
            # Scientific paper specific sections
            r"\nAbstract[:.].*\n",
            r"\nIntroduction[:.].*\n", 
            r"\nBackground[:.].*\n",
            r"\nMethodology[:.].*\n",
            r"\nMethod[:.].*\n",
            r"\nResults[:.].*\n",
            r"\nDiscussion[:.].*\n",
            r"\nConclusion[:.].*\n",
            r"\nReferences[:.].*\n",
            r"\nAcknowledgements[:.].*\n",
            
            # Structural elements
            r"\n\d+\.\d+\s+.*\n",      # Subsection headers
            r"^\s*Table\s+\d+[:.].*$",  # Table captions
            r"^\s*Figure\s+\d+[:.].*$", # Figure captions
            r"<center>.*</center>",      # Centered text (often captions)
            
            # Mathematical content
            r"\$\$.*?\$\$",            # Display equations
            r"\\\[.*?\\\]",            # Alternative equation delimiters
            r"\\begin\{.*?\}.*?\\end\{.*?\}", # LaTeX environments
            
            # Paragraph boundaries 
            r"\n\n+",                   # Multiple newlines
            
            # Citations and references
            r"\[\d+\]",                 # Reference citations
            r"^\[\d+\]\s+.*$",         # Reference list entries
        ]
        
        # Initialize LangChain's RecursiveCharacterTextSplitter as backup
        self.backup_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_text(self, text: str) -> List[str]:
        """
        Split text using boundary-aware approach with fallback to recursive character splitting.
        """
        chunks = []
        current_chunk = ""
        last_split = 0
        
        # First pass: Try to split on natural boundaries
        for pattern in self.boundary_patterns:
            matches = list(re.finditer(pattern, text))
            for match in matches:
                end_pos = match.end()
                
                # If adding the next segment would exceed chunk size
                if len(current_chunk) + (end_pos - last_split) > self.chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = text[last_split:end_pos]
                else:
                    current_chunk += text[last_split:end_pos]
                
                last_split = end_pos
        
        # Add the remaining text
        if text[last_split:]:
            if len(current_chunk) + len(text[last_split:]) <= self.chunk_size:
                current_chunk += text[last_split:]
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = text[last_split:]
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Second pass: If any chunks are still too large, use backup splitter
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size:
                final_chunks.extend(self.backup_splitter.split_text(chunk))
            else:
                final_chunks.append(chunk)
        
        return final_chunks

    def split_documents(self, documents: List[str]) -> List[str]:
        """
        Split multiple documents while maintaining document boundaries.
        """
        all_chunks = []
        for doc in documents:
            chunks = self.split_text(doc)
            all_chunks.extend(chunks)
        return all_chunks
