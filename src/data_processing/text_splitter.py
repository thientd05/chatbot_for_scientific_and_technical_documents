"""
Advanced text splitter with metadata extraction for scientific papers.
Supports boundary-aware chunking with production-grade citation/equation handling.
Features:
  - Robust citation detection (numeric, author-year, LaTeX \\cite{})
  - Multi-line LaTeX equation support
  - Chronological boundary detection (preserves text order)
  - Smart merging of small blocks for coherence
  - Rich metadata extraction for reranking
"""

from typing import List, Optional, Dict, Tuple, NamedTuple
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Metadata associated with a text chunk"""
    section: str  # Main section (Abstract, Introduction, etc.)
    subsection: str  # Subsection if available
    chunk_index: int  # Index within the section
    hierarchy_level: int  # 0 for main, 1 for subsection, etc.
    content_type: str  # "text", "equation", "figure", "table", "reference"
    has_citations: bool  # Whether chunk contains citations
    citation_count: int  # Number of citations
    has_equations: bool  # Whether chunk contains equations
    equation_count: int  # Number of equations
    is_abstract: bool  # Whether chunk is from abstract (typically more important)
    is_conclusion: bool  # Whether chunk is from conclusion
    importance_score: float  # 0.0-1.0 score based on content type and position

class BoundaryMatch(NamedTuple):
    """Represents a boundary match with its position"""
    start: int
    end: int
    pattern_type: str  # Type of boundary (e.g., "heading", "equation", "paragraph")

class BoundaryAwareTextSplitter:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        boundary_patterns: Optional[List[Tuple[str, str]]] = None,
        min_chunk_size: int = 100,
        merge_small_chunks: bool = True
    ):
        """
        Initialize boundary-aware text splitter.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            boundary_patterns: List of (pattern, pattern_type) tuples
            min_chunk_size: Minimum size before merging
            merge_small_chunks: Whether to merge small chunks for coherence
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.merge_small_chunks = merge_small_chunks
        
        # Boundary patterns with types for better chronological ordering
        self.boundary_patterns = boundary_patterns or [
            # Priority 1: Section headers (must be handled first)
            (r"^#{1,6}\s+(.+)$", "heading_markdown"),  # Markdown headers
            (r"^(\d+(?:\.\d+)*)\s+([A-Z].+?)(?:\n|$)", "heading_numbered"),  # Numbered sections
            
            # Priority 2: Major section boundaries (with case-insensitive for common variations)
            (r"^(Abstract|Introduction|Background|Methodology|Method|Results|"
             r"Discussion|Conclusion|Acknowledgments?|References|Appendix)\b", "section_boundary"),
            (r"^(?:related\s+works?|related\s+work)\b", "section_boundary", re.IGNORECASE),  # Related Work variations
            
            # Priority 3: Equations (multi-line support)
            (r"\\begin\{(?:equation|align|align\*|gather|gather\*|multline|displaymath|equation\*)\}.+?"
             r"\\end\{(?:equation|align|align\*|gather|gather\*|multline|displaymath|equation\*)\}", "equation_block"),
            (r"\$\$.+?\$\$", "equation_inline_display"),
            (r"\\\[.+?\\\]", "equation_latex_brackets"),
            
            # Priority 4: Tables and figures (CAPTION patterns must flush chunk)
            (r"^\s*(?:Table|Figure)\s+\d+(?:\.\d+)?:.*$", "caption"),
            (r"^\s*(?:Table|Figure)\s*\d+", "table_figure"),
            
            # Priority 5: Captions and centered content
            (r"<center>.+?</center>", "centered_text"),
            
            # Priority 6: Lists (numbered and bulleted)
            (r"^[\s]*[\d]+\.\s+", "numbered_list"),
            (r"^[\s]*[-•*]\s+", "bullet_list"),
            
            # Priority 7: Paragraph boundaries
            (r"\n\n+", "paragraph_break"),
            
            # Priority 8: Citations (must be last to not interfere with others)
            (r"\\\cite\{[^}]+\}", "cite_latex"),
            (r"\[(?:\d+(?:,\s*\d+)*|\d+\s*[–-]\s*\d+)\]", "cite_numeric"),
            (r"\((?:[A-Za-z\s]+(?:et\s+al\.)?),?\s*\d{4}[a-z]?\)", "cite_author_year"),
            (r"[A-Z][a-z]+\s+(?:et\s+al\.)?,\s*\d{4}", "cite_author_year_inline"),
        ]
        
        # Initialize LangChain's RecursiveCharacterTextSplitter as backup
        self.backup_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def _find_all_boundaries(self, text: str) -> List[BoundaryMatch]:
        """
        Find ALL boundaries in chronological order (by position in text).
        This is crucial for maintaining text order and avoiding duplicate splits.
        Handles regex flags properly for case-insensitive patterns.
        
        Returns:
            List of BoundaryMatch sorted by start position
        """
        all_matches = []
        
        # Use DOTALL flag to handle multi-line patterns
        for pattern_tuple in self.boundary_patterns:
            # Handle both 2-tuple and 3-tuple formats (pattern, type) or (pattern, type, flags)
            if len(pattern_tuple) == 3:
                pattern, pattern_type, flags = pattern_tuple
                base_flags = re.MULTILINE | re.DOTALL | flags
            else:
                pattern, pattern_type = pattern_tuple
                base_flags = re.MULTILINE | re.DOTALL
            
            try:
                for match in re.finditer(pattern, text, base_flags):
                    # Avoid duplicates at same position
                    if not any(m.start == match.start() and m.end == match.end() for m in all_matches):
                        all_matches.append(BoundaryMatch(
                            start=match.start(),
                            end=match.end(),
                            pattern_type=pattern_type
                        ))
            except re.error as e:
                logger.warning(f"Regex error for pattern {pattern_type}: {e}")
                continue
        
        # Sort by start position (chronological order)
        all_matches.sort(key=lambda x: (x.start, -x.end))  # Sort by start, then longer matches first
        
        # Remove overlapping boundaries (keep the first/longest one)
        filtered_matches = []
        for match in all_matches:
            # Check if this boundary overlaps with an already-added boundary
            overlaps = any(
                (fm.start <= match.start < fm.end) or (fm.start < match.end <= fm.end)
                for fm in filtered_matches
            )
            if not overlaps:
                filtered_matches.append(match)
        
        logger.debug(f"Found {len(filtered_matches)} boundaries in chronological order")
        return filtered_matches

    def _count_citations_advanced(self, chunk: str) -> int:
        """
        Count citations with support for multiple citation formats.
        Handles: [1], [1,2], [1-3], [1–3], (Smith et al., 2023), Smith (2022), \\cite{...}
        """
        count = 0
        
        # Numeric citations: [1], [1, 2], [1–3], [1-3]
        numeric_citations = re.findall(
            r"\[(?:\d+(?:[,\s]+\d+)*|(?:\d+\s*[–-]\s*\d+))\]",
            chunk
        )
        count += len(numeric_citations)
        
        # Author-year citations: (Smith et al., 2023), (Smith, 2023)
        author_year = re.findall(
            r"\([A-Za-z\s]+(?:et\s+al\.)?[,\s]*\d{4}[a-z]?\)",
            chunk
        )
        count += len(author_year)
        
        # Author-year inline: Smith et al. (2023), Smith (2022)
        author_inline = re.findall(
            r"[A-Z][a-z]+(?:\s+et\s+al\.)?[,\s]*\(\d{4}[a-z]?\)",
            chunk
        )
        count += len(author_inline)
        
        # LaTeX citations: \cite{...}
        latex_cites = re.findall(r"\\\cite\{[^}]+\}", chunk)
        count += len(latex_cites)
        
        return count

    def _count_equations_advanced(self, chunk: str) -> int:
        """
        Count equations with support for multi-line LaTeX environments.
        Handles nested structures and various equation environments.
        """
        count = 0
        
        # Display equations with $$
        count += len(re.findall(r"\$\$", chunk)) // 2
        
        # LaTeX bracket equations \\[...\\]
        count += len(re.findall(r"\\\[.*?\\\]", chunk, re.DOTALL))
        
        # Equation environments (align, gather, multline, etc.)
        count += len(re.findall(
            r"\\begin\{(?:equation|align|align\*|gather|gather\*|multline|displaymath|equation\*)\}",
            chunk
        ))
        
        # Inline math with single $
        inline_math = re.findall(r"(?<!\$)\$(?!\$)[^$]+\$(?!\$)", chunk)
        count += len(inline_math)
        
    def _extract_section_info(self, text: str, boundary_pattern_type: Optional[str] = None) -> Tuple[str, int]:
        """
        Extract section name and hierarchy level from text.
        Prioritizes boundary pattern type to avoid mis-detection in multi-line content.
        
        Args:
            text: The text chunk to analyze
            boundary_pattern_type: The pattern type from boundary detection (if available)
            
        Returns:
            Tuple of (section_name, hierarchy_level)
        """
        # Priority 1: Use boundary pattern type if available (most reliable)
        if boundary_pattern_type == "section_boundary":
            # Extract section name from text
            match = re.search(
                r"^(?:related\s+works?|related\s+work|abstract|introduction|background|methodology|method|"
                r"results|discussion|conclusion|acknowledgments?|references|appendix)\b",
                text,
                re.MULTILINE | re.IGNORECASE
            )
            if match:
                return match.group(0).title(), 1
        
        # Priority 2: Check for markdown headers
        headers = {
            r"^#\s+(.+)$": 1,
            r"^##\s+(.+)$": 2,
            r"^###\s+(.+)$": 3,
            r"^####\s+(.+)$": 4,
            r"^#####\s+(.+)$": 5,
            r"^######\s+(.+)$": 6,
        }
        
        for pattern, level in headers.items():
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                return match.group(1).strip(), level
        
        # Priority 3: Check for numbered sections
        numbered_match = re.search(r"^(\d+(?:\.\d+)*)\s+(.+)$", text, re.MULTILINE)
        if numbered_match:
            section_name = numbered_match.group(2).strip()
            level = len(numbered_match.group(1).split('.'))
            return section_name, level
        
        return "General", 0

    def _identify_content_type(self, chunk: str) -> str:
        """Identify the type of content in the chunk"""
        if re.search(r"\$\$|\\begin\{|\\end\{|\\\[|\\\]", chunk):
            return "equation"
        elif re.search(r"^(Table|Figure)\s+\d+", chunk, re.MULTILINE):
            return "table" if chunk.startswith("Table") else "figure"
        elif re.search(r"^\[\d+\]\s+", chunk, re.MULTILINE):
            return "reference"
        elif re.search(r"^(Abstract|Introduction|Methodology|Results|Discussion|Conclusion)", chunk):
            return "section_header"
        else:
            return "text"

    def _extract_metadata(
        self,
        chunk: str,
        chunk_index: int,
        current_section: str,
        boundary_pattern_type: Optional[str] = None
    ) -> Dict:
        """
        Extract comprehensive metadata for a chunk.
        
        Args:
            chunk: The text chunk
            chunk_index: Index of chunk
            current_section: Current section name
            boundary_pattern_type: Type of boundary pattern detected (for accurate section detection)
        """
        # Count citations (advanced)
        citation_count = self._count_citations_advanced(chunk)
        
        # Count equations (advanced)
        equation_count = self._count_equations_advanced(chunk)
        
        # Determine content type
        content_type = self._identify_content_type(chunk)
        
        # Extract section info - pass boundary type for accurate detection
        section_name, hierarchy_level = self._extract_section_info(chunk, boundary_pattern_type)
        
        # Determine if abstract or conclusion
        is_abstract = "abstract" in section_name.lower()
        is_conclusion = "conclusion" in section_name.lower()
        
        # Calculate importance score
        importance_score = self._calculate_importance_score(
            content_type=content_type,
            hierarchy_level=hierarchy_level,
            is_abstract=is_abstract,
            is_conclusion=is_conclusion,
            citation_count=citation_count
        )
        
        # Use detected section or keep current section
        if section_name != "General":
            current_section = section_name
        
        metadata = {
            "section": current_section,
            "subsection": section_name if section_name != "General" else "",
            "chunk_index": chunk_index,
            "hierarchy_level": hierarchy_level,
            "content_type": content_type,
            "has_citations": citation_count > 0,
            "citation_count": citation_count,
            "has_equations": equation_count > 0,
            "equation_count": equation_count,
            "is_abstract": is_abstract,
            "is_conclusion": is_conclusion,
            "importance_score": importance_score,
            "chunk_length": len(chunk),
            "word_count": len(chunk.split()),
            "boundary_pattern_type": boundary_pattern_type or "unknown"
        }
        
        return metadata

    def _calculate_importance_score(
        self,
        content_type: str,
        hierarchy_level: int,
        is_abstract: bool,
        is_conclusion: bool,
        citation_count: int
    ) -> float:
        """Calculate importance score for prioritizing chunks"""
        score = 0.5  # Base score
        
        # Content type impact
        content_weights = {
            "abstract": 1.0,
            "conclusion": 0.9,
            "equation": 0.8,
            "section_header": 0.7,
            "reference": 0.4,
            "table": 0.6,
            "figure": 0.6,
            "text": 0.5
        }
        score += content_weights.get(content_type, 0.5) * 0.3
        
        # Hierarchy level impact (lower level = higher importance)
        if hierarchy_level > 0:
            score += max(0, (4 - hierarchy_level) / 4) * 0.2
        
        # Section importance
        if is_abstract:
            score += 0.3
        if is_conclusion:
            score += 0.25
        
        # Citation impact (more citations = more important)
        citation_bonus = min(citation_count / 5, 0.5) * 0.2
        score += citation_bonus
        
        return min(1.0, score)

    def split_text(self, text: str) -> List[Tuple[str, Dict]]:
        """
        Split text using chronologically-ordered boundary detection with metadata.
        Handles multi-line equations, various citation formats, and maintains text coherence.
        Smart flush for captions to keep them separate from regular text.
        
        Returns:
            List of tuples (chunk_text, metadata)
        """
        # Find all boundaries in chronological order
        boundaries = self._find_all_boundaries(text)
        
        if not boundaries:
            # No boundaries found, treat entire text as one chunk
            metadata = self._extract_metadata(text.strip(), 0, "General")
            return [(text.strip(), metadata)]
        
        chunks_raw = []
        last_pos = 0
        current_section = "Introduction"
        
        # Split text at boundaries in chronological order
        for boundary in boundaries:
            # Add text before boundary
            chunk_before = text[last_pos:boundary.start].strip()
            if chunk_before:
                chunks_raw.append((chunk_before, "text"))
            
            # Add the boundary itself
            boundary_text = text[boundary.start:boundary.end].strip()
            if boundary_text:
                chunks_raw.append((boundary_text, boundary.pattern_type))
            
            last_pos = boundary.end
        
        # Add remaining text after last boundary
        if last_pos < len(text):
            remaining = text[last_pos:].strip()
            if remaining:
                chunks_raw.append((remaining, "text"))
        
        # Merge and filter chunks with smart caption flushing
        chunks_with_metadata = []
        current_chunk = ""
        current_chunk_type = "text"
        chunk_index = 0
        
        for i, (chunk_text, pattern_type) in enumerate(chunks_raw):
            # Smart caption handling: if current chunk has content and we encounter a caption, flush immediately
            is_caption = pattern_type == "caption"
            should_flush_for_caption = is_caption and current_chunk and len(current_chunk) > self.min_chunk_size
            
            # Try to add to current chunk
            if not should_flush_for_caption and len(current_chunk) + len(chunk_text) <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n" + chunk_text
                else:
                    current_chunk = chunk_text
                    current_chunk_type = pattern_type
            else:
                # Current chunk would exceed size or caption encountered, save it and start new one
                if current_chunk and len(current_chunk) > self.min_chunk_size:
                    metadata = self._extract_metadata(
                        current_chunk,
                        chunk_index,
                        current_section,
                        current_chunk_type
                    )
                    chunks_with_metadata.append((current_chunk, metadata))
                    if metadata["subsection"]:
                        current_section = metadata["subsection"]
                    chunk_index += 1
                
                current_chunk = chunk_text
                current_chunk_type = pattern_type
        
        # Add the last chunk
        if current_chunk and len(current_chunk) > self.min_chunk_size:
            metadata = self._extract_metadata(
                current_chunk,
                chunk_index,
                current_section,
                current_chunk_type
            )
            chunks_with_metadata.append((current_chunk, metadata))
        elif current_chunk:
            # Small chunk at end - try to merge with previous
            if chunks_with_metadata:
                prev_chunk, prev_metadata = chunks_with_metadata[-1]
                merged_chunk = prev_chunk + "\n" + current_chunk
                if len(merged_chunk) <= self.chunk_size * 1.2:  # Allow slight overflow for merging
                    merged_metadata = self._extract_metadata(
                        merged_chunk,
                        prev_metadata["chunk_index"],
                        prev_metadata["section"],
                        prev_metadata.get("boundary_pattern_type", "text")
                    )
                    chunks_with_metadata[-1] = (merged_chunk, merged_metadata)
                else:
                    metadata = self._extract_metadata(
                        current_chunk,
                        chunk_index,
                        current_section,
                        current_chunk_type
                    )
                    chunks_with_metadata.append((current_chunk, metadata))
        
        # Final pass: Split chunks that are still too large
        final_chunks = []
        for chunk, metadata in chunks_with_metadata:
            if len(chunk) > self.chunk_size:
                # Use backup splitter
                backup_chunks = self.backup_splitter.split_text(chunk)
                for i, sub_chunk in enumerate(backup_chunks):
                    new_metadata = metadata.copy()
                    new_metadata["chunk_index"] = len(final_chunks)
                    final_chunks.append((sub_chunk, new_metadata))
            else:
                final_chunks.append((chunk, metadata))
        
        logger.info(f"Split into {len(final_chunks)} chunks with smart caption separation")
        return final_chunks

    def split_documents(self, documents: List[str]) -> List[Tuple[str, Dict]]:
        """
        Split multiple documents while maintaining document boundaries.
        
        Returns:
            List of tuples (chunk_text, metadata)
        """
        all_chunks = []
        for doc in documents:
            chunks = self.split_text(doc)
            all_chunks.extend(chunks)
        return all_chunks
