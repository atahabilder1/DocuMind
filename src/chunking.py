"""
Document chunking strategies for optimal embedding and retrieval.
"""
from typing import List, Dict, Any
import re


class DocumentChunker:
    """
    Chunk documents into smaller pieces for embedding and retrieval.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target size for each chunk (in characters)
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_by_tokens(self, text: str) -> List[str]:
        """
        Chunk text by approximate token count (using character-based estimation).

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        if not text or len(text) == 0:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # If not at the end, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                period_idx = text.rfind('.', start, end)
                newline_idx = text.rfind('\n', start, end)
                break_idx = max(period_idx, newline_idx)

                if break_idx > start:
                    end = break_idx + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap

        return chunks

    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """
        Chunk text by paragraphs.

        Args:
            text: Input text to chunk

        Returns:
            List of paragraph chunks
        """
        # Split by double newlines or similar paragraph markers
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds chunk size, save current and start new
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def chunk_by_sentences(self, text: str) -> List[str]:
        """
        Chunk text by sentences.

        Args:
            text: Input text to chunk

        Returns:
            List of sentence-based chunks
        """
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def chunk_document(
        self,
        text: str,
        strategy: str = "tokens",
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document using the specified strategy.

        Args:
            text: Input text to chunk
            strategy: Chunking strategy ('tokens', 'paragraphs', or 'sentences')
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if strategy == "paragraphs":
            chunks = self.chunk_by_paragraphs(text)
        elif strategy == "sentences":
            chunks = self.chunk_by_sentences(text)
        else:
            chunks = self.chunk_by_tokens(text)

        result = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                'text': chunk,
                'chunk_id': i,
                'chunk_count': len(chunks),
                'metadata': metadata or {}
            }
            result.append(chunk_data)

        return result
