"""
Unit tests for document chunking.
"""
import unittest
from src.chunking import DocumentChunker


class TestDocumentChunker(unittest.TestCase):
    """Test cases for DocumentChunker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)

    def test_initialization(self):
        """Test chunker initialization."""
        self.assertEqual(self.chunker.chunk_size, 100)
        self.assertEqual(self.chunker.chunk_overlap, 10)

    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunks = self.chunker.chunk_by_tokens("")
        self.assertEqual(len(chunks), 0)

    def test_chunk_short_text(self):
        """Test chunking short text."""
        text = "This is a short text."
        chunks = self.chunker.chunk_by_tokens(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_chunk_document(self):
        """Test document chunking with metadata."""
        text = "This is a test. " * 20  # Create longer text
        result = self.chunker.chunk_document(text, strategy="tokens")

        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        self.assertIn('text', result[0])
        self.assertIn('chunk_id', result[0])


if __name__ == '__main__':
    unittest.main()
