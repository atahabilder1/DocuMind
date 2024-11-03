"""
Unit tests for vector store.
"""
import unittest
import numpy as np
from src.vector_store import VectorStore


class TestVectorStore(unittest.TestCase):
    """Test cases for VectorStore class."""

    def setUp(self):
        """Set up test fixtures."""
        self.store = VectorStore(dimension=128)

    def test_initialization(self):
        """Test store initialization."""
        self.assertEqual(self.store.dimension, 128)
        self.assertEqual(self.store.size(), 0)

    def test_add_document(self):
        """Test adding a document."""
        embedding = np.random.rand(128)
        self.store.add_document(
            doc_id="test_1",
            content="Test content",
            embedding=embedding,
            metadata={"source": "test"}
        )
        self.assertEqual(self.store.size(), 1)

    def test_add_document_wrong_dimension(self):
        """Test adding document with wrong embedding dimension."""
        embedding = np.random.rand(64)  # Wrong dimension
        with self.assertRaises(ValueError):
            self.store.add_document(
                doc_id="test_1",
                content="Test content",
                embedding=embedding
            )

    def test_search(self):
        """Test vector search."""
        # Add some documents
        for i in range(5):
            embedding = np.random.rand(128)
            self.store.add_document(
                doc_id=f"doc_{i}",
                content=f"Content {i}",
                embedding=embedding
            )

        # Search
        query_embedding = np.random.rand(128)
        results = self.store.search(query_embedding, top_k=3)

        self.assertEqual(len(results), 3)
        self.assertTrue(all(isinstance(score, (float, np.floating)) for _, score in results))

    def test_get_by_id(self):
        """Test retrieving document by ID."""
        embedding = np.random.rand(128)
        self.store.add_document(
            doc_id="test_1",
            content="Test content",
            embedding=embedding
        )

        doc = self.store.get_by_id("test_1")
        self.assertIsNotNone(doc)
        self.assertEqual(doc.id, "test_1")

    def test_delete(self):
        """Test deleting a document."""
        embedding = np.random.rand(128)
        self.store.add_document(
            doc_id="test_1",
            content="Test content",
            embedding=embedding
        )

        self.assertEqual(self.store.size(), 1)
        self.store.delete("test_1")
        self.assertEqual(self.store.size(), 0)


if __name__ == '__main__':
    unittest.main()
