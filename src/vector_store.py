"""
Vector store for embedding-based document retrieval.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class VectorDocument:
    """Document with associated vector embedding."""
    id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any]


class VectorStore:
    """
    In-memory vector store for semantic search over documents.
    """

    def __init__(self, dimension: int = 1536):
        """
        Initialize the vector store.

        Args:
            dimension: Embedding dimension size
        """
        self.dimension = dimension
        self.documents: List[VectorDocument] = []
        self.index_to_id: Dict[int, str] = {}

    def add_document(
        self,
        doc_id: str,
        content: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Add a document with its embedding to the store.

        Args:
            doc_id: Unique document identifier
            content: Document text content
            embedding: Vector embedding
            metadata: Optional metadata
        """
        if embedding.shape[0] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {embedding.shape[0]}")

        doc = VectorDocument(
            id=doc_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {}
        )

        self.documents.append(doc)
        self.index_to_id[len(self.documents) - 1] = doc_id

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity score
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[VectorDocument, float]]:
        """
        Search for most similar documents using cosine similarity.

        Args:
            query_embedding: Query vector embedding
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (document, similarity_score) tuples
        """
        if len(self.documents) == 0:
            return []

        # Calculate similarities
        similarities = []
        for doc in self.documents:
            similarity = self.cosine_similarity(query_embedding, doc.embedding)
            if similarity >= threshold:
                similarities.append((doc, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def get_by_id(self, doc_id: str) -> Optional[VectorDocument]:
        """
        Retrieve a document by ID.

        Args:
            doc_id: Document identifier

        Returns:
            VectorDocument or None if not found
        """
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None

    def delete(self, doc_id: str) -> bool:
        """
        Delete a document from the store.

        Args:
            doc_id: Document identifier

        Returns:
            True if deleted, False if not found
        """
        for i, doc in enumerate(self.documents):
            if doc.id == doc_id:
                del self.documents[i]
                # Rebuild index
                self.index_to_id = {j: d.id for j, d in enumerate(self.documents)}
                return True
        return False

    def clear(self) -> None:
        """Clear all documents from the store."""
        self.documents.clear()
        self.index_to_id.clear()

    def size(self) -> int:
        """
        Get the number of documents in the store.

        Returns:
            Number of documents
        """
        return len(self.documents)
