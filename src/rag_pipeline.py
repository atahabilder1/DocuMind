"""
RAG (Retrieval-Augmented Generation) pipeline for document understanding.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Document:
    """Represents a document chunk with metadata."""
    content: str
    metadata: Dict[str, Any]
    doc_id: str
    chunk_id: int


class RAGPipeline:
    """
    Orchestrates the RAG pipeline for multi-modal document understanding.
    """

    def __init__(self):
        self.documents: List[Document] = []
        self.processed_docs: Dict[str, Any] = {}

    def add_document(self, content: str, metadata: Dict[str, Any], doc_id: str) -> None:
        """
        Add a document to the pipeline.

        Args:
            content: Document text content
            metadata: Document metadata
            doc_id: Unique document identifier
        """
        doc = Document(
            content=content,
            metadata=metadata,
            doc_id=doc_id,
            chunk_id=0
        )
        self.documents.append(doc)
        self.processed_docs[doc_id] = {
            'content': content,
            'metadata': metadata,
            'chunks': []
        }

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a processed document by ID.

        Args:
            doc_id: Document identifier

        Returns:
            Document data or None if not found
        """
        return self.processed_docs.get(doc_id)

    def list_documents(self) -> List[str]:
        """
        List all document IDs in the pipeline.

        Returns:
            List of document IDs
        """
        return list(self.processed_docs.keys())

    def clear(self) -> None:
        """Clear all documents from the pipeline."""
        self.documents.clear()
        self.processed_docs.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.

        Returns:
            Dictionary containing pipeline stats
        """
        return {
            'total_documents': len(self.documents),
            'total_chunks': sum(len(doc['chunks']) for doc in self.processed_docs.values()),
            'document_ids': self.list_documents()
        }
