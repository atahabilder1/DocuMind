"""
Query processing and retrieval for RAG pipeline.
"""
from typing import List, Dict, Any, Tuple
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore, VectorDocument


class QueryProcessor:
    """
    Process user queries and retrieve relevant document chunks.
    """

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_store: VectorStore
    ):
        """
        Initialize the query processor.

        Args:
            embedding_generator: EmbeddingGenerator instance
            vector_store: VectorStore instance
        """
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store

    def process_query(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Process a user query and retrieve relevant documents.

        Args:
            query: User's question or query
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of relevant documents with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.embed_query(query)

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=threshold
        )

        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.content,
                'score': float(score),
                'metadata': doc.metadata,
                'doc_id': doc.id
            })

        return formatted_results

    def expand_query(self, query: str) -> List[str]:
        """
        Expand a query with related terms or paraphrases.

        Args:
            query: Original query

        Returns:
            List of expanded query variations
        """
        # Simple expansion - in production, use LLM for better expansions
        expanded = [query]

        # Add question variations
        if not query.endswith('?'):
            expanded.append(query + '?')

        return expanded

    def rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Re-rank retrieved results for better relevance.

        Args:
            query: Original query
            results: Initial retrieval results

        Returns:
            Re-ranked results
        """
        # Simple keyword-based re-ranking
        query_terms = set(query.lower().split())

        for result in results:
            content_terms = set(result['content'].lower().split())
            keyword_overlap = len(query_terms.intersection(content_terms))

            # Adjust score based on keyword overlap
            result['rerank_score'] = result['score'] + (keyword_overlap * 0.01)

        # Sort by rerank score
        results.sort(key=lambda x: x.get('rerank_score', x['score']), reverse=True)

        return results

    def get_context_for_generation(
        self,
        query: str,
        top_k: int = 5,
        max_context_length: int = 2000
    ) -> Dict[str, Any]:
        """
        Get context for answer generation.

        Args:
            query: User query
            top_k: Number of documents to retrieve
            max_context_length: Maximum context length in characters

        Returns:
            Dictionary with context and metadata
        """
        results = self.process_query(query, top_k=top_k)

        # Combine contexts up to max length
        context_parts = []
        total_length = 0

        for result in results:
            content = result['content']
            if total_length + len(content) <= max_context_length:
                context_parts.append(content)
                total_length += len(content)
            else:
                break

        return {
            'query': query,
            'context': '\n\n'.join(context_parts),
            'sources': results,
            'num_sources': len(context_parts)
        }
