"""
Main application entry point with optimized pipeline.
"""
from typing import Optional, Dict, Any
from src.document_processor import PDFProcessor
from src.image_processor import ImageProcessor
from src.vision_model import VisionModel
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.chunking import DocumentChunker
from src.query_processor import QueryProcessor
from src.response_generator import ResponseGenerator
from src.cache import CacheManager
from src.logger import Logger, ErrorHandler


class DocuMind:
    """
    Main DocuMind application class with optimized multi-modal RAG pipeline.
    """

    def __init__(
        self,
        use_cache: bool = True,
        cache_ttl: int = 3600,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize DocuMind application.

        Args:
            use_cache: Enable caching for performance
            cache_ttl: Cache time-to-live in seconds
            chunk_size: Document chunk size
            chunk_overlap: Chunk overlap size
        """
        # Initialize logging
        self.logger = Logger()
        self.error_handler = ErrorHandler(self.logger)

        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.image_processor = ImageProcessor()
        self.vision_model = VisionModel()
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Initialize embedding and vector store
        self.embedding_generator = EmbeddingGenerator()
        dimension = self.embedding_generator.get_embedding_dimension()
        self.vector_store = VectorStore(dimension=dimension)

        # Initialize query and response components
        self.query_processor = QueryProcessor(self.embedding_generator, self.vector_store)
        self.response_generator = ResponseGenerator()

        # Initialize cache
        self.use_cache = use_cache
        if use_cache:
            self.cache = CacheManager(ttl=cache_ttl)
        else:
            self.cache = None

        self.logger.info("DocuMind initialized successfully")

    def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Process a PDF document and add to knowledge base.

        Args:
            file_path: Path to PDF file

        Returns:
            Processing result
        """
        try:
            self.logger.info(f"Processing PDF: {file_path}")

            # Extract text and metadata
            doc_data = self.pdf_processor.process_document(file_path)

            # Chunk the document
            chunks = self.chunker.chunk_document(
                doc_data['text'],
                strategy="paragraphs",
                metadata=doc_data['metadata']
            )

            # Generate embeddings and store
            for chunk in chunks:
                # Check cache first
                if self.use_cache:
                    cached_embedding = self.cache.get_cached_embedding(chunk['text'])
                    if cached_embedding is not None:
                        embedding = cached_embedding
                    else:
                        embedding = self.embedding_generator.embed_text(chunk['text'])
                        self.cache.cache_embedding(chunk['text'], embedding)
                else:
                    embedding = self.embedding_generator.embed_text(chunk['text'])

                # Add to vector store
                doc_id = f"{file_path}_{chunk['chunk_id']}"
                self.vector_store.add_document(
                    doc_id=doc_id,
                    content=chunk['text'],
                    embedding=embedding,
                    metadata=chunk['metadata']
                )

            self.logger.info(f"Successfully processed {len(chunks)} chunks from PDF")

            return {
                'status': 'success',
                'chunks_processed': len(chunks),
                'file_path': file_path
            }

        except Exception as e:
            return self.error_handler.handle_error(e, context="process_pdf")

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query the knowledge base and generate an answer.

        Args:
            question: User's question
            top_k: Number of context chunks to retrieve

        Returns:
            Answer with sources
        """
        try:
            self.logger.info(f"Processing query: {question}")

            # Check cache for query
            if self.use_cache:
                cached_result = self.cache.get_cached_query(question)
                if cached_result is not None:
                    self.logger.info("Returning cached query result")
                    return cached_result

            # Get context
            context_data = self.query_processor.get_context_for_generation(
                question,
                top_k=top_k
            )

            # Generate answer
            answer_data = self.response_generator.generate_with_sources(
                query=question,
                context=context_data['context'],
                sources=context_data['sources']
            )

            result = {
                'status': 'success',
                'query': question,
                'answer': answer_data['answer'],
                'sources': answer_data['sources'],
                'num_sources': context_data['num_sources']
            }

            # Cache result
            if self.use_cache:
                self.cache.cache_query_result(question, result)

            self.logger.info("Query processed successfully")
            return result

        except Exception as e:
            return self.error_handler.handle_error(e, context="query")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get application statistics.

        Returns:
            Statistics dictionary
        """
        stats = {
            'documents_in_store': self.vector_store.size(),
            'cache_enabled': self.use_cache
        }

        if self.use_cache:
            stats['cache_stats'] = self.cache.get_stats()

        return stats

    def clear_cache(self) -> Dict[str, Any]:
        """
        Clear application cache.

        Returns:
            Cache clearing result
        """
        if not self.use_cache:
            return {'status': 'error', 'message': 'Cache not enabled'}

        expired = self.cache.clear_expired()
        return {
            'status': 'success',
            'expired_entries_cleared': expired
        }


if __name__ == "__main__":
    # Example usage
    app = DocuMind(use_cache=True)
    print("DocuMind initialized")
    print(f"Stats: {app.get_stats()}")
