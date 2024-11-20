"""
Basic usage example for DocuMind.
"""
from src.main import DocuMind


def main():
    """Demonstrate basic DocuMind functionality."""

    # Initialize DocuMind with caching enabled
    print("Initializing DocuMind...")
    app = DocuMind(use_cache=True, chunk_size=500)

    # Example 1: Process a PDF document
    print("\n--- Example 1: Process PDF ---")
    # Uncomment to use with actual PDF:
    # result = app.process_pdf("sample_document.pdf")
    # print(f"Status: {result['status']}")
    # print(f"Chunks processed: {result['chunks_processed']}")

    # Example 2: Query the knowledge base
    print("\n--- Example 2: Query Knowledge Base ---")
    # Uncomment to use with actual documents:
    # answer = app.query("What are the main points in the document?")
    # print(f"Query: {answer['query']}")
    # print(f"Answer: {answer['answer']}")
    # print(f"Number of sources: {answer['num_sources']}")

    # Example 3: Get application statistics
    print("\n--- Example 3: Application Statistics ---")
    stats = app.get_stats()
    print(f"Documents in store: {stats['documents_in_store']}")
    print(f"Cache enabled: {stats['cache_enabled']}")

    # Example 4: Clear cache
    print("\n--- Example 4: Clear Cache ---")
    cache_result = app.clear_cache()
    print(f"Cache cleared: {cache_result['status']}")

    print("\nDocuMind demo completed!")


if __name__ == "__main__":
    main()
