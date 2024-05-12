# DocuMind

Multi-modal AI agent that extracts information from PDFs, images, and documents to answer questions. Combines vision models with RAG architecture for intelligent document understanding.

## Features

- PDF document processing and text extraction
- Image analysis using vision models
- Multi-modal document understanding
- RAG (Retrieval-Augmented Generation) architecture
- Question answering over uploaded documents

## Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API key

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

## Usage

### Python API

```python
from src.main import DocuMind

# Initialize DocuMind
app = DocuMind(use_cache=True)

# Process a PDF document
result = app.process_pdf("path/to/document.pdf")
print(f"Processed {result['chunks_processed']} chunks")

# Query the knowledge base
answer = app.query("What is this document about?")
print(f"Answer: {answer['answer']}")
print(f"Sources: {answer['num_sources']}")

# Get application statistics
stats = app.get_stats()
print(f"Documents in store: {stats['documents_in_store']}")
```

### REST API

Start the API server:

```bash
python -m uvicorn src.api:app --reload
```

Upload a document:

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

Query documents:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "top_k": 5}'
```

### Running Tests

```bash
pytest tests/
```

## Architecture

DocuMind uses a multi-modal RAG architecture to process various document types:

1. **Document Processing**: Extract text and images from PDFs
2. **Vision Analysis**: Analyze images and diagrams using vision models
3. **Embedding Generation**: Create vector embeddings for text and visual content
4. **Retrieval**: Find relevant content based on user queries
5. **Generation**: Generate accurate answers using retrieved context

## License

MIT
