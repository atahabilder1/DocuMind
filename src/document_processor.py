"""
PDF document processing module for extracting text and metadata.
"""
from typing import List, Dict, Any
import PyPDF2
from pathlib import Path


class PDFProcessor:
    """Process PDF documents and extract text content."""

    def __init__(self):
        self.supported_formats = ['.pdf']

    def extract_text(self, file_path: str) -> str:
        """
        Extract text content from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text content
        """
        path = Path(file_path)
        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        text_content = []

        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_content.append(text)

        return "\n\n".join(text_content)

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary containing metadata
        """
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            metadata = pdf_reader.metadata

            return {
                'title': metadata.get('/Title', ''),
                'author': metadata.get('/Author', ''),
                'subject': metadata.get('/Subject', ''),
                'creator': metadata.get('/Creator', ''),
                'producer': metadata.get('/Producer', ''),
                'num_pages': len(pdf_reader.pages)
            }

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a PDF document and extract both text and metadata.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary containing text content and metadata
        """
        return {
            'text': self.extract_text(file_path),
            'metadata': self.extract_metadata(file_path),
            'file_path': file_path
        }
