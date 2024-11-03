"""
Unit tests for document processing module.
"""
import unittest
from src.document_processor import PDFProcessor


class TestPDFProcessor(unittest.TestCase):
    """Test cases for PDFProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = PDFProcessor()

    def test_initialization(self):
        """Test processor initialization."""
        self.assertIsNotNone(self.processor)
        self.assertEqual(self.processor.supported_formats, ['.pdf'])

    def test_supported_formats(self):
        """Test supported file formats."""
        self.assertIn('.pdf', self.processor.supported_formats)

    def test_extract_text_invalid_format(self):
        """Test extraction with invalid file format."""
        with self.assertRaises(ValueError):
            self.processor.extract_text('test.txt')


if __name__ == '__main__':
    unittest.main()
