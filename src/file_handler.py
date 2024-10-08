"""
File upload and storage handling for DocuMind.
"""
from typing import Dict, Any, Optional
from pathlib import Path
import shutil
import hashlib
from datetime import datetime


class FileHandler:
    """
    Handle file uploads, storage, and management.
    """

    def __init__(self, upload_dir: str = "uploads"):
        """
        Initialize the file handler.

        Args:
            upload_dir: Directory for storing uploaded files
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)

        # Create subdirectories for different file types
        (self.upload_dir / "pdfs").mkdir(exist_ok=True)
        (self.upload_dir / "images").mkdir(exist_ok=True)
        (self.upload_dir / "temp").mkdir(exist_ok=True)

    def generate_file_id(self, filename: str) -> str:
        """
        Generate a unique file ID.

        Args:
            filename: Original filename

        Returns:
            Unique file identifier
        """
        timestamp = datetime.now().isoformat()
        content = f"{filename}_{timestamp}"
        file_id = hashlib.md5(content.encode()).hexdigest()
        return file_id

    def get_file_category(self, content_type: str) -> str:
        """
        Determine file category from content type.

        Args:
            content_type: MIME type

        Returns:
            File category (pdfs, images, or temp)
        """
        if content_type == "application/pdf":
            return "pdfs"
        elif content_type.startswith("image/"):
            return "images"
        else:
            return "temp"

    def save_upload(
        self,
        file_content: bytes,
        filename: str,
        content_type: str
    ) -> Dict[str, Any]:
        """
        Save an uploaded file.

        Args:
            file_content: File content as bytes
            filename: Original filename
            content_type: MIME type

        Returns:
            Dictionary with file metadata
        """
        file_id = self.generate_file_id(filename)
        category = self.get_file_category(content_type)

        # Preserve file extension
        file_ext = Path(filename).suffix
        stored_filename = f"{file_id}{file_ext}"

        file_path = self.upload_dir / category / stored_filename

        # Save file
        with open(file_path, 'wb') as f:
            f.write(file_content)

        return {
            'file_id': file_id,
            'original_filename': filename,
            'stored_filename': stored_filename,
            'file_path': str(file_path),
            'content_type': content_type,
            'category': category,
            'size_bytes': len(file_content),
            'upload_time': datetime.now().isoformat()
        }

    def get_file_path(self, file_id: str, category: str = None) -> Optional[Path]:
        """
        Get the path to a stored file.

        Args:
            file_id: File identifier
            category: Optional file category

        Returns:
            Path to file or None if not found
        """
        if category:
            categories = [category]
        else:
            categories = ["pdfs", "images", "temp"]

        for cat in categories:
            cat_dir = self.upload_dir / cat
            for file_path in cat_dir.iterdir():
                if file_path.stem == file_id:
                    return file_path

        return None

    def delete_file(self, file_id: str) -> bool:
        """
        Delete a stored file.

        Args:
            file_id: File identifier

        Returns:
            True if deleted, False if not found
        """
        file_path = self.get_file_path(file_id)
        if file_path and file_path.exists():
            file_path.unlink()
            return True
        return False

    def list_files(self, category: str = None) -> list:
        """
        List all stored files.

        Args:
            category: Optional filter by category

        Returns:
            List of file metadata
        """
        files = []

        if category:
            categories = [category]
        else:
            categories = ["pdfs", "images", "temp"]

        for cat in categories:
            cat_dir = self.upload_dir / cat
            for file_path in cat_dir.iterdir():
                if file_path.is_file():
                    files.append({
                        'file_id': file_path.stem,
                        'filename': file_path.name,
                        'category': cat,
                        'size_bytes': file_path.stat().st_size,
                        'modified_time': datetime.fromtimestamp(
                            file_path.stat().st_mtime
                        ).isoformat()
                    })

        return files

    def cleanup_temp_files(self, older_than_hours: int = 24) -> int:
        """
        Clean up temporary files older than specified hours.

        Args:
            older_than_hours: Age threshold in hours

        Returns:
            Number of files deleted
        """
        temp_dir = self.upload_dir / "temp"
        deleted_count = 0
        current_time = datetime.now().timestamp()

        for file_path in temp_dir.iterdir():
            if file_path.is_file():
                file_age_hours = (current_time - file_path.stat().st_mtime) / 3600
                if file_age_hours > older_than_hours:
                    file_path.unlink()
                    deleted_count += 1

        return deleted_count
