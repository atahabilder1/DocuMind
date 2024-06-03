"""
Image processing module for handling various image formats.
"""
from typing import Dict, Any, Tuple
from PIL import Image
import base64
from io import BytesIO
from pathlib import Path


class ImageProcessor:
    """Process images for vision model analysis."""

    def __init__(self):
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
        self.max_size = (1024, 1024)

    def load_image(self, file_path: str) -> Image.Image:
        """
        Load an image from file path.

        Args:
            file_path: Path to the image file

        Returns:
            PIL Image object
        """
        path = Path(file_path)
        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported image format: {path.suffix}")

        return Image.open(file_path)

    def resize_image(self, image: Image.Image, max_size: Tuple[int, int] = None) -> Image.Image:
        """
        Resize image while maintaining aspect ratio.

        Args:
            image: PIL Image object
            max_size: Maximum dimensions (width, height)

        Returns:
            Resized PIL Image object
        """
        if max_size is None:
            max_size = self.max_size

        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image

    def image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 encoded string.

        Args:
            image: PIL Image object

        Returns:
            Base64 encoded string
        """
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    def get_image_info(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract image metadata and information.

        Args:
            image: PIL Image object

        Returns:
            Dictionary containing image information
        """
        return {
            'width': image.width,
            'height': image.height,
            'format': image.format,
            'mode': image.mode,
            'size_bytes': len(image.tobytes())
        }

    def process_image(self, file_path: str, resize: bool = True) -> Dict[str, Any]:
        """
        Process an image file for vision model input.

        Args:
            file_path: Path to the image file
            resize: Whether to resize the image

        Returns:
            Dictionary containing processed image data
        """
        image = self.load_image(file_path)

        if resize:
            image = self.resize_image(image)

        return {
            'image': image,
            'base64': self.image_to_base64(image),
            'info': self.get_image_info(image),
            'file_path': file_path
        }
