"""
Vision model integration for analyzing images and visual content.
"""
from typing import Dict, Any, List
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class VisionModel:
    """Wrapper for GPT-4 Vision model to analyze images."""

    def __init__(self, model_name: str = None, api_key: str = None):
        """
        Initialize the vision model.

        Args:
            model_name: Name of the vision model to use
            api_key: OpenAI API key
        """
        self.model_name = model_name or os.getenv('MODEL_NAME', 'gpt-4-vision-preview')
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)

    def analyze_image(self, image_base64: str, prompt: str = None) -> str:
        """
        Analyze an image using the vision model.

        Args:
            image_base64: Base64 encoded image
            prompt: Optional prompt for analysis

        Returns:
            Analysis result from the model
        """
        if prompt is None:
            prompt = "Describe this image in detail, including any text, diagrams, charts, or important visual elements."

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )

        return response.choices[0].message.content

    def extract_text_from_image(self, image_base64: str) -> str:
        """
        Extract text content from an image using OCR.

        Args:
            image_base64: Base64 encoded image

        Returns:
            Extracted text
        """
        prompt = "Extract all text from this image. Return only the text content, preserving formatting where possible."
        return self.analyze_image(image_base64, prompt)

    def analyze_diagram(self, image_base64: str) -> Dict[str, Any]:
        """
        Analyze diagrams, charts, and visual elements.

        Args:
            image_base64: Base64 encoded image

        Returns:
            Dictionary containing analysis results
        """
        prompt = """Analyze this diagram or chart. Provide:
        1. Type of visual (chart, diagram, flowchart, etc.)
        2. Main components or elements
        3. Key data or information presented
        4. Any text labels or annotations
        """

        analysis = self.analyze_image(image_base64, prompt)

        return {
            'type': 'diagram_analysis',
            'analysis': analysis,
            'model': self.model_name
        }

    def answer_question_about_image(self, image_base64: str, question: str) -> str:
        """
        Answer a specific question about an image.

        Args:
            image_base64: Base64 encoded image
            question: Question to answer

        Returns:
            Answer from the model
        """
        return self.analyze_image(image_base64, question)
