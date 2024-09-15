"""
Response generation using LLM with retrieved context.
"""
from typing import Dict, Any, List
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class ResponseGenerator:
    """
    Generate answers to user queries using retrieved context.
    """

    def __init__(self, model_name: str = None, api_key: str = None):
        """
        Initialize the response generator.

        Args:
            model_name: Name of the LLM model
            api_key: OpenAI API key
        """
        self.model_name = model_name or os.getenv('MODEL_NAME', 'gpt-4')
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)

    def generate_answer(
        self,
        query: str,
        context: str,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Generate an answer to a query using retrieved context.

        Args:
            query: User's question
            context: Retrieved context from documents
            include_sources: Whether to include source attribution

        Returns:
            Dictionary containing answer and metadata
        """
        system_prompt = """You are DocuMind, an AI assistant that answers questions based on provided document context.
Your task is to:
1. Answer the user's question accurately using only the provided context
2. If the context doesn't contain enough information, say so clearly
3. Be concise but thorough
4. Cite specific details from the context when relevant"""

        user_prompt = f"""Context from documents:
{context}

Question: {query}

Please answer the question based on the context provided above."""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        answer = response.choices[0].message.content

        return {
            'query': query,
            'answer': answer,
            'model': self.model_name,
            'context_used': context[:200] + '...' if len(context) > 200 else context
        }

    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """
        Generate a summary of the provided text.

        Args:
            text: Text to summarize
            max_length: Maximum length of summary

        Returns:
            Summary text
        """
        prompt = f"""Summarize the following text in approximately {max_length} characters or less:

{text}

Summary:"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=150
        )

        return response.choices[0].message.content.strip()

    def generate_with_sources(
        self,
        query: str,
        context: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate answer with source attribution.

        Args:
            query: User's question
            context: Retrieved context
            sources: List of source documents

        Returns:
            Dictionary with answer and sources
        """
        answer_data = self.generate_answer(query, context, include_sources=True)

        # Add source information
        answer_data['sources'] = [
            {
                'doc_id': src.get('doc_id'),
                'score': src.get('score'),
                'snippet': src.get('content', '')[:100] + '...'
            }
            for src in sources[:3]  # Top 3 sources
        ]

        return answer_data

    def chat(
        self,
        messages: List[Dict[str, str]],
        context: str = None
    ) -> str:
        """
        Multi-turn chat with optional context.

        Args:
            messages: List of chat messages
            context: Optional document context

        Returns:
            Assistant's response
        """
        system_message = {
            "role": "system",
            "content": "You are DocuMind, a helpful AI assistant for document understanding."
        }

        if context:
            system_message["content"] += f"\n\nRelevant context:\n{context}"

        chat_messages = [system_message] + messages

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=chat_messages,
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content
