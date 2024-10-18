"""
Caching layer for embeddings and query results.
"""
from typing import Any, Optional, Dict
import hashlib
import json
import time
from pathlib import Path
import pickle


class CacheManager:
    """
    Manage caching for embeddings and query results to improve performance.
    """

    def __init__(self, cache_dir: str = ".cache", ttl: int = 3600):
        """
        Initialize the cache manager.

        Args:
            cache_dir: Directory for cache storage
            ttl: Time-to-live for cache entries in seconds (default 1 hour)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = ttl

        # Create subdirectories
        (self.cache_dir / "embeddings").mkdir(exist_ok=True)
        (self.cache_dir / "queries").mkdir(exist_ok=True)
        (self.cache_dir / "responses").mkdir(exist_ok=True)

    def _generate_key(self, data: Any) -> str:
        """
        Generate a cache key from data.

        Args:
            data: Data to hash

        Returns:
            Hash string
        """
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)

        return hashlib.sha256(data_str.encode()).hexdigest()

    def _get_cache_path(self, key: str, category: str) -> Path:
        """
        Get the file path for a cache entry.

        Args:
            key: Cache key
            category: Cache category

        Returns:
            Path to cache file
        """
        return self.cache_dir / category / f"{key}.pkl"

    def _is_expired(self, file_path: Path) -> bool:
        """
        Check if a cache entry is expired.

        Args:
            file_path: Path to cache file

        Returns:
            True if expired, False otherwise
        """
        if not file_path.exists():
            return True

        file_age = time.time() - file_path.stat().st_mtime
        return file_age > self.ttl

    def set(self, key: str, value: Any, category: str = "general") -> None:
        """
        Set a cache entry.

        Args:
            key: Cache key
            value: Value to cache
            category: Cache category
        """
        cache_path = self._get_cache_path(key, category)

        with open(cache_path, 'wb') as f:
            pickle.dump({
                'value': value,
                'timestamp': time.time()
            }, f)

    def get(self, key: str, category: str = "general") -> Optional[Any]:
        """
        Get a cache entry.

        Args:
            key: Cache key
            category: Cache category

        Returns:
            Cached value or None if not found/expired
        """
        cache_path = self._get_cache_path(key, category)

        if self._is_expired(cache_path):
            return None

        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                return data['value']
        except (FileNotFoundError, pickle.UnpicklingError):
            return None

    def cache_embedding(self, text: str, embedding: Any) -> None:
        """
        Cache an embedding for text.

        Args:
            text: Input text
            embedding: Embedding vector
        """
        key = self._generate_key(text)
        self.set(key, embedding, category="embeddings")

    def get_cached_embedding(self, text: str) -> Optional[Any]:
        """
        Get cached embedding for text.

        Args:
            text: Input text

        Returns:
            Cached embedding or None
        """
        key = self._generate_key(text)
        return self.get(key, category="embeddings")

    def cache_query_result(self, query: str, result: Any) -> None:
        """
        Cache a query result.

        Args:
            query: Query text
            result: Query result
        """
        key = self._generate_key(query)
        self.set(key, result, category="queries")

    def get_cached_query(self, query: str) -> Optional[Any]:
        """
        Get cached query result.

        Args:
            query: Query text

        Returns:
            Cached result or None
        """
        key = self._generate_key(query)
        return self.get(key, category="queries")

    def cache_response(self, query: str, context: str, response: str) -> None:
        """
        Cache a generated response.

        Args:
            query: User query
            context: Context used
            response: Generated response
        """
        key = self._generate_key({"query": query, "context": context})
        self.set(key, response, category="responses")

    def get_cached_response(self, query: str, context: str) -> Optional[str]:
        """
        Get cached response.

        Args:
            query: User query
            context: Context used

        Returns:
            Cached response or None
        """
        key = self._generate_key({"query": query, "context": context})
        return self.get(key, category="responses")

    def clear_category(self, category: str) -> int:
        """
        Clear all entries in a category.

        Args:
            category: Category to clear

        Returns:
            Number of entries deleted
        """
        category_dir = self.cache_dir / category
        count = 0

        for cache_file in category_dir.glob("*.pkl"):
            cache_file.unlink()
            count += 1

        return count

    def clear_expired(self) -> int:
        """
        Clear all expired cache entries.

        Returns:
            Number of entries deleted
        """
        count = 0

        for category in ["embeddings", "queries", "responses"]:
            category_dir = self.cache_dir / category
            for cache_file in category_dir.glob("*.pkl"):
                if self._is_expired(cache_file):
                    cache_file.unlink()
                    count += 1

        return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        stats = {}

        for category in ["embeddings", "queries", "responses"]:
            category_dir = self.cache_dir / category
            files = list(category_dir.glob("*.pkl"))

            stats[category] = {
                'total': len(files),
                'expired': sum(1 for f in files if self._is_expired(f))
            }

        return stats
