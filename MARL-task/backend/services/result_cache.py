"""Result caching service."""

from typing import Dict, Optional
from collections import OrderedDict


class ResultCache:
    """Simple LRU cache for allocation results."""

    def __init__(self, max_size: int = 100):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of results to cache
        """
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.max_size = max_size

    def set(self, key: str, value: dict):
        """
        Store result in cache.

        Args:
            key: Result identifier
            value: Result data
        """
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        self.cache[key] = value
        self.cache.move_to_end(key)

    def get(self, key: str) -> Optional[dict]:
        """
        Retrieve result from cache.

        Args:
            key: Result identifier

        Returns:
            Result data or None if not found
        """
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def clear(self):
        """Clear all cached results."""
        self.cache.clear()

    def __len__(self):
        """Return number of cached results."""
        return len(self.cache)


# Global instance
result_cache = ResultCache()
