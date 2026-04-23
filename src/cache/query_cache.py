import hashlib
from typing import Dict, Any, Optional
from cachetools import TTLCache
from src.utils.logger import logger


class QueryResultCache:
    """LRU кэш с TTL для результатов запросов ChromaDB."""

    def __init__(self, ttl: int = 3600, maxsize: int = 1000):
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.ttl = ttl
        self.maxsize = maxsize
        logger.info(f"QueryResultCache initialized: ttl={ttl}s, maxsize={maxsize}")

    def _get_key(self, question: str, collection: str, n_results: int) -> str:
        key_str = f"{collection}:{n_results}:{question.lower().strip()}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(
        self,
        question: str,
        collection: str,
        n_results: int
    ) -> Optional[Dict[str, Any]]:
        key = self._get_key(question, collection, n_results)
        result = self.cache.get(key)

        if result is not None:
            logger.debug(f"Cache HIT for query: {question[:50]}...")
        else:
            logger.debug(f"Cache MISS for query: {question[:50]}...")

        return result

    def set(
        self,
        question: str,
        collection: str,
        n_results: int,
        result: Dict[str, Any]
    ) -> None:
        key = self._get_key(question, collection, n_results)
        self.cache[key] = result
        logger.debug(f"Cached query result for: {question[:50]}... (cache size: {len(self.cache)}/{self.maxsize})")

    def clear(self) -> None:
        self.cache.clear()
        logger.info("QueryResultCache cleared")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "size": len(self.cache),
            "maxsize": self.maxsize,
            "ttl": self.ttl,
            "fill_ratio": len(self.cache) / self.maxsize
        }


_query_cache: Optional[QueryResultCache] = None


def get_query_cache(ttl: int = 3600, maxsize: int = 1000) -> QueryResultCache:
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryResultCache(ttl=ttl, maxsize=maxsize)
    return _query_cache
