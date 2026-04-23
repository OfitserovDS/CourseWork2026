from typing import List, Tuple
from sentence_transformers import CrossEncoder
from src.utils.logger import logger


class DocumentReranker:
    """Re-ranks retrieved documents using cross-encoder for better relevance."""

    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        """Initialize with cross-encoder model."""
        try:
            self.model = CrossEncoder(model_name)
            self.model_name = model_name
            logger.info(f"Initialized DocumentReranker with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize DocumentReranker: {e}")
            self.model = None

    def rerank(
        self,
        question: str,
        documents: List[str],
        top_k: int = 3
    ) -> List[str]:
        """
        Re-rank documents by relevance to question.

        Args:
            question: User's question
            documents: List of retrieved documents
            top_k: Number of top documents to return

        Returns:
            Top-k re-ranked documents
        """
        if not self.model or not documents:
            logger.debug(f"Reranker unavailable or no documents. Returning first {top_k}")
            return documents[:top_k]

        if len(documents) <= top_k:
            logger.debug(f"Documents count ({len(documents)}) <= top_k ({top_k}). Skipping reranking")
            return documents

        try:
            pairs = [[question, doc] for doc in documents]
            scores = self.model.predict(pairs)

            ranked_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]

            reranked_docs = [documents[i] for i in ranked_indices]
            logger.debug(f"Re-ranked {len(documents)} documents to top-{top_k}")

            return reranked_docs

        except Exception as e:
            logger.error(f"Re-ranking failed: {e}. Returning original top-{top_k}")
            return documents[:top_k]

    def _get_ranked_indices(
        self,
        question: str,
        documents: List[str],
        top_k: int = 3
    ) -> List[int]:
        """Get indices of top-k re-ranked documents."""
        if not self.model or not documents or len(documents) <= top_k:
            return list(range(min(top_k, len(documents))))

        try:
            pairs = [[question, doc] for doc in documents]
            scores = self.model.predict(pairs)

            ranked_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]

            return ranked_indices

        except Exception as e:
            logger.error(f"Failed to get ranked indices: {e}")
            return list(range(min(top_k, len(documents))))
