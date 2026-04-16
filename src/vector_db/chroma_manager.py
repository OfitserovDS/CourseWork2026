from typing import List, Dict, Any, Optional, Union, Sequence
from pathlib import Path
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config.settings import settings
from src.utils.logger import logger
from src.cache.query_cache import get_query_cache


class LangchainEmbeddingAdapter(EmbeddingFunction[Documents]):
    """LangChain embeddings with ChromaDB 1.3+ API (embed_query(input=...))."""

    def __init__(self, embedding_function: HuggingFaceEmbeddings):
        self.embedding_function = embedding_function

    def __call__(self, input: Documents) -> Embeddings:
        documents = self._to_text_list(input)
        vectors = self.embedding_function.embed_documents(documents)
        return [np.array(v, dtype=np.float32) for v in vectors]

    def embed_query(self, input: Union[Documents, str]) -> Embeddings:
        query = self._to_query_text(input)
        vector = self.embedding_function.embed_query(query)
        return [np.array(vector, dtype=np.float32)]

    @staticmethod
    def _to_text_list(input: Documents) -> List[str]:
        if isinstance(input, str):
            return [input]
        return [str(item) for item in input]

    @staticmethod
    def _to_query_text(input: Union[Documents, str]) -> str:
        if isinstance(input, str):
            return input
        if isinstance(input, Sequence) and not isinstance(input, str):
            if not input:
                return ""
            return str(input[0])
        return str(input)


class ChromaManager:

    def __init__(self, persist_directory: Optional[Path] = None):
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': settings.embedding_device},
            encode_kwargs={'normalize_embeddings': True}
        )
        self._embedding_fn = LangchainEmbeddingAdapter(self.embeddings)

        self.query_cache = get_query_cache(ttl=3600, maxsize=1000)

        logger.info(f"ChromaManager initialized with persist dir: {self.persist_directory}")
        logger.info(f"Using embedding model: {settings.embedding_model}")
        logger.info(f"Query cache enabled: TTL=3600s, maxsize=1000")

    def create_collection(self, name: str, metadata: Optional[Dict] = None) -> chromadb.Collection:
        try:
            collection = self.client.get_or_create_collection(
                name=name,
                metadata=metadata or {"description": f"Chunks for {name} strategy"},
                embedding_function=self._embedding_fn,
            )
            logger.info(f"Collection '{name}' ready, contains {collection.count()} documents")
            return collection
        except Exception as e:
            logger.error(f"Failed to create collection '{name}': {e}")
            raise

    def add_documents(
            self,
            collection_name: str,
            documents: List[Document],
            batch_size: int = 100
    ) -> int:
        collection = self.create_collection(collection_name)

        total_added = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            ids = []
            texts = []
            metadatas = []

            for j, doc in enumerate(batch):
                doc_id = f"{collection_name}_{i + j}"
                ids.append(doc_id)
                texts.append(doc.page_content)

                clean_metadata = {}
                for key, value in doc.metadata.items():
                    if isinstance(value, list):
                        clean_metadata[key] = ", ".join(str(v) for v in value)
                    elif isinstance(value, dict):
                        clean_metadata[key] = str(value)
                    elif value is None:
                        clean_metadata[key] = ""
                    else:
                        clean_metadata[key] = value

                metadatas.append(clean_metadata)

            try:
                collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )
                total_added += len(batch)
                logger.debug(f"Added batch {i // batch_size + 1}: {len(batch)} docs")
            except Exception as e:
                logger.error(f"Failed to add batch: {e}")
                for k, (doc_id, text, meta) in enumerate(zip(ids, texts, metadatas)):
                    try:
                        collection.add(
                            ids=[doc_id],
                            documents=[text],
                            metadatas=[meta]
                        )
                        total_added += 1
                    except Exception as e2:
                        logger.error(f"Failed to add document {doc_id}: {e2}")
                continue

        logger.info(f"Added {total_added} documents to collection '{collection_name}'")
        return total_added

    def query(
            self,
            collection_name: str,
            query_text: str,
            n_results: int = 5,
            where_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        cache_key_extra = str(sorted(where_filter.items())) if where_filter else ""
        cached_result = self.query_cache.get(
            query_text + cache_key_extra,
            collection_name,
            n_results
        )
        if cached_result is not None:
            logger.info(
                f"Query returned from cache: {query_text[:50]}... "
                f"({len(cached_result.get('ids', [[]])[0])} results)"
                f"{f' [filtered by: {list(where_filter.keys())}]' if where_filter else ''}"
            )
            return cached_result

        collection = self.create_collection(collection_name)

        try:
            if where_filter:
                logger.debug(f"Query with filter: {where_filter}")

            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_filter
            )

            self.query_cache.set(
                query_text + cache_key_extra,
                collection_name,
                n_results,
                results
            )

            filter_info = f" [filtered by: {list(where_filter.keys())}]" if where_filter else ""
            logger.info(
                f"Query executed and cached: {query_text[:50]}... "
                f"returned {len(results['ids'][0])} results{filter_info}"
            )
            return results
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        try:
            collection = self.client.get_collection(collection_name)
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    def list_collections(self) -> List[str]:
        return [col.name for col in self.client.list_collections()]

    def delete_collection(self, name: str) -> bool:
        try:
            self.client.delete_collection(name)
            logger.info(f"Deleted collection '{name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection '{name}': {e}")
            return False


_chroma_manager: Optional[ChromaManager] = None


def get_chroma_manager() -> ChromaManager:
    global _chroma_manager
    if _chroma_manager is None:
        _chroma_manager = ChromaManager()
    return _chroma_manager
