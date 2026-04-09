#!/usr/bin/env python

import sys
from pathlib import Path
from src.config.settings import settings
from src.data.loader import get_document_loader
from src.data.preprocessor import DocumentPreprocessor
from src.chunking.strategies import get_chunking_strategy
from src.vector_db.chroma_manager import get_chroma_manager
from src.utils.logger import logger


def main():
    logger.info("=== Part 2: Creating chroma db collections ===")
    document_path = settings.raw_data_dir / settings.document_name

    if not document_path.exists():
        logger.error(f"Document not found: {document_path}")
        sys.exit(1)

    logger.info(f"Loading document: {document_path}")
    loader = get_document_loader("unstructured")
    documents = loader.load_document(document_path)
    logger.info(f"Loaded {len(documents)} documents")
    preprocessor = DocumentPreprocessor()
    processed_chunks = preprocessor.process_documents(documents)
    preprocessed_documents = preprocessor.preprocess_langchain_documents(documents)
    logger.info(
        f"Preprocessing done: base chunks={len(processed_chunks)}, docs_for_chunking={len(preprocessed_documents)}"
    )
    strategies = [
        {
            "name": "recursive",
            "class": "recursive",
            "params": {"chunk_size": 512, "chunk_overlap": 128},
            "description": "Базовый рекурсивный чанкинг"
        },
        {
            "name": "markdown_header",
            "class": "markdown_header",
            "params": {},
            "description": "Чанкинг по заголовкам (структурный)"
        },
        {
            "name": "parent_document",
            "class": "parent_document",
            "params": {"parent_chunk_size": 2000, "child_chunk_size": 300},
            "description": "Parent Document Retriever (иерархический)"
        },
        {
            "name": "semantic",
            "class": "semantic",
            "params": {},
            "description": "Семантический чанкинг (по абзацам)"
        },
        {
            "name": "legal_document",
            "class": "legal_document",
            "params": {"min_chunk_size": 80, "max_chunk_size": 600},
            "description": "Специализированный для юридических документов (ПОПАТКУС)"
        }
    ]
    chroma = get_chroma_manager()

    results = {}

    for strategy_config in strategies:
        strategy_name = strategy_config["name"]
        strategy_class = strategy_config["class"]
        strategy_params = strategy_config["params"]
        description = strategy_config["description"]

        logger.info(f"Processing strategies: {strategy_name} - {description}")

        try:
            chunker = get_chunking_strategy(strategy_class, **strategy_params)
            chunks = chunker.split(preprocessed_documents)
            logger.info(f"Created {len(chunks)} chunks")
            collection_name = f"popatkus_{strategy_name}"
            num_added = chroma.add_documents(collection_name, chunks)
            collection_info = chroma.get_collection_info(collection_name)

            results[strategy_name] = {
                "collection_name": collection_name,
                "num_chunks": len(chunks),
                "num_added": num_added,
                "collection_info": collection_info,
                "description": description
            }

            logger.info(f"Strategy '{strategy_name}' completed: {num_added} documents in collection")

        except Exception as e:
            logger.error(f"Error in processing strategy '{strategy_name}': {e}")
            results[strategy_name] = {"error": str(e)}

    logger.info("=== RESULTS ===")

    for strategy_name, result in results.items():
        if "error" in result:
            logger.error(f"{strategy_name}: ERROR - {result['error']}")
        else:
            logger.info(f"{strategy_name}:")
            logger.info(f"  - Collection: {result['collection_name']}")
            logger.info(f"  - Chunks: {result['num_chunks']}")
            logger.info(f"  - Added: {result['num_added']}")
            logger.info(f"  - Description: {result['description']}")

    logger.info("=== ALL COLLECTIONS CHROMADB: ===")
    collections = chroma.list_collections()
    for col in collections:
        info = chroma.get_collection_info(col)
        logger.info(f"  - {col}: {info.get('count', 0)} documents")

    logger.info("\n Part 2 completed")
    logger.info(f"DB saved in: {settings.chroma_persist_dir}")


if __name__ == "__main__":
    main()
