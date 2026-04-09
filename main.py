#!/usr/bin/env python

import sys
from src.config.settings import settings
from src.data.loader import get_document_loader
from src.data.preprocessor import DocumentPreprocessor
from src.utils.logger import logger


def main():
    logger.info("Starting RAG POPATKUS system")
    logger.info(f"Settings loaded: {settings.model_dump()}")

    document_path = settings.raw_data_dir / settings.document_name

    if not document_path.exists():
        logger.error(f"Document not found: {document_path}")
        logger.info(f"Please place {settings.document_name} in {settings.raw_data_dir}")
        sys.exit(1)

    logger.info(f"Loading document from {document_path}")
    loader = get_document_loader("unstructured")

    try:
        documents = loader.load_document(document_path)
        logger.info(f"Loaded {len(documents)} documents")
        preprocessor = DocumentPreprocessor()
        chunks = preprocessor.process_documents(documents)

        logger.info(f"Created {len(chunks)} chunks")

        if chunks:
            sample = chunks[0]
            logger.info(f"\nSample chunk:")
            logger.info(f"ID: {sample.id}")
            logger.info(f"Metadata: {sample.metadata.model_dump()}")
            logger.info(f"Content preview: {sample.content[:200]}...")

            if sample.metadata.extra_info:
                logger.info(f"Extra info: {sample.metadata.extra_info}")

        import json

        output_file = settings.processed_data_dir / "processed_chunks.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(
                [{
                    "id": c.id,
                    "content": c.content,
                    "metadata": c.metadata.model_dump()
                } for c in chunks],
                f,
                ensure_ascii=False,
                indent=2
            )

        logger.info(f"Saved processed chunks to {output_file}")

        stats_file = settings.processed_data_dir / "statistics.json"
        stats = {
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "avg_chunk_length": sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0,
            "document_format": settings.document_type,
            "loader_used": loader.loader_type
        }

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved statistics to {stats_file}")

    except Exception as e:
        logger.exception(f"Error processing document: {e}")
        sys.exit(1)

    logger.info("Processing completed successfully")


if __name__ == "__main__":
    main()
