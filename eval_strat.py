#!/usr/bin/env python

import pandas as pd
from src.vector_db.chroma_manager import get_chroma_manager
from src.eval.ragas_eval import RAGASEvaluator
from src.utils.logger import logger
from src.config.settings import settings


def main():
    logger.info("Этап 3: Оценка качества стратегий чанкинга с RAGAS")

    collections = [
        "popatkus_recursive",
        "popatkus_markdown_header",
        "popatkus_parent_document",
        "popatkus_semantic",
        "popatkus_legal_document"
    ]

    chroma = get_chroma_manager()
    existing_collections = chroma.list_collections()

    available = [c for c in collections if c in existing_collections]
    missing = [c for c in collections if c not in existing_collections]

    if missing:
        logger.warning(f"Missing collections: {missing}")

    if not available:
        logger.error("No collections found to evaluate!")
        return

    logger.info(f"Evaluating collections: {available}")

    evaluator = RAGASEvaluator()

    logger.info("\nGenerating test dataset...")
    test_dataset = evaluator.generate_test_dataset(
        collection_name=available[0],
        num_questions=settings.testset_size
    )
    test_dataset.to_csv("test_dataset.csv", index=False)
    logger.info(f"Test dataset saved to test_dataset.csv")
    print("\nSample test questions:")
    print(test_dataset[['question', 'ground_truth']].head())

    results = evaluator.compare_all_strategies(
        collections=available,
        test_df=None,
        num_test_questions=settings.testset_size
    )

    results.to_csv("evaluation_results.csv", index=False)
    logger.info(f"\nResults saved to evaluation_results.csv")

    if not results.empty:
        metric_cols = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
        available_metrics = [c for c in metric_cols if c in results.columns]

        if available_metrics:
            results['avg_score'] = results[available_metrics].mean(axis=1)
            best_idx = results['avg_score'].idxmax()
            best_strategy = results.loc[best_idx, 'strategy']
            best_score = results.loc[best_idx, 'avg_score']

            logger.info("ВЫВОДЫ")
            logger.info(f"Лучшая стратегия чанкинга: {best_strategy}")
            logger.info(f"Средний балл: {best_score:.3f}")

    logger.info("\nЭтап 3 завершен!")


if __name__ == "__main__":
    main()
