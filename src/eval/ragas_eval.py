import pandas as pd
from typing import List, Dict, Any, Optional
import json
import re
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.vector_db.chroma_manager import get_chroma_manager
from src.config.settings import settings
from src.utils.logger import logger
from src.vector_db.metadata_filter import parse_section_filter, extract_question_without_filter
from src.retrieval.reranker import DocumentReranker


class RAGASEvaluator:

    def __init__(self, llm_model: str = "gpt-3.5-turbo"):
        try:
            self.llm = ChatOpenAI(model=llm_model, temperature=0)
            logger.info(f"Initialized RAGASEvaluator with LLM: {llm_model}")
        except Exception as e:
            logger.warning(f"OpenAI not available: {e}")
            logger.info("Will use fallback evaluation methods")
            self.llm = None

        self.chroma = get_chroma_manager()
        self.reranker = DocumentReranker()
        logger.info("RAGASEvaluator ready")

    def generate_test_dataset(
            self,
            collection_name: str,
            num_questions: int = 100
    ) -> pd.DataFrame:
        """Generate diverse test questions using LLM based on document chunks."""
        logger.info(f"Generating {num_questions} test questions from {collection_name}")

        collection = self.chroma.client.get_collection(collection_name)
        all_ids = collection.get()['ids']

        # Sample chunks (aim for ~20-30 chunks to generate questions from)
        chunks_to_sample = min(max(20, num_questions // 4), len(all_ids))
        import random
        random.seed(42)
        sample_ids = random.sample(all_ids, min(chunks_to_sample, len(all_ids)))

        sample_chunks = collection.get(ids=sample_ids)

        test_data = {
            "question": [],
            "ground_truth": [],
            "contexts": []
        }

        question_count = 0
        for doc_id, doc_text in zip(sample_chunks['ids'], sample_chunks['documents']):
            if question_count >= num_questions:
                break

            if len(doc_text.strip()) < 50:
                continue

            # Generate 4-5 questions per chunk via LLM
            generated_questions = self._generate_questions_for_chunk(doc_text)

            for q in generated_questions:
                if question_count >= num_questions:
                    break

                test_data["question"].append(q)
                # Keep ground_truth aligned with the excerpt we used to generate the question.
                # We also cap it to avoid overly long strings.
                gt = doc_text[:1000]
                test_data["ground_truth"].append(gt)
                test_data["contexts"].append([gt])
                question_count += 1

        df = pd.DataFrame(test_data)
        logger.info(f"Generated {len(df)} test questions (requested {num_questions})")
        return df

    def _generate_questions_for_chunk(self, chunk_text: str, num_questions: int = 4) -> List[str]:
        """Generate diverse questions for a chunk using LLM, fallback to templates."""
        if len(chunk_text.strip()) < 50:
            return []

        # Try LLM-based generation first
        if self.llm:
            try:
                sentences = [s.strip() for s in chunk_text.split('.') if len(s.strip()) > 20]
                if not sentences:
                    return self._generate_fallback_questions(chunk_text, num_questions)

                prompt = f"""Based on this document excerpt, generate {num_questions} different natural questions that a student might ask:

EXCERPT:
{chunk_text[:400]}

INSTRUCTIONS:
- Generate questions that are diverse in type: definition, procedure, requirement, exception, clarification
- Questions should be natural, not templates
- Each question should be answerable from the excerpt
- Keep questions between 10-100 words
- Return ONLY a JSON array of strings (questions), no markdown formatting

EXAMPLE OUTPUT:
["Что такое академическая задолженность?", "Какой процесс пересдачи?", "Какие сроки допустимы?"]

Generate questions now:"""

                response = self.llm.invoke(prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)

                # Parse JSON from response
                json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
                if json_match:
                    questions = json.loads(json_match.group())
                    # Filter valid questions
                    questions = [
                        q.strip() for q in questions
                        if isinstance(q, str) and 10 <= len(q) <= 200 and '?' in q
                    ]
                    if questions:
                        logger.debug(f"Generated {len(questions)} questions from chunk via LLM")
                        return questions[:num_questions]

            except Exception as e:
                logger.debug(f"LLM question generation failed: {e}, using fallback")

        # Fallback: use template-based generation
        return self._generate_fallback_questions(chunk_text, num_questions)

    def _generate_fallback_questions(self, chunk_text: str, num_questions: int = 4) -> List[str]:
        """Fallback question generation using diverse templates."""
        words = chunk_text.split()
        first_10_words = ' '.join(words[:10])
        first_20_words = ' '.join(words[:20])

        question_templates = [
            f"Что такое {first_10_words}?",
            f"Каковы требования к {first_10_words}?",
            f"Какой процесс {first_10_words}?",
            f"Что необходимо знать про {first_10_words}?",
            f"Как {first_10_words}?",
            f"Какие сроки для {first_10_words}?",
            f"Каким образом {first_20_words}?",
            f"Какие исключения существуют для {first_10_words}?",
        ]

        # Shuffle and return requested number
        import random
        random.shuffle(question_templates)
        return question_templates[:num_questions]

    def query_collection(
            self,
            collection_name: str,
            question: str,
            k: int = None
    ) -> tuple[List[str], str]:
        # Align evaluation retrieval with Telegram bot:
        # - metadata filter via `section` parsing
        # - initial retrieval with `top_k_initial`
        # - reranking down to `top_k_final`
        if k is None:
            k = settings.top_k_final

        section_filter = parse_section_filter(question)
        clean_question, section_filter_alt = extract_question_without_filter(question)
        where_filter = section_filter or section_filter_alt

        query_text = clean_question or question

        results = self.chroma.query(
            collection_name,
            query_text,
            n_results=settings.top_k_initial,
            where_filter=where_filter
        )

        all_docs = results['documents'][0] if results.get('documents') else []
        all_metas = results['metadatas'][0] if results.get('metadatas') else []

        if all_docs:
            ranked_indices = self.reranker._get_ranked_indices(
                query_text,
                all_docs,
                top_k=k
            )
            contexts = [all_docs[i] for i in ranked_indices]
        else:
            contexts = []

        if contexts:
            # Use retrieved context as the "answer" text for fallback evaluation.
            joined = "\n\n".join(contexts[:k])
            answer = joined[:1200]
        else:
            answer = "В документе нет информации по вашему вопросу"

        return contexts, answer


    def evaluate_strategy(
            self,
            collection_name: str,
            test_df: pd.DataFrame
    ) -> Dict[str, float]:
        logger.info(f"Evaluating strategy: {collection_name}")
        answers = []
        contexts_list = []

        for question in test_df['question']:
            contexts, answer = self.query_collection(
                collection_name,
                question,
                k=settings.top_k_final
            )
            answers.append(answer)
            contexts_list.append(contexts)

        eval_dataset = Dataset.from_pandas(pd.DataFrame({
            "question": test_df['question'],
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": test_df['ground_truth']
        }))

        if self.llm:
            try:
                result = evaluate(
                    dataset=eval_dataset,
                    metrics=[
                        context_precision,
                        context_recall,
                        faithfulness,
                        answer_relevancy
                    ],
                    llm=self.llm
                )

                metrics = {
                    "context_precision": result["context_precision"],
                    "context_recall": result["context_recall"],
                    "faithfulness": result["faithfulness"],
                    "answer_relevancy": result["answer_relevancy"]
                }
            except Exception as e:
                logger.error(f"RAGAS evaluation failed: {e}")
                metrics = self._calculate_fallback_metrics(eval_dataset)
        else:
            metrics = self._calculate_fallback_metrics(eval_dataset)

        logger.info(f"Metrics for {collection_name}: {metrics}")
        return metrics

    def _calculate_fallback_metrics(self, dataset: Dataset) -> Dict[str, float]:
        logger.warning("Using fallback metrics (simplified)")
        total = len(dataset)
        context_precision_sum = 0.0
        context_recall_sum = 0.0
        faithfulness_sum = 0.0
        answer_relevancy_sum = 0.0

        def tokenize(s: str) -> set:
            return set(re.findall(r"[\wа-яА-ЯЁё]+", s.lower()))

        def similarity(a: str, b: str) -> float:
            ta, tb = tokenize(a), tokenize(b)
            if not ta or not tb:
                return 0.0
            return len(ta.intersection(tb)) / max(1, len(ta))

        # Threshold for "good enough" overlap.
        sim_threshold = 0.15

        for item in dataset:
            gt = item["ground_truth"] or ""
            answer = item["answer"] or ""
            contexts = item["contexts"] or []

            if contexts:
                relevant_contexts = [ctx for ctx in contexts if similarity(gt, ctx) >= sim_threshold]
                context_precision_sum += len(relevant_contexts) / max(1, len(contexts))
                context_recall_sum += 1.0 if relevant_contexts else 0.0
            else:
                context_precision_sum += 0.0
                context_recall_sum += 0.0

            faithfulness_sum += 1.0 if similarity(answer, gt) >= sim_threshold else 0.0
            # With the current "answer" construction, relevance correlates with overlap to the excerpt.
            answer_relevancy_sum += 1.0 if similarity(answer, gt) >= sim_threshold else 0.0

        return {
            "context_precision": context_precision_sum / total if total > 0 else 0,
            "context_recall": context_recall_sum / total if total > 0 else 0,
            "faithfulness": faithfulness_sum / total if total > 0 else 0,
            "answer_relevancy": answer_relevancy_sum / total if total > 0 else 0
        }

    def compare_all_strategies(
            self,
            collections: List[str],
            test_df: Optional[pd.DataFrame] = None,
            num_test_questions: int = 20
    ) -> pd.DataFrame:
        results = []

        for collection_name in collections:
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Evaluating: {collection_name}")
            logger.info(f"{'=' * 50}")
            if test_df is None:
                current_test_df = self.generate_test_dataset(collection_name, num_test_questions)
            else:
                current_test_df = test_df

            metrics = self.evaluate_strategy(collection_name, current_test_df)
            metrics["strategy"] = collection_name
            results.append(metrics)
        results_df = pd.DataFrame(results)
        logger.info("\n" + "=" * 60)
        logger.info("СРАВНЕНИЕ СТРАТЕГИЙ ЧАНКИНГА")
        logger.info("=" * 60)
        print(results_df.to_string(index=False))

        return results_df
