#!/usr/bin/env python
import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.client.default import DefaultBotProperties
from aiogram.exceptions import TelegramBadRequest

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.vector_db.chroma_manager import get_chroma_manager
from src.vector_db.metadata_filter import parse_section_filter, extract_question_without_filter
from src.retrieval.reranker import DocumentReranker
from src.utils.logger import logger
from src.config.settings import settings

from dotenv import load_dotenv

load_dotenv()


class RAGBot:
    def __init__(self):
        self.chroma = get_chroma_manager()
        self.reranker = DocumentReranker()
        self.llm = None
        self.rag_chain = None
        self._last_results = None
        self._last_question = None
        self._init_llm()
        logger.info(f"Bot initialized with collection: {settings.collection_name}")

    def _init_llm(self):
        try:
            self.llm = ChatOllama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=settings.ollama_temperature,
                num_ctx=settings.ollama_context_size
            )
            self.llm.invoke("Привет")
            logger.info(f"Ollama connected with model: {settings.ollama_model}")
            self._init_rag_chain()
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")

    def _init_rag_chain(self):
        def format_docs(docs: List) -> str:
            if not docs:
                return "Информация не найдена."
            return "\n\n---\n\n".join(docs)

        def retrieve(question: str) -> str:
            if question == self._last_question and self._last_results is not None:
                logger.info(f"Using cached Chroma results for: {question[:120]}...")
                documents = self._last_results['documents']
            else:
                logger.info(f"Chroma query: {question[:120]}...")
                section_filter = parse_section_filter(question)
                clean_question, section_filter_alt = extract_question_without_filter(question)
                where_filter = section_filter or section_filter_alt

                if where_filter:
                    logger.debug(f"Query with metadata filter: {where_filter}")

                results = self.chroma.query(
                    settings.collection_name,
                    clean_question,
                    n_results=settings.top_k_initial,
                    where_filter=where_filter
                )

                # Re-rank documents for better relevance
                documents = []
                metadatas = []
                if results['documents'] and results['documents'][0]:
                    all_docs = results['documents'][0]
                    all_metas = results['metadatas'][0] if results['metadatas'] else []
                    clean_q = clean_question or question

                    # Get re-ranked indices
                    ranked_indices = self.reranker._get_ranked_indices(clean_q, all_docs, top_k=settings.top_k_final)
                    documents = [all_docs[i] for i in ranked_indices]
                    metadatas = [all_metas[i] for i in ranked_indices] if all_metas else []
                    logger.info(f"Retrieved {len(all_docs)} docs, re-ranked to top-{settings.top_k_final}")

                self._last_results = {'documents': [documents], 'metadatas': [metadatas]}
                self._last_question = question

            if self._last_results and self._last_results['documents'] and self._last_results['documents'][0]:
                return format_docs(self._last_results['documents'][0])
            return format_docs([])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты - помощник студентов Высшей школы экономики по вопросам учебного регламента (ПОПАТКУС).

ИНСТРУКЦИИ:
1. Отвечай ДЛЯ ВСЕХ ВОПРОСОВ только на основе предоставленного контекста
2. Если информации нет в контексте, ясно скажи: "В документе нет информации по вашему вопросу"
3. Структурируй ответ логически:
   - Начни с прямого ответа на вопрос (1-2 предложения)
   - Объясни детали, условия и исключения
   - Приведи конкретные сроки, номера статей, пункты
   - Укажи связанные правила, если релевантны

4. Отвечай подробно, а не кратко - используй 3-5 предложений минимум
5. Каждый ответ должен содержать объяснение ПОЧЕМУ и НА ОСНОВЕ КАКИХ ПРАВИЛ
6. Приводи прямые цитаты или ссылки на статьи/пункты документа
7. Будь точен - не добавляй предположения, только факты из контекста

ПРИМЕРЫ ХОРОШИХ ОТВЕТОВ:
Q: Что такое академическая задолженность?
A: Академическая задолженность - это задолженность студента перед вузом по учебным обязательствам. Согласно ПОПАТКУС, она может возникнуть в результате неудовлетворительной оценки за экзамен или неявки на экзамен без уважительной причины. Студент обязан погасить такую задолженность путем пересдачи в установленные сроки (обычно до конца семестра). Неликвидация задолженности может привести к отчислению.

Q: Какой процесс пересдачи?
A: Пересдача проводится в соответствии со следующей процедурой: сначала студент подает заявление в деканат, затем назначается дата пересдачи в согласованное время. Пересдача проходит перед комиссией, состоящей из преподавателя и его ассистента. Результаты пересдачи должны быть документированы, и студент имеет право на две пересдачи в семестр.

КОНТЕКСТ ДОКУМЕНТА:
{context}"""),
            ("human", "{question}")
        ])

        self.rag_chain = (
                {"context": retrieve, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )
        logger.info("RAG chain initialized with enhanced prompt and re-ranking")


    async def generate_answer(self, question: str) -> Tuple[str, List[str]]:
        if self.rag_chain:
            try:
                answer = await asyncio.to_thread(self.rag_chain.invoke, question)
                sources = []
                if self._last_results and self._last_results['metadatas'] and self._last_results['metadatas'][0]:
                    sources = [m.get('source', 'ПОПАТКУС') for m in self._last_results['metadatas'][0]]
                    logger.debug(f"Using cached metadata for sources (query count: 1 instead of 2)")

                preview = answer[:500] + ("..." if len(answer) > 500 else "")
                logger.info(f"LLM answer ({len(answer)} chars): {preview}")
                return answer, sources
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
        return "LLM недоступен. Проверьте установку Ollama.", []

    def search_only(self, question: str) -> Tuple[str, List[str]]:
        if question == self._last_question and self._last_results is not None:
            logger.debug(f"Using cached search results for: {question[:50]}...")
            results = self._last_results
        else:
            logger.debug(f"Executing new search query for: {question[:50]}...")
            section_filter = parse_section_filter(question)
            clean_question, section_filter_alt = extract_question_without_filter(question)
            where_filter = section_filter or section_filter_alt

            if where_filter:
                logger.debug(f"Search with metadata filter: {where_filter}")

            results_raw = self.chroma.query(
                settings.collection_name,
                clean_question,
                n_results=settings.top_k_initial,
                where_filter=where_filter
            )

            # Re-rank for search mode too
            all_docs = results_raw['documents'][0] if results_raw['documents'] else []
            all_metas = results_raw['metadatas'][0] if results_raw.get('metadatas') else []

            if all_docs:
                ranked_indices = self.reranker._get_ranked_indices(clean_question or question, all_docs, top_k=settings.top_k_final)
                reranked_docs = [all_docs[i] for i in ranked_indices]
                reranked_metas = [all_metas[i] for i in ranked_indices] if all_metas else []
            else:
                reranked_docs = []
                reranked_metas = []

            results = {
                'documents': [reranked_docs],
                'metadatas': [reranked_metas]
            }
            self._last_results = results
            self._last_question = question

        documents, sources = [], []
        if results['documents'] and results['documents'][0]:
            documents = results['documents'][0]
            sources = [m.get('source', 'ПОПАТКУС') for m in results['metadatas'][0]] if results.get('metadatas') else []

        if not documents:
            return "Не удалось найти информацию.", []

        response = "Режим поиска (LLM недоступен)\n\n"
        for i, doc in enumerate(documents[:settings.top_k_final], 1):
            snippet = doc[:400].strip()
            if len(doc) > 400:
                snippet += "..."
            response += f"\nФрагмент {i}:\n{snippet}\n"
        preview = response[:500] + ("..." if len(response) > 500 else "")
        logger.info(f"Search-only answer ({len(response)} chars): {preview}")
        return response, sources


rag_bot = RAGBot()
bot = Bot(token=settings.bot_token, default=DefaultBotProperties())
dp = Dispatcher()


async def safe_answer(
    message: types.Message,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    """Send plain text; fall back if Telegram rejects formatting."""
    try:
        await message.answer(text, reply_markup=reply_markup)
    except TelegramBadRequest:
        await message.answer(text[:4096], reply_markup=reply_markup)


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    llm_status = "Активна" if rag_bot.llm else "Недоступна"
    await message.answer(
        f"RAG-бот по ПОПАТКУС\n\n"
        f"LLM: {llm_status}\n"
        f"Модель: {settings.ollama_model if rag_bot.llm else '—'}\n"
        f"База знаний: {settings.collection_name}\n\n"
        f"Примеры вопросов:\n"
        f"- Что такое академическая задолженность?\n"
        f"- Какие сроки пересдачи экзаменов?\n\n"
        f"Просто задайте свой вопрос!"
    )


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.answer(
        "Как пользоваться ботом:\n\n"
        "1. Напишите ваш вопрос о правилах аттестации\n"
        "2. Бот найдет фрагменты из ПОПАТКУС\n"
        "3. LLM сгенерирует ответ\n\n"
        "Команды:\n"
        "/start - Начать\n/help - Справка\n/info - О системе\n/stats - Статистика\n/model - Информация о LLM"
    )


@dp.message(Command("info"))
async def cmd_info(message: types.Message):
    await message.answer(
        "О системе:\n\n"
        "Технология: RAG\n Чанкинг: Рекурсивный (размер 512)\n"
        "Эмбеддинги: multilingual-e5-large\n• Векторная БД: ChromaDB\n"
        "LLM: Ollama\n Фреймворк: aiogram 3.x\n\n Система готова!"
    )


@dp.message(Command("stats"))
async def cmd_stats(message: types.Message):
    try:
        info = rag_bot.chroma.get_collection_info(settings.collection_name)
        await message.answer(
            f"Статистика:\n\n Коллекция: {settings.collection_name}\n Документов: {info.get('count', 0)}")
    except Exception as e:
        await message.answer(f"Ошибка: {e}")


@dp.message(Command("model"))
async def cmd_model(message: types.Message):
    if rag_bot.llm:
        await message.answer(f"LLM: {settings.ollama_model}\n Статус: Активна\n Температура: {settings.ollama_temperature}")
    else:
        await message.answer(
            "LLM недоступна\n\nУстановите Ollama: https://ollama.com")


@dp.message()
async def handle_question(message: types.Message):
    question = message.text
    if question.startswith('/'):
        return

    user_id = message.from_user.id if message.from_user else "unknown"
    logger.info(f"[user={user_id}] Question: {question}")

    await bot.send_chat_action(message.chat.id, action="typing")

    if rag_bot.llm:
        logger.info("Generating answer via RAG + LLM...")
        answer, sources = await rag_bot.generate_answer(question)
    else:
        logger.info("Generating answer via search-only mode...")
        answer, sources = rag_bot.search_only(question)

    if sources:
        answer += f"\n\nИсточники: {', '.join(set(sources[:3]))}"

    logger.info(
        f"[user={user_id}] Response sent ({len(answer)} chars), "
        f"sources={sources[:3] if sources else 'none'}"
    )

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Полезно", callback_data="like"),
         InlineKeyboardButton(text="Бесполезно", callback_data="dislike")]
    ])
    await safe_answer(message, answer, reply_markup=keyboard)


@dp.callback_query()
async def handle_callback(callback: types.CallbackQuery):
    await callback.answer()
    if callback.data == "like":
        await callback.message.edit_reply_markup(reply_markup=None)
        await callback.message.answer("Спасибо!")
    elif callback.data == "dislike":
        await callback.message.edit_reply_markup(reply_markup=None)
        await callback.message.answer("Жаль. Попробуйте переформулировать вопрос.")


async def main():
    logger.info("Запуск Telegram бота с LLM (Ollama)")

    logger.info("Бот запущен!")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
