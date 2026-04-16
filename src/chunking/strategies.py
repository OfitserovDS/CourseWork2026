from typing import List, Dict, Any
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter
)
from langchain_core.documents import Document as LangChainDocument
from src.utils.logger import logger
import re


class ChunkingStrategy:
    def __init__(self, name: str):
        self.name = name

    def split(self, documents: List[LangChainDocument]) -> List[LangChainDocument]:
        raise NotImplementedError


class RecursiveChunking(ChunkingStrategy):

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        super().__init__("recursive")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"Initialized RecursiveChunking: size={chunk_size}, overlap={chunk_overlap}")

    def split(self, documents: List[LangChainDocument]) -> List[LangChainDocument]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len
        )

        chunks = splitter.split_documents(documents)

        for chunk in chunks:
            chunk.metadata["chunking_strategy"] = self.name
            chunk.metadata["chunk_size"] = self.chunk_size

        logger.info(f"RecursiveChunking: created {len(chunks)} chunks")
        return chunks


class MarkdownHeaderChunking(ChunkingStrategy):
    """Experimental chunking strategy - NOT USED IN MAIN PIPELINE.

    Kept for potential future use with markdown-formatted documents.
    Main pipeline uses LegalDocumentChunking."""

    def __init__(self):
        super().__init__("markdown_header")
        logger.info("Initialized MarkdownHeaderChunking")

    def split(self, documents: List[LangChainDocument]) -> List[LangChainDocument]:
        headers_to_split_on = [
            ("#", "Раздел"),  # Разделы
            ("##", "Глава"),  # Главы
            ("###", "Статья"),  # Статьи
            ("####", "Пункт"),  # Пункты
        ]

        splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

        all_chunks = []
        for doc in documents:
            try:
                markdown_text = self._text_to_markdown(doc.page_content)
                chunks = splitter.split_text(markdown_text)

                for chunk in chunks:
                    chunk.metadata.update(doc.metadata)
                    chunk.metadata["chunking_strategy"] = self.name

                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Markdown split failed, using fallback: {e}")
                doc.metadata["chunking_strategy"] = self.name
                all_chunks.append(doc)

        logger.info(f"MarkdownHeaderChunking: created {len(all_chunks)} chunks")
        return all_chunks

    def _text_to_markdown(self, text: str) -> str:
        lines = text.split('\n')
        result = []

        for line in lines:
            if len(line) < 100 and (line.strip().endswith(':') or any(c.isdigit() for c in line[:10])):
                result.append(f"### {line.strip()}")
            else:
                result.append(line)

        return '\n'.join(result)


class ParentDocumentChunking(ChunkingStrategy):
    """Experimental chunking strategy - NOT USED IN MAIN PIPELINE.

    Hierarchical chunking with parent/child relationship tracking.
    Kept for potential future use in complex document retrieval.
    Main pipeline uses LegalDocumentChunking."""

    def __init__(self, parent_chunk_size: int = 2000, child_chunk_size: int = 300):
        super().__init__("parent_document")
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        logger.info(f"Initialized ParentDocumentChunking: parent={parent_chunk_size}, child={child_chunk_size}")

    def split(self, documents: List[LangChainDocument]) -> List[LangChainDocument]:
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". "]
        )
        parent_chunks = parent_splitter.split_documents(documents)

        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=50,
            separators=["\n", ". ", " ", ""]
        )

        all_child_chunks = []
        for i, parent in enumerate(parent_chunks):
            child_chunks = child_splitter.split_documents([parent])
            for j, child in enumerate(child_chunks):
                child.metadata["parent_id"] = f"parent_{i}"
                child.metadata["parent_content"] = parent.page_content[:500]  # превью
                child.metadata["chunking_strategy"] = self.name
                child.metadata["chunk_type"] = "child"
                all_child_chunks.append(child)

        logger.info(
            f"ParentDocumentChunking: created {len(parent_chunks)} parents and {len(all_child_chunks)} children")
        return all_child_chunks


class SemanticChunking(ChunkingStrategy):
    """Experimental chunking strategy - NOT USED IN MAIN PIPELINE.

    Simple paragraph-based splitting. Kept for potential future use.
    Main pipeline uses LegalDocumentChunking for better context preservation."""

    def __init__(self):
        super().__init__("semantic")
        logger.info("Initialized SemanticChunking")

    def split(self, documents: List[LangChainDocument]) -> List[LangChainDocument]:
        all_chunks = []

        for doc in documents:
            paragraphs = doc.page_content.split('\n\n')

            for i, para in enumerate(paragraphs):
                if para.strip():
                    chunk = LangChainDocument(
                        page_content=para.strip(),
                        metadata={
                            **doc.metadata,
                            "chunking_strategy": self.name,
                            "paragraph_index": i
                        }
                    )
                    all_chunks.append(chunk)

        logger.info(f"SemanticChunking: created {len(all_chunks)} chunks")
        return all_chunks


class LegalDocumentChunking(ChunkingStrategy):
    """Specialized chunking for legal documents (ПОПАТКУС regulations)."""

    def __init__(self, min_chunk_size: int = 80, max_chunk_size: int = 600):
        super().__init__("legal_document")
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        logger.info(f"Initialized LegalDocumentChunking: min={min_chunk_size}, max={max_chunk_size}")

    def split(self, documents: List[LangChainDocument]) -> List[LangChainDocument]:
        """
        Split legal documents preserving:
        1. Numbered lists as single chunks
        2. Hierarchical structure (phases, procedures)
        3. Rich metadata (document_phase, actor_type, action_type)
        """
        all_chunks = []

        # Define document phases (for ПОПАТКУС)
        phase_markers = {
            "определения": r"(используемые понятия|сокращение|определение|–)",
            "текущий контроль": r"(текущий контроль|текущей аттестации)",
            "промежуточная": r"(промежуточн|аттестация|экзамен|зачет)",
            "апелляция": r"(апелляция|обжалование|жалоба)",
            "сессия": r"(сессия|расписание|график)",
        }

        for doc in documents:
            text = doc.page_content
            current_phase = self._detect_phase(text, phase_markers)

            chunks = self._process_document(text, doc.metadata, current_phase)
            all_chunks.extend(chunks)

        logger.info(f"LegalDocumentChunking: created {len(all_chunks)} chunks")
        return all_chunks

    def _detect_phase(self, text: str, phase_markers: Dict[str, str]) -> str:
        """Detect document phase from text."""
        text_lower = text.lower()
        for phase, pattern in phase_markers.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                return phase
        return "общие"

    def _process_document(
        self,
        text: str,
        base_metadata: Dict,
        current_phase: str
    ) -> List[LangChainDocument]:
        """Process document text into chunks."""
        chunks = []

        # Split by paragraphs first
        paragraphs = text.split('\n\n')

        i = 0
        while i < len(paragraphs):
            para = paragraphs[i].strip()

            if not para or len(para) < 10:
                i += 1
                continue

            # Check if this is a numbered list
            if self._is_numbered_list_start(para):
                # Collect all consecutive list items
                list_items = [para]
                j = i + 1

                while j < len(paragraphs):
                    next_para = paragraphs[j].strip()
                    if self._is_list_item(next_para) or (
                        next_para and not self._is_new_section(next_para)
                    ):
                        list_items.append(next_para)
                        j += 1
                    else:
                        break

                # Combine list items
                combined = "\n".join(list_items)

                if len(combined) <= self.max_chunk_size:
                    action_type = self._detect_action_type(combined)
                    chunk = self._create_chunk(
                        combined, base_metadata, current_phase, action_type
                    )
                    chunks.append(chunk)
                else:
                    # Split large lists
                    sub_chunks = self._split_large_text(
                        combined, base_metadata, current_phase
                    )
                    chunks.extend(sub_chunks)

                i = j
            else:
                # Regular paragraph
                if len(para) < self.min_chunk_size:
                    # Merge with next paragraph
                    if i + 1 < len(paragraphs):
                        merged = para + " " + paragraphs[i + 1].strip()
                        if len(merged) <= self.max_chunk_size:
                            action_type = self._detect_action_type(merged)
                            chunk = self._create_chunk(
                                merged, base_metadata, current_phase, action_type
                            )
                            chunks.append(chunk)
                            i += 2
                            continue
                    i += 1
                    continue

                # Split if too large
                if len(para) > self.max_chunk_size:
                    sub_chunks = self._split_large_text(
                        para, base_metadata, current_phase
                    )
                    chunks.extend(sub_chunks)
                else:
                    action_type = self._detect_action_type(para)
                    chunk = self._create_chunk(
                        para, base_metadata, current_phase, action_type
                    )
                    chunks.append(chunk)

                i += 1

        return chunks

    def _is_numbered_list_start(self, text: str) -> bool:
        """Check if text starts with numbered list."""
        # Digits: "1) ...", Letters (RU): "а) ...", "А) ..."
        return bool(re.match(r"^\d+\)", text)) or bool(re.match(r"^[А-Яа-я]\)", text))

    def _is_list_item(self, text: str) -> bool:
        """Check if text is a list item."""
        # Common list markers: digits, RU letters, dash/bullet.
        return (
            bool(re.match(r"^\d+\)", text))
            or bool(re.match(r"^[А-Яа-я]\)", text))
            or bool(re.match(r"^[-–—]", text))
            or bool(re.match(r"^•", text))
        )

    def _is_new_section(self, text: str) -> bool:
        """Check if text starts a new section."""
        return (
            len(text) < 140
            and (
                text.isupper()
                or text.endswith((":","："))
                or bool(re.match(r"^[IVX]+\.", text, re.IGNORECASE))
            )
        )

    def _split_large_text(
        self,
        text: str,
        base_metadata: Dict,
        current_phase: str,
        max_size: int = None
    ) -> List[LangChainDocument]:
        """Split large text into smaller chunks."""
        max_size = max_size or self.max_chunk_size

        if len(text) <= max_size:
            return [
                self._create_chunk(
                    text,
                    base_metadata,
                    current_phase,
                    self._detect_action_type(text),
                )
            ]

        # Use recursive splitter for large texts
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_size,
            chunk_overlap=50,
            separators=["\n", ".", " ", ""],
        )

        sub_chunks = splitter.split_text(text)
        return [
            self._create_chunk(
                chunk,
                base_metadata,
                current_phase,
                self._detect_action_type(chunk),
            )
            for chunk in sub_chunks
            if len(chunk) >= self.min_chunk_size
        ]

    def _detect_action_type(self, text: str) -> str:
        """Detect action type from text."""
        text_lower = text.lower()

        if re.search(r"(запрещ|нельзя|не должн|не может)", text_lower):
            return "restriction"
        elif re.search(r"(может|возможн|допуска|разрешен)", text_lower):
            return "permission"
        elif re.search(r"(процедур|порядок|следующ|должн|обязан)", text_lower):
            return "procedure"
        elif re.search(r"(определен|понима|суть|это|являет|является)", text_lower):
            return "definition"
        elif re.search(r"(требован|условие|критерий|условие)", text_lower):
            return "requirement"
        else:
            return "general"

    def _create_chunk(
        self,
        text: str,
        base_metadata: Dict,
        document_phase: str,
        action_type: str
    ) -> LangChainDocument:
        """Create a chunk with enriched metadata."""
        # Detect actor type
        actor_type = self._detect_actor_type(text)

        # Extract keywords
        keywords = self._extract_keywords(text)

        metadata = {
            **base_metadata,
            "chunking_strategy": self.name,
            "document_phase": document_phase,
            "actor_type": actor_type,
            "action_type": action_type,
            "keywords": ", ".join(keywords) if keywords else "",
            "chunk_size": len(text),
        }

        return LangChainDocument(page_content=text.strip(), metadata=metadata)

    def _detect_actor_type(self, text: str) -> str:
        """Detect who the regulation applies to."""
        text_lower = text.lower()

        actors = []
        if re.search(r"(студент|обучающ)", text_lower):
            actors.append("student")
        if re.search(r"(преподавател|инструктор|викладач)", text_lower):
            actors.append("instructor")
        if re.search(r"(деканат|администрация|декан|уполномоченный)", text_lower):
            actors.append("administration")

        return ", ".join(actors) if actors else "general"

    def _extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """Extract important keywords from text."""
        legal_keywords = [
            "пересдача",
            "апелляция",
            "уважительная",
            "причина",
            "сроки",
            "проверка",
            "комиссия",
            "экзамен",
            "зачет",
            "оценка",
            "задолженность",
            "консультация",
            "документ",
            "заявление",
            "расписание",
        ]

        text_lower = text.lower()
        found = [kw for kw in legal_keywords if kw in text_lower]

        return found[:top_k]


def get_chunking_strategy(strategy_name: str, **kwargs) -> ChunkingStrategy:
    strategies = {
        "recursive": RecursiveChunking,
        "markdown_header": MarkdownHeaderChunking,
        "parent_document": ParentDocumentChunking,
        "semantic": SemanticChunking,
        "legal_document": LegalDocumentChunking,
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")

    return strategies[strategy_name](**kwargs)
