import re
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document as LangChainDocument
from src.models.schemas import DocumentChunk, DocumentMetadata
from src.utils.logger import logger


class DocumentPreprocessor:

    def __init__(self):
        self.section_pattern = re.compile(
            r'^\s*(?:'
            r'Раздел\s+([IVX]+(?:\.\d+)?|\d+(?:\.\d+)?)|'
            r'Глава\s+([IVX]+(?:\.\d+)?|\d+(?:\.\d+)?)|'      
            r'Статья\s+(\d+(?:\.\d+)?)|'       
            r'§\s*(\d+(?:\.\d+)?)|'                           
            r'Пункт\s+(\d+(?:\.\d+)?)|'                        
            r'Подпункт\s+([а-я]\))|'                           
            r'\((\d+)\)|'                                      
            r'([а-я]+\))\s*'                                   
            r')',
            re.MULTILINE | re.IGNORECASE
        )

        self.article_pattern = re.compile(
            r'(?:статья|ст\.)\s+(\d+(?:\.\d+)?)',
            re.IGNORECASE
        )

        self.hierarchy_pattern = re.compile(
            r'^(?:Раздел|Глава|Статья|Пункт|§)\s+[\d\.IVX]+',
            re.MULTILINE | re.IGNORECASE
        )

        self.heading_pattern = re.compile(
            r'^(\d+(?:\.\d+){0,3})[\.\s]+(.+?)$',
            re.MULTILINE
        )

        self.section_number_pattern = re.compile(
            r'^[\s]*([А-Я][А-Яа-яЁё\s]*?)\s*(\d+(?:\.\d+)?|[IVX]+(?:\.\d+)?)',
            re.MULTILINE | re.IGNORECASE
        )

        logger.info("Initialized DocumentPreprocessor with enhanced patterns for legal documents")

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\n\s*\n', '\n\n', text)

        text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)

        text = re.sub(r'[ \t]+', ' ', text)

        text = text.strip()

        return text

    def extract_hierarchy(self, text: str, metadata: Dict[str, Any] = None) -> Dict[str, Optional[str]]:
        hierarchy = {
            "section": None,
            "subsection": None,
            "article": None,
            "heading": None
        }

        if metadata and metadata.get("is_heading"):
            hierarchy["heading"] = text[:100]

        first_200_chars = text[:200]

        section_match = self.section_number_pattern.search(first_200_chars)
        if section_match:
            section_type = section_match.group(1).strip()
            section_num = section_match.group(2)
            hierarchy["section"] = f"{section_type} {section_num}"

        article_match = self.article_pattern.search(text[:500])
        if article_match:
            hierarchy["article"] = f"Статья {article_match.group(1)}"

        heading_match = self.heading_pattern.match(first_200_chars)
        if heading_match and not hierarchy["section"]:
            hierarchy["section"] = heading_match.group(1)
            heading_text = heading_match.group(2).strip()
            if heading_text:
                hierarchy["heading"] = heading_text[:100]

        return hierarchy

    def process_document(self, document: LangChainDocument) -> List[DocumentChunk]:
        cleaned_text = self.clean_text(document.page_content)

        if not cleaned_text:
            return []

        hierarchy = self.extract_hierarchy(cleaned_text, document.metadata)

        hierarchy_parts = []
        for key in ["section", "article", "heading"]:
            if hierarchy.get(key):
                hierarchy_parts.append(str(hierarchy[key]))

        hierarchy_str = " > ".join(hierarchy_parts) if hierarchy_parts else ""
        # `metadata_filter.parse_section_filter()` в боте всегда фильтрует по полю `section`,
        # поэтому для вопросов вида "Статья N" важно, чтобы `section` тоже был заполнен.
        # Если раздел не распознан, но распознана статья — продублируем.
        section_value = hierarchy["section"] if hierarchy["section"] else hierarchy["article"]

        metadata = DocumentMetadata(
            source=str(document.metadata.get("source", "unknown")),
            page=document.metadata.get("page"),
            section=str(section_value) if section_value else None,
            article=str(hierarchy["article"]) if hierarchy["article"] else None,
            heading=str(hierarchy["heading"]) if hierarchy["heading"] else None,
            hierarchy=hierarchy_str,
            chunk_strategy="preprocessed"
        )

        if document.metadata.get("is_heading") is not None:
            pass

        chunk = DocumentChunk(
            content=cleaned_text,
            metadata=metadata
        )

        return [chunk]

    def preprocess_langchain_documents(self, documents: List[LangChainDocument]) -> List[LangChainDocument]:
        """
        Enrich input LangChain Documents with hierarchy metadata (section/article/heading)
        and cleaned text. This is meant to be used as an input for chunking strategies
        so metadata filtering (e.g. by `section`) works downstream.
        """
        enriched_docs: List[LangChainDocument] = []

        for i, doc in enumerate(documents):
            try:
                cleaned_text = self.clean_text(doc.page_content)
                if not cleaned_text:
                    continue

                hierarchy = self.extract_hierarchy(cleaned_text, doc.metadata)
                hierarchy_parts = []
                for key in ["section", "article", "heading"]:
                    if hierarchy.get(key):
                        hierarchy_parts.append(str(hierarchy[key]))
                hierarchy_str = " > ".join(hierarchy_parts) if hierarchy_parts else ""
                section_value = hierarchy["section"] if hierarchy["section"] else hierarchy["article"]

                # Preserve existing loader metadata (e.g. `source`, `format`) as-is,
                # only extend with extracted hierarchy fields.
                new_metadata: Dict[str, Any] = dict(doc.metadata) if doc.metadata else {}
                new_metadata.update(
                    {
                        "section": str(section_value) if section_value else None,
                        "article": str(hierarchy["article"]) if hierarchy["article"] else None,
                        "heading": str(hierarchy["heading"]) if hierarchy["heading"] else None,
                        "hierarchy": hierarchy_str,
                        "chunk_strategy": "preprocessed",
                    }
                )

                enriched_docs.append(
                    LangChainDocument(
                        page_content=cleaned_text,
                        metadata=new_metadata,
                    )
                )
            except Exception as e:
                logger.error(f"Failed to preprocess document {i}: {e}")
                continue

        logger.info(f"Preprocessed {len(enriched_docs)} LangChain documents")
        return enriched_docs

    def process_documents(self, documents: List[LangChainDocument]) -> List[DocumentChunk]:
        all_chunks = []

        for i, doc in enumerate(documents):
            try:
                chunks = self.process_document(doc)
                for j, chunk in enumerate(chunks):
                    chunk.id = f"chunk_{i}_{j}"
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process document {i}: {e}")
                continue

        logger.info(f"Processed {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks
