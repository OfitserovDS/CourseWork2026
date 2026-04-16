import re
from typing import Dict, Any, Optional, Tuple
from src.utils.logger import logger


def parse_section_filter(question: str) -> Optional[Dict[str, Any]]:
    patterns = [
        (r'[Вв]\s+(?:Статье|Ст\.)\s+(\d+(?:\.\d+)?)', "Статья"),  # В Статье 5
        (r'(?:Статья|Ст\.)\s+(\d+(?:\.\d+)?)', "Статья"),          # Статья 5
        (r'[Вв]\s+(?:Разделе|Раздел)\s+([IVX]+|\d+)', "Раздел"),   # В Разделе I
        (r'(?:Раздел|Раздела)\s+([IVX]+|\d+)', "Раздел"),          # Раздел I
        (r'[Вв]\s+(?:Главе|Глава)\s+(\d+)', "Глава"),              # В Главе 1
        (r'(?:Глава|Главы?)\s+(\d+)', "Глава"),                    # Глава 1
        (r'[Вв]\s+(?:Пункте|Пункт)\s+(\d+(?:\.\d+)?)', "Пункт"),   # В Пункте 1
        (r'(?:Пункт|Пункта)\s+(\d+(?:\.\d+)?)', "Пункт"),         # Пункт 1
        (r'§\s*(\d+(?:\.\d+)?)', "§"),                             # § 1
    ]

    for pattern, section_type in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            section_num = match.group(1)
            section_value = f"{section_type} {section_num}"
            logger.debug(f"Parsed section filter from question: {section_value}")
            return {"section": {"$contains": section_value}}

    return None


def extract_question_without_filter(question: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    clean_question = re.sub(
        r'[Вв]\s+(?:Статье|Разделе|Главе|Пункте|Ст\.)\s+[\w\d\.\s]+:\s*',
        '',
        question
    )
    clean_question = re.sub(r'(?:Статья|Раздел|Глава|Пункт)\s+[\w\d\.]+:\s*', '', clean_question)

    where_filter = parse_section_filter(question)

    return clean_question.strip() or question, where_filter
