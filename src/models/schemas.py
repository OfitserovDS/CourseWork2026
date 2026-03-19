from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class DocumentMetadata(BaseModel):
    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    article: Optional[str] = None
    heading: Optional[str] = None
    hierarchy: Optional[str] = Field(default="")
    chunk_strategy: Optional[str] = None
    extra_info: Optional[Dict[str, Any]] = None

    class Config:
        frozen = False
        arbitrary_types_allowed = True


class DocumentChunk(BaseModel):
    content: str
    metadata: DocumentMetadata
    id: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
