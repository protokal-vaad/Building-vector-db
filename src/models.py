"""Pydantic models for all data structures in the application."""
from typing import Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A single semantic chunk extracted from a protocol PDF."""

    chunk_id: int = Field(description="Sequential identifier for the chunk")
    document_date: Optional[str] = Field(
        default=None,
        description="Date extracted from the document header"
    )
    section_type: str = Field(
        description="Section type: 'Header and Agenda', 'Topic Discussion', or 'Closing and Decisions'"
    )
    content: str = Field(description="Verbatim text content in the original language")
    source_file: Optional[str] = Field(
        default=None,
        description="Name of the source PDF file"
    )


class PDFProcessingResult(BaseModel):
    """Outcome of processing a single PDF file."""

    file_name: str = Field(description="Source PDF filename")
    chunks: list[DocumentChunk] = Field(
        default_factory=list,
        description="Extracted document chunks"
    )

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)