"""PDF Processing Agent — wraps a pydantic-ai Agent for chunking protocol PDFs."""
from google.oauth2 import service_account
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from src.models import DocumentChunk

_SYSTEM_PROMPT = """Role: You are an expert document parser and data analyst specialized in multilingual administrative protocols.

Task: Analyze the attached PDF document, extract the document date, and partition its content into logical, context-aware chunks.

Instructions for PDF Processing:

Metadata Extraction: Identify the Document Date (e.g., "תאריך", "Date") typically found in the header or the beginning of the protocol. This date must be included in every chunk.

OCR & Extraction: Extract the text from the PDF precisely as it appears. Maintain the original structure, line breaks (\\n), and numbering.

Language Consistency: The content of the chunks must be in the exact same language as the source text in the PDF. Do not translate, paraphrase, or adapt the text into another language.

Semantic Segmentation:

Header & Agenda: Group the metadata (meeting title, committee name, date, participants, and the initial list of topics) into the first chunk.

Topic-Based Breakdown: Identify sections where specific topics are discussed (e.g., sections labeled 2.1, 2.2, etc.). Each distinct topic, including its discussion and internal details, must be placed in its own individual chunk.

Closing & Decisions: Group the final "Decisions" or "Summary" section into a final chunk.

Data Integrity: Do not summarize, edit, or fix typos. The content must be a verbatim reflection of the PDF text.

JSON Output: Your response must be strictly a valid JSON array of objects. Do not include any conversational text.

Constraints:

If the document date is missing or cannot be identified, set the document_date value to null.

Ensure all characters are encoded correctly.

Preserve the vertical layout of the original document within the content field."""


class PDFAgent:
    """Wraps a pydantic-ai Agent that chunks PDF documents into DocumentChunks."""

    def __init__(self, sa_path: str, project_id: str, location: str, model_name: str):
        self.sa_path = sa_path
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self._agent: Agent | None = None

    def setup(self) -> None:
        """Build GCP credentials, provider, model and create the agent."""
        credentials = service_account.Credentials.from_service_account_file(
            self.sa_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        provider = GoogleProvider(
            credentials=credentials,
            project=self.project_id,
            location=self.location,
        )
        model = GoogleModel(self.model_name, provider=provider)
        self._agent = Agent(
            model=model,
            output_type=list[DocumentChunk],
            instructions=_SYSTEM_PROMPT,
        )

    async def process_pdf(self, pdf_content: bytes, file_name: str) -> list[DocumentChunk]:
        """Send a PDF to the agent and return extracted chunks."""
        result = await self._agent.run([
            f"Process this PDF document: {file_name}",
            BinaryContent(data=pdf_content, media_type="application/pdf"),
        ])
        return result.output
