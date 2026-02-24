"""Flow Manager — orchestrates the full pipeline: GCS → PDF Agent → Vector DB."""
import os
from typing import Optional
from google.cloud import storage

from src.logger import get_logger
from src.models import PDFProcessingResult
from src.agent import PDFAgent
from src.vector_db_service import VectorDBService


class FlowManager:
    """
    Coordinates the end-to-end pipeline:
      1. List PDF files in GCS
      2. Download each PDF
      3. Chunk via PDFAgent
      4. Embed and store in Firestore via VectorDBService
    """

    def __init__(
        self,
        sa_path: str,
        project_id: str,
        location: str,
        model_name: str,
        bucket_name: str,
        folder_path: str,
        firestore_database: str,
        firestore_collection: str,
        embedding_model: str,
    ):
        self.sa_path = sa_path
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.bucket_name = bucket_name
        self.folder_path = folder_path
        self.firestore_database = firestore_database
        self.firestore_collection = firestore_collection
        self.embedding_model = embedding_model
        self.logger = get_logger("flow_manager")
        self._storage_client: Optional[storage.Client] = None
        self._pdf_agent: Optional[PDFAgent] = None
        self._vector_db: Optional[VectorDBService] = None


    def setup(self) -> None:
        """Initialize GCS client, PDF agent, and Vector DB service."""
        self.logger.info("Setting up FlowManager...")

        # GCS credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.sa_path
        self._storage_client = storage.Client()
        self.logger.info(f"Connected to GCS bucket: {self.bucket_name}")

        # PDF chunking agent
        self._pdf_agent = PDFAgent(
            sa_path=self.sa_path,
            project_id=self.project_id,
            location=self.location,
            model_name=self.model_name,
        )
        self._pdf_agent.setup()
        self.logger.info(f"PDF agent ready (model: {self.model_name})")

        # Vector database service (Firestore)
        self._vector_db = VectorDBService(
            project_id=self.project_id,
            location=self.location,
            database_id=self.firestore_database,
            collection_name=self.firestore_collection,
            embedding_model_name=self.embedding_model,
        )
        self._vector_db.setup()

        self.logger.info("Vector DB service (Firestore) ready")

    # ------------------------------------------------------------------ #
    # GCS helpers                                                          #
    # ------------------------------------------------------------------ #

    def _list_pdfs(self) -> list[str]:
        """Return all PDF blob names under the configured folder."""
        bucket = self._storage_client.bucket(self.bucket_name)
        blobs = bucket.list_blobs(prefix=self.folder_path)
        pdf_files = [b.name for b in blobs if b.name.lower().endswith(".pdf")]
        self.logger.info(f"Found {len(pdf_files)} PDFs in {self.folder_path}")
        return pdf_files

    def _download_pdf(self, blob_name: str) -> bytes:
        """Download and return raw bytes for a single PDF blob."""
        blob = self._storage_client.bucket(self.bucket_name).blob(blob_name)
        self.logger.info(f"Downloading: {blob_name}")
        return blob.download_as_bytes()

    # ------------------------------------------------------------------ #
    # Pipeline                                                             #
    # ------------------------------------------------------------------ #

    async def _process_single_pdf(self, blob_name: str) -> PDFProcessingResult:
        """Download, chunk, embed, and store one PDF. Returns its result."""
        file_name = blob_name.split("/")[-1]

        pdf_bytes = self._download_pdf(blob_name)
        self.logger.info(f"Downloaded {file_name} ({len(pdf_bytes):,} bytes)")

        self.logger.info(f"Chunking {file_name}...")
        chunks = await self._pdf_agent.process_pdf(pdf_bytes, file_name)
        result = PDFProcessingResult(file_name=file_name, chunks=chunks)
        self.logger.info(f"Extracted {result.total_chunks} chunks from {file_name}")

        self.logger.info(f"Uploading {result.total_chunks} chunks to Firestore...")
        self._vector_db.upload_chunks(result.chunks, file_name)

        return result

    async def process_all_pdfs(self) -> list[PDFProcessingResult]:
        """Run the full pipeline over every PDF in the GCS folder."""
        pdf_files = self._list_pdfs()
        if not pdf_files:
            self.logger.warning("No PDFs found — nothing to process")
            return []

        results: list[PDFProcessingResult] = []
        for i, blob_name in enumerate(pdf_files, 1):
            self.logger.info(f"[{i}/{len(pdf_files)}] {blob_name}")
            result = await self._process_single_pdf(blob_name)
            results.append(result)

        self.logger.info(f"Pipeline complete: {len(results)}/{len(pdf_files)} files processed")
        return results
