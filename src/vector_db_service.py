"""Vector DB Service — creates embeddings using Vertex AI and upserts chunks into Firestore."""
from typing import Optional
from langchain_google_firestore import FirestoreVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
from google.cloud import firestore
from langchain_core.documents import Document

from src.logger import get_logger
from src.models import DocumentChunk


class VectorDBService:
    """Handles embedding generation via LangChain and Firestore vector store."""

    def __init__(
        self,
        project_id: str,
        location: str,
        database_id: str,
        collection_name: str,
        embedding_model_name: str,
    ):
        self.project_id = project_id
        self.location = location
        self.database_id = database_id
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.logger = get_logger("vector_db")
        self._vector_store: Optional[FirestoreVectorStore] = None


    def setup(self) -> None:
        """Initialize LangChain components."""
        self.logger.info(f"Connecting to Firestore (Database: {self.database_id}, Collection: {self.collection_name})...")
        
        # Initialize Vertex AI Embeddings (uses SA credentials from GOOGLE_APPLICATION_CREDENTIALS)
        embeddings = VertexAIEmbeddings(
            model_name=self.embedding_model_name,
            project=self.project_id,
            location=self.location,
        )

        
        # Initialize Firestore Client to specify database
        client = firestore.Client(project=self.project_id, database=self.database_id)
        
        # Initialize Vector Store
        self._vector_store = FirestoreVectorStore(
            collection=self.collection_name,
            embedding_service=embeddings,
            client=client
        )
        self.logger.info("Firestore Vector Store initialized")

    def upload_chunks(self, chunks: list[DocumentChunk], file_name: str) -> None:
        """Embed each chunk and upsert all documents to Firestore."""
        self.logger.info(f"Uploading {len(chunks)} chunks from {file_name} to Firestore...")
        
        documents = []
        ids = []
        for chunk in chunks:
            if not chunk.source_file:
                chunk.source_file = file_name

            doc_id = f"{file_name}_{chunk.chunk_id}".replace("/", "_").replace(".", "_")
            ids.append(doc_id)
            
            # LangChain Document
            doc = Document(
                page_content=chunk.content,
                metadata={
                    "source_file": chunk.source_file,
                    "chunk_id": chunk.chunk_id,
                    "document_date": chunk.document_date,
                    "section_type": chunk.section_type,
                }
            )
            documents.append(doc)
            
        # Add documents to vector store
        self._vector_store.add_documents(documents=documents, ids=ids)
        self.logger.info(f"Successfully uploaded {len(chunks)} chunks from {file_name} to Firestore")
