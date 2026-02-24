"""Verification script for Firestore Vector Store integration."""
import os
from dotenv import load_dotenv
from src.settings import Settings
from src.vector_db_service import VectorDBService
from src.models import DocumentChunk

def test_integration():
    load_dotenv()
    config = Settings()
    
    # Set GOOGLE_APPLICATION_CREDENTIALS for the client to find it
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.google_application_credentials
    
    print("Testing VectorDBService setup...")
    service = VectorDBService(
        project_id=config.gcp_project_id,
        location=config.gcp_location,
        database_id=config.firestore_database,
        collection_name=config.firestore_collection,
        embedding_model_name=config.embedding_model
    )
    service.setup()
    print("Setup successful.")
    
    print("Testing chunk upload...")
    test_chunk = DocumentChunk(
        chunk_id=999,
        document_date="2026-02-23",
        section_type="Verification Test",
        content="This is a test document to verify Firestore Vector Store integration with Gemini Embeddings.",
        source_file="test_verification.pdf"
    )
    
    try:
        service.upload_chunks([test_chunk], "test_verification.pdf")
        print("Upload successful!")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    test_integration()
