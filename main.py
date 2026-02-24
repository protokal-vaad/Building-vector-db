"""Entry point — initializes all services and runs the PDF processing pipeline."""
import asyncio

from src.settings import Settings
from src.logger import AppLogger
from src.flow_manager import FlowManager


async def main():
    # Load all config from .env
    config = Settings()

    # Initialize app-wide logger
    logger = AppLogger(level=config.log_level)
    logger.info("Starting Vector Database Builder")

    # Build and set up the pipeline (GCS → PDF Agent → Firestore)
    flow_manager = FlowManager(
        sa_path=config.google_application_credentials,
        project_id=config.gcp_project_id,
        location=config.gcp_location,
        model_name=config.model_name,
        bucket_name=config.bucket_name,
        folder_path=config.folder_path,
        firestore_database=config.firestore_database,
        firestore_collection=config.firestore_collection,
        embedding_model=config.embedding_model,
    )

    flow_manager.setup()

    # Run pipeline
    results = await flow_manager.process_all_pdfs()

    # Log summary
    logger.info("=" * 50)
    for result in results:
        logger.info(result.model_dump_json(indent=2))
    logger.info(f"Done — {len(results)} file(s) processed")


if __name__ == "__main__":
    asyncio.run(main())