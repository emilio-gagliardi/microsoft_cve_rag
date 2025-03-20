# Purpose: Entry point for the FastAPI application
# Inputs: None
# Outputs: FastAPI application instance
# Dependencies: Routes from api/routes
import os
import sys
import requests
from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging
from colorama import Fore, Style, init as colorama_init
from qdrant_client.http.models import (
    Distance,
    VectorParams,
)
from qdrant_client.async_qdrant_client import AsyncQdrantClient
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from application.services.vector_db_service import VectorDBService
from application.services.graph_db_service import (
    ensure_graph_db_constraints_exist,
    # GraphDatabase,
    # ProductService,
    # ProductBuildService,
    # KBArticleService,
    # MSRCPostService,
    # UpdatePackageService,
    # SymptomService,
    # FixService,
    # ToolService,
    # CauseService,
    # PatchManagementPostService,
    # inflate_nodes,
)

from application.api.v1.routes.vector_db import (
    router as v1_vector_router,
)
# from application.api.v1.routes.document_db import (
#     router as v1_document_router,
# )

# from application.api.v1.routes.graph_db import (
#     router as v1_graph_router,
# )

from application.api.v1.routes.etl_routes import (
    router as v1_etl_router,
)

from application.api.v1.routes.report_routes import (
    router as v1_report_router,
)
# from application.api.v1.routes.chat_routes import (
#     router as v1_chat_router,
# )

from application.app_utils import (
    # get_openai_api_key,
    get_app_config,
    get_graph_db_credentials,
    get_vector_db_credentials,
    # get_documents_db_credentials,
)

# Set variables
settings = get_app_config()

graph_db_credentials = get_graph_db_credentials()
vector_db_credentials = get_vector_db_credentials()
# Initialize colorama
colorama_init(strip=False, convert=True, autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors"""

    COLORS = {
        'DEBUG': Fore.GREEN,
        'INFO': Fore.CYAN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
        'MODULE': Fore.LIGHTBLACK_EX,
    }

    def format(self, record):
        # Check if the message is from uvicorn/FastAPI
        if not record.name.startswith(('uvicorn', 'fastapi')):
            module_name = f"{self.COLORS['MODULE']}{record.name}{Style.RESET_ALL}"
            record.name = module_name
            # Add color to levelname for our application logs
            levelname_color = self.COLORS.get(record.levelname, Fore.WHITE)
            record.levelname = f"{levelname_color}{record.levelname}{Style.RESET_ALL}"

            # Format the message
            formatted = super().format(record)

            # Color the metadata
            parts = formatted.split(" - ", 1)
            if len(parts) == 2:
                metadata, message = parts
                return f"{Fore.WHITE}{metadata}{Style.RESET_ALL} - {message}"

        return super().format(record)


def setup_logging(level=logging.INFO):
    """Configure logging settings for the application."""
    # Configure UTF-8 encoding for stdout and stderr
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

    # Remove existing handlers
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Set the logging level
    root_logger.setLevel(level)

    # Create handlers with UTF-8 encoding
    stream_handler = logging.StreamHandler(sys.stdout)  # Use stdout with UTF-8
    file_handler = logging.FileHandler("app.log", encoding='utf-8')

    # Set formatters
    colored_formatter = ColoredFormatter(
        "%(asctime)s|%(levelname)s|%(name)s| - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_formatter = logging.Formatter(  # Plain formatter for file
        "%(asctime)s|%(levelname)s|%(name)s| - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Apply formatters
    stream_handler.setFormatter(colored_formatter)
    file_handler.setFormatter(file_formatter)  # No colors in file

    # Add handlers to the root logger
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    # Adjust logging levels for specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("fastembed").setLevel(logging.WARNING)
    logging.getLogger("dotenv").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    app_logger = logging.getLogger("microsoft_cve_rag")
    app_logger.setLevel(level)


log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
setup_logging(log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Begin lifespan tasks")

    # Load configurations
    embedding_config = settings["EMBEDDING_CONFIG"]
    vectordb_config = settings["VECTORDB_CONFIG"]
    graphdb_config = settings["GRAPHDB_CONFIG"]
    collection_name = vectordb_config["tier1_collection"]

    # GRAPH DB validation
    logger.info("Validating Graph DB setup...")
    graph_db_credentials = get_graph_db_credentials()
    graph_db_uri = f"{graph_db_credentials.protocol}://{graph_db_credentials.host}:{graph_db_credentials.port}"
    graph_db_auth = (graph_db_credentials.username, graph_db_credentials.password)
    db_status = ensure_graph_db_constraints_exist(
        graph_db_uri, graph_db_auth, graphdb_config
    )
    if db_status["status"] != "success":
        logger.error(f"Database setup failed: {db_status['message']}")
        logger.error(f"Constraints status: {db_status['constraints_status']}")
        raise RuntimeError("Graph DB setup failed")
    logger.info("Graph DB validation completed successfully")

    # VECTOR DB validation
    logger.info("Validating Vector DB setup...")
    vectordb_credentials = get_vector_db_credentials()
    async_client = AsyncQdrantClient(
        host=vectordb_credentials.host,
        port=vectordb_credentials.port,
    )

    try:
        # Ensure the collection exists
        if not await async_client.collection_exists(collection_name):
            embedding_length = embedding_config.get("embedding_length", 1024)
            await async_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_length,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Collection '{collection_name}' created successfully")
        else:
            logger.info(f"Collection '{collection_name}' already exists")
    finally:
        # Ensure client is closed even if an error occurs
        await async_client.close()
        logger.info("Vector DB validation completed")

    logger.info("All lifespan tasks completed successfully")
    yield


logging.info("Building FastAPI application")
app = FastAPI(lifespan=lifespan)


# Include routers
# app.include_router(v1_chat_router, prefix="/api/v1", tags=["Chat v1"])
app.include_router(v1_etl_router, prefix="/api/v1", tags=["ETL v1"])
# app.include_router(v1_graph_router, prefix="/api/v1", tags=["Graph Service v1"])
app.include_router(v1_vector_router, prefix="/api/v1", tags=["Vector Service v1"])
# app.include_router(v1_document_router, prefix="/api/v1", tags=["Document Service v1"])
# app.include_router(v2_chat_router, prefix="/api/v2", tags=["Chat v2"])

app.include_router(v1_report_router, prefix="/api/v1", tags=["Reports v1"])

@app.get("/")
async def root():
    print("Executing root route.")
    return {"message": "Welcome to the AI-powered Knowledge Graph API"}


@app.get("/system_test")
async def system_test():
    base_url = "http://localhost:7501"  # Adjust if your server is running on a different address
    test_results = {}

    # Test document route
    document_data = {
        "text": "Sample document text",
        "metadata": {
            "title": "Sample Document",
            "description": "This is a sample document.",
        },
    }
    response = requests.post(
        f"{base_url}/api/v1/document/documents/", json=document_data
    )
    test_results["document"] = {
        "status_code": response.status_code,
        "response_type": str(type(response.json())),
        "response": response.json(),
    }

    # Add similar tests for other routes (vector, graph, sql, rag)
    # ...

    return test_results


# @app.get("/custom_encoder")
# def get_encodable_data():
#     data = {"id": ObjectId(), "date": datetime.now()}
#     return JSONResponse(
#         content=jsonable_encoder(data, custom_encoder=MongoJSONEncoder().default)
#     )
