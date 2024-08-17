# Purpose: Entry point for the FastAPI application
# Inputs: None
# Outputs: FastAPI application instance
# Dependencies: Routes from api/routes
import os
import sys
import requests

# from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from application.services.vector_db_service import VectorDBService
from application.api.v1.routes.vector_db import (
    router as v1_vector_router,
)
from application.api.v1.routes.document_db import (
    router as v1_document_router,
)

# from application.api.v1.routes.graph_db import (
#     router as v1_graph_router,
# )
# from application.api.v1.routes.etl_routes import (
#     router as v1_etl_router,
# )
# from application.api.v1.routes.chat_routes import (
#     router as v1_chat_router,
# )


from application.app_utils import (
    get_openai_api_key,
    setup_logger,
    get_app_config,
)


# Set variables
settings = get_app_config()
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"
os.environ["OPENAI_API_KEY"] = get_openai_api_key()

logger = setup_logger(__name__)

logger.info("Loaded app configuration")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize a temporary VectorDBService instance for setup
    print("begin lifespan tasks")
    collection = "tier1_collection"
    embedding_config = settings["EMBEDDING_CONFIG"]
    vectordb_config = settings["VECTORDB_CONFIG"]

    temp_vector_db_service = VectorDBService(
        collection=collection,
        embedding_config=embedding_config,
        vectordb_config=vectordb_config,
    )

    # Ensure the collection exists
    await temp_vector_db_service.ensure_collection_exists_async()

    # Clean up the temporary instance
    del temp_vector_db_service

    yield


logger.info("Building FastAPI application")
app = FastAPI(lifespan=lifespan)


# Include routers
# app.include_router(v1_chat_router, prefix="/api/v1", tags=["Chat v1"])
# app.include_router(v1_etl_router, prefix="/api/v1", tags=["ETL v1"])
# app.include_router(v1_graph_router, prefix="/api/v1", tags=["Graph Service v1"])
app.include_router(v1_vector_router, prefix="/api/v1", tags=["Vector Service v1"])
# app.include_router(v1_document_router, prefix="/api/v1", tags=["Document Service v1"])
# app.include_router(v2_chat_router, prefix="/api/v2", tags=["Chat v2"])


@app.get("/")
async def root():

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
