# Purpose: Entry point for the FastAPI application
# Inputs: None
# Outputs: FastAPI application instance
# Dependencies: Routes from api/routes
import os
from fastapi import FastAPI
from application.api.v1.routes import (
    vector_db as v1_vector_router,
    graph_db as v1_graph_router,
    chat as v1_chat_router,
    etl as v1_etl_router,
)

# from application.api.v2.routes import (
#     vector_db as v2_vector_router,
#     graph_db as v2_graph_router,
#     chat as v2_chat_router,
#     etl as v2_etl_router,
# )
from app_utils import (
    get_openai_api_key,
    get_vector_db_credentials,
    get_graph_db_credentials,
    get_documents_db_credentials,
    get_sql_db_credentials,
    setup_logger,
)
from application.config import DEFAULT_EMBEDDING_CONFIG

# Set variables
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"
os.environ["OPENAI_API_KEY"] = get_openai_api_key()
embedding_config = DEFAULT_EMBEDDING_CONFIG

logger = setup_logger()

app = FastAPI()

# Include routers
app.include_router(v1_chat_router, prefix="/api/v1", tags=["Chat v1"])
app.include_router(v1_etl_router, prefix="/api/v1", tags=["ETL v1"])
app.include_router(v1_graph_router, prefix="/api/v1", tags=["Graph Service v1"])
app.include_router(v1_vector_router, prefix="/api/v1", tags=["Vector Service v1"])
# app.include_router(v2_chat_router, prefix="/api/v2", tags=["Chat v2"])


# Test function to print credentials
def print_credentials():
    vector_credentials = get_vector_db_credentials()
    logger.info(f"The vector db credentials:\n{vector_credentials}")

    graph_credentials = get_graph_db_credentials()
    logger.info(f"The graph db credentials:\n{graph_credentials}")

    documents_credentials = get_documents_db_credentials()
    logger.info(f"The documents db credentials:\n{documents_credentials}")

    sql_credentials = get_sql_db_credentials()
    logger.info(f"The sql db credentials:\n{sql_credentials}")


@app.get("/")
async def root():
    print_credentials()
    return {"message": "Welcome to the AI-powered Knowledge Graph API"}
