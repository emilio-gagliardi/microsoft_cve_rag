import os
from dotenv import load_dotenv, find_dotenv
import logging
from core.schemas.environment_schemas import (
    VectorDBCredentialsSchema,
    GraphDBCredentialsSchema,
    DocumentsDBCredentialsSchema,
    SQLDBCredentialsSchema,
)
from pydantic import ValidationError


# Configure the logger
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler(),
        ],
    )

    # Create a logger object
    logger = logging.getLogger(__name__)
    return logger


env_loaded = False


def load_env():
    global env_loaded

    if env_loaded:
        return  # Environment variables are already loaded

    environment = os.getenv("ENVIRONMENT", "local")
    # environment = "production"
    # Map environment to the correct .env file
    env_file_map = {
        "local": ".env.local",
        "docker": ".env.docker",
    }
    dotenv_file = env_file_map.get(environment)
    if dotenv_file is None:
        raise ValueError(
            f"Environment '{environment}' is not recognized. Please set up the environment before using the app."
        )

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dotenv_path = os.path.join(project_root, dotenv_file)

    if not os.path.exists(dotenv_path):
        raise FileNotFoundError(f"The .env file at path {dotenv_path} does not exist.")

    if load_dotenv(dotenv_path):
        logging.info(f"Environment variable(s) loaded successfully from {dotenv_file}")
        env_loaded = True
    else:
        logging.warning(f"No {dotenv_file} file found or failed to load")
        env_loaded = False


def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        logging.error("OPENAI_API_KEY environment variable is not set")
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return openai_api_key


def get_serper_api_key():
    load_env()
    serper_api_key = os.getenv("SERPER_API_KEY")
    if serper_api_key is None:
        logging.error("SERPER_API_KEY environment variable is not set")
        raise ValueError("SERPER_API_KEY environment variable is not set")
    return serper_api_key


def get_groq_api_key():
    load_env()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key is None:
        logging.error("GROQ_API_KEY environment variable is not set")
        raise ValueError("GROQ_API_KEY environment variable is not set")
    return groq_api_key


def get_exa_api_key():
    load_env()
    exa_api_key = os.getenv("EXA_API_KEY")
    if exa_api_key is None:
        logging.error("EXA_API_KEY environment variable is not set")
        raise ValueError("EXA_API_KEY environment variable is not set")
    return exa_api_key


def get_notion_api_key():
    load_env()
    notion_api_key = os.getenv("NOTION_API_KEY")
    if notion_api_key is None:
        logging.error("NOTION_API_KEY environment variable is not set")
        raise ValueError("NOTION_API_KEY environment variable is not set")
    return notion_api_key


def get_tavily_api_key():
    load_env()
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if tavily_api_key is None:
        logging.error("TAVILY_API_KEY environment variable is not set")
        raise ValueError("TAVILY_API_KEY environment variable is not set")
    return tavily_api_key


def get_langtrace_api_key():
    load_env()
    langtrace_api_key = os.getenv("LANGTRACE_API_KEY")
    if langtrace_api_key is None:
        logging.error("LANGTRACE_API_KEY environment variable is not set")
        raise ValueError("LANGTRACE_API_KEY environment variable is not set")
    return langtrace_api_key


def get_brave_api_key():
    load_env()
    brave_api_key = os.getenv("BRAVE_API_KEY")
    if brave_api_key is None:
        logging.error("BRAVE_API_KEY environment variable is not set")
        raise ValueError("BRAVE_API_KEY environment variable is not set")
    return brave_api_key


def get_fal_api_key():
    # prompt to image service
    load_env()
    fal_api_key = os.getenv("FAL_API_KEY")
    if fal_api_key is None:
        logging.error("FAL_API_KEY environment variable is not set")
        raise ValueError("FAL_API_KEY environment variable is not set")
    return fal_api_key


def get_e2b_api_key():
    # prompt to image service
    load_env()
    e2b_api_key = os.getenv("E2B_API_KEY")
    if e2b_api_key is None:
        logging.error("E2B_API_KEY environment variable is not set")
        raise ValueError("E2B_API_KEY environment variable is not set")
    return e2b_api_key


def get_vector_db_credentials() -> VectorDBCredentialsSchema:
    load_env()
    try:
        credentials = VectorDBCredentialsSchema()
        return credentials
    except ValidationError as e:
        logging.error("Vector database environment credentials are not set")
        logging.error(e)
        raise ValueError(
            "VECTOR_DATABASE_USERNAME or VECTOR_DATABASE_PASSWORD environment variable is not set"
        )


def get_graph_db_credentials() -> GraphDBCredentialsSchema:
    load_env()
    try:
        credentials = GraphDBCredentialsSchema(
            username=os.getenv("GRAPH_DATABASE_USERNAME"),
            password=os.getenv("GRAPH_DATABASE_PASSWORD"),
            host=os.getenv("GRAPH_DATABASE_HOST"),
            port=int(os.getenv("GRAPH_DATABASE_PORT")),
            protocol=os.getenv("GRAPH_DATABASE_PROTOCOL"),
        )
        return credentials
    except ValidationError as e:
        logging.error("Graph database environment credentials are not set")
        logging.error(e)
        raise ValueError(
            "GRAPH_DATABASE_USERNAME or GRAPH_DATABASE_PASSWORD environment variable is not set"
        )


def get_documents_db_credentials() -> DocumentsDBCredentialsSchema:
    load_env()
    try:
        credentials = DocumentsDBCredentialsSchema(
            username=os.getenv("DOCUMENTS_DATABASE_USERNAME"),
            password=os.getenv("DOCUMENTS_DATABASE_PASSWORD"),
            db_cluster=os.getenv("DOCUMENTS_DATABASE_CLUSTER"),
            db_cluster_id=os.getenv("DOCUMENTS_DATABASE_CLUSTER_ID"),
            protocol=os.getenv("DOCUMENTS_DATABASE_PROTOCOL"),
        )
        return credentials
    except ValidationError as e:
        logging.error("Documents database environment credentials are not set")
        logging.error(e)
        raise ValueError("DOCUMENTS_DATABASE_ environment variables are not set")


def get_sql_db_credentials() -> SQLDBCredentialsSchema:
    load_env()
    try:
        credentials = SQLDBCredentialsSchema(
            username=os.getenv("SQL_DATABASE_USERNAME"),
            password=os.getenv("SQL_DATABASE_PASSWORD"),
            host=os.getenv("SQL_DATABASE_HOST"),
            port=os.getenv("SQL_DATABASE_PORT"),
            protocol=os.getenv("SQL_DATABASE_PROTOCOL"),
        )
        return credentials
    except ValidationError as e:
        logging.error("SQL database environment credentials are not set")
        logging.error(e)
        raise ValueError("SQL_DATABASE_ environment variables are not set")
