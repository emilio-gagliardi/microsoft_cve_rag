import os
from pathlib import Path
from dotenv import load_dotenv
import logging
import yaml
from application.core.schemas.environment_schemas import (
    VectorDBCredentialsSchema,
    GraphDBCredentialsSchema,
    DocumentsDBCredentialsSchema,
    SQLDBCredentialsSchema,
)
from pydantic import ValidationError
from typing import Dict

PROJECT_ROOT: Path | None = None
INNER_PROJECT_DIR: Path | None = None
APP_DIR: Path | None = None
DATA_DIR: Path | None = None
REPORTS_DIR: Path | None = None
CONF_DIR: Path | None = None
KEYS_DIR: Path | None = None

# --- Environment Configuration ---
# Mapping environment names to their corresponding .env file names
ENV_FILE_MAP: Dict[str, str] = {
    "local": ".env.local",
    "dev": ".env.dev",
    "docker": ".env.docker",
    "staging": ".env.staging",
    "prod": ".env.production",
}

_env_loaded = False


def load_app_config() -> dict:
    """
    Loads the application configuration from 'config.yaml' located
    in the same directory as this script (app_utils.py).
    """
    try:
        # Get the directory containing this script (app_utils.py)
        current_script_dir = Path(__file__).resolve().parent
        # Construct the path to config.yaml within the same directory
        config_path = current_script_dir / "config.yaml"

        logging.debug(f"Attempting to load config using resolved path: {config_path}")

        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found at expected path: {config_path}")

        with open(config_path, "r") as file:
            config_data = yaml.safe_load(file)
            if not config_data:
                raise ValueError(f"Config file at {config_path} is empty or invalid YAML.")
            logging.info(f"Application config loaded successfully from {config_path}")
            return config_data
    except FileNotFoundError as e:
        logging.error(f"FATAL: Application config file error: {e}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"FATAL: Error parsing YAML config file at {config_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"FATAL: Unexpected error loading config file {config_path}: {e}")
        raise


# Configure the logger
def setup_logger(name=None, level=logging.INFO):
    """
    Create and configure a logger object with the specified name.
    This logger can be imported and used anywhere in the application.

    :param name: Optional name for the logger (module-level granularity)
    :return: Configured logger instance
    """
    # Create a logger object with the specified name or use the root logger
    logger_name = name if name else "app_logger"
    logger = logging.getLogger(logger_name)

    # Make sure the logger doesn't propagate to the root logger
    # logger.propagate = False

    # # Check if the logger has already been configured
    # if not logger.hasHandlers():
    #     logger.setLevel(level)  # Set the logging level

    #     # Create handlers
    #     stream_handler = logging.StreamHandler()
    #     file_handler = logging.FileHandler("app.log")

    #     # Set the logging format with colors for the stream handler
    #     colored_formatter = colorlog.ColoredFormatter(
    #         "%(log_color)s%(asctime)s|%(levelname)s|/%(name)s/ - %(reset)s%(message)s",
    #         datefmt="%Y-%m-%d %H:%M:%S",
    #         log_colors={
    #             "DEBUG": "cyan",
    #             "INFO": "blue",
    #             "WARNING": "yellow",
    #             "ERROR": "red",
    #             "CRITICAL": "red,bg_white",
    #         },
    #         style="%",
    #         reset=True,
    #     )

    #     # Set a simple formatter for the file handler
    #     file_formatter = logging.Formatter(
    #         "%(asctime)s|%(levelname)s|%(name)s| - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    #     )

    #     # Apply formatters to handlers
    #     stream_handler.setFormatter(colored_formatter)
    #     file_handler.setFormatter(file_formatter)

    #     # Add handlers to the logger
    #     logger.addHandler(stream_handler)
    #     logger.addHandler(file_handler)

    #     print(f"Logger '{logger_name}' initialized with handlers.")  # Debug statement

    return logger


def initialize_environment_and_paths():
    """
    Loads application config, sets env vars, determines project root from config,
    defines key project Paths, and loads the appropriate .env file.
    """
    global _env_loaded, PROJECT_ROOT, INNER_PROJECT_DIR, APP_DIR, DATA_DIR, REPORTS_DIR, CONF_DIR, KEYS_DIR

    if _env_loaded:
        logging.debug("Environment and paths already initialized.")
        return

    logging.info("Initializing environment and paths...")

    try:
        # --- 1. Load Application Config (now using the reliable method) ---
        app_config = load_app_config()

        # --- 2. Determine Project Root from Config ---
        project_path_str = app_config.get("PROJECT_PATH")
        if not project_path_str or not isinstance(project_path_str, str):
            raise ValueError("PROJECT_PATH not found or is invalid in config.yaml")

        PROJECT_ROOT = Path(project_path_str).resolve()  # e.g., C:\Users\...\microsoft_cve_rag

        if not PROJECT_ROOT.is_dir():
            raise FileNotFoundError(f"Project root from config ('{project_path_str}') -> '{PROJECT_ROOT}' does not exist or is not a directory.")

        logging.info(f"Project Root determined from config as: {PROJECT_ROOT}")

        # --- 3. Define Core Paths Relative to PROJECT_ROOT ---
        INNER_PROJECT_DIR = PROJECT_ROOT / "microsoft_cve_rag"  # The directory containing app, conf etc.
        APP_DIR = INNER_PROJECT_DIR / "application"
        DATA_DIR = APP_DIR / "data"
        REPORTS_DIR = DATA_DIR / "reports"
        # Where is 'conf'? Assuming it's parallel to 'application'
        CONF_DIR = INNER_PROJECT_DIR / "conf"
        KEYS_DIR = CONF_DIR / "local" / "keys"

        # --- Validate crucial directories ---
        if not INNER_PROJECT_DIR.is_dir(): logging.warning(f"INNER_PROJECT_DIR may not exist: {INNER_PROJECT_DIR}")  # noqa E701
        if not APP_DIR.is_dir(): logging.warning(f"APP_DIR may not exist: {APP_DIR}")  # noqa E701
        if not DATA_DIR.is_dir(): logging.warning(f"DATA_DIR may not exist: {DATA_DIR}")  # noqa E701
        if not CONF_DIR.is_dir(): logging.warning(f"CONF_DIR may not exist: {CONF_DIR}")  # noqa E701
        # Consider creating REPORTS_DIR if needed:
        # if not REPORTS_DIR.is_dir(): os.makedirs(REPORTS_DIR, exist_ok=True)

        # --- 4. Load Config Values into Environment Variables ---
        logging.debug("Loading config values into environment variables...")
        for key, value in app_config.items():
            os.environ[key] = str(value)
            logging.debug(f"Set env var from config: {key}=***")  # Avoid logging sensitive values

        # --- 5. Load .env File (Relative to PROJECT_ROOT) ---
        environment = os.getenv("ENVIRONMENT", "local").lower()
        logging.info(f"Running in ENVIRONMENT: {environment}")

        dotenv_filename = ENV_FILE_MAP.get(environment)
        dotenv_path = PROJECT_ROOT / (dotenv_filename if dotenv_filename else ".env")

        if dotenv_filename and not dotenv_path.exists():
            logging.warning(f"Specific env file '{dotenv_filename}' not found at '{dotenv_path}'. Trying default '.env'...")
            dotenv_path = PROJECT_ROOT / ".env"

        if not dotenv_path.exists():
            logging.warning(f"No .env file found at '{dotenv_path}'. Proceeding without loading .env.")
        else:
            if load_dotenv(dotenv_path=dotenv_path, override=True):
                logging.info(f"Env vars loaded/updated from {dotenv_path}")
            else:
                logging.warning(f"dotenv.load_dotenv returned False for path: {dotenv_path}")

        _env_loaded = True
        logging.info("Environment and paths successfully initialized.")

    except (FileNotFoundError, ValueError, KeyError, TypeError) as e:
        logging.exception(f"FATAL: Failed to initialize environment or paths: {e}")
        raise RuntimeError(f"Environment/Path initialization failed: {e}") from e
    except Exception as e:
        logging.exception(f"FATAL: Unexpected error during env init: {e}")
        raise RuntimeError(f"Unexpected initialization error: {e}") from e


def get_app_config():

    return load_app_config()


def get_openai_api_key():
    initialize_environment_and_paths()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        logging.error("OPENAI_API_KEY environment variable is not set")
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return openai_api_key


def get_serper_api_key():
    initialize_environment_and_paths()
    serper_api_key = os.getenv("SERPER_API_KEY")
    if serper_api_key is None:
        logging.error("SERPER_API_KEY environment variable is not set")
        raise ValueError("SERPER_API_KEY environment variable is not set")
    return serper_api_key


def get_groq_api_key():
    initialize_environment_and_paths()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key is None:
        logging.error("GROQ_API_KEY environment variable is not set")
        raise ValueError("GROQ_API_KEY environment variable is not set")
    return groq_api_key


def get_exa_api_key():
    initialize_environment_and_paths()
    exa_api_key = os.getenv("EXA_API_KEY")
    if exa_api_key is None:
        logging.error("EXA_API_KEY environment variable is not set")
        raise ValueError("EXA_API_KEY environment variable is not set")
    return exa_api_key


def get_notion_api_key():
    initialize_environment_and_paths()
    notion_api_key = os.getenv("NOTION_API_KEY")
    if notion_api_key is None:
        logging.error("NOTION_API_KEY environment variable is not set")
        raise ValueError("NOTION_API_KEY environment variable is not set")
    return notion_api_key


def get_tavily_api_key():
    initialize_environment_and_paths()
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if tavily_api_key is None:
        logging.error("TAVILY_API_KEY environment variable is not set")
        raise ValueError("TAVILY_API_KEY environment variable is not set")
    return tavily_api_key


def get_langtrace_api_key():
    initialize_environment_and_paths()
    langtrace_api_key = os.getenv("LANGTRACE_API_KEY")
    if langtrace_api_key is None:
        logging.error("LANGTRACE_API_KEY environment variable is not set")
        raise ValueError("LANGTRACE_API_KEY environment variable is not set")
    return langtrace_api_key


def get_brave_api_key():
    initialize_environment_and_paths()
    brave_api_key = os.getenv("BRAVE_API_KEY")
    if brave_api_key is None:
        logging.error("BRAVE_API_KEY environment variable is not set")
        raise ValueError("BRAVE_API_KEY environment variable is not set")
    return brave_api_key


def get_fal_api_key():
    # prompt to image service
    initialize_environment_and_paths()
    fal_api_key = os.getenv("FAL_API_KEY")
    if fal_api_key is None:
        logging.error("FAL_API_KEY environment variable is not set")
        raise ValueError("FAL_API_KEY environment variable is not set")
    return fal_api_key


def get_e2b_api_key():
    # prompt to image service
    initialize_environment_and_paths()
    e2b_api_key = os.getenv("E2B_API_KEY")
    if e2b_api_key is None:
        logging.error("E2B_API_KEY environment variable is not set")
        raise ValueError("E2B_API_KEY environment variable is not set")
    return e2b_api_key


def get_vector_db_credentials() -> VectorDBCredentialsSchema:
    initialize_environment_and_paths()
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
    initialize_environment_and_paths()
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
    initialize_environment_and_paths()
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
    initialize_environment_and_paths()
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
