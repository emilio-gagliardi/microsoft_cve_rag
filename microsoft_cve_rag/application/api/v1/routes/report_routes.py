"""Handle report generation operations via API."""
import asyncio
import os
from datetime import datetime
from typing import Dict, Any, Optional
import logging
import json
import base64
import io
from functools import lru_cache, partial
from PIL import Image, UnidentifiedImageError  # Import Pillow components
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import math
import pandas as pd
from jinja2 import (
    Environment as JinjaEnvironment,
    FileSystemLoader,
    select_autoescape
)
from application.services.document_service import DocumentService
from application.services.template_service import TemplateService
from application.reports.kb_report_generator import (
    extract_kb_articles_for_report,
    extract_cve_details_for_report
)
from application.reports.kb_report_generator import (
    process_cve_data_for_kb_report,
    transform_kb_data_for_kb_report
)
from application.reports.quarterly_deep_dive_generator import (
    ReportConfig as DeepDiveReportConfig,
    GeneratedReportAssets as DeepDiveAssets,
    QuarterlyDeepDiveReportGenerator,
    generate_synthetic_cve_data,
)
from application.services.azure_storage_blob_service import (
    AzureStorageSettings,
    AzureBlobStorageService,
    get_azure_settings
)
from application.services.sftp_service import (
    SFTPSettings,
    get_sftp_settings,
    SFTPService
)
from application.services.chat_service import LLMClient
from application.app_utils import (
    initialize_environment_and_paths,
    get_metrics_credentials,
    APP_DIR,
    DATA_DIR,
    REPORTS_DIR
)
from application.services.metrics_service import DuckDBMetricsService
from crawl4ai import AsyncWebCrawler, CrawlResult
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI Request Models ---------------------------------------


class KBReportRequest(BaseModel):
    """Request model for KB report generation."""
    start_date: datetime
    end_date: datetime


class QuarterlyReportRequest(BaseModel):
    """Request model for Quarterly report generation."""
    start_date: datetime
    end_date: datetime
    config_override: Optional[DeepDiveReportConfig] = Field(None, description="Optional overrides for default report configuration.")


QuarterlyReportRequest.model_rebuild()

# Utility Functions ---------------------------------------------


def _sanitize_for_json(obj):
    """Recursively sanitize objects for JSON serialization.

    Handles:
    - datetime objects (converts to ISO format)
    - dict objects (recursively sanitizes values)
    - list objects (recursively sanitizes elements)
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: _sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    return obj


# --- Document Service Dependency ---
async def get_document_service() -> DocumentService:
    """Provides an instance of the DocumentService."""
    # Consider using settings/env vars for db_name/collection_name
    try:
        service = DocumentService(
            db_name="report_docstore",  # Use config/env var preferably
            collection_name="docstore"  # Use config/env var preferably
        )
        return service
    except ImportError:
        logger.error("ERROR: Failed to import DocumentService.")
        raise HTTPException(status_code=500, detail="DocumentService configuration error.")
    except Exception as e:
        logger.error(f"ERROR: Failed to instantiate DocumentService: {e}")
        raise HTTPException(status_code=500, detail=f"DocumentService instantiation error: {e}")


# --- LLM Client Dependency ---
@lru_cache
def get_llm_client() -> LLMClient:
    """Provides an instance of the LLMClient."""
    logger.debug("Executing get_llm_client dependency function.")
    try:
        # Reading env var is synchronous
        mock_mode_str = os.getenv("LLM_MOCK_MODE", "False") # Get as string
        mock_mode = mock_mode_str.lower() in ('true', '1', 't', 'yes') # Robust boolean conversion
        logger.info(f"LLM Mock Mode: {mock_mode}")

        # Assuming LLMClient.__init__ is synchronous
        client_instance = LLMClient(mock_mode=mock_mode)
        logger.info(f"LLMClient instantiated: Type={type(client_instance)}")

        # Directly return the instance, not a coroutine
        return client_instance
    except Exception as e:
        logger.exception(f"Failed to instantiate LLMClient: {e}") # Use logger.exception for traceback
        raise HTTPException(status_code=500, detail="LLM Client configuration error")


# --- Jinja Environment Dependency ---
async def get_jinja_environment() -> JinjaEnvironment:
    """Provides a configured Jinja2 Environment instance."""
    template_dir = DATA_DIR / "templates"
    if not template_dir.is_dir():
        logging.error(f"Jinja templates directory not found: {template_dir}")
        raise HTTPException(status_code=500, detail="Server configuration error: Templates directory missing.")

    logging.debug(f"Initializing Jinja2 environment with loader path: {template_dir}")
    try:
        env = JinjaEnvironment(
            # Use FileSystemLoader pointing to the base 'templates' directory
            loader=FileSystemLoader(str(template_dir)),
            # Enable autoescaping for security, include .j2 for prompt templates
            autoescape=select_autoescape(['html', 'xml', 'j2'])
        )
        # Optional: Add global functions or filters if needed later
        # env.globals['now'] = datetime.utcnow
        # env.filters['format_date'] = lambda d: d.strftime('%Y-%m-%d')
        return env
    except Exception as e:
        logger.error(f"Failed to initialize Jinja2 environment: {e}")
        raise HTTPException(status_code=500, detail="Server configuration error: Jinja initialization failed.")


# --- Generator Dependency ---
def get_report_generator(
    llm_client=Depends(get_llm_client),
    jinja_env: JinjaEnvironment = Depends(get_jinja_environment)
) -> QuarterlyDeepDiveReportGenerator:

    """
    Dependency that initializes the environment (if needed) and then
    creates and returns a fully configured QuarterlyDeepDiveReportGenerator.
    """
    try:
        # 1. Ensure environment and paths are initialized *first*
        # This function should ideally be idempotent (safe to call multiple times)
        initialize_environment_and_paths()

        # 2. Check if required paths were actually set (they should be now)
        if REPORTS_DIR is None or not REPORTS_DIR.is_dir():
            logging.error(f"REPORTS_DIR is not initialized or not a directory after env init: {REPORTS_DIR}")
            raise HTTPException(status_code=500, detail="Server configuration error: Reports directory not available.")
        if DATA_DIR is None or not DATA_DIR.is_dir():
            logging.error(f"DATA_DIR is not initialized or not a directory after env init: {DATA_DIR}")
            raise HTTPException(status_code=500, detail="Server configuration error: Data directory not available.")
        if APP_DIR is None or not APP_DIR.is_dir():
            logging.error(f"APP_DIR is not initialized or not a directory after env init: {APP_DIR}")
            raise HTTPException(status_code=500, detail="Server configuration error: App directory not available.")

        # 3. Define the base templates directory using the now-available DATA_DIR
        # Make sure this path is correct according to your structure
        templates_base_dir = DATA_DIR / "templates"
        if not templates_base_dir.is_dir():
            logging.error(f"Templates base directory not found at expected location: {templates_base_dir}")
            raise HTTPException(status_code=500, detail="Server configuration error: Templates directory not found.")

        # --- Debugging: Verify the type of jinja_env received ---
        logger.info(f"Type of jinja_env received in get_report_generator: {type(jinja_env)}")
        if not isinstance(jinja_env, JinjaEnvironment):
            logger.error(f"Incorrect type received for jinja_env dependency: {type(jinja_env)}. Expected JinjaEnvironment.")
            # This indicates a deeper issue in FastAPI setup or the dependency function itself
            raise HTTPException(status_code=500, detail="Internal server error: Invalid Jinja configuration.")
        # --- End Debugging ---

        # 4. Instantiate the generator, passing the *required* absolute paths
        generator = QuarterlyDeepDiveReportGenerator(
            jinja_env=jinja_env,
            llm_client=llm_client,
            global_reports_dir=REPORTS_DIR,       # Pass the actual Path object
            global_templates_dir=templates_base_dir  # Pass the actual Path object
            # source_static_dir is handled within the generator's __init__ using APP_DIR
        )
        return generator

    except RuntimeError as e:
        # Catch errors specifically from initialize_environment_and_paths
        logging.exception("Fatal error during environment initialization within dependency.")
        raise HTTPException(status_code=500, detail=f"Server initialization failed: {e}")
    except Exception as e:
        logging.exception("Error creating report generator dependency.")
        raise HTTPException(status_code=500, detail=f"Could not create report generator: {e}")

# Metrics depenendency
def get_metrics_service() -> DuckDBMetricsService:
    """
    FastAPI dependency function to get a cached DuckDBMetricsService instance.

    Uses lru_cache to instantiate the service only once on the first call.
    NOTE: Does not handle graceful shutdown (calling service.close()).
    """
    logger.info("Attempting to create/retrieve cached DuckDBMetricsService instance...")
    try:
        metrics_creds = get_metrics_credentials()
        # Instantiate the service using the path from the credentials object
        service_instance = DuckDBMetricsService(db_path=metrics_creds.db_path)
        logger.info("DuckDBMetricsService instance created and cached.")
        return service_instance
    except ValueError as e:
        # Catch error if METRICS_DATABASE_PATH is missing
        logger.error(f"Failed to get metrics credentials for service instantiation: {e}")
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail=f"Metrics database configuration error: {e}"
        )
    except Exception as e:
        # Catch potential errors during service __init__
        logger.exception(f"Failed to instantiate DuckDBMetricsService: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to initialize metrics database service."
        )


# End Utility Functions -----------------------------------------

# Begin FastAPI Router ------------------------------------------


router = APIRouter()

# Begin Report Generation Routes --------------------------------


@router.post("/reports/kb")
async def generate_kb_report(
    request: KBReportRequest,
    sftp_settings: SFTPSettings = Depends(get_sftp_settings),
    azure_settings: AzureStorageSettings = Depends(get_azure_settings)
) -> Dict[str, Any]:
    """Generate a KB report for the specified date range.

    Args:
        request: Contains start_date and end_date for report generation

    Returns:
        Dict containing report status, file path, and processing summary

    Raises:
        HTTPException: If report generation fails
    """
    try:
        # Initialize services
        kb_service = DocumentService(
            db_name="report_docstore",
            collection_name="microsoft_kb_articles"
        )
        docstore_service = DocumentService(
            db_name="report_docstore",
            collection_name="docstore"
        )
        template_service = TemplateService()
        # Extract data
        kb_articles = extract_kb_articles_for_report(
            start_date=request.start_date,
            end_date=request.end_date,
            kb_service=kb_service
        )

        if not kb_articles:
            return {
                "status": "success",
                "data": [],
                "message": "No KB articles found"
            }

        # Get unique CVE IDs from KB articles
        cve_ids = {
            cve_id
            for article in kb_articles
            for cve_id in article.get("cve_ids", [])
        }
        kb_ids = {article.get("kb_id") for article in kb_articles}

        # Extract CVE details
        cve_details = extract_cve_details_for_report(
            list(cve_ids),
            list(kb_ids),
            docstore_service
        )

        # Transform data
        cve_lookup = process_cve_data_for_kb_report(cve_details)
        kb_data = await transform_kb_data_for_kb_report(kb_articles, cve_lookup)

        # Convert DataFrame to records and handle JSON serialization
        kb_records = kb_data.replace({float('inf'): None, float('-inf'): None, float('nan'): None}).to_dict(orient="records")
        # TODO: create new dataframe and drop columns ['cve_ids', 'cve_details', 'summary_html', 'report_new_features', 'report_bug_fixes', 'report_known_issues_workarounds']

        # Log data structure before sanitization
        # if kb_records:
        #     sample = kb_records[0]
        #     logging.info("Sample record before sanitization:")
        #     logging.info(f"Keys: {list(sample.keys())}")
        #     logging.info(f"os_classification: {sample.get('os_classification', 'NOT FOUND')}")
        #     logging.info(f"cve_details: {sample.get('cve_details', {})}")

        # Recursively sanitize all data including nested structures
        kb_records = _sanitize_for_json(kb_records)

        # Log data structure after sanitization
        if kb_records:
            sample = kb_records[0]
            logging.info("Sample record after sanitization:")
            logging.info(f"Keys: {list(sample.keys())}")
            logging.info(f"os_classification: {sample.get('os_classification', 'NOT FOUND')}")
            logging.info(f"cve_details: {sample.get('cve_details', {})}")

        # Render template and generate report
        try:
            report_file_content = template_service.render_kb_report(
                kb_data=kb_records,
                report_date=request.end_date,
                report_title="PortalFuse Weekly KB Update Report"
            )
        except Exception as e:
            report_file_content = None
            logging.error(f"Template rendering failed: {str(e)}")
            logging.error(f"Error type: {type(e).__name__}")
            if hasattr(e, '__traceback__'):
                import traceback
                logging.error("Full traceback:")
                logging.error(traceback.format_exc())

                # Log a sample record for debugging
                if kb_records:
                    sample = kb_records[0]
                    logging.error("Sample record that caused error:")
                    logging.error(f"cve_details structure: {json.dumps(sample.get('cve_details', {}), indent=2)}")
                    logging.error(f"cve_ids: {sample.get('cve_ids', [])}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to render report template: {str(e)}"
            )

        report_date_format = '%B %d, %Y'
        # start_date and end_date should be datetime objects by the timeout
        # they reach the function due to pydantic response object specification
        if not request.end_date or not isinstance(request.end_date, datetime):
            if isinstance(request.end_date, str):
                report_date = datetime.strptime(request.end_date, report_date_format)
            else:
                report_date = datetime.now()
                logging.warning("Report date not provided or invalid, using current date.")
        else:
            report_date = request.end_date
        # begin sftp workflow
        output_file_name = f"kb_report_{report_date.strftime('%Y%m%d')}.html"
        local_base_path = REPORTS_DIR / "weekly_kb_report"
        report_html_path = local_base_path / "html"
        report_thumbnail_path = local_base_path / "thumbnails"
        report_markdown_path = local_base_path / "markdown"
        sftp_remote_base_dir = "www/portalfuse.io/public_html/wp-content/uploads/kb_weekly"
        base_url = "https://portalfuse.io/wp-content/uploads/kb_weekly"
        report_html_file = report_html_path / output_file_name
        markdown_output_filename = None
        screenshot_output_filename = None
        # Write to file
        with open(report_html_file, "w", encoding="utf-8") as f:
            f.write(report_file_content)
        # SFTP uploads
        sftp_service = SFTPService(sftp_settings)
        with sftp_service:
            sftp_service.upload_file(
                local_path=report_html_file,
                remote_path=f"{sftp_remote_base_dir}/html/{output_file_name}"
            )

        browser_cfg = BrowserConfig(
            headless=False,
            viewport_width=1140,
            viewport_height=1280
        )

        run_cfg = CrawlerRunConfig(
            excluded_tags=["footer"],
            word_count_threshold=2,
            screenshot=True,
            screenshot_wait_for=1.33,
            verbose=True
        )

        async def _run_crawl_async_task():
            # This will run within the executor thread's event loop context
            async with AsyncWebCrawler(config=browser_cfg) as crawler:
                return await crawler.arun(
                    url=f"{base_url}/html/{output_file_name}",
                    config=run_cfg
                )

        def _run_crawl_sync_wrapper():
            # This function is called by the executor. It sets up and runs
            # the async task within the current thread.
            try:
                # asyncio.run() creates a *new* event loop for this thread
                # and runs the async function until it completes.
                return asyncio.run(_run_crawl_async_task())
            except Exception as e:
                # Log or capture the exception from within the thread
                logging.error(f"Error inside _run_crawl_sync_wrapper: {e}", exc_info=True)
                # Re-raise or return an error indicator if needed
                raise  # Re-raise to be caught by the main await

        loop = asyncio.get_running_loop()
        crawl_result: CrawlResult = await loop.run_in_executor(None, _run_crawl_sync_wrapper)
        azure_service = AzureBlobStorageService(azure_settings)
        container_name = "kb-weekly-report"

        if crawl_result.markdown:
            markdown_output_filename = f"kb_report_{request.end_date.strftime('%Y%m%d')}.md"
            markdown_output_path = report_markdown_path / markdown_output_filename
            with open(markdown_output_path, "w", encoding="utf-8") as f:
                f.write(crawl_result.markdown)
            with sftp_service:
                sftp_service.upload_file(
                    local_path=markdown_output_path,
                    remote_path=f"{sftp_remote_base_dir}/markdown/{markdown_output_filename}"
                )
        if crawl_result.screenshot:
            print("Screenshot captured (base64, length):", len(crawl_result.screenshot))
            CONTENT_WIDTH = 1000
            LETTER_ASPECT_RATIO_H_W = 11 / 8.5
            MAX_CROP_HEIGHT = 1500
            base64_string = crawl_result.screenshot
            image_bytes = None
            if base64_string:
                try:
                    image_bytes = base64.b64decode(base64_string)
                except base64.binascii.Error as e:
                    print(f"Error decoding base64 string: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred during decoding: {e}")
            else:
                print("Screenshot data is missing (None).")

            if image_bytes:
                try:
                    # --- Use BytesIO to treat the bytes like a file ---
                    image_stream = io.BytesIO(image_bytes)

                    # --- Open the image using Pillow ---
                    # Pillow will attempt to identify the format from the byte stream
                    img = Image.open(image_stream)

                    # --- Get the detected format ---
                    # detected_format = img.format  # e.g., 'PNG', 'JPEG'
                    # print(f"Detected image format: {detected_format}")

                    # Get the original size
                    original_width, original_height = img.size
                    # print(f"Original image size: {original_width}x{original_height}")
                    # --- Calculate Crop Dimensions ---
                    # 1. Determine Crop Width: Use content width, but don't exceed original
                    crop_width = min(CONTENT_WIDTH, original_width)

                    # 2. Determine Crop Height based on aspect ratio and width
                    # Calculate ideal height based on width and aspect ratio
                    ideal_crop_height = crop_width * LETTER_ASPECT_RATIO_H_W
                    # Apply max height constraint and ensure it doesn't exceed original height
                    crop_height = math.ceil(min(ideal_crop_height, MAX_CROP_HEIGHT, original_height))

                    # 3. Calculate Centered Horizontal Position (left, right)
                    # Find the starting point to center the crop_width within the original_width
                    left_offset = (original_width - crop_width) / 2
                    # Ensure left is an integer and not negative (if original < crop_width somehow)
                    left = max(0, int(left_offset))
                    right = left + crop_width
                    # Adjust right slightly if rounding caused it to exceed original width
                    right = min(right, original_width)
                    # Recalculate actual crop_width based on integer coords if needed (usually minor)
                    # actual_crop_width = right - left

                    # 4. Calculate Vertical Position (upper, lower) - Start from top
                    upper = 0
                    lower = upper + crop_height  # Height already constrained by original_height

                    # --- Assemble the Crop Box ---
                    crop_box = (left, upper, right, lower)
                    # print(f"Calculated crop box (L, U, R, L): {crop_box}")
                    # print(f"Effective crop dimensions: {actual_crop_width}x{crop_height}")

                    # --- Perform the Crop ---
                    cropped_img = img.crop(crop_box)

                    # --- Determine the file extension ---
                    # Create a mapping or use lower()
                    # extension_map = {
                    #     'PNG': '.png',
                    #     'JPEG': '.jpg',
                    #     'GIF': '.gif',
                    #     'BMP': '.bmp',
                    #     # Add other formats Pillow might detect if needed
                    # }
                    # file_extension = extension_map.get(detected_format, '.png')
                    file_extension = ".png"
                    # --- Save the image ---
                    screenshot_output_filename = f"thumbnail_{request.end_date.strftime('%Y_%m_%d')}{file_extension}"
                    screenshot_local_path = report_thumbnail_path / screenshot_output_filename

                    # You can save in the original format:
                    # img.save(screenshot_local_path, format=detected_format)
                    # Or convert to a specific format (e.g., always save as PNG):
                    cropped_img.save(screenshot_local_path, format='PNG')

                    print(f"Screenshot saved successfully to: {screenshot_local_path}")

                    # --- Cleanup ---
                    img.close()
                    cropped_img.close()

                except UnidentifiedImageError:
                    logger.error("Error: Pillow could not identify the image format from the decoded data.")
                except FileNotFoundError:  # From Image.open if stream is invalid? Unlikely but possible
                    logger.error("Error: Could not process image stream.")
                except Exception as e:
                    logger.error(f"An error occurred processing the image: {e}")
            with sftp_service:
                sftp_service.upload_file(
                    local_path=screenshot_local_path,
                    remote_path=f"{sftp_remote_base_dir}/thumbnails/{screenshot_output_filename}"
                )

            html_blob_name = f"html/{output_file_name}"
            with AzureBlobStorageService(azure_settings) as azure_service:
                azure_service.upload_blob(
                    container_name=container_name,
                    blob_name=html_blob_name,
                    data=report_html_file,
                    overwrite=True
                )

            thumbnail_blob_name = f"thumbnails/{screenshot_output_filename}"

            with AzureBlobStorageService(azure_settings) as azure_service:
                azure_service.upload_blob(
                    container_name=container_name,
                    blob_name=thumbnail_blob_name,
                    data=screenshot_local_path,
                    overwrite=True
                )

        return {
            "status": "success",
            "file_path": report_html_file,
            "message": f"Successfully processed {len(kb_articles)} KB articles"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate KB report: {str(e)}"
        )


@router.post("/reports/quarterly-deep-dive/generate", tags=["Reports"])
async def generate_quarterly_report_prod(
    request: QuarterlyReportRequest,
    doc_service: DocumentService = Depends(get_document_service),
    generator: QuarterlyDeepDiveReportGenerator = Depends(get_report_generator)
    # Inject upload services here if needed for step 4
    # sftp_service = Depends(get_sftp_service),
    # azure_service = Depends(get_azure_service),
) -> Dict[str, Any]:
    """
    Generates the Quarterly Deep Dive report using REAL data for the specified date range.
    """
    logger.info(f"Received request for production report: {request.start_date} to {request.end_date}")
    import traceback
    # --- 1. Fetch Real Data ---
    try:
        query = {"metadata.published": {"$gte": request.start_date, "$lte": request.end_date}}
        projection = {
            "_id": 0, "embedding": 0, "excluded_embed_metadata_keys": 0, "excluded_llm_metadata_keys": 0,
            "relationships": 0, "hash": 0, "start_char_idx": 0, "end_char_idx": 0, "text_template": 0,
            "metadata_template": 0, "metadata_seperator": 0, "class_name": 0,
        }
        bound_func = partial(doc_service.query_documents, query=query, projection=projection)
        loop = asyncio.get_running_loop()
        report_data_list = await loop.run_in_executor(None, bound_func)
        if not report_data_list:
            logger.warning("Warning: No documents found in MongoDB for the specified date range.")
            unflattened_df = pd.DataFrame()
        else:
            unflattened_df = pd.DataFrame(report_data_list)
            logger.info(f"Fetched {len(unflattened_df)} records from DocumentService.")
    except Exception as e:
        logger.error(f"Error fetching data from DocumentService: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Database fetch error: {str(e)}")

    # --- 2. Determine Report Configuration ---
    final_config = DeepDiveReportConfig()  # Start with defaults
    if request.config_override:
        update_data = request.config_override.model_dump(exclude_unset=True)
        final_config = final_config.model_copy(update=update_data)

    # --- 3. Generate Report Assets ---
    try:
        # Run synchronously. Add BackgroundTasks if generation is long.
        generated_assets: DeepDiveAssets = generator.generate_report(
            report_data=unflattened_df,
            config=final_config,
            start_date=request.start_date,
            end_date=request.end_date
        )
        logger.info(f"Report generation complete. Assets in: {generated_assets.base_directory}")
    except Exception as e:
        logger.error(f"Error during report generation: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

    # --- 4. OPTIONAL: Trigger Uploads ---
    # ... (Your logic here, using generated_assets object and injected upload services) ...

    # --- 5. Return Success Response ---
    return {
        "message": "Report generation successful.",
        "report_period_start": request.start_date.isoformat(),
        "report_period_end": request.end_date.isoformat(),
        "output_base_directory": str(generated_assets.base_directory),
        "generated_files": [str(f) for f in generated_assets.get_all_files() if f]
    }


# --- Development Route using SYNTHETIC Data ---
@router.post("/reports/quarterly-deep-dive/generate-dev-synthetic", tags=["Reports Development"])
async def generate_quarterly_report_dev_synthetic(
    request: QuarterlyReportRequest,
    generator: QuarterlyDeepDiveReportGenerator = Depends(get_report_generator),
    metrics_service: DuckDBMetricsService = Depends(get_metrics_service)
) -> Dict[str, Any]:
    """
    Generates the Quarterly Deep Dive report using SYNTHETIC data
    for template/styling development. Runs synchronously.
    """
    logger.info(f"Received request for SYNTHETIC report: {request.start_date} to {request.end_date}")
    import traceback

    # --- 1. Generate Synthetic Data ---
    try:
        # Can add num_records to request body if desired
        loop = asyncio.get_running_loop()
        synthetic_df = await loop.run_in_executor(None, generate_synthetic_cve_data, request.start_date, request.end_date, 200)
    except Exception as e:
        logger.error(f"Error generating synthetic data: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Synthetic data generation failed: {str(e)}")

    # --- 2. Determine Report Configuration ---
    final_config = DeepDiveReportConfig()  # Start with defaults
    if request.config_override:
        update_data = request.config_override.model_dump(exclude_unset=True)
        final_config = final_config.model_copy(update=update_data)

    # --- 3. Generate Report Assets ---
    try:
        # Ensure start_date and end_date are datetime.date objects
        # QuarterlyReportRequest likely defines them as such.
        report_start_date = request.start_date
        report_end_date = request.end_date

        generated_assets: DeepDiveAssets = await generator.generate_report(
            report_data=synthetic_df,  # Pass synthetic data
            config=final_config,
            start_date=report_start_date,
            end_date=report_end_date,
            metrics_service=metrics_service # Pass the metrics_service instance
        )
        logger.info(f"Synthetic report generation complete. Assets in: {generated_assets.base_directory}")
    except Exception as e:
        logger.error(f"Error during synthetic report generation: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Synthetic report generation failed: {str(e)}")

    # --- 4. Return Success Response (No Uploads for Dev) ---
    return {
        "message": "SYNTHETIC report generation successful.",
        "report_period_start": request.start_date.isoformat(),
        "report_period_end": request.end_date.isoformat(),
        "output_base_directory": str(generated_assets.base_directory),
        "generated_files": [str(f) for f in generated_assets.get_all_files() if f]
    }

# End Report Generation Routes ------------------------------------
