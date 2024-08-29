# Purpose: Handle ETL operations via API
# Inputs: ETL job configurations
# Outputs: Job status
# Dependencies: ETL components

from fastapi import APIRouter, HTTPException, BackgroundTasks
from application.core.schemas.etl_schemas import (
    ETLJobConfig,
    ETLJobStatus,
    FullETLRequest,
)
from microsoft_cve_rag.application.etl.pipelines import (
    incremental_ingestion_pipeline,
    full_ingestion_pipeline,
)
from uuid import uuid4
from datetime import datetime

router = APIRouter()


@router.post("/etl/start", response_model=ETLJobStatus)
async def start_etl_job(config: ETLJobConfig, background_tasks: BackgroundTasks):
    try:
        job_id = "job_" + str(uuid4())  # Generate a unique job ID
        background_tasks.add_task(
            incremental_ingestion_pipeline,
            config.source,
            config.destination,
            config.transform_type,
        )
        return ETLJobStatus(job_id=job_id, status="Started")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/etl/full_etl")
def start_full_etl_pipeline(request: FullETLRequest):
    """
    Initiates a full ETL pipeline process.

    This route will go to the document store, extract documents across all relevant collections, transform documents into objects suitable for insertion into vector and graph databases, generate embeddings in the case of the vector db, and load the objects into their respective repositories.

    For convenience, the caller can pass a start_date alone (which assumes today as the end_date) or both start_date and end_date to precisely set the date range of the ingestion.

    Inputs:
    - start_date (datetime): The start date for the ETL process.
    - end_date (datetime, optional): The end date for the ETL process. Defaults to today if not provided.

    Outputs:
    - dict: A response dictionary containing the status and message of the ETL process.
    """
    start_date = request.start_date
    end_date = request.end_date or datetime.now()
    print(
        f"request data: {type(start_date)} - {type(end_date)}\n{start_date} - {end_date}"
    )
    response = full_ingestion_pipeline(start_date, end_date)
    if response["code"] == 200:
        print(f"status: {response['status']} message: {response['message']}")
    else:
        print("full etl failed.")

    return response


@router.get("/etl/status/{job_id}", response_model=ETLJobStatus)
async def get_etl_job_status(job_id: str):
    # Implement ETL job status check logic here
    # For now, we'll return a dummy status
    return ETLJobStatus(job_id=job_id, status="Completed")
