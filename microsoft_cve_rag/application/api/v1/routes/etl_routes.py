# Purpose: Handle ETL operations via API
# Inputs: ETL job configurations
# Outputs: Job status
# Dependencies: ETL components

from fastapi import APIRouter, HTTPException, BackgroundTasks
from application.core.schemas.etl_schemas import ETLJobConfig, ETLJobStatus
from microsoft_cve_rag.application.etl.pipelines import (
    incremental_ingestion_pipeline,
    full_ingestion_pipeline,
)
from uuid import uuid4

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
def start_full_etl_pipeline():

    response = full_ingestion_pipeline()
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
