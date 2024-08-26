from application.etl.extractor import extract_from_mongo
from application.etl.transformer import transform
from application.etl.loader import load_to_vector_db, load_to_graph_db
from typing import List, Dict, Any


def incremental_ingestion_pipeline(
    db_name: str, collection_name: str, query: Dict[str, Any]
):
    response = {
        "message": "incremental ingestion pipeline complete.",
        "status": "success",
        "code": 200,
    }
    # Extract
    data = extract_from_mongo(db_name, collection_name, query)

    # Transform
    transformed_data = transform(data)

    # Load
    load_to_vector_db(transformed_data)
    load_to_graph_db(transformed_data)
    return response


def full_ingestion_pipeline():
    response = {
        "message": "full ingestion pipeline complete.",
        "status": "success",
        "code": 200,
    }
    return response
