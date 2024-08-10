from application.etl.extractor import extract_from_mongo
from application.etl.transformer import transform
from application.etl.loader import load_to_vector_db, load_to_graph_db
from typing import List, Dict, Any


def run_etl_workflow(db_name: str, collection_name: str, query: Dict[str, Any]):
    # Extract
    data = extract_from_mongo(db_name, collection_name, query)

    # Transform
    transformed_data = transform(data)

    # Load
    load_to_vector_db(transformed_data)
    load_to_graph_db(transformed_data)
