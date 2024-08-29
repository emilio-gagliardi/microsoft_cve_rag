from application.etl import extractor
from application.etl.transformer import transform
from application.etl.loader import load_to_vector_db, load_to_graph_db
from typing import List, Dict, Any
from datetime import datetime
import time


def incremental_ingestion_pipeline(
    db_name: str, collection_name: str, query: Dict[str, Any]
):
    response = {
        "message": "incremental ingestion pipeline complete.",
        "status": "success",
        "code": 200,
    }
    # Extract
    # data = extract_from_mongo(db_name, collection_name, query)

    # Transform
    # transformed_data = transform(data)

    # Load
    # load_to_vector_db(transformed_data)
    # load_to_graph_db(transformed_data)
    return response


def full_ingestion_pipeline(start_date: datetime, end_date: datetime = None):
    response = {
        "message": "full ingestion pipeline complete.",
        "status": "in progress",
        "code": 200,
    }
    print("Begin data extraction.")
    start_time = time.time()
    product_docs = extractor.extract_products()
    # print(f"product docs received: {len(product_docs)}")

    product_build_docs = extractor.extract_product_builds(start_date, end_date, None)
    # print(f"product_build_docs received: {len(product_build_docs)}")
    # for pb in product_build_docs:
    #     print(f"cve: {pb['cve_id']} build: {pb['build_number']}")

    # print("sorted next\n")
    product_build_docs_by_build = sorted(product_build_docs, key=sort_by_build_number)
    # for item in product_build_docs_by_build:
    #     print(f"cve: {item['cve_id']} build: {item['build_number']}")

    kb_article_docs_windows, kb_article_docs_edge = extractor.extract_kb_articles(
        start_date, end_date, None
    )

    update_package_docs = extractor.extract_update_packages(start_date, end_date, None)

    msrc_posts = extractor.extract_msrc_posts(start_date, end_date, None)
    patch_posts = extractor.extract_patch_posts(start_date, end_date, None)

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Time taken to extract all data: {int(minutes)} min : {int(seconds)} sec")
    print("Begin data transformation.")
    start_time = time.time()

    # product_docs - perform ops to prepare to insert into vector db and graph db
    print(f"Product from mongo:\n{product_docs[0]}")
    # product_build_docs - perform ops to prepare to insert into vector db and graph db

    # kb_article_docs_windows & kb_article_docs_edge - perform ops to prepare to insert into vector db and graph db

    # update_package_docs - perform ops to prepare to insert into vector db and graph db

    # msrc_posts - perform ops to prepare to insert into vector db and graph db

    # patch_posts - perform ops to prepare to insert into vector db and graph db

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Time taken to transform all data: {int(minutes)} min : {int(seconds)} sec")
    return response


# Custom sorting function
def sort_by_build_number(item):
    return tuple(item["build_number"])
