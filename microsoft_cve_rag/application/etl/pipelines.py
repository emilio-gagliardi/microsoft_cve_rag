from application.etl import extractor
from application.etl import transformer
from application.etl import loader

from application.services.graph_db_service import build_relationships, SERVICES_MAPPING

from typing import Dict, Any
from datetime import datetime
import time
import asyncio


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
    #
    return response


async def full_ingestion_pipeline(start_date: datetime, end_date: datetime = None):
    response = {
        "message": "full ingestion pipeline complete.",
        "status": "in progress",
        "code": 200,
    }
    print("Begin data extraction ===============================================")
    start_time = time.time()

    product_docs = await asyncio.to_thread(extractor.extract_products, 2)
    product_build_docs = await asyncio.to_thread(
        extractor.extract_product_builds, start_date, end_date, 20
    )
    product_build_docs_by_build = sorted(product_build_docs, key=sort_by_build_number)
    # kb_article_docs_windows, kb_article_docs_edge = await asyncio.to_thread(
    #     extractor.extract_kb_articles, start_date, end_date, None
    # )
    # update_package_docs = extractor.extract_update_packages(start_date, end_date, None)
    # msrc_docs = extractor.extract_msrc_posts(start_date, end_date, None)
    # patch_docs = extractor.extract_patch_posts(start_date, end_date, None)

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Time taken to extract all data: {int(minutes)} min : {int(seconds)} sec")

    print("Begin data transformation ==========================================")
    start_time = time.time()
    products_df = await asyncio.to_thread(transformer.transform_products, product_docs)
    product_builds_df = await asyncio.to_thread(
        transformer.transform_product_builds, product_build_docs_by_build
    )
    # kb_articles_combined_df = await asyncio.to_thread(
    #     transformer.transform_kb_articles, kb_article_docs_windows, kb_article_docs_edge
    # )
    # update_packages_df = transformer.transform_update_packages(update_package_docs)
    # msrc_posts_df = transformer.transform_msrc_posts(msrc_docs)
    # patch_posts_df = transformer.transform_patch_posts(patch_docs)

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Time taken to transform all data: {int(minutes)} min : {int(seconds)} sec")

    print("Begin data loading =================================================")
    start_time = time.time()

    product_load_response = await loader.load_products_graph_db(products_df)
    product_nodes = product_load_response["nodes"]
    # product_node_ids = product_load_response["insert_ids"]
    # print(product_node_ids)

    product_build_load_response = await loader.load_product_builds_graph_db(
        product_builds_df
    )
    product_build_nodes = product_build_load_response["nodes"]
    # product_build_node_ids = product_build_load_response["insert_ids"]

    # # KBArticle nodes <class 'application.core.models.graph_db_models.KBArticle'>
    # kb_article_load_response = await loader.load_kbs_graph_db(kb_articles_combined_df)
    # kb_nodes = kb_article_load_response["nodes"]
    # kb_node_ids = kb_article_load_response["insert_ids"]

    # update_packages_load_response = await loader.load_update_packages_graph_db(
    #     update_packages_df
    # )

    # UpdatePackage nodes <class 'application.core.models.graph_db_models.UpdatePackage'>
    # update_package_nodes = update_packages_load_response["nodes"]
    # update_package_node_ids = update_packages_load_response["insert_ids"]

    # msrc_posts_load_response = await loader.load_msrc_posts_graph_db(msrc_posts_df)

    # MSRCPost nodes <class 'application.core.models.graph_db_models.MSRCPost'>
    # msrc_post_nodes = msrc_posts_load_response["nodes"]
    # msrc_post_node_ids = msrc_posts_load_response["insert_ids"]

    # patch_posts_load_response = await loader.load_patch_posts_graph_db(patch_posts_df)

    # PatchManagementPost nodes <class 'application.core.models.graph_db_models.PatchManagementPost'>
    # patch_post_nodes = patch_posts_load_response["nodes"]
    # patch_post_node_ids = patch_posts_load_response["insert_ids"]
    nodes_dict = {
        "Product": product_nodes,
        "ProductBuild": product_build_nodes,
        # "UpdatePackage": update_package_nodes,
        # "KBArticle": kb_nodes,
        # "MSRCPost": msrc_post_nodes,
        # "PatchManagementPost": patch_post_nodes,
    }
    # services_mapping = SERVICES_MAPPING
    await build_relationships(nodes_dict)

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Time taken to load all data: {int(minutes)} min : {int(seconds)} sec")
    print("Full Ingestion complete ============================================")
    response = {
        "message": "full ingestion pipeline complete.",
        "status": "Complete",
        "code": 200,
    }
    return response


# Custom sorting function
def sort_by_build_number(item):
    return tuple(item["build_number"])
