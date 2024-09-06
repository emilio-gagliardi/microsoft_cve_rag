# Purpose: Load transformed data into databases
# Inputs: Transformed data
# Outputs: Loading status
# Dependencies: VectorDBService, GraphDBService

from typing import Any, Dict, List
from neo4j import GraphDatabase
from neomodel import config as NeomodelConfig  # required by AsyncDatabase
from neomodel.async_.core import AsyncDatabase  # required for db CRUD
from datetime import datetime
import pandas as pd
from application.app_utils import (
    get_app_config,
    get_vector_db_credentials,
    get_graph_db_credentials,
)
from application.services.vector_db_service import VectorDBService
from application.services.graph_db_service import (
    GraphDatabaseManager,
    ProductService,
    ProductBuildService,
    KBArticleService,
    UpdatePackageService,
    MSRCPostService,
    PatchManagementPostService,
    inflate_nodes,
)


vector_db_credentials = get_vector_db_credentials()
settings = get_app_config()
embedding_config = settings["EMBEDDING_CONFIG"]
vectordb_config = settings["VECTORDB_CONFIG"]
graphdb_config = settings["GRAPHDB_CONFIG"]
# from application.services.graph_db_service import GraphDBService
db = AsyncDatabase()
db_manager = GraphDatabaseManager(db)


def get_graph_db_uri():
    credentials = get_graph_db_credentials()
    host = credentials.host
    port = credentials.port
    protocol = credentials.protocol
    username = credentials.username
    password = credentials.password
    uri = f"{protocol}://{username}:{password}@{host}:{port}"
    return uri


def set_graph_db_uri():
    NeomodelConfig.DATABASE_URL = get_graph_db_uri()


def get_neo4j_driver():
    # TODO: move to graph_db_service
    credentials = get_graph_db_credentials()
    username = credentials["GRAPH_DATABASE_USERNAME"]
    password = credentials["GRAPH_DATABASE_PASSWORD"]
    set_graph_db_uri()
    return GraphDatabase.driver(get_graph_db_uri(), auth=(username, password))


async def lookup_thread_emails(thread_id):
    patch_posts_service = PatchManagementPostService(db_manager)
    existing_nodes = await patch_posts_service.get_by_thread_id(thread_id)
    return [
        {"node_id": node.node_id, "receivedDateTime": node.receivedDateTime}
        for node in existing_nodes
    ]


async def determine_email_sequence(all_emails):
    all_emails.sort(
        key=lambda x: datetime.fromisoformat(
            x["receivedDateTime"].replace("Z", "+00:00")
        )
    )

    updates = []
    for i, email in enumerate(all_emails):
        updates.append(
            {
                "node_id": email["node_id"],
                "previous_id": all_emails[i - 1]["node_id"] if i > 0 else None,
                "next_id": (
                    all_emails[i + 1]["node_id"] if i < len(all_emails) - 1 else None
                ),
            }
        )

    return updates


async def update_email_sequence(updates, patch_posts_service):
    print("update patch sequence")
    for update in updates:
        await patch_posts_service.update_sequence(
            update["node_id"], update["previous_id"], update["next_id"]
        )


# Usage:
# update_email_sequence(sequence_updates)


# load products into graph db
async def load_products_graph_db(products: pd.DataFrame):
    response = {
        "code": 404,
        "status": "Complete",
        "message": "No products to insert",
        "insert_ids": [],
        "nodes": None,
    }
    product_service = ProductService(db_manager)
    if not products.empty:
        print(f"received products: {products.shape[0]}")
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()
        # bulk_create() -> List[<class 'application.core.models.graph_db_models.Product'>]
        new_products_list = await product_service.bulk_create(
            products.to_dict(orient="records")
        )

        if new_products_list:
            node_ids = [node.node_id for node in new_products_list]
            response["code"] = 200
            response["message"] = "Products inserted"
            response["insert_ids"] = node_ids
            response["nodes"] = new_products_list
    else:
        print("Received products: 0")

    print(f"Total Products Loaded: {len(new_products_list)}")
    return response


# load product_builds into graph db
async def load_product_builds_graph_db(product_builds: pd.DataFrame):
    response = {
        "code": 404,
        "status": "Complete",
        "message": "No product builds to insert",
        "insert_ids": [],
        "nodes": None,
    }
    product_build_service = ProductBuildService(db_manager)
    if not product_builds.empty:
        print(f"received product_builds: {product_builds.shape[0]}")
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()
        # bulk_create() -> List[<class 'application.core.models.graph_db_models.Product'>]
        new_product_builds_list = await product_build_service.bulk_create(
            product_builds.to_dict(orient="records")
        )

        if new_product_builds_list:
            node_ids = [node.node_id for node in new_product_builds_list]
            response["code"] = 200
            response["message"] = "Product Builds inserted"
            response["insert_ids"] = node_ids
            response["nodes"] = new_product_builds_list
        else:
            print("No Nodes created or returned.")
    else:
        print("Received product builds: 0")

    print(f"Total Product Builds Loaded: {len(new_product_builds_list)}")
    return response


# load kb articles into graph db
async def load_kbs_graph_db(kb_articles: pd.DataFrame):
    response = {
        "code": 404,
        "status": "Complete",
        "message": "No KB Articles to insert",
        "insert_ids": [],
        "nodes": [],
    }
    if not kb_articles.empty:
        kb_article_service = KBArticleService(db_manager)
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()
        new_kb_article_nodes_list = await kb_article_service.bulk_create(
            kb_articles.to_dict(orient="records")
        )

        if new_kb_article_nodes_list:
            node_ids = [node.node_id for node in new_kb_article_nodes_list]
            response["code"] = 200
            response["message"] = "Product Builds inserted"
            response["insert_ids"] = node_ids
            response["nodes"] = new_kb_article_nodes_list
            print(f"Total KB Articles Returned: {len(new_kb_article_nodes_list)}.")
        else:
            print(
                "WARNING. KB Articles -> No Nodes created or returned. bulk_create() is intended to return nodes that it creates and finds."
            )
    else:
        print("kb articles dataframe is empty.")

    return response


# load update_packages into graph db
async def load_update_packages_graph_db(update_packages: pd.DataFrame):
    response = {
        "code": 404,
        "status": "Complete",
        "message": "No update packages to insert",
        "insert_ids": [],
        "nodes": None,
    }
    update_packages_service = UpdatePackageService(db_manager)
    if not update_packages.empty:
        print(f"received update_packages: {update_packages.shape[0]}")
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()
        # bulk_create() -> List[<class 'application.core.models.graph_db_models.Product'>]
        new_update_packages_list = await update_packages_service.bulk_create(
            update_packages.to_dict(orient="records")
        )

        if new_update_packages_list:
            node_ids = [node.node_id for node in new_update_packages_list]
            response["code"] = 200
            response["message"] = "Update Packages inserted"
            response["insert_ids"] = node_ids
            response["nodes"] = new_update_packages_list
    else:
        print("Received Update Packages: 0")

    print(f"Total Update Packages Returned: {len(new_update_packages_list)}")
    return response


# load msrc posts into graph db
async def load_msrc_posts_graph_db(msrc_posts: pd.DataFrame):
    response = {
        "code": 404,
        "status": "Complete",
        "message": "No msrc posts to insert",
        "insert_ids": [],
        "nodes": None,
    }
    msrc_posts_service = MSRCPostService(db_manager)
    if not msrc_posts.empty:
        print(f"received msrc_posts: {msrc_posts.shape[0]}")
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()
        new_msrc_posts_list = await msrc_posts_service.bulk_create(
            msrc_posts.to_dict(orient="records")
        )

        if new_msrc_posts_list:
            node_ids = [node.node_id for node in new_msrc_posts_list]
            response["code"] = 200
            response["message"] = "MSRC Posts inserted"
            response["insert_ids"] = node_ids
            response["nodes"] = new_msrc_posts_list
    else:
        print(f"received msrc_posts: 0")

    print(f"Total MSRC Posts Returned: {len(new_msrc_posts_list)}")
    return response


# load patch posts into graph db
async def load_patch_posts_graph_db(patch_posts: pd.DataFrame):
    print("loading patch posts")
    response = {
        "code": 404,
        "status": "Complete",
        "message": "No patch posts to insert",
        "insert_ids": [],
        "nodes": None,
    }
    patch_posts_service = PatchManagementPostService(db_manager)
    if not patch_posts.empty:
        print(f"received patch_posts: {patch_posts.shape[0]}")
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()

        # Group patch posts by thread_id
        grouped_posts = patch_posts.groupby("thread_id")

        all_nodes = []
        for thread_id, group in grouped_posts:
            # Look up existing emails for this thread
            existing_emails = await lookup_thread_emails(thread_id)

            # Create new nodes
            batch_nodes = await patch_posts_service.bulk_create(
                group.to_dict(orient="records")
            )

            # Prepare new emails data for sequencing
            all_emails = existing_emails + [
                {"node_id": node.node_id, "receivedDateTime": node.receivedDateTime}
                for node in batch_nodes
            ]
            print("starting to determine email sequence")
            # Determine the correct sequence
            updates = await determine_email_sequence(all_emails)

            print("starting to update the sequence")
            # Update the sequence in the database
            await update_email_sequence(updates, patch_posts_service)

            all_nodes.extend(batch_nodes)

        if all_nodes:
            node_ids = [node.node_id for node in all_nodes]
            response["code"] = 200
            response["message"] = "Patch Management Posts inserted and updated"
            response["insert_ids"] = node_ids
            response["nodes"] = all_nodes
    else:
        print("received patch_posts: 0")
    print(f"Total Patch Posts Returned: {len(all_nodes)}")
    return response


# ===============================================
# begin vector loading
# ===============================================

# VectorDBService loads credentials internally
vector_db_service = VectorDBService(
    embedding_config, vectordb_config, vectordb_config["tier1_collection"]
)


def load_to_vector_db(data: List[Dict[str, Any]]) -> bool:
    for record in data:
        if "vector" in record:
            vector_db_service.create_vector(record["vector"])
    return True


# load products into graph db


# load product_builds into graph db


# load kb articles into graph db


# load update_packages into graph db


# load msrc posts into graph db


# load patch posts into graph db
