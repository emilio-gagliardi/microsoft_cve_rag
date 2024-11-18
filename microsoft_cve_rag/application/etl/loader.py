# Purpose: Load transformed data into databases
# Inputs: Transformed data
# Outputs: Loading status
# Dependencies: VectorDBService, GraphDBService
import os
import logging
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
    setup_logger,
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
    SymptomService,
    CauseService,
    FixService,
    ToolService,
    TechnologyService,
    inflate_nodes,
)
import json

# Get the logging level from the environment variable, default to INFO
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
# Convert the string to a logging level
log_level = getattr(logging, log_level, logging.INFO)
logger = setup_logger(__name__, level=log_level)

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


def custom_json_serializer(obj):
    """Custom JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


async def lookup_thread_emails(thread_id):
    patch_posts_service = PatchManagementPostService(db_manager)
    existing_nodes = await patch_posts_service.get_by_thread_id(thread_id)
    return [
        {"node_id": node.node_id, "receivedDateTime": node.receivedDateTime}
        for node in existing_nodes
    ]


async def determine_email_sequence(all_emails):
    # print(f"determine_email_sequence:\n")
    try:
        all_emails.sort(
            key=lambda x: (
                x["receivedDateTime"]
                if isinstance(x["receivedDateTime"], pd.Timestamp)
                else datetime.fromisoformat(
                    x["receivedDateTime"].replace("Z", "+00:00")
                )
            )
        )
    except Exception as e:
        # print(f"determine_email_sequence:{e}\n{all_emails}")
        return []
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
    # print("update patch sequence")
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
    # The ProductService ensures that the key-value pairs in the dictionary are used to create an instance of AsyncStructuredNode for Product
    product_service = ProductService(db_manager)

    if not products.empty:
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()

        # Bulk create products and return newly inserted nodes
        new_products_list = await product_service.bulk_create(
            products.to_dict(orient="records")
        )

        if new_products_list:
            node_ids = [node.node_id for node in new_products_list]
            response["code"] = 200
            response["message"] = "Products inserted"
            response["insert_ids"] = node_ids
            response["nodes"] = new_products_list

            # Construct the Cypher query to update the published field for the new nodes
            # update_published_query = """
            # MATCH (n:Product)
            # WHERE n.node_id IN $node_ids
            # SET n.published = datetime({epochSeconds: toInteger(n.published)})
            # RETURN n.node_id, n.published
            # """

            # Execute the Cypher query using the product_service
            # try:
            #     await product_service.cypher(
            #         update_published_query, {"node_ids": node_ids}
            #     )
            #     print("Updated published field for new Product nodes.")
            # except Exception as e:
            #     print(f"Failed to update published field: {e}")
        else:
            print("No new products were inserted.")
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

    # Instantiate the ProductBuildService
    product_build_service = ProductBuildService(db_manager)

    # Check if the DataFrame is not empty
    if not product_builds.empty:
        print(f"received product_builds: {product_builds.shape[0]}")

        # Set the database URL
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()

        # Bulk create product builds in the database
        new_product_builds_list = await product_build_service.bulk_create(
            product_builds.to_dict(orient="records")
        )

        # If there are newly inserted product builds, update the response and run the Cypher query
        if new_product_builds_list:
            node_ids = [node.node_id for node in new_product_builds_list]
            response["code"] = 200
            response["message"] = "Product Builds inserted"
            response["insert_ids"] = node_ids
            response["nodes"] = new_product_builds_list

            # Construct the Cypher query to update the published field for the new nodes
            # update_published_query = """
            # MATCH (n:ProductBuild)
            # WHERE n.node_id IN $node_ids
            # SET n.published = datetime({epochSeconds: toInteger(n.published)})
            # RETURN n.node_id, n.published
            # """

            # Execute the Cypher query to update the published field for the new nodes
            # try:
            #     await product_build_service.cypher(
            #         update_published_query, {"node_ids": node_ids}
            #     )
            #     print("Updated published field for new ProductBuild nodes.")
            # except Exception as e:
            #     print(f"Failed to update published field: {e}")
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

    # Check if the DataFrame is not empty
    if not kb_articles.empty:
        kb_article_service = KBArticleService(db_manager)

        # Set the Neo4j database URL
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()

        # Bulk create KB Articles in the graph database
        new_kb_article_nodes_list = await kb_article_service.bulk_create(
            kb_articles.to_dict(orient="records")
        )

        # If new KB articles were created, update the response and run the Cypher query
        if new_kb_article_nodes_list:
            node_ids = [node.node_id for node in new_kb_article_nodes_list]
            response["code"] = 200
            response["message"] = "KB Articles inserted"
            response["insert_ids"] = node_ids
            response["nodes"] = new_kb_article_nodes_list
            print(f"Total KB Articles Returned: {len(new_kb_article_nodes_list)}.")

            # Construct the Cypher query to update the published field for the new nodes
            # update_published_query = """
            # MATCH (n:KBArticle)
            # WHERE n.node_id IN $node_ids
            # SET n.published = datetime({epochSeconds: toInteger(n.published)})
            # RETURN n.node_id, n.published
            # """

            # Execute the Cypher query to update the published field for the new KB Articles
            # try:
            #     await kb_article_service.cypher(
            #         update_published_query, {"node_ids": node_ids}
            #     )
            #     print("Updated published field for new KBArticle nodes.")
            # except Exception as e:
            #     print(f"Failed to update published field for KB Articles: {e}")
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

    # Instantiate the UpdatePackageService
    update_packages_service = UpdatePackageService(db_manager)

    # Check if the DataFrame is not empty
    if not update_packages.empty:
        print(f"received update_packages: {update_packages.shape[0]}")

        # Set the Neo4j database URL
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()

        # Bulk create update packages in the graph database
        new_update_packages_list = await update_packages_service.bulk_create(
            update_packages.to_dict(orient="records")
        )

        # If new update packages were created, update the response and run the Cypher query
        if new_update_packages_list:
            node_ids = [node.node_id for node in new_update_packages_list]
            response["code"] = 200
            response["message"] = "Update Packages inserted"
            response["insert_ids"] = node_ids
            response["nodes"] = new_update_packages_list

            # Construct the Cypher query to update the published field for the new nodes
            # update_published_query = """
            # MATCH (n:UpdatePackage)
            # WHERE n.node_id IN $node_ids
            # SET n.published = datetime({epochSeconds: toInteger(n.published)})
            # RETURN n.node_id, n.published
            # """

            # Execute the Cypher query to update the published field for the new update packages
            # try:
            #     await update_packages_service.cypher(
            #         update_published_query, {"node_ids": node_ids}
            #     )
            #     print("Updated published field for new UpdatePackage nodes.")
            # except Exception as e:
            #     print(f"Failed to update published field for Update Packages: {e}")
        else:
            print("No Nodes created or returned.")
    else:
        print("Received Update Packages: 0")

    print(f"Total Update Packages Returned: {len(new_update_packages_list)}")
    return response


# load msrc posts into graph db
async def load_msrc_posts_graph_db(msrc_posts: pd.DataFrame):
    response = {
        "code": 404,
        "status": "Complete",
        "message": "No MSRC posts to insert",
        "insert_ids": [],
        "nodes": None,
    }

    # Instantiate the MSRCPostService
    msrc_posts_service = MSRCPostService(db_manager)

    # Check if the DataFrame is valid and not empty
    if isinstance(msrc_posts, pd.DataFrame) and not msrc_posts.empty:
        print(f"received msrc_posts: {msrc_posts.shape[0]}")

        # Set the Neo4j database URL
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()

        # Bulk create MSRC posts in the graph database
        new_msrc_posts_list = await msrc_posts_service.bulk_create(
            msrc_posts.to_dict(orient="records")
        )

        # If new MSRC posts were created, update the response and run the Cypher query
        if new_msrc_posts_list:
            node_ids = [node.node_id for node in new_msrc_posts_list]
            response["code"] = 200
            response["message"] = "MSRC Posts inserted"
            response["insert_ids"] = node_ids
            response["nodes"] = new_msrc_posts_list

            # Construct the Cypher query to update the published field for the new nodes
            # update_published_query = """
            # MATCH (n:MSRCPost)
            # WHERE n.node_id IN $node_ids
            # SET n.published = datetime({epochSeconds: toInteger(n.published)})
            # RETURN n.node_id, n.published
            # """

            # Execute the Cypher query to update the published field for the new MSRC posts
            # try:
            #     await msrc_posts_service.cypher(
            #         update_published_query, {"node_ids": node_ids}
            #     )
            #     print("Updated published field for new MSRCPost nodes.")
            # except Exception as e:
            #     print(f"Failed to update published field for MSRC Posts: {e}")

        print(f"Total MSRC Posts Returned: {len(new_msrc_posts_list)}")

    else:
        print("received msrc_posts: 0")
        response["code"] = 500
        response["message"] = "No MSRC data loaded"
        response["insert_ids"] = []
        response["nodes"] = []

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

    # Instantiate the PatchManagementPostService
    patch_posts_service = PatchManagementPostService(db_manager)

    if not patch_posts.empty:
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

            # Determine the correct sequence
            updates = await determine_email_sequence(all_emails)

            # Update the sequence in the database
            await update_email_sequence(updates, patch_posts_service)

            all_nodes.extend(batch_nodes)

        # If new patch posts were created, update the response and run the Cypher query
        if all_nodes:
            node_ids = [node.node_id for node in all_nodes]
            response["code"] = 200
            response["message"] = "Patch Management Posts inserted and updated"
            response["insert_ids"] = node_ids
            response["nodes"] = all_nodes

            # Construct the Cypher query to update the published field for the new nodes
            # update_published_query = """
            # MATCH (n:PatchManagementPost)
            # WHERE n.node_id IN $node_ids
            # SET n.published = datetime({epochSeconds: toInteger(n.published)})
            # RETURN n.node_id, n.published
            # """

            # Execute the Cypher query to update the published field for the new patch posts
            # try:
            #     await patch_posts_service.cypher(
            #         update_published_query, {"node_ids": node_ids}
            #     )
            #     print("Updated published field for new PatchManagementPost nodes.")
            # except Exception as e:
            #     print(
            #         f"Failed to update published field for Patch Management Posts: {e}"
            #     )
    else:
        print("received patch_posts: 0")

    print(f"Total Patch Posts Returned: {len(all_nodes)}")
    return response


async def load_symptoms_graph_db(symptoms: pd.DataFrame):
    response = {
        "code": 404,
        "status": "Complete",
        "message": "No symptoms to insert",
        "insert_ids": [],
        "nodes": None,
    }
    # Ensures that the list of dicts is properly converted into an AsyncStructuredNode Symptom when the bulk_create() is called.
    symptom_service = SymptomService(db_manager)

    if not symptoms.empty:
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()

        # Bulk create products and return newly inserted nodes
        new_symptoms_list = await symptom_service.bulk_create(
            symptoms.to_dict(orient="records")
        )

        if new_symptoms_list:
            node_ids = [node.node_id for node in new_symptoms_list]
            response["code"] = 200
            response["message"] = "Symptoms inserted"
            response["insert_ids"] = node_ids
            response["nodes"] = new_symptoms_list
            print(f"Total symptoms Loaded: {len(new_symptoms_list)}")
        else:
            print("No new symptoms were inserted.")
    else:
        print("Received symptoms: 0")

    return response


async def load_causes_graph_db(causes: pd.DataFrame):
    response = {
        "code": 404,
        "status": "Complete",
        "message": "No causes to insert",
        "insert_ids": [],
        "nodes": None,
    }
    # Ensures that the list of dicts is properly converted into an AsyncStructuredNode Cause when the bulk_create() is called.
    cause_service = CauseService(db_manager)

    if not causes.empty:
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()

        # Bulk create products and return newly inserted nodes
        new_causes_list = await cause_service.bulk_create(
            causes.to_dict(orient="records")
        )

        if new_causes_list:
            node_ids = [node.node_id for node in new_causes_list]
            response["code"] = 200
            response["message"] = "Causes inserted"
            response["insert_ids"] = node_ids
            response["nodes"] = new_causes_list
            print(f"Total Causes Loaded: {len(new_causes_list)}")
        else:
            print("No new Causes were inserted.")
    else:
        print("Received Causes: 0")

    return response


async def load_fixes_graph_db(fixes: pd.DataFrame):
    response = {
        "code": 404,
        "status": "Complete",
        "message": "No causes to insert",
        "insert_ids": [],
        "nodes": None,
    }
    # Ensures that the list of dicts is properly converted into an AsyncStructuredNode Cause when the bulk_create() is called.
    fix_service = FixService(db_manager)

    if not fixes.empty:
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()

        # Bulk create products and return newly inserted nodes
        new_fixes_list = await fix_service.bulk_create(fixes.to_dict(orient="records"))

        if new_fixes_list:
            node_ids = [node.node_id for node in new_fixes_list]
            response["code"] = 200
            response["message"] = "Fixes inserted"
            response["insert_ids"] = node_ids
            response["nodes"] = new_fixes_list
            print(f"Total Fixes Loaded: {len(new_fixes_list)}")
        else:
            print("No new Fixes were inserted.")
    else:
        print("Received Fixes: 0")

    return response


async def load_tools_graph_db(tools: pd.DataFrame):
    response = {
        "code": 404,
        "status": "Complete",
        "message": "No tools to insert",
        "insert_ids": [],
        "nodes": None,
    }
    # Ensures that the list of dicts is properly converted into an AsyncStructuredNode Cause when the bulk_create() is called.
    tool_service = ToolService(db_manager)

    if not tools.empty:
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()

        # Bulk create products and return newly inserted nodes
        new_tools_list = await tool_service.bulk_create(tools.to_dict(orient="records"))

        if new_tools_list:
            node_ids = [node.node_id for node in new_tools_list]
            response["code"] = 200
            response["message"] = "Tools inserted"
            response["insert_ids"] = node_ids
            response["nodes"] = new_tools_list
            print(f"Total Tools Loaded: {len(new_tools_list)}")
        else:
            print("No new Tools were inserted.")
    else:
        print("Received Tools: 0")

    return response


async def load_technologies_graph_db(technologies: pd.DataFrame):
    response = {
        "code": 404,
        "status": "Complete",
        "message": "No technologies to insert",
        "insert_ids": [],
        "nodes": None,
    }
    # Ensures that the list of dicts is properly converted into an AsyncStructuredNode Cause when the bulk_create() is called.
    technology_service = TechnologyService(db_manager)

    if not technologies.empty:
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()

        # Bulk create products and return newly inserted nodes
        new_technologies_list = await technology_service.bulk_create(
            technologies.to_dict(orient="records")
        )

        if new_technologies_list:
            node_ids = [node.node_id for node in new_technologies_list]
            response["code"] = 200
            response["message"] = "technologies inserted"
            response["insert_ids"] = node_ids
            response["nodes"] = new_technologies_list
            print(f"Total technologies Loaded: {len(new_technologies_list)}")
        else:
            print("No new technologies were inserted.")
    else:
        print("Received technologies: 0")

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
