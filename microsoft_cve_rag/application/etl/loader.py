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

logging.getLogger(__name__)

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
        "nodes": [],
        "errors": [],
    }
    # The ProductService ensures that the key-value pairs in the dictionary are used to create an instance of AsyncStructuredNode for Product
    product_service = ProductService(db_manager)
    new_products_list = []
    if isinstance(products, pd.DataFrame) and not products.empty:
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()

        # Bulk create products and return newly inserted nodes and errors
        new_products_list, errors = await product_service.bulk_create(
            products.to_dict(orient="records")
        )

        if new_products_list:
            response["code"] = 200
            response["status"] = "Complete"
            response["message"] = f"Successfully inserted {len(new_products_list)} products"
            response["insert_ids"] = [str(product.node_id) for product in new_products_list]
            response["nodes"] = new_products_list
        else:
            response["code"] = 404
            response["status"] = "Complete"
            response["message"] = "No new products inserted"

        if errors:
            response["code"] = 207  # Partial Content
            response["status"] = "Partial"
            response["message"] = f"Inserted {len(new_products_list)} products with {len(errors)} errors"
            response["errors"] = [{"node_id": err.node_id, "message": err.message} for err in errors]

    return response


# load product_builds into graph db
async def load_product_builds_graph_db(product_builds: pd.DataFrame):
    response = {
        "code": 404,
        "status": "Complete",
        "message": "No Product Builds to insert",
        "insert_ids": [],
        "nodes": [],
        "errors": [],
    }
    product_build_service = ProductBuildService(db_manager)
    new_product_builds_list = []
    # Check if the DataFrame is not empty
    if isinstance(product_builds, pd.DataFrame) and not product_builds.empty:
        # Set the database URL
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()

        # Bulk create product builds and return newly inserted nodes and errors
        new_product_builds_list, errors = await product_build_service.bulk_create(
            product_builds.to_dict(orient="records")
        )

        # If there are newly inserted product builds, update the response
        if new_product_builds_list:
            response["code"] = 200
            response["status"] = "Complete"
            response["message"] = f"Successfully inserted {len(new_product_builds_list)} product builds"
            response["insert_ids"] = [str(build.node_id) for build in new_product_builds_list]
            response["nodes"] = new_product_builds_list
        else:
            response["code"] = 404
            response["status"] = "Complete"
            response["message"] = "No new product builds inserted"

        if errors:
            response["code"] = 207  # Partial Content
            response["status"] = "Partial"
            response["message"] = f"Inserted {len(new_product_builds_list)} product builds with {len(errors)} errors"
            response["errors"] = [{"node_id": err.node_id, "message": err.message} for err in errors]

    return response


# load kb articles into graph db
async def load_kbs_graph_db(kb_articles: pd.DataFrame):
    response = {
        "code": 404,
        "status": "Complete",
        "message": "No KB Articles to insert",
        "insert_ids": [],
        "nodes": [],
        "errors": [],
    }
    new_kb_article_nodes_list = []
    # Check if the DataFrame is not empty
    if isinstance(kb_articles, pd.DataFrame) and not kb_articles.empty:
        kb_article_service = KBArticleService(db_manager)

        # Set the Neo4j database URL
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()

        # Bulk create KB Articles and return newly inserted nodes and errors
        new_kb_article_nodes_list, errors = await kb_article_service.bulk_create(
            kb_articles.to_dict(orient="records")
        )

        # If new KB articles were created, update the response
        if new_kb_article_nodes_list:
            response["code"] = 200
            response["status"] = "Complete"
            response["message"] = f"Successfully inserted {len(new_kb_article_nodes_list)} KB articles"
            response["insert_ids"] = [str(article.node_id) for article in new_kb_article_nodes_list]
            response["nodes"] = new_kb_article_nodes_list
        else:
            response["code"] = 404
            response["status"] = "Complete"
            response["message"] = "No new KB articles inserted"

        if errors:
            response["code"] = 207  # Partial Content
            response["status"] = "Partial"
            response["message"] = f"Inserted {len(new_kb_article_nodes_list)} KB articles with {len(errors)} errors"
            response["errors"] = [{"node_id": err.node_id, "message": err.message} for err in errors]

    return response


# load update_packages into graph db
async def load_update_packages_graph_db(update_packages: pd.DataFrame):
    response = {
        "code": 404,
        "status": "Complete",
        "message": "No update packages to insert",
        "insert_ids": [],
        "nodes": [],
        "errors": [],
    }

    # Instantiate the UpdatePackageService
    update_packages_service = UpdatePackageService(db_manager)
    new_update_packages_list = []
    # Check if the DataFrame is not empty
    if isinstance(update_packages, pd.DataFrame) and not update_packages.empty:
        print(f"received update_packages: {update_packages.shape[0]}")

        # Set the Neo4j database URL
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()

        # Bulk create update packages in the graph database
        new_update_packages_list, errors = await update_packages_service.bulk_create(
            update_packages.to_dict(orient="records")
        )

        # If new update packages were created, update the response
        if new_update_packages_list:
            response["code"] = 200
            response["status"] = "Complete"
            response["message"] = f"Successfully inserted {len(new_update_packages_list)} update packages"
            response["insert_ids"] = [str(package.node_id) for package in new_update_packages_list]
            response["nodes"] = new_update_packages_list
        else:
            response["code"] = 404
            response["status"] = "Complete"
            response["message"] = "No new update packages inserted"

        if errors:
            response["code"] = 207  # Partial Content
            response["status"] = "Partial"
            response["message"] = f"Inserted {len(new_update_packages_list)} update packages with {len(errors)} errors"
            response["errors"] = [{"node_id": err.node_id, "message": err.message} for err in errors]

    return response


# load msrc posts into graph db
async def load_msrc_posts_graph_db(msrc_posts: pd.DataFrame):
    response = {
        "code": 404,
        "status": "Complete",
        "message": "No MSRC posts to insert",
        "insert_ids": [],
        "nodes": [],
        "errors": [],
    }

    # Instantiate the MSRCPostService
    msrc_posts_service = MSRCPostService(db_manager)
    new_msrc_posts_list = []
    # Check if the DataFrame is valid and not empty
    if isinstance(msrc_posts, pd.DataFrame) and not msrc_posts.empty:
        # rename column source to source_url
        msrc_posts = msrc_posts.rename(columns={"source": "source_url"})
        # Set the Neo4j database URL
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()

        # Convert DataFrame to records and handle datetime serialization
        records = msrc_posts.to_dict(orient="records")
        for record in records:
            # Convert datetime fields to proper python datetime objects
            for datetime_field in ['published', 'nvd_published_date']:
                if datetime_field in record and isinstance(record[datetime_field], (str, pd.Timestamp)):
                    if isinstance(record[datetime_field], str):
                        record[datetime_field] = datetime.fromisoformat(record[datetime_field].replace('Z', '+00:00'))
                    else:  # pd.Timestamp
                        record[datetime_field] = record[datetime_field].to_pydatetime()

            if 'cve_category' in record:
                if isinstance(record['cve_category'], float):
                    record['cve_category'] = 'NC'
                    logging.info(f"Invalid cve_category id: {record['node_id']}. Setting to 'NC'.")
                elif record['cve_category'] is None:
                    record['cve_category'] = 'NC'
                    logging.info(f"None cve_category id: {record['node_id']}. Setting to 'NC'.")

        # Bulk create MSRC posts and return newly inserted nodes and errors
        new_msrc_posts_list, errors = await msrc_posts_service.bulk_create(records)

        # If new MSRC posts were created, update the response
        if new_msrc_posts_list:
            response["code"] = 200
            response["status"] = "Complete"
            response["message"] = f"Successfully inserted {len(new_msrc_posts_list)} MSRC posts"
            response["insert_ids"] = [str(post.node_id) for post in new_msrc_posts_list]
            response["nodes"] = new_msrc_posts_list
        else:
            response["code"] = 404
            response["status"] = "Complete"
            response["message"] = "No new MSRC posts inserted"

        if errors:
            response["code"] = 207  # Partial Content
            response["status"] = "Partial"
            response["message"] = f"Inserted {len(new_msrc_posts_list)} MSRC posts with {len(errors)} errors"
            response["errors"] = [{"node_id": err.node_id, "message": err.message} for err in errors]

    return response


# load patch posts into graph db
async def load_patch_posts_graph_db(patch_posts: pd.DataFrame):
    response = {
        "code": 404,
        "status": "Complete",
        "message": "No patch posts to insert",
        "insert_ids": [],
        "nodes": [],
        "errors": [],
    }

    # Instantiate the PatchPostService
    patch_posts_service = PatchManagementPostService(db_manager)
    new_patch_posts_list = []
    # Check if the DataFrame is valid and not empty
    if isinstance(patch_posts, pd.DataFrame) and not patch_posts.empty:
        # Set the Neo4j database URL
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()

        # Convert DataFrame to records and handle datetime serialization
        records = patch_posts.to_dict(orient="records")
        for record in records:
            # Convert datetime fields to proper python datetime objects
            for datetime_field in ['published']:
                if datetime_field in record and isinstance(record[datetime_field], (str, pd.Timestamp)):
                    if isinstance(record[datetime_field], str):
                        record[datetime_field] = datetime.fromisoformat(record[datetime_field].replace('Z', '+00:00'))
                    else:  # pd.Timestamp
                        record[datetime_field] = record[datetime_field].to_pydatetime()

            # Check for thread_id and lookup emails
            if "thread_id" in record and record["thread_id"]:
                thread_id = record["thread_id"]
                thread_emails = await lookup_thread_emails(thread_id)
                if thread_emails:
                    record["thread_emails"] = thread_emails

        # Bulk create patch posts and return newly inserted nodes and errors
        new_patch_posts_list, errors = await patch_posts_service.bulk_create(records)

        # If new patch posts were created, update the response
        if new_patch_posts_list:
            response["code"] = 200
            response["status"] = "Complete"
            response["message"] = f"Successfully inserted {len(new_patch_posts_list)} patch posts"
            response["insert_ids"] = [str(post.node_id) for post in new_patch_posts_list]
            response["nodes"] = new_patch_posts_list
        else:
            response["code"] = 404
            response["status"] = "Complete"
            response["message"] = "No new patch posts inserted"

        if errors:
            response["code"] = 207  # Partial Content
            response["status"] = "Partial"
            response["message"] = f"Inserted {len(new_patch_posts_list)} patch posts with {len(errors)} errors"
            response["errors"] = [{"node_id": err.node_id, "message": err.message} for err in errors]

    return response


async def load_symptoms_graph_db(symptoms: pd.DataFrame):
    response = {
        "code": 404,
        "status": "Complete",
        "message": "No symptoms to insert",
        "insert_ids": [],
        "nodes": [],
        "errors": [],
    }

    # Instantiate the SymptomService
    symptom_service = SymptomService(db_manager)
    new_symptoms_list = []
    # Check if the DataFrame is not empty
    if isinstance(symptoms, pd.DataFrame) and not symptoms.empty:
        # Set the Neo4j database URL
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()

        # Bulk create symptoms and return newly inserted nodes and errors
        new_symptoms_list, errors = await symptom_service.bulk_create(
            symptoms.to_dict(orient="records")
        )

        # If new symptoms were created, update the response
        if new_symptoms_list:
            response["code"] = 200
            response["status"] = "Complete"
            response["message"] = f"Successfully inserted {len(new_symptoms_list)} symptoms"
            response["insert_ids"] = [str(symptom.node_id) for symptom in new_symptoms_list]
            response["nodes"] = new_symptoms_list
        else:
            response["code"] = 404
            response["status"] = "Complete"
            response["message"] = "No new symptoms inserted"

        if errors:
            response["code"] = 207  # Partial Content
            response["status"] = "Partial"
            response["message"] = f"Inserted {len(new_symptoms_list)} symptoms with {len(errors)} errors"
            response["errors"] = [{"source_id": err.source_id, "message": err.message} for err in errors]

    return response


async def load_causes_graph_db(causes: pd.DataFrame):
    response = {
        "code": 404,
        "status": "Complete",
        "message": "No causes to insert",
        "insert_ids": [],
        "nodes": [],
        "errors": [],
    }

    # Instantiate the CauseService
    cause_service = CauseService(db_manager)
    new_causes_list = []
    # Check if the DataFrame is not empty
    if isinstance(causes, pd.DataFrame) and not causes.empty:
        # Set the Neo4j database URL
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()

        # Bulk create causes and return newly inserted nodes and errors
        new_causes_list, errors = await cause_service.bulk_create(
            causes.to_dict(orient="records")
        )

        # If new causes were created, update the response
        if new_causes_list:
            response["code"] = 200
            response["status"] = "Complete"
            response["message"] = f"Successfully inserted {len(new_causes_list)} causes"
            response["insert_ids"] = [str(cause.node_id) for cause in new_causes_list]
            response["nodes"] = new_causes_list
        else:
            response["code"] = 404
            response["status"] = "Complete"
            response["message"] = "No new causes inserted"

        if errors:
            response["code"] = 207  # Partial Content
            response["status"] = "Partial"
            response["message"] = f"Inserted {len(new_causes_list)} causes with {len(errors)} errors"
            response["errors"] = [{"source_id": err.source_id, "message": err.message} for err in errors]

    return response


async def load_fixes_graph_db(fixes: pd.DataFrame):
    response = {
        "code": 404,
        "status": "Complete",
        "message": "No fixes to insert",
        "insert_ids": [],
        "nodes": [],
        "errors": [],
    }

    # Instantiate the FixService
    fix_service = FixService(db_manager)
    new_fixes_list = []
    # Check if the DataFrame is not empty
    if isinstance(fixes, pd.DataFrame) and not fixes.empty:
        # Set the Neo4j database URL
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()

        # Bulk create fixes and return newly inserted nodes and errors
        new_fixes_list, errors = await fix_service.bulk_create(
            fixes.to_dict(orient="records")
        )

        # If new fixes were created, update the response
        if new_fixes_list:
            response["code"] = 200
            response["status"] = "Complete"
            response["message"] = f"Successfully inserted {len(new_fixes_list)} fixes"
            response["insert_ids"] = [str(fix.node_id) for fix in new_fixes_list]
            response["nodes"] = new_fixes_list
        else:
            response["code"] = 404
            response["status"] = "Complete"
            response["message"] = "No new fixes inserted"

        if errors:
            response["code"] = 207  # Partial Content
            response["status"] = "Partial"
            response["message"] = f"Inserted {len(new_fixes_list)} fixes with {len(errors)} errors"
            response["errors"] = [{"source_id": err.source_id, "message": err.message} for err in errors]

    return response


async def load_tools_graph_db(tools: pd.DataFrame):
    response = {
        "code": 404,
        "status": "Complete",
        "message": "No tools to insert",
        "insert_ids": [],
        "nodes": [],
        "errors": []
    }

    # Instantiate the ToolService
    tool_service = ToolService(db_manager)
    new_tools_list = []
    # Check if the DataFrame is not empty
    if isinstance(tools, pd.DataFrame) and not tools.empty:
        # Set the Neo4j database URL
        NeomodelConfig.DATABASE_URL = get_graph_db_uri()
        if 'created_on' in tools.columns:
            tools = tools.drop(columns=['created_on', 'last_verified_on'])
        # Bulk create tools and return newly inserted nodes and errors
        new_tools_list, errors = await tool_service.bulk_create(
            tools.to_dict(orient="records")
        )

        # If new tools were created, update the response
        if new_tools_list:
            response["code"] = 200
            response["status"] = "Complete"
            response["message"] = f"Successfully inserted {len(new_tools_list)} tools"
            response["insert_ids"] = [str(tool.node_id) for tool in new_tools_list]
            response["nodes"] = new_tools_list
        else:
            response["code"] = 404
            response["status"] = "Complete"
            response["message"] = "No new tools inserted"

        if errors:
            response["code"] = 207  # Partial Content
            response["status"] = "Partial"
            response["message"] = f"Inserted {len(new_tools_list)} tools with {len(errors)} errors"
            response["errors"] = [{"source_id": err.source_id, "message": err.message} for err in errors]

    return response


async def load_technologies_graph_db(technologies: pd.DataFrame):
    response = {
        "code": 404,
        "status": "Complete",
        "message": "No technologies to insert",
        "insert_ids": [],
        "nodes": [],
    }
    # Ensures that the list of dicts is properly converted into an AsyncStructuredNode Cause when the bulk_create() is called.
    technology_service = TechnologyService(db_manager)

    if isinstance(technologies, pd.DataFrame) and not technologies.empty:
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
