# Purpose: Manage graph database operations
# Inputs: Graph queries
# Outputs: Query results
# Dependencies: None (external graph database library)

# services/graph_db_service.py
import os
# import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# print(sys.path)
import traceback

# import asyncio
from typing import Type, TypeVar, Generic, List, Dict, Optional, Any, Tuple, Union, Set
from neomodel import (
    AsyncStructuredNode,
    DoesNotExist,
    DeflateError,
    # db,
)
from neomodel.exceptions import UniqueProperty
import uuid
import logging
import math
import numpy as np
from application.app_utils import (
    get_app_config,
    get_graph_db_credentials,
)
from application.core.models import graph_db_models

# import asyncio  # required for testing
from neomodel import config as NeomodelConfig  # required by AsyncDatabase
from neomodel.async_.core import AsyncDatabase  # required for db CRUD
from neo4j import GraphDatabase  # required for constraints check
from neo4j.exceptions import ClientError
import time
import re
from datetime import datetime
import json
import hashlib
from tqdm import tqdm

settings = get_app_config()
graph_db_settings = settings["GRAPHDB_CONFIG"]
credentials = get_graph_db_credentials()
logging.getLogger(__name__)

username = credentials.username
password = credentials.password
host = credentials.host
port = credentials.port
protocol = credentials.protocol
db_uri = f"{protocol}://{username}:{password}@{host}:{port}"


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


# Async context manager for database operations
class GraphDatabaseManager:
    def __init__(self, db: AsyncDatabase) -> None:
        self._db = db
        # print(f"db is type: {type(self._db)}, config is type: {type(NeomodelConfig)}")

    async def __aenter__(self):
        await self._db.set_connection(NeomodelConfig.DATABASE_URL)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._db.close_connection()


# BaseService definition

T = TypeVar("T", bound=AsyncStructuredNode)


class BaseService(Generic[T]):
    def __init__(self, model: Type[T], db_manager: GraphDatabaseManager):
        """
        Initialize the service with a specific Neomodel node class.

        :param model: The Neomodel node class to manage.
        """
        self.model = model
        self.db_manager = db_manager

    async def create(self, **properties) -> Tuple[Union[T, None], str, int]:
        """
        Create a new node instance.

        :param properties: Key-value pairs of properties to set on the node.
        :return: A tuple containing (node instance or None, status message, status code)
        """

        try:
            node = self.model(**properties)
            if "build_numbers" in properties:
                build_numbers = properties["build_numbers"]
                node.set_build_numbers(build_numbers)
            await node.save()
            logging.debug(f"Created {self.model.__name__}\n{node}")
            return node, "Node created successfully", 201
        except UniqueProperty as e:
            logging.warning(
                f"Duplicate {self.model.__name__} node. Skipping create...\n{str(e)}"
            )
            return None, f"Duplicate entry: {str(e)}", 409
        except DeflateError as e:
            logging.error(
                f"Choice constraint violation in {self.model.__name__}node: {node}\n{str(e)}"
            )
            return None, f"Choice constraint violation: {str(e)}", 422
        except Exception as e:
            logging.error(f"Error creating {self.model.__name__} node: {node}\n{str(e)}")
            return None, f"Error creating node: {str(e)}", 500

    async def get(self, node_id: str) -> Optional[T]:
        """
        Retrieve a node by its node_id.

        :param node_id: The unique id of the node to retrieve.
        :return: The node instance if found, otherwise None.
        """
        node = await self.model.nodes.get_or_none(node_id=node_id)
        return node

    async def update(self, node_id: str, **properties) -> Dict:
        """
        Update an existing node's properties.

        :param node_id: The unique id of the node to update.
        :param properties: Key-value pairs of properties to update.
        :return: A response dictionary with keys 'node_id', 'status', and 'message'.
        """
        response = {"node_id": node_id, "status": "success", "message": ""}
        node = await self.get(node_id)

        if node:
            for key, value in properties.items():
                setattr(node, key, value)
            await node.save()
            response["message"] = "Successfully updated node."
        else:
            response["status"] = "error"
            response["message"] = "No node found"

        return response

    async def delete(self, node_id: uuid.UUID) -> bool:
        """
        Delete a node by its UUID.

        :param node_id: The UUID of the node to delete.
        :return: True if the node was deleted, False if it was not found.
        """
        node = await self.get(node_id)
        if node:
            await node.delete()
            return True
        return False

    async def get_or_create(self, **properties) -> Tuple[T, str, int]:
        """
        Retrieve a node with matching properties, or create it if it doesn't exist.

        :param properties: Key-value pairs of properties to match or set.
        :return: A tuple containing (node instance, status message, status code)
        """
        # print(f"\nAttempting to get or create with properties: {properties}")

        try:
            node_label = properties.get("node_label", None)
            search_dict = self._build_search_dict(node_label, properties)
            # print(f"search_dict: {search_dict}")
            # print("attempting service based approach")
            existing_node_service = await self.model.nodes.get_or_none(**search_dict)
            if existing_node_service:
                # print(f"Node found by: {search_dict}")
                return existing_node_service, "Existing node retrieved by node_id", 200

            # print("attempting class based approach")
            if properties["node_label"] == "Product":
                existing_node_class = await graph_db_models.Product.nodes.get_or_none(
                    **search_dict
                )
            elif properties["node_label"] == "ProductBuild":
                existing_node_class = (
                    await graph_db_models.ProductBuild.nodes.get_or_none(**search_dict)
                )
            else:
                existing_node_class = None

            if existing_node_class:
                # print(f"Node found by: {search_dict}")
                return existing_node_class, "Existing node retrieved by node_id", 200

            # If not found, create a new node
            if "build_numbers" in properties:
                properties.pop("build_numbers")
            node_new = self.model(**properties)
            await node_new.save()
            return node_new, "Node created successfully", 201

        except UniqueProperty as e:
            print(f"UniqueProperty exception: {str(e)}")
            # In case of a unique property violation, return the existing node
            return None, f"Unique property violation: {str(e)}", 409

        except Exception as e:
            print(f"General exception: {str(e)}")
            traceback.print_exc()
            return None, f"Error creating/retrieving node: {str(e)}", 500

    def _build_search_dict(self, node_label: str, properties: dict) -> dict:
        """
        Construct the search dictionary based on the node_label.
        """
        search_dict = {}
        if node_label == "Product":
            search_dict = {
                "product_name": properties.get("product_name"),
                "product_architecture": properties.get("product_architecture"),
                "product_version": properties.get("product_version"),
            }
        elif node_label == "ProductBuild":
            search_dict = {
                "product_name": properties.get("product_name"),
                "product_version": properties.get("product_version"),
                "product_architecture": properties.get("product_architecture"),
                "cve_id": properties.get("cve_id"),
                "product_build_id": properties.get("product_build_id"),
            }
        elif node_label == "KBArticle":
            search_dict = {
                "node_id": properties.get("node_id"),
            }
        else:
            # For all other types, assume node_id is sufficient
            search_dict = {"node_id": properties.get("node_id")}

        return search_dict

    async def execute_cypher(self, query: str, params: Optional[Dict[str, Any]] = None):
        print(f"query type: {type(query)} type params: {type(params)}\n{params}")
        results = db.cypher_query(query, params)
        return results

    async def inflate_results(self, results: List[Any]) -> List[T]:
        inflated = []
        for row in results:
            if isinstance(row, tuple):
                inflated.append(tuple(await self.inflate_item(item) for item in row))
            else:
                inflated.append(await self.inflate_item(row))
        return inflated

    async def inflate_item(self, item: Any) -> Union[T, Dict[str, Any]]:
        if isinstance(item, dict) and "node_id" in item:
            return await self.model.inflate(item)
        elif isinstance(item, AsyncStructuredNode):
            return item
        else:
            return dict(item)

    async def cypher(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Union[T, Dict[str, Any]]], Dict[str, Any]]:
        try:
            results, meta = await self.execute_cypher(query, params)
            # processed_results = await self.inflate_results(results)
            print(f"Query execution metadata: {meta}")
            return results, meta
        except Exception as e:
            print(f"Error executing Cypher query: {e}")
            raise

    async def get_by_property(self, **properties) -> List[T]:
        """
        Retrieve nodes by matching properties.
        https://neomodel.readthedocs.io/en/latest/queries.html#node-sets-and-filtering
        :param properties: Key-value pairs of properties to match.
        :return: A list of node instances that match the properties.
        """
        return await self.model.nodes.filter(**properties)

    async def get_by_relationship_property(
        self, relationship: str, **properties
    ) -> List[T]:
        """
        Retrieve nodes by matching properties on a relationship.

        :param relationship: The name of the relationship to search through.
        :param properties: Key-value pairs of properties to match on the relationship.
        :return: A list of node instances that have matching relationship properties.
        """
        query = f"""
        MATCH (n)-[r:{relationship}]->(m)
        WHERE {" AND ".join([f'r.{k} = ${k}' for k in properties.keys()])}
        RETURN n
        """
        params = {k: v for k, v in properties.items()}
        results = await self.cypher(query, params)
        return [self.model.inflate(node) for node in results]

    async def bulk_create(self, items: List[Dict[str, Any]]) -> List[T]:
        results = []
        errors = []
        async with self.db_manager:
            async with db.transaction:
                for item in items:
                    try:
                        # Handle None or empty values for embedding
                        if "embedding" in item:
                            if isinstance(item["embedding"], float) and math.isnan(
                                item["embedding"]
                            ):
                                del item["embedding"]
                            elif item["embedding"] is None:
                                del item["embedding"]
                            elif (
                                isinstance(item["embedding"], (list, np.ndarray))
                                and len(item["embedding"]) == 0
                            ):
                                del item["embedding"]
                        if "cve_ids" in item:
                            if item["cve_ids"] is None or (
                                isinstance(item["cve_ids"], float)
                                and math.isnan(item["cve_ids"])
                            ):
                                item["cve_ids"] = []
                            elif isinstance(item["cve_ids"], str):
                                item["cve_ids"] = [item["cve_ids"]]
                            elif not isinstance(item["cve_ids"], list):
                                item["cve_ids"] = [str(item["cve_ids"])]
                        if "kb_ids" in item:
                            if item["kb_ids"] is None or (
                                isinstance(item["kb_ids"], float)
                                and math.isnan(item["kb_ids"])
                            ):
                                item["kb_ids"] = []
                            elif isinstance(item["kb_ids"], str):
                                item["kb_ids"] = [item["kb_ids"]]
                            elif not isinstance(item["kb_ids"], list):
                                item["kb_ids"] = [str(item["kb_ids"])]
                        if "product_build_ids" in item:
                            if item["product_build_ids"] is None or (
                                isinstance(item["product_build_ids"], float)
                                and math.isnan(item["product_build_ids"])
                            ):
                                item["product_build_ids"] = []
                            elif isinstance(item["product_build_ids"], str):
                                item["product_build_ids"] = [item["product_build_ids"]]
                            elif not isinstance(item["product_build_ids"], list):
                                item["product_build_ids"] = [str(item["product_build_ids"])]
                        if "product_mentions" in item:
                            if item["product_mentions"] is None or (
                                isinstance(item["product_mentions"], float)
                                and math.isnan(item["product_mentions"])
                            ):
                                item["product_mentions"] = []
                            elif isinstance(item["product_mentions"], str):
                                item["product_mentions"] = [item["product_mentions"]]
                            elif not isinstance(item["product_mentions"], list):
                                item["product_mentions"] = [
                                    str(item["product_mentions"])
                                ]
                        if "tags" in item:
                            if item["tags"] is None or (
                                isinstance(item["tags"], float)
                                and math.isnan(item["tags"])
                            ):
                                item["tags"] = []
                            elif isinstance(item["tags"], str):
                                item["tags"] = [item["tags"]]
                            elif not isinstance(item["tags"], list):
                                item["tags"] = [str(item["tags"])]
                        if "keywords" in item:
                            if item["keywords"] is None or (
                                isinstance(item["keywords"], float)
                                and math.isnan(item["keywords"])
                            ):
                                item["keywords"] = []
                            elif isinstance(item["keywords"], str):
                                item["keywords"] = [item["keywords"]]
                            elif not isinstance(item["keywords"], list):
                                item["keywords"] = [str(item["keywords"])]
                        if "noun_chunks" in item:
                            if item["noun_chunks"] is None or (
                                isinstance(item["noun_chunks"], float)
                                and math.isnan(item["noun_chunks"])
                            ):
                                item["noun_chunks"] = []
                            elif isinstance(item["noun_chunks"], str):
                                item["noun_chunks"] = [item["noun_chunks"]]
                            elif not isinstance(item["noun_chunks"], list):
                                item["noun_chunks"] = [str(item["noun_chunks"])]
                        if "post_type" in item:
                            if item["post_type"] is None or (
                                isinstance(item["post_type"], float)
                                and math.isnan(item["post_type"])
                            ):
                                item["post_type"] = ""
                        node, message, code = await self.get_or_create(**item)
                        if node:
                            # Handle specific node types if needed
                            if (
                                hasattr(node, "set_build_numbers")
                                and "build_numbers" in item
                            ):
                                node.set_build_numbers(item["build_numbers"])
                                await node.save()
                            if (
                                hasattr(node, "set_downloadable_packages")
                                and "downloadable_packages" in item
                            ):
                                node.set_downloadable_packages(
                                    item["downloadable_packages"]
                                )
                                await node.save()
                            results.append(node)
                        else:
                            errors.append((message, code, item))
                    except UniqueProperty as e:
                        print(f"UniqueProperty error: {str(e)}")
                        # Try to retrieve the existing node
                        existing_node = await self.model.nodes.get_or_none(**item)
                        print("UniqueProperty violation...attempting lookup again:")
                        if existing_node:
                            results.append(existing_node)
                        else:
                            errors.append((str(e), 409, item))
                    except Exception as e:
                        # logging.error(f"Error creating/retrieving node: {str(e)}")
                        errors.append((str(e), 500, item))

        if errors:
            for error in errors:
                logging.warning(f"bulk_create: {error[1]} - {error[0]}\n{error[2]}")
                # pass
        if results:
            logging.info(
                f"Successfully created/retrieved {len(results)} nodes of type: {type(results[0]).__name__}"
            )
        else:
            logging.warning("No nodes were created or retrieved.")

        return results

    async def find_related_nodes(
        self, node: T, target_model: Type[AsyncStructuredNode], rel_type: str
    ) -> List[AsyncStructuredNode]:
        """
        Find related nodes based on the relationship type and the target node type.

        :param node: The source node instance.
        :param target_model: The target Neomodel node class.
        :param rel_type: The relationship type to search by.
        :return: A list of related node instances.
        """
        try:
            # Query to find related nodes using the relationship type and target model
            query = f"""
            MATCH (n:{node.__label__})-[r:{rel_type}]->(m:{target_model.__label__})
            WHERE n.node_id = $node_id
            RETURN m
            """
            params = {"node_id": node.node_id}
            results = await self.db_manager.cypher(query, params)
            return [target_model.inflate(record[0]) for record in results]
        except Exception as e:
            logging.error(f"Error finding related nodes: {str(e)}")
            return []


class MSRCPostService(BaseService[graph_db_models.MSRCPost]):
    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.MSRCPost, db_manager)


class ProductService(BaseService[graph_db_models.Product]):
    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.Product, db_manager)


class ProductBuildService(BaseService[graph_db_models.ProductBuild]):
    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.ProductBuild, db_manager)


class SymptomService(BaseService[graph_db_models.Symptom]):
    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.Symptom, db_manager)


class CauseService(BaseService[graph_db_models.Cause]):
    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.Cause, db_manager)


class FixService(BaseService[graph_db_models.Fix]):
    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.Fix, db_manager)


class FAQService(BaseService[graph_db_models.FAQ]):
    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.FAQ, db_manager)


class ToolService(BaseService[graph_db_models.Tool]):
    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.Tool, db_manager)


class KBArticleService(BaseService[graph_db_models.KBArticle]):
    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.KBArticle, db_manager)


class UpdatePackageService(BaseService[graph_db_models.UpdatePackage]):
    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.UpdatePackage, db_manager)


class PatchManagementPostService(BaseService[graph_db_models.PatchManagementPost]):
    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.PatchManagementPost, db_manager)

    async def update_sequence(self, node_id, previous_id, next_id):
        node = await self.get(node_id=node_id)
        if node:
            node.previous_id = previous_id
            node.next_id = next_id
            await node.save()
        return node

    async def get_by_thread_id(self, thread_id):
        return await self.model.nodes.filter(thread_id=thread_id)


class TechnologyService(BaseService[graph_db_models.Technology]):
    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.Technology, db_manager)


# Function to check if a constraint exists in the graph database
def graph_db_constraint_exists(session, constraint_name):
    query = "SHOW CONSTRAINTS"
    result = session.run(query)
    constraints = [record.get("name") for record in result if record.get("name")]
    return constraint_name in constraints


def graph_db_index_exists(session, index_name):
    query = "SHOW INDEXES"
    result = session.run(query)
    indexes = [record.get("name") for record in result if record.get("name")]
    return index_name in indexes


# Function to ensure the constraint exists in the graph database
# called by fastapi at runtime to validate the graph database
# Implements neo4j package not neomodel
def ensure_graph_db_constraints_exist(uri: str, auth: tuple, graph_db_settings: dict):
    try:
        driver = GraphDatabase.driver(uri, auth=auth)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to connect to the graph database: {str(e)}",
        }

    constraints_status = []
    indexes_status = []
    try:
        with driver.session() as session:
            for constraint in graph_db_settings["constraints"]:
                constraint_name = constraint["name"]
                constraint_cypher = constraint["cypher"]
                if not graph_db_constraint_exists(session, constraint_name):
                    session.run(constraint_cypher)
                    constraints_status.append(
                        {
                            "constraint_name": constraint_name,
                            "status": "created",
                            "message": f"Constraint '{constraint_name}' created successfully.",
                        }
                    )
                else:
                    constraints_status.append(
                        {
                            "constraint_name": constraint_name,
                            "status": "exists",
                            "message": f"Constraint '{constraint_name}' already exists.",
                        }
                    )

            for index in graph_db_settings.get("indexes", []):
                index_name = index["name"]
                index_cypher = index["cypher"]
                if not graph_db_index_exists(session, index_name):
                    session.run(index_cypher)
                    indexes_status.append(
                        {
                            "index_name": index_name,
                            "status": "created",
                            "message": f"Index '{index_name}' created successfully.",
                        }
                    )
                else:
                    indexes_status.append(
                        {
                            "index_name": index_name,
                            "status": "exists",
                            "message": f"Index '{index_name}' already exists.",
                        }
                    )

    except Exception as e:
        constraints_status.append(
            {
                "constraint_name": "unknown",
                "status": "error",
                "message": f"Failed to apply constraints: {str(e)}",
            }
        )
    finally:
        driver.close()

    all_statuses = constraints_status + indexes_status
    overall_status = (
        "success"
        if all(status["status"] in ["created", "exists"] for status in all_statuses)
        else "partial_success"
    )
    return {
        "status": overall_status,
        "message": "Constraints and indexes checked and applied.",
        "constraints_status": constraints_status,
        "indexes_status": indexes_status,
    }


# Neomodel AsynchStructuredNode instances
async def convert_nodes_to_products(node_results):
    products = []
    for result in node_results:
        node = result[0]  # Assuming each result is a list with one Node
        product = graph_db_models.Product.inflate(node)
        products.append(product)
    return products


async def convert_nodes_to_product_builds(node_results):
    product_builds = []
    for result in node_results:
        node = result[0]  # Assuming each result is a list with one Node
        product_build = graph_db_models.ProductBuild.inflate(node)
        product_builds.append(product_build)
    return product_builds


async def convert_nodes_to_kbs(node_results):
    kb_articles = []
    for result in node_results:
        node = result[0]  # Assuming each result is a list with one Node
        kb_article = graph_db_models.KBArticle.inflate(node)
        kb_articles.append(kb_article)
    return kb_articles


async def convert_nodes_to_update_packages(node_results):
    update_packages = []
    for result in node_results:
        node = result[0]  # Assuming each result is a list with one Node
        update_package = graph_db_models.UpdatePackage.inflate(node)
        update_packages.append(update_package)
    return update_packages


async def convert_nodes_to_msrc_posts(node_results):
    msrc_posts = []
    for result in node_results:
        node = result[0]  # Assuming each result is a list with one Node
        msrc_post = graph_db_models.MSRCPost.inflate(node)
        msrc_posts.append(msrc_post)
    return msrc_posts


async def convert_nodes_to_patch_posts(node_results):
    patch_posts = []
    for result in node_results:
        node = result[0]  # Assuming each result is a list with one Node
        patch_post = graph_db_models.PatchManagementPost.inflate(node)
        patch_posts.append(patch_post)
    return patch_posts


async def inflate_nodes(node_results):
    inflated_objects = []
    for node in node_results:
        labels = node.labels
        if labels:
            label = next(iter(labels))  # Get the first label
            if label in graph_db_models.LABEL_TO_CLASS_MAP:
                cls = graph_db_models.LABEL_TO_CLASS_MAP[label]
                inflated_object = cls.inflate(node)
                inflated_objects.append(inflated_object)
            else:
                print(f"No class found for label: {label}")
        else:
            print("Node has no labels")
    return inflated_objects


SERVICES_MAPPING = {
    "Product": ProductService,
    "ProductBuild": ProductBuildService,
    "MSRCPost": MSRCPostService,
    "Symptom": SymptomService,
    "Cause": CauseService,
    "Fix": FixService,
    "FAQ": FAQService,
    "Tool": ToolService,
    "KBArticle": KBArticleService,
    "UpdatePackage": UpdatePackageService,
    "PatchManagementPost": PatchManagementPostService,
}


def get_graph_db_uri():
    credentials = get_graph_db_credentials()
    host = credentials.host
    port = credentials.port
    protocol = credentials.protocol
    username = credentials.username
    password = credentials.password
    uri = f"{protocol}://{username}:{password}@{host}:{port}"
    return uri


async def sort_and_update_msrc_nodes(
    msrc_posts: List[graph_db_models.MSRCPost],
) -> List[graph_db_models.MSRCPost]:
    post_groups = {}
    for post in msrc_posts:
        if post.post_id not in post_groups:
            post_groups[post.post_id] = []
        post_groups[post.post_id].append(post)

    updated_msrc_posts = []
    for post_id, posts in post_groups.items():
        sorted_posts = sorted(posts, key=lambda x: Decimal(x.revision), reverse=True)
        for i, post in enumerate(sorted_posts):
            if i < len(sorted_posts) - 1:
                post.previous_version_id = sorted_posts[i + 1].node_id
                await post.save()
            else:
                post.previous_version_id = None
                await post.save()
            updated_msrc_posts.append(post)

    return updated_msrc_posts


def build_number_in_list(target_build_number, build_numbers_list):
    """
    Check if the target build number exists in the list of build numbers.

    :param target_build_number: A list of integers representing the target build number.
    :param build_numbers_list: A list of lists of integers representing multiple build numbers.
    :return: True if the target build number is found in the list, False otherwise.
    """
    # Convert the list of lists to a set of tuples for fast membership testing
    build_numbers_set = {tuple(build_number) for build_number in build_numbers_list}
    # Convert the target build number to a tuple
    target_build_number_tuple = tuple(target_build_number)
    # Check if the target build number is in the set
    return target_build_number_tuple in build_numbers_set


def build_number_id_in_list(target_build_number_id, build_number_ids_list):
    """
    Check if the target build number exists in the list of build numbers.

    :param target_build_number_id: A list of integers representing the target build number.
    :param build_number_ids_list: A list of lists of integers representing multiple build numbers.
    :return: True if the target build number id is found in the list, False otherwise.
    """
    if isinstance(target_build_number_id, list) and target_build_number_id:
        t_product_build_id = target_build_number_id[0]
        # print(f"lookup id is: {t_product_build_id}")
    else:
        t_product_build_id = target_build_number_id
        # print(f"lookup id is: {t_product_build_id}")

    if isinstance(build_number_ids_list, str):
        print("list is a string requires conversion")

    elif isinstance(build_number_ids_list, list):

        if t_product_build_id in build_number_ids_list:
            # print("product_build_id in list")
            return True
    else:
        pass

    return False


async def check_has_symptom(
    source_node: Any,
    target_node: Any,
    rel_info: graph_db_models.AsyncSymptomCauseFixRel,
) -> bool:
    return (
        target_node.source_id == source_node.node_id
    )


async def check_has_cause(
    source_node: Any,
    target_node: Any,
    rel_info: graph_db_models.AsyncSymptomCauseFixRel,
) -> bool:
    return target_node.source_id == source_node.node_id


async def check_has_fix(
    source_node: Any,
    target_node: Any,
    rel_info: graph_db_models.AsyncSymptomCauseFixRel,
) -> bool:
    return target_node.source_id == source_node.node_id


async def check_has_faq(
    source_node: Any, target_node: Any, rel_info: graph_db_models.AsyncZeroToManyRel
) -> bool:
    return target_node.source_id in source_node.faq_ids


async def check_has_tool(
    source_node: Any, target_node: Any, rel_info: graph_db_models.AsyncZeroToManyRel
) -> bool:
    return target_node.source_id == source_node.node_id


async def check_has_kb(
    source_node: Any, target_node: Any, rel_info: graph_db_models.AsyncZeroToManyRel
) -> bool:
    if isinstance(source_node, graph_db_models.MSRCPost):
        if isinstance(target_node, graph_db_models.KBArticle):
            return target_node.kb_id in source_node.kb_ids
    return False


async def check_has_update_package(
    source_node: Any,
    target_node: Any,
    rel_info: graph_db_models.AsyncHasUpdatePackageRel,
) -> bool:
    if isinstance(source_node, graph_db_models.KBArticle):
        source_kb_id = source_node.kb_id.replace("-", "")
        target_url = target_node.package_url
        if not target_url:
            return False
        kb_id_pattern = re.compile(re.escape(source_kb_id))
        response = bool(kb_id_pattern.search(target_url))
        # print(
        #     f"KBArticle({source_node.product_build_id}) -(IN:{response})> UpdatePackage\n{target_node.product_build_ids}"
        # )
        return response
    elif isinstance(source_node, graph_db_models.ProductBuild):
        response = source_node.product_build_id in target_node.product_build_ids
        # print(
        #     f"ProductBuild({source_node.product_build_id}) -(IN:{response})> UpdatePackage\n{target_node.product_build_ids}"
        # )
        return response
    elif isinstance(source_node, graph_db_models.MSRCPost):
        # MSRCPost now has product_build_ids, an array of all product_builds that reference
        # UpdatePackage has product_build_ids, an array of all product_builds that reference an update package
        source_ids = source_node.product_build_ids
        target_ids = set(target_node.product_build_ids)
        response = any(item in target_ids for item in source_ids)
        # print(
        #     f"MSRCPost({source_node.product_build_ids}) -(IN:{response})> UpdatePackage\n{target_node.product_build_ids}"
        # )
        return response
    print(f"source_node unexpected: {type(source_node)}")
    return False


async def match_product(source_node, target_node) -> bool:
    """
    This function compares product_mentions from a source_node with the target_node (a Product)
    to determine if they match based on product_name, product_version, and product_architecture.

    :param source_node: The node containing the product_mentions list.
    :param target_node: A single Product node to compare against.
    :return: True if a match is found, otherwise False.
    """
    product_mentions = source_node.product_mentions

    # Iterate over product mentions in the source node
    for mention in product_mentions:
        # Split the mention into components by '_'
        parts = mention.split("_")

        # Windows logic (e.g., windows_10_21H2_x64)
        if parts[0].lower() == "windows":
            product_name = f"windows_{parts[1]}"  # e.g., windows_10, windows_11
            product_version = parts[2] if len(parts) > 2 else None
            product_architecture = parts[3] if len(parts) > 3 else None

            # Check the target product for an exact product_name match first
            if target_node.product_name == product_name:
                # Now check for product_version and product_architecture if available
                if (
                    not product_version
                    or target_node.product_version == product_version
                ) and (
                    not product_architecture
                    or target_node.product_architecture == product_architecture
                ):
                    return True  # Exact match found

        # Edge logic (simple match based on product_name)
        elif parts[0].lower() == "edge":
            product_name = "edge"

            # Check the target product for an exact product_name match
            if target_node.product_name == product_name:
                return True  # Exact match found

    return False


async def check_affects_product(
    source_node: Any, target_node: Any, rel_info: graph_db_models.AsyncAffectsProductRel
) -> bool:
    if isinstance(source_node, graph_db_models.MSRCPost):
        return any(
            msrc_build in target_node.build_numbers
            for msrc_build in source_node.build_numbers
        )
    elif isinstance(source_node, graph_db_models.Symptom):
        return source_node.source_id == target_node.node_id
    elif isinstance(source_node, graph_db_models.KBArticle):
        return source_node.kb_id in target_node.kb_ids
    elif isinstance(source_node, graph_db_models.PatchManagementPost):
        response = await match_product(source_node, target_node)
        # print(f"check_affects_product: Patch-[affects]>Product = {response}")
        return response
    return False


from decimal import Decimal


async def check_previous_version(
    source_node: Any, target_node: Any, rel_info: Any
) -> bool:
    if not isinstance(source_node, graph_db_models.MSRCPost) or not isinstance(
        target_node, graph_db_models.MSRCPost
    ):
        print("MSRC Previous check failed: not both msrc")
        return False

    if source_node.post_id != target_node.post_id:
        # print(
        #     f"MSRC previous check: unequal post_ids\n{source_node.post_id} -> {target_node.post_id}"
        # )
        return False

    source_revision = Decimal(source_node.revision)
    target_revision = Decimal(target_node.revision)
    # print(f"MSRC version check: {source_revision} < {target_revision}")
    return target_revision < source_revision


async def check_previous_message(
    source_node: Any,
    target_node: Any,
    rel_info: graph_db_models.AsyncPreviousMessageRel,
) -> bool:
    if source_node.node_id == target_node.node_id:
        return False
    return target_node.node_id == source_node.previous_id


async def check_references(
    source_node: Any, target_node: Any, rel_info: graph_db_models.AsyncReferencesRel
) -> bool:
    # if isinstance(source_node, graph_db_models.Fix):
    #     if isinstance(target_node, graph_db_models.KBArticle):
    #         return target_node.kb_id[0] in source_node.kb_ids

    if isinstance(source_node, graph_db_models.KBArticle):
        if isinstance(target_node, graph_db_models.ProductBuild):
            print(f"This is the From direction. No relationship to create.")
            # return target_node.product_build_id in source_node.product_build_id
            return False
    
    elif isinstance(source_node, graph_db_models.MSRCPost):
        if isinstance(target_node, graph_db_models.KBArticle):
            return target_node.kb_id in source_node.kb_ids
    
    elif isinstance(source_node, graph_db_models.PatchManagementPost):
        if isinstance(target_node, graph_db_models.KBArticle):
            """
            elif isinstance(source_node, graph_db_models.PatchManagementPost):
                print(f"Patch -[HAS_KB]-> KB checking...")
                response = target_node.kb_id in source_node.kb_ids
                if response:
                    print(f"source: {source_node}\ntarget:{target_node}")
                    print(
                        f"\n=========================================================\nPatch -[HAS_KB]->KB: {response}\n=========================================================\n"
                    )
                    time.sleep(10)
                else:
                    print(f"Patch -[HAS_KB]->KB: False")
                    print(f"source: {source_node}\ntarget:{target_node}")
                return response
            else:
                print(f"Unexpected:\nsource: {source_node}\ntarget:{target_node}")
            """
            # print("Patch-[REFERENCES]->KB checking...")
            return target_node.kb_id in source_node.kb_ids
        elif isinstance(target_node, graph_db_models.MSRCPost):
            return target_node.post_id in source_node.cve_ids

    elif isinstance(source_node, graph_db_models.ProductBuild):
        if isinstance(target_node, graph_db_models.MSRCPost):
            return target_node.post_id in source_node.cve_id
        elif isinstance(target_node, graph_db_models.KBArticle):
            # print(f"source_node->target_node")
            # print(
            #     f"{source_node.node_label} - {source_node.kb_id}\n{target_node.node_label} - {target_node.kb_id}"
            # )
            return target_node.kb_id in source_node.kb_id
    return False


async def check_has_build(
    source_node: Any, target_node: Any, rel_info: graph_db_models.AsyncZeroToManyRel
) -> bool:
    # print(
    #     f"check has build:\n{source_node.node_label}\n{target_node.node_label}\n{rel_info}"
    # )
    if isinstance(source_node, graph_db_models.Product):
        # print(f"source_node is Product.")
        build_number_to_check = target_node.build_number
        build_numbers_list = source_node.get_build_numbers()
        # print(
        #     f"source: {source_node.product_name}|{source_node.product_architecture}|{source_node.product_version} - target: {target_node.product_name}|{target_node.product_architecture}|{target_node.product_version}"
        # )
        found = any(sublist == build_number_to_check for sublist in build_numbers_list)
        # print(
        #     f"answer: {any(sublist == build_number_to_check for sublist in build_numbers_list)}"
        # )
        return found

    return False


async def should_be_related(
    source_node: AsyncStructuredNode,
    target_node: AsyncStructuredNode,
    rel_info: Tuple[str, str, type],
) -> bool:
    """
    Determine if two nodes should be related based on specific criteria.
    :param source_node: The source node instance.
    :param target_node: The target node instance.
    :param rel_info: A tuple containing (target_model, relation_type, rel_model).
    :return: True if the nodes should be related, False otherwise.
    """
    # print("SBR 1. should be related?")
    target_model, relation_type, _ = rel_info
    # print(f"SBR 2. rel_type: {relation_type}\ntarget:\n{target_model} - ")
    relation_checkers = {
        "HAS_SYMPTOM": check_has_symptom,
        "HAS_CAUSE": check_has_cause,
        "HAS_FIX": check_has_fix,
        "HAS_FAQ": check_has_faq,
        "HAS_TOOL": check_has_tool,
        "HAS_KB": check_has_kb,
        "HAS_UPDATE_PACKAGE": check_has_update_package,
        "AFFECTS_PRODUCT": check_affects_product,
        "PREVIOUS_VERSION": check_previous_version,
        "PREVIOUS_MESSAGE": check_previous_message,
        "REFERENCES": check_references,
        "HAS_BUILD": check_has_build,
    }

    checker = relation_checkers.get(relation_type)
    if checker:
        return await checker(source_node, target_node, rel_info)
    else:
        print(f"SBR 3. what is checker: {checker}")

    print(f"SBR 4. No checker found for relationship type: {relation_type}")
    return False


async def create_and_return_relationship(
    relationship, source_node, target_node, rel_class
):
    # Generate a unique identifier for the relationship
    unique_id = hashlib.sha256(
        f"{source_node.node_id}_{target_node.node_id}".encode()
    ).hexdigest()

    # Connect the relationship and assign the unique ID
    await relationship.connect(target_node, {"relationship_id": unique_id})

    # Get the actual relationship type from the rel_class
    rel_type = relationship.definition["relation_type"]
    # print(f"computed rel_type: {rel_type}")
    # Construct the Cypher query to retrieve the relationship
    query = f"""
    MATCH (source:{source_node.__label__})-[r:{rel_type}]->(target:{target_node.__label__})
    WHERE source.node_id = $source_id AND target.node_id = $target_id AND r.relationship_id = $rel_id
    RETURN r
    """
    params = {
        "source_id": source_node.node_id,
        "target_id": target_node.node_id,
        "rel_id": unique_id,
    }

    # Execute the query
    results, _ = await db.cypher_query(query, params)

    if results:
        # Create an instance of the relationship class
        relationship_instance = rel_class.inflate(results[0][0])
        # print(f"Returning relationship instance: {relationship_instance}")
        return relationship_instance
    else:
        # print("No relationship found")
        return None


# END RELATIONSHIP HELPER FUNCTIONS =============================================


# BEGIN BUILD RELATIONSHIP FUNCTION ==============================================


async def build_relationships(
    nodes_dict: dict[str, list[graph_db_models.AsyncStructuredNode]]
):
    print("begin build relationships")
    tracker = RelationshipTracker()
    all_node_ids = [node.node_id for nodes in nodes_dict.values() for node in nodes]
    await tracker.fetch_existing_relationships(list(nodes_dict.keys()), all_node_ids)

    # Sort & update MSRCPost nodes
    if "MSRCPost" in nodes_dict:
        updated_msrc_posts = await sort_and_update_msrc_nodes(nodes_dict["MSRCPost"])
        nodes_dict["MSRCPost"] = updated_msrc_posts

    for node_type_str, nodes in nodes_dict.items():
        if node_type_str in graph_db_models.RELATIONSHIP_MAPPING:
            # print(f"BR 1: node_type_str: {node_type_str}")
            for node in tqdm(nodes, desc=f"Processing {node_type_str} nodes"):
                for rel_name, rel_info in graph_db_models.RELATIONSHIP_MAPPING[
                    node_type_str
                ].items():
                    target_model, rel_type, rel_class = rel_info
                    target_nodes = nodes_dict.get(target_model, [])

                    for target_node in target_nodes:
                        if await should_be_related(node, target_node, rel_info):
                            relationship = getattr(node, rel_name)

                            # Check if the relationship already exists
                            if not tracker.relationship_exists(
                                node, rel_type, target_node
                            ):
                                # print(
                                #     f"Creating new relationship: {node.node_label} -{rel_type}-> {target_node.node_label}"
                                # )

                                # Disconnect all relationships if it is a one-to-one relationship
                                if issubclass(
                                    rel_class, graph_db_models.AsyncOneToOneRel
                                ):
                                    # print("detected 1-to-1 relationship")
                                    await relationship.disconnect_all()

                                # Connect the new relationship
                                relationship_instance = (
                                    await create_and_return_relationship(
                                        relationship, node, target_node, rel_class
                                    )
                                )
                                # print(
                                #     f"Received relationship instance: {relationship_instance}"
                                # )
                                tracker.add_relationship(node, rel_type, target_node)

                                # Set properties for custom relationship classes
                                if rel_class == graph_db_models.AsyncSymptomCauseFixRel:
                                    # print("BR 3: rel_class = AsyncSymptomCauseFixRel")
                                    await set_symptom_cause_fix_properties(
                                        relationship_instance, node, target_node
                                    )
                                elif (
                                    rel_class == graph_db_models.AsyncPreviousVersionRel
                                ):
                                    # print("BR 4: rel_class = AsyncPreviousVersionRel")
                                    await set_previous_version_properties(
                                        relationship_instance, node, target_node
                                    )
                                elif (
                                    rel_class == graph_db_models.AsyncPreviousMessageRel
                                ):
                                    # print("BR 4: rel_class = AsyncPreviousMessageRel")
                                    await set_previous_message_properties(
                                        relationship_instance, node, target_node
                                    )
                                elif (
                                    rel_class == graph_db_models.AsyncAffectsProductRel
                                ):
                                    # print("BR 5: rel_class = AsyncAffectsProductRel")
                                    await set_affects_product_properties(
                                        relationship_instance, node, target_node
                                    )
                                elif (
                                    rel_class
                                    == graph_db_models.AsyncHasUpdatePackageRel
                                ):
                                    # print("BR 6: rel_class = AsyncHasUpdatePackageRel")
                                    await set_has_update_package_properties(
                                        relationship_instance, node, target_node
                                    )

                                elif rel_class == graph_db_models.AsyncReferencesRel:
                                    # print("BR 7: rel_class = AsyncReferencesRel")
                                    await set_references_properties(
                                        relationship_instance, node, target_node
                                    )
                            else:
                                # print(
                                #     f"Relationship already exists: {node.node_id} -{rel_type}-> {target_node.node_id}"
                                # )
                                pass
        else:
            print(f"BR 9: shouldn't be here: node_type_str: {node_type_str}")


# END BUILD RELATIONSHIP FUNCTION ================================================


# BEGIN RELATIONSHIP SETTER FUNCTIONS ============================================


async def calculate_confidence(source_node, target_node):
    """
    Calculate the confidence level for a SymptomCauseFix relationship.
    This function determines how confident we are about the relationship between a symptom, cause, or fix.
    """
    # Extract relevant properties
    source_reliability = getattr(source_node, "reliability", "MEDIUM")
    target_reliability = getattr(target_node, "reliability", "MEDIUM")

    # Simple average as a placeholder
    confidence = f"{source_reliability}-{target_reliability}"
    return confidence


async def generate_description(source_node, target_node):
    """
    Generate a description for a SymptomCauseFix relationship.
    This function creates a brief explanation of how the source and target nodes are related.
    """
    source_name = getattr(source_node, "node_label", "Unknown")
    target_name = getattr(target_node, "node_label", "Unknown")
    return f"Relationship between {source_name} and {target_name}"


async def get_reported_date(source_node, target_node):
    """
    Determine the reported date for a SymptomCauseFix relationship.
    This function decides which date to use as the reported date for the relationship.
    """
    # source_date = getattr(source_node, "created_date", None)
    target_date = getattr(target_node, "created_on", None)
    return target_date or datetime.now().strftime("%Y-%m-%d")


async def calculate_impact_rating(source_node, target_node):
    """
    Calculate the impact level for an AffectsProduct relationship.
    This function determines how severely a product is affected.
    """
    source_impact = getattr(source_node, "impact_type", "NIT")
    source_severity = getattr(source_node, "severity_type", "NST")
    impact_rating = f"{source_impact}-{source_severity}"
    return impact_rating


async def get_affected_versions(source_node, target_node):
    """
    Determine the affected versions for an AffectsProduct relationship.
    This function lists all product versions affected by an issue.
    """
    return getattr(target_node, "build_numbers", [])


async def get_patched_version(source_node, target_node):
    """
    Determine the patched version for an AffectsProduct relationship.
    This function identifies the version where the issue was resolved.
    If KBArticle, there is no post_type
    """
    patched_version = "No Details"
    remediation_level = getattr(source_node, "post_type", "")
    if remediation_level:
        if "official_fix" in remediation_level:
            patched_version = source_node.revision

    return patched_version


async def get_release_date(source_node, target_node):
    """
    Determine the release date for a HasUpdatePackage relationship.
    This function decides which date to use as the release date for the update package.
    """
    return getattr(
        target_node, "published", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


async def has_cumulative(source_node, target_node):
    """
    Determine if an update package is cumulative for a HasUpdatePackage relationship.
    This function checks if the update includes all previous updates.
    """
    downloadable_packages_raw = target_node.downloadable_packages
    # is downloadable_packages a string
    if isinstance(downloadable_packages_raw, str):
        # print(f"downloadable_packages type: {type(downloadable_packages_raw)}")
        try:
            data = json.loads(downloadable_packages_raw)
        except json.JSONDecodeError:
            return False
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            for item in data:
                if "update_type" in item and "cumulative" in item["update_type"]:
                    print(item["update_type"])
                    return True

    # Is downloadable_packages a list of dictionaries?
    if (
        isinstance(downloadable_packages_raw, list)
        and downloadable_packages_raw
        and all(isinstance(item, dict) for item in downloadable_packages_raw)
    ):
        print(
            f"downloadable_packages type: {type(downloadable_packages_raw)} - All Dicts"
        )
        # cannot pass list to json.loads()
        for item in downloadable_packages_raw:
            # print(f"type of downloadable: {type(item)}")
            if "update_type" in item and "cumulative" in item["update_type"]:
                # print(item["update_type"])
                return True
    elif (
        isinstance(downloadable_packages_raw, list)
        and downloadable_packages_raw
        and all(isinstance(item, str) for item in downloadable_packages_raw)
    ):
        # downloadable_packages is a list of json strings
        # print(f"downloadable_packages type: {type(downloadable_packages_raw)} - List")
        # print(f"downloadable_packages_raw: len({len(downloadable_packages_raw)})")

        for item in downloadable_packages_raw:

            data = json.loads(item)

            if "update_type" in data and data["update_type"].lower() == "cumulative":

                return True

    else:
        print(f"downloadable_packages is misbehaving: {downloadable_packages_raw}")

    return False


async def has_dynamic(source_node, target_node):
    """
    Determine if an update package is cumulative for a HasUpdatePackage relationship.
    This function checks if the update includes all previous updates.
    """
    downloadable_packages_raw = target_node.downloadable_packages

    # Is downloadable_packages a string?
    if isinstance(downloadable_packages_raw, str):
        # print(f"downloadable_packages type: {type(downloadable_packages_raw)}")
        try:
            data = json.loads(downloadable_packages_raw)
        except json.JSONDecodeError:
            return False
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            for item in data:
                if "update_type" in item and "cumulative" in item["update_type"]:

                    return True

    # Is downloadable_packages a list of dictionaries?
    if (
        isinstance(downloadable_packages_raw, list)
        and downloadable_packages_raw
        and all(isinstance(item, dict) for item in downloadable_packages_raw)
    ):
        # print(
        #     f"downloadable_packages type: {type(downloadable_packages_raw)} - All Dicts"
        # )
        # cannot pass list to json.loads()
        print(f"downloadable_packages_raw: len({len(downloadable_packages_raw)})")
        for item in downloadable_packages_raw:
            # print(f"type of downloadable: {type(item)}")
            if "update_type" in item and "cumulative" in item["update_type"]:
                # print(item["update_type"])
                return True
    elif (
        isinstance(downloadable_packages_raw, list)
        and downloadable_packages_raw
        and all(isinstance(item, str) for item in downloadable_packages_raw)
    ):
        # Is downloadable_packages a list of strings?
        # print(f"downloadable_packages type: {type(downloadable_packages_raw)} - List")
        # print(f"downloadable_packages_raw: len({len(downloadable_packages_raw)})")
        for item in downloadable_packages_raw:
            data = json.loads(item)
            # print(f"did json.loads() work? {isinstance(data, dict)}\n{data.keys()}")
            if "update_type" in data and "dynamic" in data.get("update_type").lower():

                return True

    # print("has_dynamic: returns -> False")
    return False


async def calculate_version_difference(source_node, target_node):
    """
    Calculate the version difference for a PreviousVersion relationship.
    This function determines the difference between two versions.
    """
    source_version_str = getattr(source_node, "revision", "")
    target_version_str = getattr(target_node, "revision", "")

    try:
        # Convert the string to a float
        source_value = float(source_version_str)
        target_value = float(target_version_str)
        # Round the float to two decimal places
        source_value = round(source_value, 2)
        target_value = round(target_value, 2)

    except ValueError as ve:
        # Handle the error if the conversion fails
        print(
            f"Error: '{source_version_str}' or '{target_version_str}' is not a valid float string.\n{ve}"
        )
        source_value = source_version_str[:3]
        target_value = target_version_str[:3]

    return f"Difference between {source_value} and {target_value}"


async def generate_changes_summary(source_node, target_node):
    """
    Generate a summary of changes for a PreviousVersion relationship.
    This function creates a brief overview of changes between versions.
    """
    return getattr(source_node, "description", "")


async def calculate_relevance_score(source_node, target_node):
    """
    Calculate the relevance score for a References relationship.
    This function determines how relevant the reference is to the source.
    """
    if isinstance(source_node, graph_db_models.Fix):
        return 90
    if isinstance(source_node, graph_db_models.ProductBuild):
        return 100
    if isinstance(source_node, graph_db_models.KBArticle):
        return 100
    if isinstance(source_node, graph_db_models.PatchManagementPost):
        return 80


async def extract_context(source_node, target_node):
    """
    Extract the context for a References relationship.
    This function provides context about how the reference is related to the source.
    """
    if isinstance(source_node, graph_db_models.Fix):
        return "Referenced in a Fix"
    if isinstance(source_node, graph_db_models.ProductBuild):
        return "Referenced in a product build generated by Microsoft"
    if isinstance(source_node, graph_db_models.KBArticle):
        return "Referenced in a KB Article published by Microsoft"
    if isinstance(source_node, graph_db_models.PatchManagementPost):
        return "Referenced by a community member"


async def extract_cited_section(source_node, target_node):
    """
    Extract the cited section for a References relationship.
    This function identifies the specific section of the target that is referenced.
    """
    return "Whole document"


async def calculate_severity(source_node, target_node):
    # Implement your logic here to calculate severity based on source and target node properties
    # Symptoms, Causes, Fixes & affecting products
    if hasattr(source_node, "severity_type"):
        return source_node.severity_type
    return "NST"  # default value


async def get_previous_message_id(source_node, target_node):
    if hasattr(target_node, "previous_id"):
        return target_node.previous_id
    return None


# Updated utility functions
async def set_symptom_cause_fix_properties(relationship, source_node, target_node):
    print(
        f"set_symptom_cause_fix_properties called with source_node: {type(source_node).__name__}, target_node: {type(target_node).__name__}"
    )
    if relationship:
        severity = await calculate_severity(source_node, target_node)
        reliability = await calculate_confidence(source_node, target_node)
        description = await generate_description(source_node, target_node)
        reported_date = await get_reported_date(source_node, target_node)

        source_node_type = source_node.__class__.__name__
        target_node_type = target_node.__class__.__name__
        rel_name = next(
            name
            for name, (target, _, rel_class) in graph_db_models.RELATIONSHIP_MAPPING[
                source_node_type
            ].items()
            if target == target_node.__class__.__name__
            and rel_class == graph_db_models.AsyncSymptomCauseFixRel
        )
        if rel_name:
            await set_relationship_type(relationship, source_node_type, rel_name)
            relationship.severity = severity
            relationship.reliability = reliability
            relationship.description = description
            relationship.reported_date = reported_date
            
            await relationship.save()


async def set_affects_product_properties(relationship, source_node, target_node):
    # print(
    #     f"set_affects_product_properties called with source_node: {type(source_node).__name__}, target_node: {type(target_node).__name__}"
    # )
    if relationship:
        source_node_type = source_node.__class__.__name__
        target_node_type = target_node.__class__.__name__
        if source_node_type == "PatchManagementPost":
            # print("Detected PatchPost -[AFFECTS]-> Product")
            impact_level = ""
            severity = ""
            affected_versions = source_node.build_numbers
            patched_version = ""

            rel_name = next(
                name
                for name, (
                    target,
                    _,
                    rel_class,
                ) in graph_db_models.RELATIONSHIP_MAPPING[source_node_type].items()
                if target == target_node_type
                and rel_class == graph_db_models.AsyncAffectsProductRel
            )
        else:
            impact_level = await calculate_impact_rating(source_node, target_node)
            severity = await calculate_severity(source_node, target_node)
            affected_versions = await get_affected_versions(source_node, target_node)
            patched_version = await get_patched_version(source_node, target_node)

            source_node_type = source_node.__class__.__name__
            target_node_type = target_node.__class__.__name__
            rel_name = next(
                name
                for name, (
                    target,
                    _,
                    rel_class,
                ) in graph_db_models.RELATIONSHIP_MAPPING[source_node_type].items()
                if target == target_node_type
                and rel_class == graph_db_models.AsyncAffectsProductRel
            )
        if rel_name:
            await set_relationship_type(relationship, source_node_type, rel_name)
            relationship.impact_level = impact_level
            relationship.severity = severity
            relationship.affected_versions = affected_versions
            relationship.patched_in_version = patched_version
            await relationship.save()
    # else:
    #     print("No relationship -> set_affects_product()")


async def set_has_update_package_properties(relationship, source_node, target_node):
    # print(
    #     f"set_has_update_package_properties called with source_node: {type(source_node).__name__}, target_node: {type(target_node).__name__}"
    # )
    if relationship:
        release_date = await get_release_date(source_node, target_node)
        cumulative = await has_cumulative(source_node, target_node)
        dynamic = await has_dynamic(source_node, target_node)
        # print(f"set_update() received: {cumulative} & {dynamic}")
        source_node_type = source_node.__class__.__name__
        target_node_type = target_node.__class__.__name__
        rel_name = next(
            (
                name
                for name, (
                    target,
                    _,
                    rel_class,
                ) in graph_db_models.RELATIONSHIP_MAPPING[source_node_type].items()
                if target == target_node_type
                and rel_class == graph_db_models.AsyncHasUpdatePackageRel
            ),
            None,  # Default value if no match is found
        )
        if rel_name:
            await set_relationship_type(relationship, source_node_type, rel_name)
            relationship.release_date = release_date
            relationship.has_cumulative = cumulative
            relationship.has_dynamic = dynamic
            # print(
            #     f"Before save: release_date={relationship.release_date}, has_cumulative={relationship.has_cumulative}, has_dynamic={relationship.has_dynamic}"
            # )
            await relationship.save()
            # print(
            #     f"After save: release_date={relationship.release_date}, has_cumulative={relationship.has_cumulative}, has_dynamic={relationship.has_dynamic}"
            # )
        # else:
        #     print(
        #         f"No matching relationship found for {source_node_type} -> {target_node_type}"
        #     )


async def set_previous_version_properties(relationship, source_node, target_node):
    # print(
    #     f"set_previous_version_properties called with source_node: {type(source_node).__name__}, target_node: {type(target_node).__name__}"
    # )
    if relationship:
        # print("set_previous: relationship exists...gathering properties")
        version_difference = await calculate_version_difference(
            source_node, target_node
        )
        changes_summary = await generate_changes_summary(source_node, target_node)

        source_node_type = source_node.__class__.__name__
        target_node_type = target_node.__class__.__name__
        rel_name = next(
            (
                name
                for name, (
                    target,
                    _,
                    rel_class,
                ) in graph_db_models.RELATIONSHIP_MAPPING[source_node_type].items()
                if target == target_node_type
                and rel_class == graph_db_models.AsyncPreviousVersionRel
            ),
            None,  # Default value if no match is found
        )
        if rel_name:
            await set_relationship_type(relationship, source_node_type, rel_name)
            relationship.version_difference = version_difference
            relationship.changes_summary = changes_summary
            relationship.previous_version_id = target_node.node_id
            # print(
            #     f"Before save: version_difference={relationship.version_difference}, changes_summary={relationship.changes_summary}"
            # )
            await relationship.save()
            # print(
            #     f"After save: version_difference={relationship.version_difference}, changes_summary={relationship.changes_summary}"
            # )
        # else:
        #     print(
        #         f"No matching relationship found for {source_node_type} -> {target_node_type}"
        #     )
    # else:
    #     print("Relationship is None")


async def set_previous_message_properties(relationship, source_node, target_node):
    # print(
    #     f"set_previous_message_properties called with source_node: {type(source_node).__name__}, target_node: {type(target_node).__name__}"
    # )
    if relationship:
        # print("set_previous_message: relationship exists...gathering properties")
        previous_id = target_node.node_id

        source_node_type = source_node.__class__.__name__
        target_node_type = target_node.__class__.__name__
        rel_name = next(
            (
                name
                for name, (
                    target,
                    _,
                    rel_class,
                ) in graph_db_models.RELATIONSHIP_MAPPING[source_node_type].items()
                if target == target_node_type
                and rel_class == graph_db_models.AsyncPreviousMessageRel
            ),
            None,  # Default value if no match is found
        )
        if rel_name:
            await set_relationship_type(relationship, source_node_type, rel_name)
            relationship.previous_id = previous_id
            await relationship.save()
    # else:
    #     print("Relationship is None")


async def set_references_properties(relationship, source_node, target_node):
    print(
        f"set_references_properties called with source_node: {type(source_node).__name__}, target_node: {type(target_node).__name__}"
    )
    if relationship:
        relevance_score = await calculate_relevance_score(source_node, target_node)
        context = await extract_context(source_node, target_node)
        cited_section = await extract_cited_section(source_node, target_node)

        print(f"relationship type: {type(relationship)}")

        relationship.relevance_score = relevance_score
        relationship.context = context
        relationship.cited_section = cited_section

        source_node_type = source_node.__class__.__name__
        target_node_type = target_node.__class__.__name__
        rel_name = next(
            name
            for name, (target, _, rel_class) in graph_db_models.RELATIONSHIP_MAPPING[
                source_node_type
            ].items()
            if target == target_node_type
            and rel_class == graph_db_models.AsyncReferencesRel
        )
        await set_relationship_type(relationship, source_node_type, rel_name)

        # Don't forget to save the changes
        await relationship.save()


async def set_relationship_type(relationship, source_node_type, rel_name):
    # print(
    #     f"set_relationship_type called with source_node_type: {source_node_type}, rel_name: {rel_name}"
    # )
    """
    Set the relationship_type property for a given relationship based on the RELATIONSHIP_MAPPING.

    Args:
    relationship: The relationship object to set the type for.
    source_node_type (str): The type of the source node.
    rel_name (str): The name of the relationship as defined in the node model.
    """
    if (
        source_node_type in graph_db_models.RELATIONSHIP_MAPPING
        and rel_name in graph_db_models.RELATIONSHIP_MAPPING[source_node_type]
    ):
        rel_type = graph_db_models.RELATIONSHIP_MAPPING[source_node_type][rel_name][1]
        relationship.relationship_type = rel_type
    else:
        # Default to the rel_name if not found in RELATIONSHIP_MAPPING
        relationship.relationship_type = rel_name.upper()


# END RELATIONSHIP SETTER FUNCTIONS ==============================================

# BEGIN RELATIONSHIP TRACKER =====================================================

db = AsyncDatabase()
NeomodelConfig.DATABASE_URL = get_graph_db_uri()


class RelationshipTracker:
    def __init__(self):
        self.existing_relationships: Dict[str, Set[Tuple[str, str, str]]] = {}
        self.batch_relationships: Dict[str, Set[Tuple[str, str, str]]] = {}

    async def fetch_existing_relationships(
        self, node_types: List[str], node_ids: List[str]
    ):
        # print("Start fetch_existing")
        # print(f"node_types: {node_types} num ids: {len(node_ids)}")
        query = """
        UNWIND $node_types AS node_type
        MATCH (n)
        WHERE n.node_id IN $node_ids AND any(label IN labels(n) WHERE label = node_type)
        MATCH (n)-[r]->(m)
        RETURN n, type(r), m, labels(m) AS target_labels, node_type
        """
        results, _ = await db.cypher_query(
            query, {"node_types": node_types, "node_ids": node_ids}
        )
        print(f"Number of relationships fetched: {len(results)}")
        for source, rel_type, target, target_labels, node_type in results:
            source_label = list(source.labels)[0] if source.labels else "UnknownLabel"
            target_label = list(target.labels)[0] if target.labels else "UnknownLabel"
            source_node_id = source["node_id"]
            target_node_id = target["node_id"]

            # print(
            #     f"Relationship: {source_label}({source_node_id}) -[{rel_type}]-> {target_label}({target_node_id})"
            # )
        # time.sleep(10)
        for source, rel_type, target, target_labels, node_type in results:
            source_node = self._convert_to_neomodel_node(source, node_type)
            target_node_type = target_labels[
                0
            ]  # Assuming the first label is the node type.
            target_node = self._convert_to_neomodel_node(target, target_node_type)

            # Add the relationship to the tracker
            if node_type not in self.existing_relationships:
                self.existing_relationships[node_type] = set()

            key = self._create_relationship_key(source_node, rel_type, target_node)
            self.existing_relationships[node_type].add(key)
        # print("in fetch, checking existing rels added...")
        # for node_type, relationships in self.existing_relationships.items():
        #     print(
        #         f"Node type: {node_type}, Total relationships: {len(relationships)}\n{relationships}"
        #     )
        # print("if nothing printed for Node type:. Total relationships. take note")
        # time.sleep(15)

    def _convert_to_neomodel_node(self, neo4j_node, node_type):
        """
        Converts a native Neo4j Node to a corresponding Neomodel AsyncStructuredNode instance.
        """
        properties = dict(neo4j_node._properties)

        # Map node_type to the appropriate Neomodel class
        if node_type == "Product":
            return graph_db_models.Product(**properties)
        elif node_type == "ProductBuild":
            return graph_db_models.ProductBuild(**properties)
        elif node_type == "MSRCPost":
            return graph_db_models.MSRCPost(**properties)
        elif node_type == "Symptom":
            return graph_db_models.Symptom(**properties)
        elif node_type == "Cause":
            return graph_db_models.Cause(**properties)
        elif node_type == "Fix":
            return graph_db_models.Fix(**properties)
        elif node_type == "FAQ":
            return graph_db_models.FAQ(**properties)
        elif node_type == "Tool":
            return graph_db_models.Tool(**properties)
        elif node_type == "KBArticle":
            return graph_db_models.KBArticle(**properties)
        elif node_type == "UpdatePackage":
            return graph_db_models.UpdatePackage(**properties)
        elif node_type == "PatchManagementPost":
            return graph_db_models.PatchManagementPost(**properties)
        else:
            raise ValueError(f"Unsupported node type: {node_type}")

    def _create_relationship_key(self, source, rel_type, target) -> Tuple[str, ...]:
        """
        Create a unique key to track the relationships, based on node types and properties.
        """
        # print("_create_relationship_key")
        # print(f"source type: {type(source)}\ntarget type: {type(target)}")
        if isinstance(source, graph_db_models.Product) and isinstance(
            target, graph_db_models.ProductBuild
        ):
            # Generate a unique key for Product -> ProductBuild based on the 7 properties
            unique_string = (
                f"{source.product_name}_{source.product_architecture}_{source.product_version}_"
                f"{target.product_name}_{target.product_architecture}_{target.product_version}_{target.cve_id}"
            )
            # Create a hash of the string to keep the key consistent and simple
            unique_hash = hashlib.sha256(unique_string.encode()).hexdigest()

            # print(f"Product -> ProductBuild Key (hash): {unique_hash}")
            # Return the key in the same format as other relationships
            return (source.node_id, rel_type, unique_hash)
        if isinstance(source, graph_db_models.Product) and isinstance(
            target, graph_db_models.Product
        ):
            print(
                "Warning: Attempting to create a Product -> Product relationship, which is unexpected."
            )
        else:
            # print(f"Node({source.node_label}) -> Node({target.node_label})")
            # print(
            #     f"_create_relationship_key: {(source.node_id, rel_type, target.node_id)}"
            # )
            return (source.node_id, rel_type, target.node_id)

    def relationship_exists(self, source: Any, rel_type: str, target: Any) -> bool:
        node_type = type(source).__name__
        key = self._create_relationship_key(source, rel_type, target)
        # print(f"relationship_exists key: {key}")
        exists_in_existing = key in self.existing_relationships.get(node_type, set())
        exists_in_batch = key in self.batch_relationships.get(node_type, set())
        # print(
        #     f"Exists in existing relationships: {exists_in_existing}, Exists in batch relationships: {exists_in_batch}"
        # )

        return exists_in_existing or exists_in_batch

    def add_relationship(self, source: Any, rel_type: str, target: Any):
        node_type = type(source).__name__
        key = self._create_relationship_key(source, rel_type, target)
        # print(f"add_relationship: {node_type}:{key}")
        if node_type not in self.batch_relationships:
            self.batch_relationships[node_type] = set()
        self.batch_relationships[node_type].add(key)
        # print(f"batch_relationships updated: {self.batch_relationships[node_type]}\n")


# =================== END RELATIONSHIP TRACKER ==========================

# =================== BEGIN MIGRATION RELATIONSHIP BUILD ================


async def build_migration_relationships(
    nodes_dict: Dict[str, List[graph_db_models.AsyncStructuredNode]],
    relationships: List[Dict],
):
    print("Begin building migration relationships")

    # Build a mapping from node_id to node instances
    node_id_to_node = {}
    for node_list in nodes_dict.values():
        for node in node_list:
            node_id_to_node[node.node_id] = node

    # Build RELATIONSHIP_TYPE_MAPPING
    RELATIONSHIP_TYPE_MAPPING = {}
    for node_type_str, rels in graph_db_models.RELATIONSHIP_MAPPING.items():
        for rel_name, rel_info in rels.items():
            target_model, rel_type, rel_class = rel_info
            key = (node_type_str, rel_type)
            RELATIONSHIP_TYPE_MAPPING[key] = (rel_name, target_model, rel_class)

    for rel in relationships:
        source_id = rel["source_id"]
        target_id = rel["target_id"]
        rel_type = rel["type"]
        properties = rel.get("properties", {})

        source_node = node_id_to_node.get(source_id)
        target_node = node_id_to_node.get(target_id)

        if not source_node:
            print(f"Source node with ID {source_id} not found.")
            continue
        if not target_node:
            print(f"Target node with ID {target_id} not found.")
            continue

        source_node_type = source_node.__class__.__name__
        key = (source_node_type, rel_type)

        if key not in RELATIONSHIP_TYPE_MAPPING:
            print(f"No relationship mapping found for {key}")
            continue

        rel_name, target_model, rel_class = RELATIONSHIP_TYPE_MAPPING[key]

        if target_node.__class__.__name__ != target_model:
            print(
                f"Target node type mismatch: expected {target_model}, got {target_node.__class__.__name__}"
            )
            continue

        relationship = getattr(source_node, rel_name)

        # Check if the relationship already exists
        existing_rel = await relationship.is_connected(target_node)
        if existing_rel:
            # print(
            #     f"Relationship already exists: {source_id} -[{rel_type}]-> {target_id}"
            # )
            continue

        # Disconnect existing relationships if it's a one-to-one relationship
        if issubclass(rel_class, graph_db_models.AsyncOneToOneRel):
            await relationship.disconnect_all()

        # Connect the new relationship
        await relationship.connect(target_node)

        # Fetch the relationship instance to set properties
        rel_instance = await relationship.relationship(target_node)

        # Set properties from the properties dict
        if properties:
            for prop_name, prop_value in properties.items():
                setattr(rel_instance, prop_name, prop_value)
            await rel_instance.save()

        # Set properties for custom relationship classes
        if rel_class == graph_db_models.AsyncSymptomCauseFixRel:
            await set_symptom_cause_fix_properties(
                rel_instance, source_node, target_node
            )
        elif rel_class == graph_db_models.AsyncPreviousVersionRel:
            await set_previous_version_properties(
                rel_instance, source_node, target_node
            )
        elif rel_class == graph_db_models.AsyncPreviousMessageRel:
            await set_previous_message_properties(
                rel_instance, source_node, target_node
            )
        elif rel_class == graph_db_models.AsyncAffectsProductRel:
            await set_affects_product_properties(rel_instance, source_node, target_node)
        elif rel_class == graph_db_models.AsyncHasUpdatePackageRel:
            await set_has_update_package_properties(
                rel_instance, source_node, target_node
            )
        elif rel_class == graph_db_models.AsyncReferencesRel:
            await set_references_properties(rel_instance, source_node, target_node)

        print(f"Created relationship: {source_id} -[{rel_type}]-> {target_id}")


# =================== END MIGRATION RELATIONSHIP BUILD ==================


async def test_neomodel_get_or_none():
    db = AsyncDatabase()
    db_manager = GraphDatabaseManager(db)

    print("created database")
    NeomodelConfig.DATABASE_URL = get_graph_db_uri()
    print("created config")

    print("testing begins")
    # Initialize the ProductService for a single product
    product_build_service = ProductBuildService(db_manager)
    try:
        # Replace these with actual values from your dataset
        test_properties = {
            "node_id": "86ccde33-1c3f-7a68-a842-2b24dc8a2b8f",
            "product_name": "windows_10",
            "product_version": "NV",
            "product_architecture": "x86",
        }

        print(
            "\nAttempting to retrieve a ProductBuild node using get_or_none() with test_properties:"
        )
        print(f"Test properties: {test_properties}")

        print("\nTesting direct access using ProductBuild.nodes.get_or_none()")
        try:
            node = await graph_db_models.ProductBuild.nodes.get_or_none(
                node_id=test_properties["node_id"]
            )
            if node:
                print("Node retrieved successfully (direct):", node.__properties__)
            else:
                print("No node found (direct).")
        except Exception as e:
            print(f"Error during direct test: {e}")

        # Test case 2: Using ProductBuildService to access the model and get_or_none()
        print("\nTesting access via ProductBuildService.model.nodes.get_or_none()")
        try:
            # Here we're calling the internal model's nodes.get_or_none() via the service instance
            node = await product_build_service.model.nodes.get_or_none(
                node_id=test_properties["node_id"]
            )
            if node:
                print("Node retrieved successfully (via service):", node.__properties__)
            else:
                print("No node found (via service).")
        except Exception as e:
            print(f"Error during service test: {e}")

        # Further test with more complex unique property combinations if needed
        # You can adapt this part according to the unique properties in your model
        print(
            "\nAttempting to retrieve a ProductBuild node using get_or_none() with product_name, product_version, and product_architecture:"
        )
        node = await graph_db_models.ProductBuild.nodes.get_or_none(
            product_name=test_properties["product_name"],
            product_version=test_properties["product_version"],
            product_architecture=test_properties["product_architecture"],
        )
        if node:
            print(f"Node retrieved successfully with multiple properties: {node}")
        else:
            print("No node found with the given properties.")

    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback

        traceback.print_exc()


async def main():
    from datetime import datetime

    # install_labels(graph_db_models.Product)
    print("past install_labels")
    # testing constraints
    graph_db_uri = f"{credentials.protocol}://{credentials.host}:{credentials.port}"
    graph_db_auth = (credentials.username, credentials.password)
    db_status = ensure_graph_db_constraints_exist(
        graph_db_uri, graph_db_auth, graph_db_settings
    )
    if db_status["status"] != "success":
        print(f"Database setup failed: {db_status['message']}")
        print(f"Constraints status: {db_status['constraints_status']}")
    db = AsyncDatabase()
    db_manager = GraphDatabaseManager(db)

    print("created database")
    NeomodelConfig.DATABASE_URL = db_uri
    print("created config")

    print("testing begins")
    # Initialize the ProductService for a single product
    product_service = ProductService(db_manager)
    print(
        "ProductService created. Handles database interactions on behalf of the node instance"
    )
    # Example data for a new Product
    product_data = {
        "product_name": "windows_10",
        "product_architecture": "x64",
        "product_version": "22H2",
        "description": "Windows 11 Version 22H2 for x64-based Systems",
        "node_label": "windows_11 x64 22H2",
    }
    products_data = [
        {
            "product_name": "windows_11",
            "product_architecture": "x64",
            "product_version": "22H2",
            "description": "Windows 11 Version 22H2 for x64-based Systems",
            "node_label": "windows_11 x64 22H2",
        },
        {
            "product_name": "windows_11",
            "product_architecture": "x64",
            "product_version": "23H2",
            "description": "Windows 11 Version 23H2 for x64-based Systems",
            "node_label": "windows_11 x64 23H2",
        },
        {
            "product_name": "windows_11",
            "product_architecture": "x64",
            "product_version": "24H2",
            "description": "Windows 11 Version 24H2 for x64-based Systems",
            "node_label": "windows_11 x64 24H2",
        },
        {
            "product_name": "windows_11",
            "product_architecture": "x64",
            "product_version": "25H2",
            "description": "Windows 11 Version 25H2 for x64-based Systems",
            "node_label": "windows_11 x64 25H2",
        },
        {
            "product_name": "windows_11",
            "product_architecture": "x64",
            "product_version": "23H2",
            "description": "Windows 11 Version 23H2 for x64-based Systems",
            "node_label": "windows_11 x64 23H2",
        },
    ]
    new_nodes = await product_service.bulk_create(products_data)
    for node in new_nodes:
        print(node)

    # Create the product

    product, message, status = await product_service.create(**product_data)

    if status == 201:
        print(f"Product created")
        # Set build numbers (example data)
        if product:
            product.set_build_numbers([[10, 0, 19041, 450], [10, 0, 19042, 450]])
            print(f"Build Numbers: {product.get_build_numbers()}")
            await product.save()
    else:
        print(f"Error: {message}")
        product = await product_service.get("86b28b3e-c61e-4fa1-8445-158dd8dc06c6")

    print(f"type: {type(product)}\n{product}")

    product_build_data = {
        "node_id": "fdbb56d3-d1aa-6819-59bd-dc326d0b64c0",
        "product": "Windows 11 Version 22H2 for x64-based Systems",
        "product_name": "windows_11",
        "product_architecture": "x64",
        "product_version": "22H2",
        "node_label": "ProductBuild [10,0,19045,4046]",
        "build_number": [10, 0, 19045, 4046],
        "cve_id": "CVE-2024-38027",
        "kb_id": "kb5040442",
        "published": datetime.now(),
        "article_url": "https://support.microsoft.com/help/5034763",
        "cve_url": "https://msrc.microsoft.com/update-guide/vulnerability/CVE-2024-21406",
        "impact_type": "spoofing",
        "severity_type": "important",
    }
    # create product build service for a single product_build
    product_build_service = ProductBuildService(db_manager)
    product_build, message, status = await product_build_service.create(
        **product_build_data
    )
    if status == 201:
        print(f"Product Build created")
        # Set build numbers (example data)
        if product_build:
            # no post creation operations yet
            await product_build.save()
    else:
        print(f"Error: {message}")
        product_build = await product_build_service.get(
            "fdbb56d3-d1aa-6819-59bd-dc326d0b64c0"
        )
        if product_build:
            print(f"product_build found: {product_build}")
        else:
            print("product_build not found")

    # Create a relationship between the Product and the ProductBuild
    if product:
        existing_product_rel_build = await product.has_builds.relationship(
            product_build
        )
        if not existing_product_rel_build:
            await product.has_builds.connect(
                product_build,
                {
                    "relationship_type": "HAS_BUILD",
                },
            )
        else:
            print("Relationship between product and build exists.")
        # Update the product description
        response = await product_service.update(
            product.node_id,
            description="UPDATE UPDATE UPDATE description for Windows 10",
        )
        print(f"Updated Product response:\n{response}")
    else:
        product = await product_service.get("1f635c63-45e2-457e-a808-8df84951c5b1")
    # create kb article
    kb_article_service = KBArticleService(db_manager)

    kb_article_data = {
        "node_id": "1be277b3-2092-dc43-9dab-add5fd0eaeb1",
        "kb_id": "5040442",
        "build_number": [10, 0, 22621, 3880],
        "node_label": "KB5040442",
        "published": datetime.strptime("2024-07-09 12:00:00", "%Y-%m-%d %H:%M:%S"),
        "product_build_id": "fdbb56d3-d1aa-6819-59bd-dc326d0b64c0",
        "article_url": "https://support.microsoft.com/help/5040442",
        "embedding": [],
        "text": """ \
You're invited to try Microsoft 365 for free
Unlock now
Release Date:
7/9/2024
Version:
OS Builds 22621.3880 and 22631.3880
NEW
 07/09/24---
END OF SERVICE NOTICE
---
IMPORTANT
Home and Pro editions of Windows 11, version 22H2 will reach end of service on October 8, 2024. Until then, these editions will only receive security updates. They will not receive non-security, preview updates. To continue receiving security and non-security updates after October 8, 2024, we recommend that you update to the latest version of Windows.
Note
We will continue to support Enterprise and Education editions after October 8, 2024. For information about Windows update terminology, see the article about the types of Windows updates and the monthly quality update types. For an overview of Windows 11, version 23H2, see its update history page.
Note
Follow
@WindowsUpdate
to find out when new content is published to the Windows release health dashboard.Highlights
Below is a summary of the key issues that this update addresses when you install this KB. If there are new features, it lists them as well.
Taskbar (known issue)
 You might not be able to view or interact with the taskbar after you install KB5039302. This issue occurs on devices that run the Windows N edition. This edition is like other editions but lacks most media-related tools. The issue also occurs if you turn off “Media Features” from the Control Panel.
Improvements
Note:
To view the list of addressed issues, click or tap the OS name to expand the collapsible section.
Windows 11, version 23H2
        """,
        "title": "July 9, 2024—KB5040442 (OS Builds 22621.3880 and 22631.3880) - Microsoft Support",
    }

    kb_article, message, status = await kb_article_service.create(**kb_article_data)
    print(f"{status}: {message}")

    if kb_article:
        print(f"Created KB Article: {kb_article}")

    else:
        kb_article = await kb_article_service.get(
            "1be277b3-2092-dc43-9dab-add5fd0eaeb1"
        )
    # Check and create relationship between ProductBuild and KBArticle
    existing_product_build_rel_kb = await product_build.references_kbs.relationship(
        kb_article
    )
    if not existing_product_build_rel_kb:
        await product_build.references_kbs.connect(
            kb_article,
            {
                "relationship_type": "REFERENCES",
            },
        )

    # Check and create relationship between KBArticle and Product
    existing_kb_article_rel_product = await kb_article.affects_product.relationship(
        product
    )
    if not existing_kb_article_rel_product:
        await kb_article.affects_product.connect(
            product,
            {
                "relationship_type": "AFFECTS_PRODUCT",
            },
        )
    # create msrc post
    msrc_post_data = {
        "node_id": "6f5403e0-d9b1-804a-67db-fa99ba5f7b44",
        "embedding": [],
        "metadata": {"key": "value"},
        "published": datetime.strptime("2024-07-09 00:00:00", "%Y-%m-%d %H:%M:%S"),
        "revision": "1.0",
        "title": "CVE-2024-38027 Windows Line Printer Daemon Service Denial of Service Vulnerability",
        "description": "Information published.",
        "node_label": "CVE-2024-38027",
        "source": "https://msrc.microsoft.com/update-guide/vulnerability/CVE-2024-38027",
        "impact_type": "dos",
        "severity_type": "critical",
        "post_type": "Solution provided",
        "attack_complexity": "low",
        "attack_vector": "network",
        "exploit_code_maturity": "Proof of Concept",
        "exploitability": "Functional",
        "post_id": "CVE-2024-38027",
        "summary": "Sample Summary",
        "build_numbers": [[10, 0, 22631, 3880], [10, 0, 22621, 3880]],
        "text": """ \
Security Vulnerability
Released: 9 Jul 2024
Assigning CNA:
Microsoft
CVE-2024-38027
Impact: Denial of Service
Max Severity: Important
Weakness:
CWE-400: Uncontrolled Resource Consumption
CVSS Source:
Microsoft
CVSS:3.1 6.5 / 5.7
Base score metrics: 6.5 / Temporal score metrics: 5.7
Base score metrics: 6.5 / Temporal score metrics: 5.7
Base score metrics
(8)
Attack Vector
This metric reflects the context by which vulnerability exploitation is possible. The Base Score increases the more remote (logically, and physically) an attacker can be in order to exploit the vulnerable component.
Adjacent
The vulnerable component is bound to the network stack, but the attack is limited at the protocol level to a logically adjacent topology. This can mean an attack must be launched from the same shared physical (e.g., Bluetooth or IEEE 802.11) or logical (e.g., local IP subnet) network, or from within a secure or otherwise limited administrative domain (e.g., MPLS, secure VPN to an administrative network zone)
Attack Complexity
This metric describes the conditions beyond the attacker's control that must exist in order to exploit the vulnerability. Such conditions may require the collection of more information about the target or computational exceptions. The assessment of this metric excludes any requirements for user interaction in order to exploit the vulnerability. If a specific configuration is required for an attack to succeed, the Base metrics should be scored assuming the vulnerable component is in that configuration.
Low
Specialized access conditions or extenuating circumstances do not exist. An attacker can expect repeatable success against the vulnerable component.

Privileges Required
This metric describes the level of privileges an attacker must possess before successfully exploiting the vulnerability.
None
The attacker is unauthorized prior to attack, and therefore does not require any access to settings or files to carry out an attack.
User Interaction
This metric captures the requirement for a user, other than the attacker, to participate in the successful compromise the vulnerable component. This metric determines whether the vulnerability can be exploited solely at the will of the attacker, or whether a separate user (or user-initiated process) must participate in some manner.
None
The vulnerable system can be exploited without any interaction from any user.
         """,
    }

    msrc_service = MSRCPostService(db_manager)
    msrc_post, message, status = await msrc_service.create(**msrc_post_data)
    print(f"{status}: {message}")
    if msrc_post:
        print(f"MSRC Post: {msrc_post}")

    else:
        msrc_post = await msrc_service.get("6f5403e0-d9b1-804a-67db-fa99ba5f7b44")

    # create multiple relationships
    existing_product_rel = await msrc_post.affects_products.relationship(product)

    if not existing_product_rel:
        await msrc_post.affects_products.connect(
            product,
            {
                "relationship_type": "AFFECTS_PRODUCT",
            },
        )
    else:
        print("Product relationship already exists between MSRCPost and Product.")

    # Check if the KBArticle relationship exists between MSRCPost and KBArticle
    existing_kb_rel = await msrc_post.has_kb_articles.relationship(kb_article)

    if not existing_kb_rel:
        await msrc_post.has_kb_articles.connect(
            kb_article,
            {
                "relationship_type": "HAS_KB",
            },
        )
    else:
        print("KBArticle relationship already exists between MSRCPost and KBArticle.")

    # ===========================
    # update package service test
    # ===========================
    update_package_service = UpdatePackageService(db_manager)

    update_package_data = {
        "node_id": "ff0c2b87-a977-e36d-0ff6-e67a123fea87",
        "package_type": "security_update",
        "package_url": "https://catalog.update.microsoft.com/v7/site/Search.aspx?q=KB5040442",
        "build_number": [10, 0, 19041, 508],
        "product_build_id": "fdbb56d3-d1aa-6819-59bd-dc326d0b64c0",
        "node_label": "UpdatePackage",
        "downloadable_packages": [
            {
                "package_name": "windows_10",
                "package_version": "21H2",
                "package_architecture": "x64",
                "update_type": "Cumulative",
                "file_size": "200MB",
                "install_resources_text": "Restart behavior: Can request restart May request user input: No",
            },
            {
                "package_name": "windows_10",
                "package_version": "22H2",
                "package_architecture": "x64",
                "update_type": "Cumulative",
                "file_size": "200MB",
                "install_resources_text": "Restart behavior: Can request restart May request user input: No",
            },
        ],
    }

    update_package, message, status = await update_package_service.create(
        **update_package_data
    )
    print(f"{status}: {message}")
    print(f"update_package type: {type(update_package)}")
    if update_package:
        print(f"Created UpdatePackage: {update_package}")

    else:
        update_package = await update_package_service.get(
            "ff0c2b87-a977-e36d-0ff6-e67a123fea87"
        )

    # Check if the UpdatePackage relationship exists between MSRCPost and UpdatePackage
    existing_update_rel_msrc = await msrc_post.has_update_packages.relationship(
        update_package
    )

    if not existing_update_rel_msrc:
        await msrc_post.has_update_packages.connect(
            update_package, {"relationship_type": "HAS_UPDATE_PACKAGE"}
        )
    else:
        print(
            "Update package relationship already exists between MSRCPost and UpdatePackage."
        )

    # Check if the UpdatePackage relationship exists between ProductBuild and UpdatePackage
    existing_update_rel_product_build = (
        await product_build.has_update_packages.relationship(update_package)
    )

    if not existing_update_rel_product_build:
        await product_build.has_update_packages.connect(
            update_package, {"relationship_type": "HAS_UPDATE_PACKAGE"}
        )
    else:
        print(
            "Update package relationship already exists between ProductBuild and UpdatePackage."
        )

    # Check if the UpdatePackage relationship exists between KBArticle and UpdatePackage
    existing_update_rel_kb = await kb_article.has_update_packages.relationship(
        update_package
    )

    if not existing_update_rel_kb:
        await kb_article.has_update_packages.connect(
            update_package, {"relationship_type": "HAS_UPDATE_PACKAGE"}
        )
    else:
        print(
            "Update package relationship already exists between KBArticle and UpdatePackage."
        )

    # ====================================
    # Symptom Service test
    # ====================================
    symptom_service = SymptomService(db_manager)

    symptom_data = {
        "node_id": "ElevationOfPrivilegeVulnerability",
        "description": "Unsafe default configurations for LDAP channel binding and LDAP signing exist on Active Directory domain controllers that let LDAP clients communicate with them without enforcing LDAP channel binding and LDAP signing. This can open Active Directory domain controllers to an elevation of privilege vulnerability.",
        "node_label": "Symptom",
        "source_id": "6f5403e0-d9b1-804a-67db-fa99ba5f7b44",
        "source_type": "MSRC",
        "embedding": [],
    }

    symptom, message, status = await symptom_service.create(**symptom_data)
    print(f"{status}: {message}")

    if symptom:
        print(f"Created Symptom: {symptom}")
    else:
        symptom = await symptom_service.get("ElevationOfPrivilegeVulnerability")

    # Check if the Symptom relationship exists between MSRCPost and Symptom
    existing_symptom_rel_msrc = await msrc_post.has_symptoms.relationship(symptom)

    if not existing_symptom_rel_msrc:
        await msrc_post.has_symptoms.connect(
            symptom, {"relationship_type": "HAS_SYMPTOM"}
        )
    else:
        print("Symptom relationship already exists between MSRCPost and Symptom.")

    # Check if the Symptom relationship exists between KBArticle and Symptom
    existing_symptom_rel_kbarticle = await kb_article.has_symptoms.relationship(symptom)

    if not existing_symptom_rel_kbarticle:
        await kb_article.has_symptoms.connect(
            symptom, {"relationship_type": "HAS_SYMPTOM"}
        )
    else:
        print("Symptom relationship already exists between KBArticle and Symptom.")

    # Check if the Symptom relationship exists between Product and Symptom
    existing_product_rel_symptom = await symptom.affects_products.relationship(product)

    if not existing_product_rel_symptom:
        await symptom.affects_products.connect(
            product, {"relationship_type": "AFFECTS_PRODUCT"}
        )
    else:
        print("Symptom relationship already exists between Product and Symptom.")

    # ====================================
    # Cause Service test
    # ====================================
    cause_service = CauseService(db_manager)

    cause_data = {
        "description": "Driver incompatibility with the new update.",
        "source_id": "6f5403e0-d9b1-804a-67db-fa99ba5f7b44",
        "source_type": "MSRCPost",
        "node_label": "Cause",
        "embedding": [],
    }

    cause, message, status = await cause_service.create(**cause_data)
    print(f"{status}: {message}")

    if cause:
        print(f"Created Cause: {cause}")
    else:
        cause = await cause_service.get("047800df-c1b6-4f8a-8775-2b49b5d3813e")

    # Check if the Cause relationship exists between MSRCPost and Cause
    existing_cause_rel_msrc = await msrc_post.has_causes.relationship(cause)

    if not existing_cause_rel_msrc:
        await msrc_post.has_causes.connect(cause, {"relationship_type": "HAS_CAUSE"})
    else:
        print("Cause relationship already exists between MSRCPost and Cause.")

    # Check if the Cause relationship exists between KBArticle and Cause
    existing_cause_rel_kbarticle = await kb_article.has_causes.relationship(cause)

    if not existing_cause_rel_kbarticle:
        await kb_article.has_causes.connect(cause, {"relationship_type": "HAS_CAUSE"})
    else:
        print("Cause relationship already exists between KBArticle and Cause.")

    # ====================================
    # Fix Service test
    # ====================================
    fix_service = FixService(db_manager)

    fix_data = {
        "description": "Apply the updated driver from the vendor website.",
        "source_id": "",
        "source_type": "",
        "node_label": "Fix",
        "embedding": [],
    }

    fix, message, status = await fix_service.create(**fix_data)
    print(f"{status}: {message}")

    if fix:
        print(f"Created Fix: {fix}")
    else:
        fix = fix_service.get("0beec6e2-e9b5-4d65-8dcd-aff150b31579")

    # Check if the Fix relationship exists between MSRCPost and Fix
    existing_fix_rel_msrc = await msrc_post.has_fixes.relationship(fix)

    if not existing_fix_rel_msrc:
        await msrc_post.has_fixes.connect(fix, {"relationship_type": "HAS_FIX"})
    else:
        print("Fix relationship already exists between MSRCPost and Fix.")

    # Check if the Fix relationship exists between KBArticle and Fix
    existing_fix_rel_kbarticle = await kb_article.has_fixes.relationship(fix)

    if not existing_fix_rel_kbarticle:
        await kb_article.has_fixes.connect(fix, {"relationship_type": "HAS_FIX"})
    else:
        print("Fix relationship already exists between KBArticle and Fix.")

    # ====================================
    # FAQ Service test
    # ====================================
    faq_service = FAQService(db_manager)

    faq_data = {
        "question": "Why does my system slow down after the update?",
        "answer": "The slowdown may be caused by driver incompatibilities.",
        "source_id": "",
        "source_type": "",
        "node_label": "FAQ",
    }

    faq, message, status = await faq_service.create(**faq_data)
    print(f"{status}: {message}")

    if faq:
        print(f"Created FAQ: {faq}")

    else:
        faq = await faq_service.get("abb36c12-15e0-4f61-9b9b-d09e89592048")

    # Check if the FAQ relationship exists between MSRCPost and FAQ
    existing_faq_rel_msrc = await msrc_post.has_faqs.relationship(faq)

    if not existing_faq_rel_msrc:
        await msrc_post.has_faqs.connect(faq, {"relationship_type": "HAS_FAQ"})
    else:
        print("FAQ relationship already exists between MSRCPost and FAQ.")

    # ====================================
    # Tool Service test
    # ====================================
    tool_service = ToolService(db_manager)

    tool_data = {
        "name": "Windows Update Troubleshooter",
        "description": "A tool to diagnose and fix update-related issues.",
        "download_url": "https://support.microsoft.com/help/troubleshooter",
        "source_id": "",
        "source_type": "PatchManagementPost",
        "node_label": "Tool",
    }

    tool, message, status = await tool_service.create(**tool_data)
    print(f"{status}: {message}")

    if tool:
        print(f"Created Tool: {tool}")
    else:
        tool = tool_service.get("491e1f55-368b-4182-a1f3-44cef66a2ce7")

    # Check if the Tool relationship exists between MSRCPost and Tool
    existing_tool_rel_msrc = await msrc_post.has_tools.relationship(tool)

    if not existing_tool_rel_msrc:
        await msrc_post.has_tools.connect(tool, {"relationship_type": "HAS_TOOL"})
    else:
        print("Tool relationship already exists between MSRCPost and Tool.")

    # Check if the Tool relationship exists between KBArticle and Tool
    existing_tool_rel_kb = await kb_article.has_tools.relationship(tool)

    if not existing_tool_rel_kb:
        await kb_article.has_tools.connect(tool, {"relationship_type": "HAS_TOOL"})
    else:
        print("Tool relationship already exists between KBArticle and Tool.")

    # ====================================
    # Patch Service test
    # ====================================
    patch_management_post_service = PatchManagementPostService(db_manager)

    patch_management_post_data = {
        "node_id": "db8232be-16a0-aebe-7856-e6a79300149b",
        "receivedDateTime": "2024-08-16T08:54:23+00:00",
        "published": datetime.strptime("2024-08-16 00:00:00", "%Y-%m-%d %H:%M:%S"),
        "topic": "",
        "subject": "Re: [patchmanagement] Windows 10 Security Patch Issues",
        "text": "Hello team, we're seeing issues after applying the latest Windows 10 security patch...",
        "post_type": "Problem Statement",
        "conversation_link": "https://groups.google.com/d/msgid/patchmanagement/112233445566778899aabbccddeeff00",
        "cve_mentions": "CVE-2024-12345",
        "keywords": [
            "server 2008 r2 machine",
            "patch",
            "may contain copyright material",
            "third party",
        ],
        "noun_chunks": ["these patches", "Server 2008 R2 machines", "the environment"],
        "metadata": {
            "collection": "patch_management",
        },
        "embedding": [],
        "node_label": "PatchManagement",
    }
    patch_management_post, message, status = await patch_management_post_service.create(
        **patch_management_post_data
    )
    print(f"{status}: {message}")

    if patch_management_post:
        print(f"Created PatchManagementPost: {patch_management_post}")
    else:
        patch_management_post = await patch_management_post_service.get(
            "db8232be-16a0-aebe-7856-e6a79300149b"
        )

    # Symptom relationship
    existing_symptom_rel = await patch_management_post.has_symptoms.relationship(
        symptom
    )
    if not existing_symptom_rel:
        await patch_management_post.has_symptoms.connect(
            symptom, {"relationship_type": "HAS_SYMPTOM"}
        )
    else:
        print("Symptom relationship already exists.")

    # Cause relationship
    existing_cause_rel = await patch_management_post.has_causes.relationship(cause)
    if not existing_cause_rel:
        await patch_management_post.has_causes.connect(
            cause, {"relationship_type": "HAS_CAUSE"}
        )
    else:
        print("Cause relationship already exists.")

    # Fix relationship
    existing_fix_rel = await patch_management_post.has_fixes.relationship(fix)
    if not existing_fix_rel:
        await patch_management_post.has_fixes.connect(
            fix, {"relationship_type": "HAS_FIX"}
        )
    else:
        print("Fix relationship already exists.")

    # KBArticle relationship
    existing_kb_rel = await patch_management_post.references_kb_articles.relationship(
        kb_article
    )
    if not existing_kb_rel:
        await patch_management_post.references_kb_articles.connect(
            kb_article, {"relationship_type": "REFERENCES"}
        )
    else:
        print("KBArticle relationship already exists.")

    # MSRCPost relationship
    existing_msrc_rel = await patch_management_post.references_msrc_posts.relationship(
        msrc_post
    )
    if not existing_msrc_rel:
        await patch_management_post.references_msrc_posts.connect(
            msrc_post, {"relationship_type": "REFERENCES"}
        )
    else:
        print("MSRCPost relationship already exists.")

    # Product relationship
    existing_product_rel = await patch_management_post.affects_products.relationship(
        product
    )
    if not existing_product_rel:
        await patch_management_post.affects_products.connect(
            product, {"relationship_type": "AFFECTS_PRODUCT"}
        )
    else:
        print("Product relationship already exists.")

    # Tool relationship
    existing_tool_rel = await patch_management_post.has_tools.relationship(tool)
    if not existing_tool_rel:
        await patch_management_post.has_tools.connect(
            tool, {"relationship_type": "HAS_TOOL"}
        )
    else:
        print("Tool relationship already exists.")

    # # Delete the product
    # success = await product_service.delete(product.node_id)
    # if success:
    #     print(f"Product with ID {product.node_id} has been deleted.")
    #     del product
    # else:
    #     print(f"Failed to delete Product with ID {product.node_id}.")

    # success = await product_build_service.delete(product_build.node_id)
    # if success:
    #     print(f"ProductBuild with ID {product_build.node_id} has been deleted.")
    #     del product_build
    # else:
    #     print(f"Failed to delete ProductBuild with ID {product_build.node_id}.")

    # success = await msrc_service.delete(msrc_post.node_id)
    # if success:
    #     print(f"MSRCPost with ID {msrc_post.node_id} has been deleted.")
    #     del msrc_post
    # else:
    #     print(f"Failed to delete MSRCPost with ID {msrc_post.node_id}.")

    # success = await symptom_service.delete(symptom.node_id)
    # if success:
    #     print(f"Symptom with ID {symptom.node_id} has been deleted.")
    #     del symptom
    # else:
    #     print(f"Failed to delete Symptom with ID {symptom.node_id}.")

    # success = await cause_service.delete(cause.node_id)
    # if success:
    #     print(f"Cause with ID {cause.node_id} has been deleted.")
    #     del cause
    # else:
    #     print(f"Failed to delete Cause with ID {cause.node_id}.")

    # success = await fix_service.delete(fix.node_id)
    # if success:
    #     print(f"Fix with ID {fix.node_id} has been deleted.")
    #     del fix
    # else:
    #     print(f"Failed to delete Fix with ID {fix.node_id}.")

    # success = await faq_service.delete(faq.node_id)
    # if success:
    #     print(f"FAQ with ID {faq.node_id} has been deleted.")
    #     del faq
    # else:
    #     print(f"Failed to delete FAQ with ID {faq.node_id}.")

    # success = await tool_service.delete(tool.node_id)
    # if success:
    #     print(f"Tool with ID {tool.node_id} has been deleted.")
    #     del tool
    # else:
    #     print(f"Failed to delete Tool with ID {tool.node_id}.")

    # success = await kb_article_service.delete(kb_article.node_id)
    # if success:
    #     print(f"KBArticle with ID {kb_article.node_id} has been deleted.")
    #     del kb_article
    # else:
    #     print(f"Failed to delete KBArticle with ID {kb_article.node_id}.")

    # success = await update_package_service.delete(update_package.node_id)
    # if success:
    #     print(f"UpdatePackage with ID {update_package.node_id} has been deleted.")
    #     del update_package
    # else:
    #     print(f"Failed to delete UpdatePackage with ID {update_package.node_id}.")

    # success = await patch_management_post_service.delete(patch_management_post.node_id)
    # if success:
    #     print(
    #         f"PatchManagementPost with ID {patch_management_post.node_id} has been deleted."
    #     )
    #     del patch_management_post
    # else:
    #     print(
    #         f"Failed to delete PatchManagementPost with ID {patch_management_post.node_id}."
    #     )

    query = """
MATCH (p:Product)
WHERE p.product_architecture = $architecture
AND p.product_version IN $versions
RETURN p
"""
    parameters = {
        "architecture": "x64",
        "versions": ["23H2", "22H2"],
    }
    nodes, meta = await product_service.execute_cypher(query, parameters)
    print(f"cypher results type: {type(nodes)}\nsample:\n{nodes}")
    # products = await convert_nodes_to_products(nodes)
    inflated_objects = await inflate_nodes(nodes)
    for obj in inflated_objects:
        if isinstance(obj, graph_db_models.Product):
            print(f"Product: {obj.product_name} {obj.product_version}")
        if isinstance(obj, graph_db_models.ProductBuild):
            print(f"ProductBuild: {obj.product} {obj.product_version}")
        elif isinstance(obj, graph_db_models.KBArticle):
            print(f"KBArticle: {obj.node_id} {obj.article_url}")


# Run the main function using asyncio
if __name__ == "__main__":

    print("Starting main()")
    # asyncio.run(test_neomodel_get_or_none())
    # asyncio.run(main())
    print("Finished main()")
    # print("This module doesn't run on its own.")
