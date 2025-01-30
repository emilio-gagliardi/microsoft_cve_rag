# Purpose: Manage graph database operations
# Inputs: Graph queries
# Outputs: Query results
# Dependencies: None (external graph database library)

import difflib
import hashlib
import json
import logging
import math

# services/graph_db_service.py
import os
import re
import time

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# print(sys.path)
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

# import asyncio
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import yaml
from application.app_utils import get_app_config, get_graph_db_credentials
from application.core.models import graph_db_models
from application.etl.transformer import make_json_safe_metadata
from neo4j import GraphDatabase  # required for constraints check

# import asyncio  # required for testing
from neomodel import AsyncStructuredNode, AsyncStructuredRel, DeflateError
from neomodel import config as NeomodelConfig  # required by AsyncDatabase
from neomodel.async_.core import AsyncDatabase  # required for db CRUD
from neomodel.async_.relationship_manager import AsyncRelationshipTo
from neomodel.exceptions import UniqueProperty
from rapidfuzz import fuzz
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
    """
    A context manager for managing the lifecycle of a connection to the graph database.

    The context manager is used to ensure that the connection is properly established and
    closed when the context is exited.  It is necessary to use this context manager to ensure
    that the connection is properly set up and torn down.
    """

    def __init__(self, db: AsyncDatabase) -> None:
        self._db = db
        # print(f"db is type: {type(self._db)}, config is type: {type(NeomodelConfig)}")

    async def __aenter__(self):
        """
        Establish a connection to the graph database.

        This method sets up the connection using the database URL defined in the Neomodel configuration.
        It should be called when entering the context manager to ensure the database connection is ready for use.
        """
        await self._db.set_connection(NeomodelConfig.DATABASE_URL)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Close the connection to the graph database.

        This method is called when exiting the context manager. It ensures that any resources
        associated with the database connection are properly released. If an exception occurred
        during the context, it can be handled or logged here if necessary.
        """
        await self._db.close_connection()


# BaseService definition

T = TypeVar("T", bound=AsyncStructuredNode)


@dataclass
class NodeCreationResult:
    """Result of node creation/retrieval operation"""

    node: Optional[T]
    source_id: Optional[str]
    node_id: Optional[str]
    message: str
    status_code: int


class BaseService(Generic[T]):
    """
    BaseService provides a generic service for managing Neomodel node classes.

    This service includes methods for creating, retrieving, updating, and deleting
    node instances in a graph database. It also supports executing Cypher queries,
    inflating query results into model instances, and handling bulk operations.

    Attributes:
        model: The Neomodel node class this service manages.
        db_manager: An instance of GraphDatabaseManager for database operations.
    """

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
                f"Duplicate {self.model.__name__} node. Skipping"
                f" create...\n{str(e)}"
            )
            return None, f"Duplicate entry: {str(e)}", 409
        except DeflateError as e:
            logging.error(
                f"Choice constraint violation in {self.model.__name__}node:"
                f" {node}\n{str(e)}"
            )
            return None, f"Choice constraint violation: {str(e)}", 422
        except Exception as e:
            logging.error(
                f"Error creating {self.model.__name__} node: {node}\n{str(e)}"
            )
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

    async def get_or_create(self, **properties) -> NodeCreationResult:
        """
        Retrieve a node with matching properties, or create it if it doesn't exist.

        :param properties: Key-value pairs of properties to match or set.
        :return: NodeCreationResult containing node instance and status information
        """
        try:
            node_label = properties.get("node_label", None)
            search_dict = self._build_search_dict(node_label, properties)

            existing_node_service = await self.model.nodes.get_or_none(
                **search_dict
            )
            if existing_node_service:
                return NodeCreationResult(
                    node=existing_node_service,
                    source_id=properties.get("source_id"),
                    node_id=existing_node_service.node_id,
                    message="Existing node retrieved by node_id",
                    status_code=200,
                )

            if properties["node_label"] == "Product":
                existing_node_class = (
                    await graph_db_models.Product.nodes.get_or_none(
                        **search_dict
                    )
                )
            elif properties["node_label"] == "ProductBuild":
                existing_node_class = (
                    await graph_db_models.ProductBuild.nodes.get_or_none(
                        **search_dict
                    )
                )
            else:
                existing_node_class = None

            if existing_node_class:
                return NodeCreationResult(
                    node=existing_node_class,
                    source_id=properties.get("source_id"),
                    node_id=existing_node_class.node_id,
                    message="Existing node retrieved by node_id",
                    status_code=200,
                )

            if "build_numbers" in properties:
                properties.pop("build_numbers")
            node_new = self.model(**properties)
            await node_new.save()
            return NodeCreationResult(
                node=node_new,
                source_id=properties.get("source_id"),
                node_id=node_new.node_id,
                message="Node created successfully",
                status_code=201,
            )

        except UniqueProperty as e:
            return NodeCreationResult(
                node=None,
                source_id=properties.get("source_id"),
                node_id=properties.get("node_id"),
                message=f"Unique property violation: {str(e)}",
                status_code=409,
            )

        except Exception as e:
            print(f"General exception: {str(e)}")
            traceback.print_exc()
            return NodeCreationResult(
                node=None,
                source_id=properties.get("source_id"),
                node_id=properties.get("node_id"),
                message=f"Error creating/retrieving node: {str(e)}",
                status_code=500,
            )

    def _build_search_dict(self, node_label: str, properties: dict) -> dict:
        """Construct the search dictionary based on the node_label."""
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

    async def execute_cypher(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Union[T, Dict[str, Any]]], Dict[str, Any]]:
        """
        Execute a Cypher query with optional parameters and return the results.

        Args:
            query (str): The Cypher query to execute.
            params (Optional[Dict[str, Any]]): A dictionary of parameters to
                replace placeholders in the query string.

        Returns:
            Tuple[List[Union[T, Dict[str, Any]]], Dict[str, Any]]: A tuple containing
                the query results and the query execution metadata.
        """
        results = self.db_manager._db.cypher_query(query, params)
        return results

    async def inflate_results(self, results: List[Any]) -> List[T]:
        """
        Inflate a list of raw query results into a list of model instances.

        Given a list of query results, this method will inflate each item
        into a model instance using the `inflate_item` method. If the item
        is a tuple, it will be inflated as a tuple of model instances.

        Args:
            results (List[Any]): The list of query results to inflate.

        Returns:
            List[T]: A list of inflated model instances.
        """
        inflated = []
        for row in results:
            if isinstance(row, tuple):
                inflated.append(
                    tuple(await self.inflate_item(item) for item in row)
                )
            else:
                inflated.append(await self.inflate_item(row))
        return inflated

    async def inflate_item(self, item: Any) -> Union[T, Dict[str, Any]]:
        """
        Inflate a single query result into a model instance.

        Args:
            item (Any): The query result to inflate.

        Returns:
            Union[T, Dict[str, Any]]: The inflated model instance or a dictionary
                representation of the query result.
        """
        if isinstance(item, dict) and "node_id" in item:
            return await self.model.inflate(item)
        elif isinstance(item, AsyncStructuredNode):
            return item
        else:
            return dict(item)

    async def cypher(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Union[T, Dict[str, Any]]], Dict[str, Any]]:
        """
        Execute a Cypher query with optional parameters and return the results.

        Args:
            query (str): The Cypher query to execute.
            params (Optional[Dict[str, Any]]): A dictionary of parameters to
                replace placeholders in the query string.

        Returns:
            Tuple[List[Union[T, Dict[str, Any]]], Dict[str, Any]]: A tuple containing
                the query results and the query execution metadata.
        """
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

    async def bulk_create(
        self, items: List[Dict[str, Any]]
    ) -> Tuple[List[T], List[NodeCreationResult]]:
        """
        Bulk create nodes from a list of items.

        Args:
            items (List[Dict[str, Any]]): List of items to create nodes from

        Returns:
            Tuple[List[T], List[NodeCreationResult]]: Tuple of (successful nodes, failed nodes with errors)
        """
        results = []
        errors = []
        async with self.db_manager:
            async with self.db_manager._db.transaction:
                for item in items:
                    try:
                        # Handle metadata conversion
                        if "metadata" in item:
                            if isinstance(item["metadata"], dict):
                                # Keep as dictionary but ensure values are JSON-safe
                                item["metadata"] = make_json_safe_metadata(
                                    item["metadata"]
                                )
                            elif isinstance(item["metadata"], str):
                                # If it's a string, try to parse it as JSON
                                try:
                                    metadata_dict = json.loads(
                                        item["metadata"]
                                    )
                                    item["metadata"] = make_json_safe_metadata(
                                        metadata_dict
                                    )
                                except json.JSONDecodeError:
                                    logging.warning(
                                        "Invalid JSON in metadata:"
                                        f" {item['metadata']}"
                                    )
                                    item["metadata"] = {}
                            else:
                                item["metadata"] = {}

                        # Handle None or empty values for embedding
                        if "embedding" in item:
                            if isinstance(
                                item["embedding"], float
                            ) and math.isnan(item["embedding"]):
                                del item["embedding"]
                            elif item["embedding"] is None:
                                del item["embedding"]
                            elif (
                                isinstance(
                                    item["embedding"], (list, np.ndarray)
                                )
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
                                cve_ids = [
                                    cve_id.strip()
                                    for cve_id in item["cve_ids"].split()
                                    if cve_id.strip()
                                ]
                                # Only keep items that match CVE pattern
                                cve_pattern = re.compile(
                                    r'CVE-\d{4}-\d{4,7}', re.IGNORECASE
                                )
                                item["cve_ids"] = [
                                    cve_id
                                    for cve_id in cve_ids
                                    if cve_pattern.match(cve_id)
                                ]
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
                                item["product_build_ids"] = [
                                    item["product_build_ids"]
                                ]
                            elif not isinstance(
                                item["product_build_ids"], list
                            ):
                                item["product_build_ids"] = [
                                    str(item["product_build_ids"])
                                ]
                        if "product_mentions" in item:
                            if item["product_mentions"] is None or (
                                isinstance(item["product_mentions"], float)
                                and math.isnan(item["product_mentions"])
                            ):
                                item["product_mentions"] = []
                            elif isinstance(item["product_mentions"], str):
                                item["product_mentions"] = [
                                    item["product_mentions"]
                                ]
                            elif not isinstance(
                                item["product_mentions"], list
                            ):
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
                                item["noun_chunks"] = [
                                    str(item["noun_chunks"])
                                ]
                        if "post_type" in item:
                            if item["post_type"] is None or (
                                isinstance(item["post_type"], float)
                                and math.isnan(item["post_type"])
                            ):
                                item["post_type"] = ""

                        result = await self.get_or_create(**item)
                        if result.node:
                            # Handle specific node types if needed
                            if (
                                hasattr(result.node, "set_build_numbers")
                                and "build_numbers" in item
                            ):
                                result.node.set_build_numbers(
                                    item["build_numbers"]
                                )
                                await result.node.save()
                            if (
                                hasattr(
                                    result.node, "set_downloadable_packages"
                                )
                                and "downloadable_packages" in item
                            ):
                                result.node.set_downloadable_packages(
                                    item["downloadable_packages"]
                                )
                                await result.node.save()
                            results.append(result.node)
                        else:
                            errors.append(result)
                    except UniqueProperty as e:
                        print(f"UniqueProperty error: {str(e)}")
                        # Try to retrieve the existing node
                        existing_node = await self.model.nodes.get_or_none(
                            **item
                        )
                        print(
                            "UniqueProperty violation...attempting lookup"
                            " again:"
                        )
                        if existing_node:
                            results.append(existing_node)
                        else:
                            errors.append(
                                NodeCreationResult(
                                    node=None,
                                    source_id=item.get("source_id", "unknown"),
                                    node_id=None,
                                    message=str(e),
                                    status_code=409,
                                )
                            )
                    except Exception as e:
                        error_msg = f"Error processing item: {str(e)}"
                        errors.append(
                            NodeCreationResult(
                                node=None,
                                source_id=item.get("source_id", "unknown"),
                                node_id=None,
                                message=error_msg,
                                status_code=500,
                            )
                        )
                        logging.error(f"{error_msg}\n{traceback.format_exc()}")

        if errors:
            logging.warning(
                f"Encountered {len(errors)} errors during bulk create"
            )
            for error in errors:
                logging.warning(
                    f"Error: {error.message} for item with source_id:"
                    f" {error.source_id}"
                )

        return results, errors

    def _process_array_field(self, value: Any, field_name: str) -> List[str]:
        """Helper method to process array fields consistently"""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return []
        if isinstance(value, str):
            if field_name == "cve_ids":
                cve_ids = [
                    cve_id.strip()
                    for cve_id in value.split()
                    if cve_id.strip()
                ]
                cve_pattern = re.compile(r'CVE-\d{4}-\d{4,7}', re.IGNORECASE)
                return [
                    cve_id for cve_id in cve_ids if cve_pattern.match(cve_id)
                ]
            return [value]
        if not isinstance(value, list):
            return [str(value)]
        return value

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
            results = await self.cypher(query, params)
            return [target_model.inflate(record[0]) for record in results]
        except Exception as e:
            logging.error(f"Error finding related nodes: {str(e)}")
            return []


class MSRCPostService(BaseService[graph_db_models.MSRCPost]):
    """Service class for MSRCPost nodes."""

    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.MSRCPost, db_manager)


class ProductService(BaseService[graph_db_models.Product]):
    """Service class for Product nodes."""

    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.Product, db_manager)


class ProductBuildService(BaseService[graph_db_models.ProductBuild]):
    """Service class for ProductBuild nodes."""

    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.ProductBuild, db_manager)


class SymptomService(BaseService[graph_db_models.Symptom]):
    """Service class for Symptom nodes."""

    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.Symptom, db_manager)


class CauseService(BaseService[graph_db_models.Cause]):
    """Service class for Cause nodes."""

    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.Cause, db_manager)


class FixService(BaseService[graph_db_models.Fix]):
    """Service class for Fix nodes."""

    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.Fix, db_manager)


class FAQService(BaseService[graph_db_models.FAQ]):
    """Service class for FAQ nodes."""

    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.FAQ, db_manager)


class ToolService(BaseService[graph_db_models.Tool]):
    """Service class for Tool nodes."""

    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.Tool, db_manager)
        self._cached_tools = None
        self._last_cache_time = None
        self._cache_ttl = 300  # 5 minutes TTL
        self._compound_words, self._acronyms = self._load_compound_words()

    def _load_compound_words(self) -> tuple[set, set]:
        """
        Load compound words and acronyms from YAML file.
        Returns tuple of (compound_words, acronyms)
        """
        try:
            compound_words_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "etl",
                "microsoft_compound_phrases.yaml",
            )
            with open(compound_words_path, 'r') as f:
                data = yaml.safe_load(f)
                compound_words = set(data.get("_COMPOUND_WORDS", []))
                acronyms = set(data.get("_ACRONYMS", []))
                logging.info(
                    f"Loaded {len(compound_words)} compound words and"
                    f" {len(acronyms)} acronyms"
                )
                return compound_words, acronyms
        except Exception as e:
            logging.error(
                f"Error loading compound words from {compound_words_path}:"
                f" {str(e)}"
            )
            return set(), set()

    async def _load_all_tools(
        self, force_refresh: bool = False
    ) -> List[graph_db_models.Tool]:
        """
        Load all tools from the database with caching.

        Args:
            force_refresh: If True, force a cache refresh even if TTL hasn't expired

        Returns:
            List of Tool nodes
        """
        current_time = time.time()

        # Check if cache is valid
        if (
            not force_refresh
            and self._cached_tools is not None
            and self._last_cache_time is not None
            and current_time - self._last_cache_time < self._cache_ttl
        ):
            return self._cached_tools

        # Cache miss or forced refresh - load from database
        try:
            self._cached_tools = await self.model.nodes.all()
            self._last_cache_time = current_time
            logging.info(f"Loaded {len(self._cached_tools)} tools into cache")
            return self._cached_tools
        except Exception as e:
            logging.error(f"Error loading tools from database: {str(e)}")
            # If we have cached data, return it as fallback
            if self._cached_tools is not None:
                logging.info("Returning cached tools as fallback")
                return self._cached_tools
            raise

    async def invalidate_cache(self):
        """Force the tool cache to be refreshed on next access."""
        self._cached_tools = None
        self._last_cache_time = None

    async def find_similar_tool(
        self, tool_label: str, threshold: float = 0.85, max_candidates: int = 5
    ) -> Optional[graph_db_models.Tool]:
        """
        Find existing tools with similar labels using string similarity matching.
        This method performs the following steps:
        1. Normalizes the input tool label to snake_case
        2. Retrieves candidate tools from the database cache
        3. Uses SequenceMatcher for string similarity comparison
        4. Returns the most similar tool if it meets the threshold

        Args:
            tool_label: The tool label to match against
            threshold: Similarity threshold (0-1), higher means stricter matching
            max_candidates: Maximum number of candidates to compare

        Returns:
            Optional[Tool]: The most similar tool if one exists above threshold, else None

        Example:
            >>> tool = await tool_service.find_similar_tool("update_edge", threshold=0.85)
            >>> if tool:
            >>>     print(f"Found similar tool: {tool.tool_label}")
        """
        if not tool_label or len(tool_label.strip()) == 0:
            logging.warning("Empty tool label provided to find_similar_tool")
            return None

        def normalize_label(text: str) -> str:
            """Convert any string to snake_case format"""
            if not text:
                return text

            # If already in snake_case, return as-is
            if is_snake_case(text):
                return text

            # Remove any special characters and extra whitespace
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text.strip())

            # Convert CamelCase to space-separated words
            text = re.sub(r'([A-Z])', r' \1', text).strip()

            # Convert to lowercase and replace spaces with underscores
            return re.sub(r'\s+', '_', text.lower())

        def is_snake_case(text: str) -> bool:
            """
            Check if a string is already in valid snake_case format.
            Returns True if:
            - Contains only lowercase letters, numbers, and underscores
            - No consecutive underscores
            - No leading/trailing underscores
            - No spaces
            """
            if not text:
                return True

            # Check for uppercase letters or spaces
            if any(c.isupper() or c.isspace() for c in text):
                return False

            # Check for valid characters (lowercase, numbers, single underscores)
            if not re.match(r'^[a-z0-9]+(?:_[a-z0-9]+)*$', text):
                return False

            return True

        def combined_similarity(
            input_label: str, candidate_label: str
        ) -> float:
            """
            Calculate similarity between two labels using both character and token-based matching.

            Args:
                input_label: The input label to compare
                candidate_label: The candidate label to compare against

            Returns:
                float: Combined similarity score between 0 and 1
            """
            # Character-based ratio using SequenceMatcher
            char_ratio = difflib.SequenceMatcher(
                None, input_label, candidate_label
            ).ratio()

            # Token-based ratios using RapidFuzz (returns scores in 0..100)
            token_set_ratio = (
                fuzz.token_set_ratio(input_label, candidate_label) / 100.0
            )
            token_sort_ratio = (
                fuzz.token_sort_ratio(input_label, candidate_label) / 100.0
            )

            # Common suffixes to consider less important
            common_suffixes = [
                'available',
                'enabled',
                'disabled',
                'running',
                'complete',
                'failed',
                'tool',
                'utility',
            ]

            # Boost score if the only difference is a common suffix
            input_parts = input_label.split('_')
            candidate_parts = candidate_label.split('_')

            # If one label contains a common suffix that the other doesn't, boost the token scores
            suffix_boost = 0
            if len(input_parts) != len(candidate_parts):
                extra_words = set(input_parts) ^ set(
                    candidate_parts
                )  # Words that differ
                if all(word in common_suffixes for word in extra_words):
                    suffix_boost = 0.1  # Boost score by 0.1 if only difference is common suffix

            # Use the higher of the token ratios and apply suffix boost
            token_ratio = max(token_set_ratio, token_sort_ratio) + suffix_boost
            token_ratio = min(1.0, token_ratio)  # Cap at 1.0

            # Weighted average (token matching given more weight)
            weight_char = 0.3  # Reduced character-level weight
            weight_token = 0.7  # Increased token-level weight
            final_score = (char_ratio * weight_char) + (
                token_ratio * weight_token
            )
            if final_score > 0.6:
                # Detailed logging for development/tuning
                logging.info("Similarity Analysis:")
                logging.info(f"  Input:     '{input_label}'")
                logging.info(f"  Candidate: '{candidate_label}'")
                logging.info(f"  Char Score:      {char_ratio:.3f}")
                logging.info(f"  Token Set:       {token_set_ratio:.3f}")
                logging.info(f"  Token Sort:      {token_sort_ratio:.3f}")
                logging.info(f"  Suffix Boost:    {suffix_boost:.3f}")
                logging.info(f"  Final Score:     {final_score:.3f}")

            return final_score

        try:
            # Get all tools from cache
            all_tools = await self._load_all_tools()
            if not all_tools:
                logging.debug("No existing tools found in database")
                return None

            # Normalize input label
            normalized_input = normalize_label(tool_label)
            logging.info(f"Normalized input: '{normalized_input}'")

            # Calculate similarity scores using combined similarity
            candidates = []
            for idx, tool in enumerate(all_tools):
                normalized_tool = (
                    tool.tool_label.lower()
                )  # Tool labels are already in snake_case
                score = combined_similarity(normalized_input, normalized_tool)
                if (
                    score > threshold * 0.7
                ):  # Lower threshold for development/monitoring
                    candidates.append((tool, score))

            if not candidates:
                logging.info(f"No candidates passed threshold ({threshold})")
                return None

            # Sort by score and get top candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[:max_candidates]

            # Return the best match if it meets the final threshold
            best_match = candidates[0]
            if best_match[1] >= threshold:
                logging.info(
                    f"Found similar tool: '{best_match[0].tool_label}' with"
                    f" score {best_match[1]}"
                )
                return best_match[0]

            logging.info("No candidates passed final threshold")
            return None

        except Exception as e:
            logging.error(f"Error finding similar tool: {str(e)}")
            return None


class KBArticleService(BaseService[graph_db_models.KBArticle]):
    """Service class for KBArticle nodes."""

    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.KBArticle, db_manager)


class UpdatePackageService(BaseService[graph_db_models.UpdatePackage]):
    """Service class for UpdatePackage nodes."""

    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.UpdatePackage, db_manager)


class PatchManagementPostService(
    BaseService[graph_db_models.PatchManagementPost]
):
    """Service class for PatchManagementPost nodes."""

    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.PatchManagementPost, db_manager)

    async def update_sequence(self, node_id, previous_id, next_id):
        """
        Updates the sequence of patch posts in the graph database.

        Args:
            node_id: The node_id of the PatchManagementPost to update.
            previous_id: The node_id of the previous PatchManagementPost in the sequence.
            next_id: The node_id of the next PatchManagementPost in the sequence.

        Returns:
            The updated PatchManagementPost instance.
        """
        node = await self.get(node_id=node_id)
        if node:
            node.previous_id = previous_id
            node.next_id = next_id
            await node.save()
        return node

    async def get_by_thread_id(self, thread_id):
        """
        Retrieve nodes with a specific thread ID.

        Args:
            thread_id: The ID of the thread to filter nodes by.

        Returns:
            A list of node instances with the given thread ID.
        """
        return await self.model.nodes.filter(thread_id=thread_id)


class TechnologyService(BaseService[graph_db_models.Technology]):
    """Service class for Technology nodes."""

    def __init__(self, db_manager: GraphDatabaseManager):
        super().__init__(graph_db_models.Technology, db_manager)


# Function to check if a constraint exists in the graph database
def graph_db_constraint_exists(session, constraint_name):
    """
    Checks if a constraint exists in the graph database.

    Args:
        session: The session to use for the query.
        constraint_name: The name of the constraint to check for.

    Returns:
        True if the constraint exists, False otherwise.
    """
    query = "SHOW CONSTRAINTS"
    result = session.run(query)
    constraints = [
        record.get("name") for record in result if record.get("name")
    ]
    return constraint_name in constraints


def graph_db_index_exists(session, index_name):
    """
    Checks if an index exists in the graph database.

    Args:
        session: The session to use for the query.
        index_name: The name of the index to check for.

    Returns:
        True if the index exists, False otherwise.
    """
    query = "SHOW INDEXES"
    result = session.run(query)
    indexes = [record.get("name") for record in result if record.get("name")]
    return index_name in indexes


# Function to ensure the constraint exists in the graph database
# called by fastapi at runtime to validate the graph database
# Implements neo4j package not neomodel
def ensure_graph_db_constraints_exist(
    uri: str, auth: tuple, graph_db_settings: dict
):
    """Ensure all required constraints and indexes exist in the Neo4j database.

    This function checks for and creates any missing constraints and indexes
    specified in the graph_db_settings.

    Args:
        uri (str): The Neo4j database URI
        auth (tuple): Authentication tuple (username, password)
        graph_db_settings (dict): Dictionary containing constraint and index definitions

    Returns:
        dict: Status report containing success/failure information for each
        constraint and index operation
    """
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
                            "message": (
                                f"Constraint '{constraint_name}' created"
                                " successfully."
                            ),
                        }
                    )
                else:
                    constraints_status.append(
                        {
                            "constraint_name": constraint_name,
                            "status": "exists",
                            "message": (
                                f"Constraint '{constraint_name}' already exists."
                            ),
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
                            "message": (
                                f"Index '{index_name}' created successfully."
                            ),
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
        if all(
            status["status"] in ["created", "exists"]
            for status in all_statuses
        )
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
    """
    Convert a list of query results into a list of Product instances.

    Args:
        node_results (List[Tuple[Node]]): A list of query results where each
            result is a list with one Node.

    Returns:
        List[graph_db_models.Product]: A list of Product instances.
    """
    products = []
    for result in node_results:
        node = result[0]  # Assuming each result is a list with one Node
        product = graph_db_models.Product.inflate(node)
        products.append(product)
    return products


async def convert_nodes_to_product_builds(node_results):
    """
    Convert a list of query results into a list of ProductBuild instances.

    Args:
        node_results (List[Tuple[Node]]): A list of query results where each
            result is a list with one Node.

    Returns:
        List[graph_db_models.ProductBuild]: A list of ProductBuild instances.
    """
    product_builds = []
    for result in node_results:
        node = result[0]  # Assuming each result is a list with one Node
        product_build = graph_db_models.ProductBuild.inflate(node)
        product_builds.append(product_build)
    return product_builds


async def convert_nodes_to_kbs(node_results):
    """
    Convert a list of query results into a list of KBArticle instances.

    Args:
        node_results (List[Tuple[Node]]): A list of query results where each
            result is a list with one Node.

    Returns:
        List[graph_db_models.KBArticle]: A list of KBArticle instances.
    """
    kb_articles = []
    for result in node_results:
        node = result[0]  # Assuming each result is a list with one Node
        kb_article = graph_db_models.KBArticle.inflate(node)
        kb_articles.append(kb_article)
    return kb_articles


async def convert_nodes_to_update_packages(node_results):
    """
    Convert a list of query results into a list of UpdatePackage instances.

    Args:
        node_results (List[Tuple[Node]]): A list of query results where each
            result is a list with one Node.

    Returns:
        List[graph_db_models.UpdatePackage]: A list of UpdatePackage instances.
    """
    update_packages = []
    for result in node_results:
        node = result[0]  # Assuming each result is a list with one Node
        update_package = graph_db_models.UpdatePackage.inflate(node)
        update_packages.append(update_package)
    return update_packages


async def convert_nodes_to_msrc_posts(node_results):
    """
    Convert a list of query results into a list of MSRCPost instances.

    Args:
        node_results (List[Tuple[Node]]): A list of query results where each
            result is a list with one Node.

    Returns:
        List[graph_db_models.MSRCPost]: A list of MSRCPost instances.
    """
    msrc_posts = []
    for result in node_results:
        node = result[0]  # Assuming each result is a list with one Node
        msrc_post = graph_db_models.MSRCPost.inflate(node)
        msrc_posts.append(msrc_post)
    return msrc_posts


async def convert_nodes_to_patch_posts(node_results):
    """
    Convert a list of query results into a list of PatchManagementPost instances.

    Args:
        node_results (List[Tuple[Node]]): A list of query results where each
            result is a list with one Node.

    Returns:
        List[graph_db_models.PatchManagementPost]: A list of PatchManagementPost instances.
    """
    patch_posts = []
    for result in node_results:
        node = result[0]  # Assuming each result is a list with one Node
        patch_post = graph_db_models.PatchManagementPost.inflate(node)
        patch_posts.append(patch_post)
    return patch_posts


async def inflate_nodes(node_results):
    """
    Inflate a list of query results into a list of model instances.

    Args:
        node_results (List[Tuple[Node]]): A list of query results where each
            result is a list with one Node.

    Returns:
        List: A list of inflated model instances.
    """
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


async def sort_and_update_msrc_nodes(
    msrc_posts: List[graph_db_models.MSRCPost],
) -> List[graph_db_models.MSRCPost]:
    """
    Sort and update MSRCPost nodes by post_id and revision.

    This function takes a list of MSRCPost nodes and sorts them by post_id and revision.
    It then updates the previous_version_id field of each node to point to the next newest
    version of the same post. If the node is the newest version, the previous_version_id
    field is set to None.

    Args:
        msrc_posts (List[graph_db_models.MSRCPost]): A list of MSRCPost nodes to be sorted and updated.

    Returns:
        List[graph_db_models.MSRCPost]: The sorted and updated list of MSRCPost nodes.
    """
    post_groups = {}
    for post in msrc_posts:
        if post.post_id not in post_groups:
            post_groups[post.post_id] = []
        # Clean and validate the revision string before converting to Decimal
        revision = post.revision.strip() if post.revision else "0"
        try:
            # Store the Decimal value temporarily for sorting
            post._temp_revision_decimal = Decimal(revision)
        except (InvalidOperation, TypeError):
            # Handle invalid revision formats by setting a default
            post._temp_revision_decimal = Decimal("0")
        post_groups[post.post_id].append(post)

    updated_msrc_posts = []
    for post_id, posts in post_groups.items():
        # Sort using the temporary Decimal value
        sorted_posts = sorted(
            posts, key=lambda x: x._temp_revision_decimal, reverse=True
        )
        for i, post in enumerate(sorted_posts):
            # Clean up temporary attribute
            delattr(post, '_temp_revision_decimal')
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
    build_numbers_set = {
        tuple(build_number) for build_number in build_numbers_list
    }
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


# BEGIN RELATIONSHIP TRACKER =====================================================


db = AsyncDatabase()
NeomodelConfig.DATABASE_URL = get_graph_db_uri()


class RelationshipTracker:
    """
    A high-performance relationship tracking system for Neo4j graph database operations.

    This class manages the creation and tracking of relationships between nodes in a Neo4j
    database, with specific optimizations for handling both Neo4j native relationships and
    Neomodel AsyncStructuredRel relationships. It includes batching capabilities and caching
    mechanisms to prevent Cartesian products and improve performance.

    Attributes:
        existing_relationships (Dict[str, Set[Tuple[str, str, str]]]): Tracks existing relationships by node type
        batch_relationships (Dict[str, Set[Tuple[str, str, str]]]): Tracks new relationships to be created
        batch_size (int): Maximum number of nodes to process in a single batch
        _relationship_cache (dict): Cache for frequently checked relationships
    """

    def __init__(self, batch_size: int = 1000):
        """
        Initialize the RelationshipTracker with specified batch size.

        Args:
            batch_size (int, optional): Maximum number of nodes to process in a single batch.
                                      Defaults to 1000.
        """
        self.batch_size = batch_size
        self.batch_relationships = {}
        self._relationship_cache = {}
        self.existing_relationships = {}
        self.current_relationships = set()
        self.one_to_one_rel_types = {"PREVIOUS_VERSION", "PREVIOUS_MESSAGE"}

    async def fetch_existing_relationships(
        self,
        node_types: List[str],
        node_ids: List[str],
        relationship_types: Optional[List[str]] = None,
    ):
        """
        Fetch existing relationships in batches with improved filtering and caching.

        Args:
            node_types (List[str]): List of node types to fetch relationships for
            node_ids (List[str]): List of node IDs to fetch relationships for
            relationship_types (Optional[List[str]]): Specific relationship types to fetch,
                if None, fetches all relationships
        """
        logging.info(
            f"Fetching relationships for {len(node_types)} node types and"
            f" {len(node_ids)} nodes"
        )

        # Define relationship cardinality constraints
        one_to_one_relationships = {
            "MSRCPost": {
                "HAS_SYMPTOM",
                "HAS_CAUSE",
                "HAS_FIX",
                "HAS_TOOL",
                "PREVIOUS_VERSION",  # Ensure sequence relationship is included
            },
            "PatchManagementPost": {
                "HAS_SYMPTOM",
                "HAS_CAUSE",
                "HAS_FIX",
                "HAS_TOOL",
                "PREVIOUS_MESSAGE",  # Ensure sequence relationship is included
            },
        }

        for node_type in node_types:
            if node_type not in self.existing_relationships:
                self.existing_relationships[node_type] = {}

            for i in range(0, len(node_ids), self.batch_size):
                batch_ids = node_ids[i : i + self.batch_size]

                try:
                    # Optimized query with relationship type filtering
                    query = """
                        MATCH (source)
                        WHERE source.node_id IN $batch_ids
                        WITH source
                        MATCH (source)-[r]->(target)
                        WHERE CASE
                            WHEN size($rel_types) > 0 THEN type(r) IN $rel_types
                            ELSE true
                        END
                        RETURN source.node_id as source_id,
                            type(r) as rel_type,
                            target.node_id as target_id,
                            labels(source) as source_labels,
                            labels(target) as target_labels
                        """
                    params = {
                        "batch_ids": batch_ids,
                        "rel_types": (
                            relationship_types if relationship_types else []
                        ),
                    }

                    results, _ = await db.cypher_query(query, params)

                    for (
                        source_id,
                        rel_type,
                        target_id,
                        source_labels,
                        target_labels,
                    ) in results:
                        # Store only necessary information without full node conversion
                        # source_type = source_labels[0]
                        target_type = target_labels[0]

                        # Initialize relationship type set if not exists
                        if (
                            rel_type
                            not in self.existing_relationships[node_type]
                        ):
                            self.existing_relationships[node_type][
                                rel_type
                            ] = set()

                        # Check cardinality constraints
                        if (
                            node_type in one_to_one_relationships
                            and rel_type in one_to_one_relationships[node_type]
                        ):
                            # For one-to-one relationships, ensure uniqueness
                            existing_relationships = (
                                self.existing_relationships[node_type][
                                    rel_type
                                ]
                            )
                            for existing_rel in existing_relationships:
                                if existing_rel[0] == source_id:
                                    logging.warning(
                                        "Duplicate one-to-one relationship"
                                        " found:"
                                        f" {source_id}-[{rel_type}]->{target_id}"
                                    )
                                    continue

                        # Store relationship with minimal information
                        relationship_key = (source_id, target_id)
                        self.existing_relationships[node_type][rel_type].add(
                            relationship_key
                        )

                        # Cache node existence
                        self._cache_node_existence(source_id, node_type)
                        self._cache_node_existence(target_id, target_type)

                except Exception as e:
                    logging.error(
                        f"Error fetching relationships for {node_type}:"
                        f" {str(e)}"
                    )
                    raise

    def _cache_node_existence(self, node_id: str, node_type: str):
        """Cache node existence to prevent duplicate node creation."""
        if not hasattr(self, '_node_existence_cache'):
            self._node_existence_cache = {}
        self._node_existence_cache[node_id] = node_type

    def _convert_to_neomodel_node(
        self, neo4j_node, node_type: str
    ) -> AsyncStructuredNode:
        """
        Convert a native Neo4j Node to a corresponding Neomodel AsyncStructuredNode instance.

        This method ensures proper Object-Graph Mapping (OGM) integration by converting
        native Neo4j nodes to their corresponding Neomodel class instances, with caching
        and minimal property conversion.

        Args:
            neo4j_node: Native Neo4j node instance
            node_type (str): Type of the node to convert to

        Returns:
            AsyncStructuredNode: Converted Neomodel node instance

        Raises:
            ValueError: If the node_type is not supported or required properties are missing
        """
        # Check node existence cache first
        node_id = neo4j_node.get('node_id')
        if node_id and hasattr(self, '_node_existence_cache'):
            if node_id in self._node_existence_cache:
                cached_type = self._node_existence_cache[node_id]
                if cached_type == node_type:
                    logging.debug(
                        f"Node cache hit for {node_id} of type {node_type}"
                    )
                    return self._get_cached_node(node_id, node_type)

        # Define required properties for each node type
        required_properties = {
            "MSRCPost": {"node_id", "text"},
            "Symptom": {"node_id", "description", "symptom_label"},
            "Cause": {"node_id", "description"},
            "Fix": {"node_id", "description"},
            "Tool": {"node_id", "description", "tool_label"},
            "KBArticle": {"node_id", "product_build_id"},
            "UpdatePackage": {"node_id", "product_build_ids"},
            "PatchManagementPost": {"node_id"},
            "Product": {"node_id", "product_name"},
            "ProductBuild": {"node_id", "build_number"},
            "FAQ": {"node_id", "question", "answer"},
        }

        try:
            # Extract properties with type conversion
            properties = self._extract_node_properties(
                neo4j_node, required_properties.get(node_type, set())
            )

            # Get model class from map
            model_class = self._get_model_class(node_type)

            # Create instance with validated properties
            instance = model_class(**properties)

            # Cache the node
            self._cache_node_existence(properties['node_id'], node_type)
            self._cache_node_instance(instance)

            logging.debug(
                f"Converted node of type {node_type} with ID:"
                f" {properties.get('node_id')}"
            )
            return instance

        except Exception as e:
            error_msg = f"Error converting node of type {node_type}: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg) from e

    def _extract_node_properties(
        self, neo4j_node, required_properties: Set[str]
    ) -> Dict:
        """Extract and validate node properties."""
        properties = dict(neo4j_node._properties)

        # Validate required properties
        missing_props = required_properties - set(properties.keys())
        if missing_props:
            raise ValueError(f"Missing required properties: {missing_props}")

        # Convert array properties if they exist
        array_properties = {
            'embedding',
            'build_number',
            'tags',
            'product_build_ids',
        }
        for prop in array_properties:
            if prop in properties and properties[prop] is not None:
                if not isinstance(properties[prop], (list, set)):
                    properties[prop] = [properties[prop]]

        return properties

    def _get_model_class(self, node_type: str) -> Type[AsyncStructuredNode]:
        """Get the appropriate model class for the node type."""
        model_map = {
            "Product": graph_db_models.Product,
            "ProductBuild": graph_db_models.ProductBuild,
            "MSRCPost": graph_db_models.MSRCPost,
            "Symptom": graph_db_models.Symptom,
            "Cause": graph_db_models.Cause,
            "Fix": graph_db_models.Fix,
            "FAQ": graph_db_models.FAQ,
            "Tool": graph_db_models.Tool,
            "KBArticle": graph_db_models.KBArticle,
            "UpdatePackage": graph_db_models.UpdatePackage,
            "PatchManagementPost": graph_db_models.PatchManagementPost,
        }

        model_class = model_map.get(node_type)
        if not model_class:
            raise ValueError(f"Unsupported node type: {node_type}")
        return model_class

    def _cache_node_instance(self, node: AsyncStructuredNode):
        """Cache a node instance for future use."""
        if not hasattr(self, '_node_instance_cache'):
            self._node_instance_cache = {}
        self._node_instance_cache[node.node_id] = node

    def _get_cached_node(
        self, node_id: str, node_type: str
    ) -> Optional[AsyncStructuredNode]:
        """Retrieve a cached node instance."""
        if (
            hasattr(self, '_node_instance_cache')
            and node_id in self._node_instance_cache
        ):
            return self._node_instance_cache[node_id]
        return None

    def _create_relationship_key(
        self,
        source: AsyncStructuredNode,
        rel_type: str,
        target: AsyncStructuredNode,
    ) -> Tuple[str, str]:
        """
        Create a unique key for tracking relationships.

        Creates a consistent and reliable key for relationship tracking using node_id
        instead of element_id. Handles special cases for Product->ProductBuild relationships
        by creating a deterministic hash based on business keys.

        Args:
            source (AsyncStructuredNode): Source node of the relationship
            rel_type (str): Type of relationship
            target (AsyncStructuredNode): Target node of the relationship

        Returns:
            Tuple[str, str]: Tuple of (source_id, target_id) representing the relationship

        Raises:
            ValueError: If required node properties are missing
        """
        source_id = str(source.node_id)
        target_id = str(target.node_id)

        # Special case for Product->ProductBuild relationships
        if (
            isinstance(source, graph_db_models.Product)
            and isinstance(target, graph_db_models.ProductBuild)
            and rel_type == "HAS_BUILD"
        ):

            try:
                # Create deterministic key using business properties
                product_key = {
                    "name": source.product_name,
                    "version": getattr(source, "product_version", ""),
                    "arch": getattr(source, "product_architecture", ""),
                }
                build_key = {
                    "number": (
                        target.build_number[0]
                        if isinstance(target.build_number, list)
                        else target.build_number
                    )
                }

                # Create deterministic hash
                product_hash = self._create_deterministic_hash(product_key)
                build_hash = self._create_deterministic_hash(build_key)

                logging.debug(
                    "Created special Product->ProductBuild key: "
                    f"{product_hash} -> {build_hash}"
                )
                return (product_hash, build_hash)

            except AttributeError as e:
                error_msg = (
                    "Missing required property for Product->ProductBuild"
                    f" relationship: {str(e)}"
                )
                logging.error(error_msg)
                raise ValueError(error_msg) from e

        return (source_id, target_id)

    def _create_deterministic_hash(self, data: Dict) -> str:
        """
        Create a deterministic hash from a dictionary.

        Args:
            data (Dict): Dictionary of data to hash

        Returns:
            str: Deterministic hash of the data
        """
        # Sort dictionary to ensure deterministic output
        sorted_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(sorted_str.encode()).hexdigest()[:32]

    def relationship_exists(
        self,
        source: AsyncStructuredNode,
        rel_type: str,
        target: AsyncStructuredNode,
    ) -> bool:
        """
        Check if a relationship exists using hierarchical caching and cardinality rules.

        Args:
            source (AsyncStructuredNode): Source node of the relationship
            rel_type (str): Type of relationship
            target (AsyncStructuredNode): Target node of the relationship

        Returns:
            bool: True if the relationship exists or violates cardinality, False otherwise
        """
        node_type = type(source).__name__
        source_id = str(source.node_id)
        target_id = str(target.node_id)
        cache_key = (source_id, rel_type, target_id)

        # Check memory cache first
        if cache_key in self._relationship_cache:
            logging.debug(f"Cache hit for relationship: {cache_key}")
            return self._relationship_cache[cache_key]

        # Define one-to-one relationship constraints
        one_to_one_relationships = {
            "MSRCPost": {
                "HAS_SYMPTOM",
                "HAS_CAUSE",
                "HAS_FIX",
                "HAS_TOOL",
                "PREVIOUS_VERSION",  # Added sequence relationship constraint
            },
            "PatchManagementPost": {
                "HAS_SYMPTOM",
                "HAS_CAUSE",
                "HAS_FIX",
                "HAS_TOOL",
                "PREVIOUS_MESSAGE",  # Added sequence relationship constraint
            },
        }

        # Check cardinality constraints
        if (
            node_type in one_to_one_relationships
            and rel_type in one_to_one_relationships[node_type]
        ):

            # Check if source node already has this type of relationship
            if (
                node_type in self.existing_relationships
                and rel_type in self.existing_relationships[node_type]
            ):

                existing_relationships = self.existing_relationships[
                    node_type
                ][rel_type]
                for existing_rel in existing_relationships:
                    if existing_rel[0] == source_id:
                        # Special logging for sequence relationships
                        if rel_type in {
                            "PREVIOUS_VERSION",
                            "PREVIOUS_MESSAGE",
                        }:
                            logging.warning(
                                "Sequence relationship violation:"
                                f" {source_id} already has a"
                                f" {rel_type} relationship. Only one"
                                f" {rel_type} allowed."
                            )
                        else:
                            logging.warning(
                                "One-to-one relationship violation:"
                                f" {source_id} already has a"
                                f" {rel_type} relationship"
                            )
                        return True

        # Check existence in current relationships
        exists_in_existing = False
        if (
            node_type in self.existing_relationships
            and rel_type in self.existing_relationships[node_type]
        ):
            exists_in_existing = (
                source_id,
                target_id,
            ) in self.existing_relationships[node_type][rel_type]

        # Check existence in batch relationships
        exists_in_batch = False
        if (
            node_type in self.batch_relationships
            and rel_type in self.batch_relationships[node_type]
        ):
            exists_in_batch = (
                source_id,
                target_id,
            ) in self.batch_relationships[node_type][rel_type]

        result = exists_in_existing or exists_in_batch
        self._relationship_cache[cache_key] = result

        logging.debug(f"Relationship {cache_key} exists: {result}")
        return result

    def add_relationship(
        self,
        source: AsyncStructuredNode,
        rel_type: str,
        target: AsyncStructuredNode,
    ) -> bool:
        """
        Add a new relationship to the tracker if it doesn't violate cardinality constraints.

        Updates both the tracking set and cache. The relationship will be created in the
        database when the batch is committed.

        Args:
            source (AsyncStructuredNode): Source node of the relationship
            rel_type (str): Type of relationship
            target (AsyncStructuredNode): Target node of the relationship

        Returns:
            bool: True if relationship was added successfully, False if it violates constraints
        """
        node_type = type(source).__name__
        source_id = str(source.node_id)
        target_id = str(target.node_id)

        # Check if relationship already exists or violates constraints
        if self.relationship_exists(source, rel_type, target):
            logging.warning(
                f"Relationship not added: {source_id} -[{rel_type}]->"
                f" {target_id} already exists or violates constraints"
            )
            return False

        # Initialize relationship type set if not exists
        if node_type not in self.batch_relationships:
            self.batch_relationships[node_type] = {}
        if rel_type not in self.batch_relationships[node_type]:
            self.batch_relationships[node_type][rel_type] = set()

        # Store relationship with minimal information
        relationship_key = (source_id, target_id)
        self.batch_relationships[node_type][rel_type].add(relationship_key)

        # Update cache
        cache_key = (source_id, rel_type, target_id)
        self._relationship_cache[cache_key] = True

        # Cache node existence
        self._cache_node_existence(source_id, node_type)
        self._cache_node_existence(target_id, type(target).__name__)

        logging.debug(
            f"Added relationship: {source_id} -[{rel_type}]-> {target_id}"
        )
        return True


# =================== END RELATIONSHIP TRACKER ==========================


async def check_has_symptom(
    source_node: Any,
    target_node: Any,
    rel_info: graph_db_models.AsyncSymptomRel,
) -> bool:
    return target_node.source_id == source_node.node_id


async def check_has_cause(
    source_node: Any,
    target_node: Any,
    rel_info: graph_db_models.AsyncCauseRel,
) -> bool:
    return target_node.source_id == source_node.node_id


async def check_has_fix(
    source_node: Any,
    target_node: Any,
    rel_info: graph_db_models.AsyncFixRel,
) -> bool:
    return target_node.source_id == source_node.node_id


async def check_has_faq(
    source_node: Any,
    target_node: Any,
    rel_info: graph_db_models.AsyncZeroToManyRel,
) -> bool:
    return target_node.source_id in source_node.faq_ids


async def check_has_tool(
    source_node: Any, target_node: Any, rel_info: graph_db_models.AsyncToolRel
) -> bool:
    # Check direct source_id match
    direct_match = target_node.source_id == source_node.node_id

    # Check source_ids array if property exists
    array_match = (
        source_node.node_id in target_node.source_ids
        if hasattr(target_node, 'source_ids')
        else False
    )

    return direct_match or array_match


async def check_has_kb(
    source_node: Any,
    target_node: Any,
    rel_info: graph_db_models.AsyncReferencesRel,
) -> bool:
    if isinstance(source_node, graph_db_models.MSRCPost):
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
                        f"\n======\nPatch -[HAS_KB]->KB: {response}\n======\n"
                    )
                    time.sleep(10)
                else:
                    print(f"Patch -[HAS_KB]->KB: False")
                    print(f"source: {source_node}\ntarget:{target_node}")
            """
            # print("Patch-[REFERENCES]->KB checking...")
            return target_node.kb_id in source_node.kb_ids
        elif isinstance(target_node, graph_db_models.MSRCPost):
            return target_node.post_id in source_node.cve_ids

    elif isinstance(source_node, graph_db_models.ProductBuild):
        if isinstance(target_node, graph_db_models.MSRCPost):
            return target_node.post_id in source_node.cve_id
        elif isinstance(target_node, graph_db_models.KBArticle):
            return target_node.kb_id in source_node.kb_id
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
        response = (
            source_node.product_build_id in target_node.product_build_ids
        )
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
            product_name = f"windows_{parts[1]}"  # e.g., windows_10
            # Check if the last part is an architecture
            if len(parts) > 2 and parts[-1].lower() in ["x86", "x64"]:
                product_architecture = parts[-1].lower()
                # Version is everything between product name and architecture (if any)
                product_version = "_".join(parts[2:-1]) if len(parts) > 3 else None
            else:
                # No architecture specified
                product_architecture = None
                # Version is everything after product name (if any)
                product_version = "_".join(parts[2:]) if len(parts) > 2 else None

            # Check the target product for an exact product_name match first
            if target_node.product_name == product_name:
                # Now check for product_version and product_architecture
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
    source_node: Any,
    target_node: Any,
    rel_info: graph_db_models.AsyncAffectsProductRel,
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
    source_node: Any,
    target_node: Any,
    rel_info: graph_db_models.AsyncReferencesRel,
) -> bool:
    # if isinstance(source_node, graph_db_models.Fix):
    #     if isinstance(target_node, graph_db_models.KBArticle):
    #         return target_node.kb_id[0] in source_node.kb_ids

    if isinstance(source_node, graph_db_models.KBArticle):
        if isinstance(target_node, graph_db_models.ProductBuild):
            logging.warning(
                "This is the From direction. No relationship to create."
            )

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
                        f"\n======\nPatch -[HAS_KB]->KB: {response}\n======\n"
                    )
                    time.sleep(10)
                else:
                    print(f"Patch -[HAS_KB]->KB: False")
                    print(f"source: {source_node}\ntarget:{target_node}")
            """
            # print("Patch-[REFERENCES]->KB checking...")
            return target_node.kb_id in source_node.kb_ids
        elif isinstance(target_node, graph_db_models.MSRCPost):
            return target_node.post_id in source_node.cve_ids

    elif isinstance(source_node, graph_db_models.ProductBuild):
        if isinstance(target_node, graph_db_models.MSRCPost):
            return target_node.post_id in source_node.cve_id
        elif isinstance(target_node, graph_db_models.KBArticle):
            return target_node.kb_id in source_node.kb_id
    return False


async def check_has_build(
    source_node: Any,
    target_node: Any,
    rel_info: graph_db_models.AsyncZeroToManyRel,
) -> bool:
    if isinstance(source_node, graph_db_models.Product):
        build_number_to_check = target_node.build_number
        build_numbers_list = source_node.get_build_numbers()

        found = any(
            sublist == build_number_to_check for sublist in build_numbers_list
        )
        return found

    return False


async def should_be_related(
    source_node: AsyncStructuredNode,
    target_node: AsyncStructuredNode,
    rel_info: Tuple[str, str, type],
) -> bool:
    """
    Check if a relationship between two nodes should be created based on the
    relationship type and the node types.

    :param source_node: The source node of the relationship.
    :param target_node: The target node of the relationship.
    :param rel_info: A tuple containing the target node model, the relationship type,
                     and the relationship class.
    :return: True if the relationship should be created, False otherwise.
    """
    target_model, relation_type, _ = rel_info
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
    """
    Creates a relationship between two nodes and returns the created relationship
    instance.

    :param relationship: The relationship object from the relationship mapping.
    :param source_node: The source node of the relationship.
    :param target_node: The target node of the relationship.
    :param rel_class: The relationship class.
    :return: The created relationship instance.
    """
    try:
        # Get the actual relationship type from the relationship definition
        rel_type = relationship.definition["relation_type"]

        # Create a deterministic relationship key based on business properties
        rel_key = {}

        # For Product->ProductBuild relationships, use business keys
        if isinstance(source_node, graph_db_models.Product) and isinstance(
            target_node, graph_db_models.ProductBuild
        ):
            rel_key = {
                "product_name": source_node.product_name,
                "product_version": source_node.product_version,
                "product_architecture": source_node.product_architecture,
                "build_number": target_node.build_number,
                "rel_type": rel_type,
            }
        else:
            # For other relationships, use node IDs
            rel_key = {
                "source_id": source_node.node_id,
                "target_id": target_node.node_id,
                "rel_type": rel_type,
            }

        # Create a deterministic hash of the relationship key
        unique_id = hashlib.sha256(
            json.dumps(rel_key, sort_keys=True).encode()
        ).hexdigest()

        # Initialize base properties
        rel_properties = {
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "relationship_id": unique_id,
            "source_node_id": source_node.node_id,
            "target_node_id": target_node.node_id,
        }

        # Add relationship-specific properties
        if hasattr(relationship, "confidence"):
            rel_properties["confidence"] = relationship.confidence
        if hasattr(relationship, "tags"):
            rel_properties["tags"] = relationship.tags
        if hasattr(relationship, "relationship_type"):
            rel_properties["relationship_type"] = (
                relationship.relationship_type
            )
        if hasattr(relationship, "severity"):
            rel_properties["severity"] = relationship.severity
        if hasattr(relationship, "description"):
            rel_properties["description"] = relationship.description
        if hasattr(relationship, "reported_date"):
            rel_properties["reported_date"] = relationship.reported_date

        # Use an optimized query that avoids Cartesian products and enforces uniqueness
        query = (
            """
        MATCH (source)
        WHERE source.node_id = $source_id
        WITH source
        MATCH (target)
        WHERE target.node_id = $target_id
        MERGE (source)-[r:`%s` {relationship_id: $rel_id}]->(target)
        SET r += $properties
        RETURN r
        """
            % rel_type
        )

        params = {
            "source_id": source_node.node_id,
            "target_id": target_node.node_id,
            "rel_id": unique_id,
            "properties": rel_properties,
        }

        results, _ = await db.cypher_query(query, params)

        if results and results[0]:
            # Convert the neo4j relationship to a neomodel relationship instance
            rel_instance = rel_class.inflate(results[0][0])
            logging.info(
                f"Successfully created and retrieved relationship {rel_type} "
                f"from {source_node.node_id} to {target_node.node_id}"
            )
            return rel_instance
        else:
            logging.error(
                f"Failed to retrieve relationship {rel_type} "
                f"from {source_node.node_id} to {target_node.node_id} "
                "after creation"
            )
            return None

    except Exception as e:
        logging.error(
            f"Error in create_and_return_relationship: {str(e)}\n"
            f"Source: {source_node.node_id}, Target: {target_node.node_id}, "
            f"Rel Type: {rel_class.__name__}"
        )
        raise


# END RELATIONSHIP HELPER FUNCTIONS =============================================


# BEGIN BUILD RELATIONSHIP FUNCTION ==============================================


async def build_relationships(
    nodes_dict: Dict[str, List[graph_db_models.AsyncStructuredNode]],
) -> None:
    """
    Build relationships between nodes based on defined mappings and business rules.

    This function handles relationship creation with proper validation, deduplication,
    and property setting. It uses a RelationshipTracker to prevent duplicate relationships
    and enforce cardinality constraints.

    Args:
        nodes_dict: Dictionary mapping node types to lists of nodes

    Raises:
        ValueError: If invalid node types or relationship mappings are encountered
    """
    logging.info("Starting relationship building process")
    tracker = RelationshipTracker()

    try:
        # Get all possible relationship types for the node types
        relationship_types = set()
        for node_type in nodes_dict.keys():
            if node_type in graph_db_models.RELATIONSHIP_MAPPING:
                for _, rel_info in graph_db_models.RELATIONSHIP_MAPPING[
                    node_type
                ].items():
                    relationship_types.add(
                        rel_info[1]
                    )  # Add the relationship type string

        # Initialize relationship tracking
        all_node_ids = [
            node.node_id for nodes in nodes_dict.values() for node in nodes
        ]
        await tracker.fetch_existing_relationships(
            list(nodes_dict.keys()), all_node_ids, list(relationship_types)
        )

        # Pre-process MSRCPost nodes if present
        if "MSRCPost" in nodes_dict:
            nodes_dict["MSRCPost"] = await sort_and_update_msrc_nodes(
                nodes_dict["MSRCPost"]
            )

        # Process each node type
        for node_type_str, nodes in nodes_dict.items():
            if node_type_str not in graph_db_models.RELATIONSHIP_MAPPING:
                logging.warning(
                    "No relationship mappings found for node type:"
                    f" {node_type_str}"
                )
                continue
            logging.info("Processing node type: " + node_type_str)
            await _process_node_type_relationships(
                node_type_str=node_type_str,
                nodes=nodes,
                nodes_dict=nodes_dict,
                tracker=tracker,
            )

    except Exception as e:
        error_msg = f"Error building relationships: {str(e)}"
        logging.error(error_msg)
        raise ValueError(error_msg) from e


async def _process_node_type_relationships(
    node_type_str: str,
    nodes: List[AsyncStructuredNode],
    nodes_dict: Dict[str, List[AsyncStructuredNode]],
    tracker: RelationshipTracker,
) -> None:
    """Process relationships for a specific node type."""
    for node in tqdm(nodes, desc=f"Processing {node_type_str} relationships"):
        for rel_name, rel_info in graph_db_models.RELATIONSHIP_MAPPING[
            node_type_str
        ].items():
            target_model, rel_type, rel_class = rel_info
            target_nodes = nodes_dict.get(target_model, [])
            logging.info("_process_node_type_relationships about to await")
            await _process_node_relationships(
                source_node=node,
                target_nodes=target_nodes,
                rel_name=rel_name,
                rel_info=rel_info,
                rel_type=rel_type,
                rel_class=rel_class,
                tracker=tracker,
            )


async def _process_node_relationships(
    source_node: graph_db_models.AsyncStructuredNode,
    target_nodes: List[graph_db_models.AsyncStructuredNode],
    rel_name: str,
    rel_info: Any,
    rel_type: str,
    rel_class: Type[graph_db_models.AsyncStructuredRel],
    tracker: RelationshipTracker,
) -> None:
    """Process relationships between a source node and potential target nodes."""
    relationship = getattr(source_node, rel_name)

    for target_node in target_nodes:
        try:
            if not await should_be_related(source_node, target_node, rel_info):
                continue
            logging.info(
                "_process_node_relationships about to call"
                " tracker.relationship_exists"
            )
            if tracker.relationship_exists(source_node, rel_type, target_node):
                logging.debug(
                    f"Skipping existing relationship: {source_node.node_id} "
                    f"-[{rel_type}]-> {target_node.node_id}"
                )
                continue
            logging.info(
                "_process_node_relationships about to call"
                " _create_and_configure_relationship"
            )
            # Create and configure the relationship
            await _create_and_configure_relationship(
                relationship=relationship,
                source_node=source_node,
                target_node=target_node,
                rel_class=rel_class,
                rel_type=rel_type,
            )

        except Exception as e:
            logging.error(
                f"Error processing relationship {rel_type} between "
                f"{source_node.node_id} and {target_node.node_id}: {str(e)}"
            )


async def _create_and_configure_relationship(
    relationship,
    source_node: graph_db_models.AsyncStructuredNode,
    target_node: graph_db_models.AsyncStructuredNode,
    rel_class: Type[graph_db_models.AsyncStructuredRel],
    rel_type: str,
) -> None:
    """Create and configure a relationship with appropriate properties."""
    try:
        logging.debug(
            f"Creating relationship of type {rel_class.__name__}-"
            f" relationship:{relationship}"
        )
        logging.debug(
            "Source node:"
            f" {source_node.node_id} ({type(source_node).__name__})"
        )
        logging.debug(
            "Target node:"
            f" {target_node.node_id} ({type(target_node).__name__})"
        )

        # Find the relationship attribute name that matches our criteria
        rel_attr_name = next(
            name
            for name, rel in source_node.defined_properties(
                aliases=True, properties=False
            ).items()
            if isinstance(rel, AsyncRelationshipTo)
            and rel.definition["relation_type"] == rel_type
            and rel.definition["node_class"] == type(target_node)
        )

        # Get the actual relationship manager instance using the attribute name
        rel_manager = getattr(source_node, rel_attr_name)
        if not hasattr(rel_manager, "connect"):
            raise ValueError(
                f"Relationship manager for '{rel_attr_name}' does not support"
                f" 'connect'. Got {type(rel_manager).__name__}. Ensure the"
                " relationship is correctly initialized."
            )
        # Base properties that all relationships need
        base_properties = {
            'source_node_id': source_node.node_id,
            'target_node_id': target_node.node_id,
            'relationship_type': rel_type,
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
        }

        # Create and configure the relationship based on type
        if rel_class == graph_db_models.AsyncSymptomRel:
            severity = await calculate_severity(source_node, target_node)
            confidence = await calculate_confidence(source_node, target_node)
            description = await generate_description(source_node, target_node)
            reported_date = await get_reported_date(source_node, target_node)

            initial_props = {
                "severity": severity or "medium",  # Default to medium if None
                "confidence": (
                    confidence if confidence is not None else 50
                ),  # Default to 50% if None
                "description": (
                    description or ""
                ),  # Default to empty string if None
                "reported_date": reported_date
                or datetime.now().strftime(
                    "%Y-%m-%d"
                ),  # Default to current date if None
            }
            properties = (
                {**base_properties, **initial_props}
                if initial_props
                else base_properties
            )
            rel_instance = await rel_manager.connect(target_node, properties)
            if not isinstance(rel_instance, AsyncStructuredRel):
                raise ValueError(
                    "Failed to instantiate relationship as AsyncStructuredRel"
                )
            logging.debug(f"Symptom properties set: {properties}")

        elif rel_class == graph_db_models.AsyncCauseRel:
            severity = await calculate_severity(source_node, target_node)
            confidence = await calculate_confidence(source_node, target_node)
            description = await generate_description(source_node, target_node)
            reported_date = await get_reported_date(source_node, target_node)

            initial_props = {
                "severity": severity or "medium",  # Default to medium if None
                "confidence": (
                    confidence if confidence is not None else 50
                ),  # Default to 50% if None
                "description": (
                    description or ""
                ),  # Default to empty string if None
                "reported_date": reported_date
                or datetime.now().strftime(
                    "%Y-%m-%d"
                ),  # Default to current date if None
            }
            properties = (
                {**base_properties, **initial_props}
                if initial_props
                else base_properties
            )
            rel_instance = await rel_manager.connect(target_node, properties)
            if not isinstance(rel_instance, AsyncStructuredRel):
                raise ValueError(
                    "Failed to instantiate relationship as AsyncStructuredRel"
                )
            logging.debug(f"Cause properties set: {properties}")

        elif rel_class == graph_db_models.AsyncFixRel:
            severity = await calculate_severity(source_node, target_node)
            confidence = await calculate_confidence(source_node, target_node)
            description = await generate_description(source_node, target_node)
            reported_date = await get_reported_date(source_node, target_node)

            initial_props = {
                "severity": severity or "medium",  # Default to medium if None
                "confidence": (
                    confidence if confidence is not None else 50
                ),  # Default to 50% if None
                "description": (
                    description or ""
                ),  # Default to empty string if None
                "reported_date": reported_date
                or datetime.now().strftime(
                    "%Y-%m-%d"
                ),  # Default to current date if None
            }
            properties = (
                {**base_properties, **initial_props}
                if initial_props
                else base_properties
            )
            rel_instance = await rel_manager.connect(target_node, properties)
            if not isinstance(rel_instance, AsyncStructuredRel):
                raise ValueError(
                    "Failed to instantiate relationship as AsyncStructuredRel"
                )
            logging.debug(f"Fix properties set: {properties}")

        elif rel_class == graph_db_models.AsyncToolRel:
            confidence = await calculate_confidence(source_node, target_node)
            description = await generate_description(source_node, target_node)
            reported_date = await get_reported_date(source_node, target_node)

            initial_props = {
                "confidence": (
                    confidence if confidence is not None else 50
                ),  # Default to 50% if None
                "description": (
                    description or ""
                ),  # Default to empty string if None
                "reported_date": reported_date
                or datetime.now().strftime(
                    "%Y-%m-%d"
                ),  # Default to current date if None
            }
            properties = (
                {**base_properties, **initial_props}
                if initial_props
                else base_properties
            )
            rel_instance = await rel_manager.connect(target_node, properties)
            if not isinstance(rel_instance, AsyncStructuredRel):
                raise ValueError(
                    "Failed to instantiate relationship as AsyncStructuredRel"
                )
            logging.debug(f"Tool properties set: {properties}")

        elif rel_class == graph_db_models.AsyncAffectsProductRel:
            initial_props = {
                'impact_rating': await calculate_impact_rating(
                    source_node, target_node
                ),
                'severity': await calculate_severity(
                    source_node, target_node, rel_class
                ),
                'affected_versions': await get_affected_versions(
                    source_node, target_node
                ),
                'patched_in_version': await get_patched_version(
                    source_node, target_node
                ),
            }
            properties = {**base_properties, **initial_props}
            rel_instance = await rel_manager.connect(target_node, properties)
            if not isinstance(rel_instance, AsyncStructuredRel):
                raise ValueError(
                    "Failed to instantiate relationship as AsyncStructuredRel"
                )
            logging.debug(f"Product properties set: {properties}")

        elif rel_class == graph_db_models.AsyncHasUpdatePackageRel:
            release_date = await get_release_date(source_node, target_node)
            is_cumulative = await has_cumulative(source_node, target_node)
            is_dynamic = await has_dynamic(source_node, target_node)

            initial_props = {
                "release_date": release_date
                or datetime.now().strftime(
                    "%Y-%m-%d"
                ),  # Default to current date
                "has_cumulative": (
                    is_cumulative if is_cumulative is not None else False
                ),
                "has_dynamic": is_dynamic if is_dynamic is not None else False,
            }
            properties = (
                {**base_properties, **initial_props}
                if initial_props
                else base_properties
            )
            rel_instance = await rel_manager.connect(target_node, properties)
            if not isinstance(rel_instance, AsyncStructuredRel):
                raise ValueError(
                    "Failed to instantiate relationship as AsyncStructuredRel"
                )
            logging.debug(f"Update package properties set: {properties}")

        elif rel_class == graph_db_models.AsyncPreviousVersionRel:
            if not target_node or not target_node.node_id:
                raise ValueError(
                    "Target node with valid node_id is required for previous"
                    " version relationship"
                )

            version_difference = await calculate_version_difference(
                source_node, target_node
            )
            changes_summary = await generate_changes_summary(
                source_node, target_node
            )

            initial_props = {
                "version_difference": (
                    version_difference or ""
                ),  # Optional, default to empty string
                "changes_summary": (
                    changes_summary or ""
                ),  # Optional, default to empty string
                "previous_version_id": (
                    target_node.node_id
                ),  # Required, from target node
            }
            properties = (
                {**base_properties, **initial_props}
                if initial_props
                else base_properties
            )
            rel_instance = await rel_manager.connect(target_node, properties)
            if not isinstance(rel_instance, AsyncStructuredRel):
                raise ValueError(
                    "Failed to instantiate relationship as AsyncStructuredRel"
                )
            logging.debug(f"Previous version properties set: {properties}")

        elif rel_class == graph_db_models.AsyncPreviousMessageRel:
            if not target_node or not target_node.node_id:
                raise ValueError(
                    "Target node with valid node_id is required for previous"
                    " message relationship"
                )

            initial_props = {
                "previous_id": (
                    target_node.node_id
                ),  # Required, from target node
            }
            properties = (
                {**base_properties, **initial_props}
                if initial_props
                else base_properties
            )
            rel_instance = await rel_manager.connect(target_node, properties)
            if not isinstance(rel_instance, AsyncStructuredRel):
                raise ValueError(
                    "Failed to instantiate relationship as AsyncStructuredRel"
                )
            logging.debug(f"Previous message properties set: {properties}")

        elif rel_class == graph_db_models.AsyncReferencesRel:
            relevance_score = await calculate_relevance_score(
                source_node, target_node
            )
            context = await extract_context(source_node, target_node)
            cited_section = await extract_cited_section(
                source_node, target_node
            )

            initial_props = {
                "relevance_score": max(0, min(100, relevance_score or 0)),
                "context": context or "",
                "cited_section": cited_section or "",
            }
            properties = (
                {**base_properties, **initial_props}
                if initial_props
                else base_properties
            )
            rel_instance = await rel_manager.connect(target_node, properties)
            if not isinstance(rel_instance, AsyncStructuredRel):
                raise ValueError(
                    "Failed to instantiate relationship as AsyncStructuredRel"
                )
            logging.debug(f"References properties set: {properties}")

        else:
            # Default case for AsyncZeroToManyRel and unknown types
            rel_instance = await rel_manager.connect(
                target_node, base_properties
            )
            if not isinstance(rel_instance, AsyncStructuredRel):
                raise ValueError(
                    "Failed to instantiate relationship as AsyncStructuredRel"
                )
            logging.debug(f"Basic properties set: {base_properties}")

        logging.debug(
            f"Successfully created and configured relationship: {rel_type}"
        )
        return rel_instance

    except Exception as e:
        logging.error(f"Error in _create_and_configure_relationship: {str(e)}")
        logging.error(f"Source node: {source_node.node_id}")
        logging.error(f"Target node: {target_node.node_id}")
        logging.error(f"Relationship class: {rel_class.__name__}")
        raise


# END BUILD RELATIONSHIP FUNCTION ================================================


# BEGIN RELATIONSHIP SETTER FUNCTIONS ============================================
async def _get_relationship_name(
    source_node: graph_db_models.AsyncStructuredNode,
    target_node: graph_db_models.AsyncStructuredNode,
    rel_class: Type[graph_db_models.AsyncStructuredRel],
) -> Optional[str]:
    """
    Get the relationship name from RELATIONSHIP_MAPPING based on source and target nodes.

    Args:
        source_node: Source node of the relationship
        target_node: Target node of the relationship
        rel_class: Expected relationship class

    Returns:
        str: Relationship name if found, None otherwise
    """
    source_node_type = source_node.__class__.__name__
    target_node_type = target_node.__class__.__name__

    try:
        return next(
            name
            for name, (target, _, cls) in graph_db_models.RELATIONSHIP_MAPPING[
                source_node_type
            ].items()
            if target == target_node_type and cls == rel_class
        )
    except (KeyError, StopIteration):
        logging.warning(
            "No relationship mapping found for"
            f" {source_node_type}->{target_node_type} with class"
            f" {rel_class.__name__}"
        )
        return None


async def _save_relationship_properties(
    relationship: AsyncStructuredRel, properties: Dict[str, Any]
):
    try:
        logging.debug(
            "Setting properties for relationship"
            f" {type(relationship).__name__}"
        )
        for key, value in properties.items():
            # Get property definition to access default value
            prop_def = getattr(type(relationship), key, None)

            if value is None and prop_def and hasattr(prop_def, 'default'):
                # Use default value if property has one defined
                value = prop_def.default
                if callable(value):
                    value = value()
                logging.info(f"Using default value {value} for property {key}")
            elif value is None:
                logging.warning(f"Skipping None value for property {key}")
                continue

            logging.debug(f"Setting {key}={value} (type: {type(value)})")
            setattr(relationship, key, value)

        await relationship.save()
    except Exception as e:
        logging.error(f"Error saving relationship properties: {str(e)}")
        logging.error(f"Properties that failed: {properties}")
        raise


async def calculate_confidence(source_node, target_node):
    """
    Calculate the confidence level for a SymptomCauseFix relationship.
    This function determines how confident we are about the relationship between a symptom, cause, or fix.
    """
    try:
        # Extract relevant properties with default values
        source_reliability = getattr(source_node, "reliability", "MEDIUM")
        target_reliability = getattr(target_node, "reliability", "MEDIUM")

        def get_reliability_value(reliability):
            if isinstance(reliability, (int, float)):
                return float(reliability)
            reliability_map = {
                "HIGH": 90,
                "MEDIUM": 50,
                "LOW": 10,
                "": 50,  # Default value for empty string
            }
            return reliability_map.get(str(reliability).upper(), 50)

        # Get numeric values with default of 50 for unknown values
        source_value = get_reliability_value(source_reliability)
        target_value = get_reliability_value(target_reliability)
        confidence_value = int((source_value + target_value) / 2)
        # Return average confidence score
        return confidence_value if confidence_value is not None else 25

    except Exception as e:
        logging.error(f"Error calculating confidence: {e}")
        return 25  # Return default value on error


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
    source_impact = getattr(
        source_node, "impact_type", "medium"
    )  # Default to medium if not specified
    source_severity = getattr(
        source_node, "severity_type", "medium"
    )  # Default to medium if not specified

    # Map impact and severity to standard levels
    impact_map = {
        "NIT": "medium",  # Default NIT (Not Impact Type) to medium
        "low": "low",
        "medium": "medium",
        "high": "high",
        "critical": "critical",
    }

    # Use the higher of impact_type or severity_type
    severity_levels = ["low", "medium", "high", "critical"]
    impact_level = impact_map.get(source_impact, "medium")
    severity_level = impact_map.get(source_severity, "medium")

    # Return the higher severity level
    if severity_levels.index(severity_level) > severity_levels.index(
        impact_level
    ):
        return severity_level
    return impact_level


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
        if isinstance(data, list) and all(
            isinstance(item, dict) for item in data
        ):
            for item in data:
                if (
                    "update_type" in item
                    and "cumulative" in item["update_type"]
                ):
                    print(item["update_type"])
                    return True

    # Is downloadable_packages a list of dictionaries?
    if (
        isinstance(downloadable_packages_raw, list)
        and downloadable_packages_raw
        and all(isinstance(item, dict) for item in downloadable_packages_raw)
    ):
        print(
            f"downloadable_packages type: {type(downloadable_packages_raw)} -"
            " All Dicts"
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

            if (
                "update_type" in data
                and data["update_type"].lower() == "cumulative"
            ):

                return True

    else:
        print(
            "downloadable_packages is misbehaving:"
            f" {downloadable_packages_raw}"
        )

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
        if isinstance(data, list) and all(
            isinstance(item, dict) for item in data
        ):
            for item in data:
                if (
                    "update_type" in item
                    and "cumulative" in item["update_type"]
                ):

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
        print(
            f"downloadable_packages_raw: len({len(downloadable_packages_raw)})"
        )
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
            if (
                "update_type" in data
                and "dynamic" in data.get("update_type").lower()
            ):

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
            f"Error: '{source_version_str}' or '{target_version_str}' is not a"
            f" valid float string.\n{ve}"
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


async def calculate_severity(source_node, target_node, relationship_type=None):
    """Calculate severity based on source node properties and relationship type."""
    try:
        # First try to get severity from source node
        if hasattr(source_node, "severity_type"):
            severity = source_node.severity_type

            # For AffectsProduct relationships, map NST/nst to medium
            if relationship_type == graph_db_models.AsyncAffectsProductRel:
                if severity in ["NST", "nst"]:
                    return "medium"
                # Map important to high for product relationships
                if severity == "important":
                    return "high"
                # Only return valid choices for product relationships
                if severity in ["low", "medium", "high", "critical"]:
                    return severity
                return "medium"  # Default for product relationships

            # For Symptom/Cause/Fix relationships, return as is if valid
            if severity in [
                "low",
                "medium",
                "high",
                "important",
                "nst",
                "NST",
            ]:
                return severity

        # Default values based on relationship type
        if relationship_type == graph_db_models.AsyncAffectsProductRel:
            return "medium"  # Default for product relationships

        return "NST"  # Default for symptom/cause/fix relationships

    except Exception as e:
        logging.error(f"Error calculating severity: {str(e)}")
        # Return safe defaults based on relationship type
        return (
            "medium"
            if relationship_type == graph_db_models.AsyncAffectsProductRel
            else "NST"
        )


async def get_previous_message_id(source_node, target_node):
    if hasattr(target_node, "previous_id"):
        return target_node.previous_id
    return None


# Updated utility functions
async def set_symptom_cause_fix_tool_properties(
    relationship, source_node, target_node
):
    """Set properties for Symptom, Cause, Fix, and Tool relationships."""
    try:
        # Determine the relationship class and name if we have a relationship instance
        rel_class = None
        rel_name = None
        if relationship is not None:
            if isinstance(relationship, graph_db_models.AsyncSymptomRel):
                rel_class = graph_db_models.AsyncSymptomRel
            elif isinstance(relationship, graph_db_models.AsyncCauseRel):
                rel_class = graph_db_models.AsyncCauseRel
            elif isinstance(relationship, graph_db_models.AsyncFixRel):
                rel_class = graph_db_models.AsyncFixRel
            elif isinstance(relationship, graph_db_models.AsyncToolRel):
                rel_class = graph_db_models.AsyncToolRel
            else:
                logging.warning(
                    f"Unknown relationship type: {type(relationship)}"
                )
                return None

            rel_name = await _get_relationship_name(
                source_node, target_node, rel_class
            )
            if not rel_name:
                return None

            logging.info(
                "Setting properties for existing"
                f" {rel_class.__name__} relationship"
            )
        else:
            logging.info(
                "Preparing properties for new symptom/cause/fix/tool"
                " relationship"
            )

        logging.info(
            "Source node:"
            f" {source_node.node_id} ({type(source_node).__name__})"
        )
        logging.info(
            "Target node:"
            f" {target_node.node_id} ({type(target_node).__name__})"
        )

        # Compute property values
        severity = await calculate_severity(source_node, target_node)
        confidence = await calculate_confidence(source_node, target_node)
        description = await generate_description(source_node, target_node)
        reported_date = await get_reported_date(source_node, target_node)

        # Properties common to all these relationships
        properties = {
            "severity": severity or "medium",  # Default to medium if None
            "confidence": (
                confidence if confidence is not None else 50
            ),  # Default to 50% if None
            "description": (
                description or ""
            ),  # Default to empty string if None
            "reported_date": reported_date
            or datetime.now().strftime(
                "%Y-%m-%d"
            ),  # Default to current date if None
        }

        # If we have an existing relationship, set its properties and type
        if relationship is not None and rel_class and rel_name:
            await set_relationship_type(
                relationship, source_node.__class__.__name__, rel_name
            )
            await _save_relationship_properties(relationship, properties)
            logging.info(
                f"Updated properties for {rel_class.__name__} relationship"
            )
        else:
            logging.info(
                "Prepared properties for new symptom/cause/fix/tool"
                " relationship"
            )

        return properties

    except Exception as e:
        logging.error(
            "Error setting symptom/cause/fix/tool properties between"
            f" {source_node.node_id} and {target_node.node_id}: {str(e)}"
        )
        raise


async def set_affects_product_properties(
    relationship, source_node, target_node
):
    """Set properties for AffectsProduct relationships."""
    if not relationship:
        return

    try:
        rel_name = await _get_relationship_name(
            source_node, target_node, graph_db_models.AsyncAffectsProductRel
        )
        if not rel_name:
            return

        # Handle PatchManagementPost specially
        if source_node.__class__.__name__ == "PatchManagementPost":
            properties = {
                "impact_rating": (
                    "medium"
                ),  # Default to medium for PatchManagementPost
                "severity": (
                    "medium"
                ),  # Default to medium for PatchManagementPost
                "affected_versions": source_node.build_numbers,
                "patched_in_version": "",
            }
        else:
            properties = {
                "impact_rating": await calculate_impact_rating(
                    source_node, target_node
                ),
                "severity": await calculate_severity(source_node, target_node),
                "affected_versions": await get_affected_versions(
                    source_node, target_node
                ),
                "patched_in_version": await get_patched_version(
                    source_node, target_node
                ),
            }

        await set_relationship_type(
            relationship, source_node.__class__.__name__, rel_name
        )
        await _save_relationship_properties(relationship, properties)

    except Exception as e:
        logging.error(
            "Error setting affects_product properties between"
            f" {source_node.node_id} and {target_node.node_id}: {str(e)}"
        )
        raise


async def set_has_update_package_properties(
    relationship: graph_db_models.AsyncHasUpdatePackageRel,
    source_node: AsyncStructuredNode,
    target_node: AsyncStructuredNode,
) -> Dict[str, str]:
    """
    Set properties for an existing HasUpdatePackage relationship instance.

    Args:
        relationship (AsyncHasUpdatePackageRel): The relationship instance to update.
        source_node (AsyncStructuredNode): The source node in the relationship.
        target_node (AsyncStructuredNode): The target node in the relationship.

    Returns:
        Dict[str, str]: A dictionary of updated properties.
    """
    try:
        # Validate the relationship instance type
        if not isinstance(
            relationship, graph_db_models.AsyncHasUpdatePackageRel
        ):
            raise ValueError(
                "Expected instance of AsyncHasUpdatePackageRel, got"
                f" {type(relationship).__name__}"
            )

        # Compute or retrieve properties
        release_date = await get_release_date(source_node, target_node)
        is_cumulative = await has_cumulative(source_node, target_node)
        is_dynamic = await has_dynamic(source_node, target_node)

        properties = {
            "release_date": release_date
            or datetime.now().strftime("%Y-%m-%d"),  # Default to current date
            "has_cumulative": (
                is_cumulative if is_cumulative is not None else False
            ),
            "has_dynamic": is_dynamic if is_dynamic is not None else False,
        }

        # Update the relationship properties
        for key, value in properties.items():
            setattr(relationship, key, value)

        # Save the updated relationship
        await relationship.save()
        logging.info(
            "Updated properties for HasUpdatePackage relationship:"
            f" {relationship}"
        )

        return properties

    except Exception as e:
        logging.error(
            "Error updating HasUpdatePackage properties between"
            f" {source_node.node_id} and {target_node.node_id}: {str(e)}"
        )
        raise


async def set_previous_version_properties(
    relationship: graph_db_models.AsyncPreviousVersionRel,
    source_node: AsyncStructuredNode,
    target_node: AsyncStructuredNode,
) -> Dict[str, str]:
    """
    Set properties for an existing PreviousVersion relationship instance.

    Args:
        relationship (AsyncPreviousVersionRel): The relationship instance to update.
        source_node (AsyncStructuredNode): The source node in the relationship.
        target_node (AsyncStructuredNode): The target node in the relationship.

    Returns:
        Dict[str, str]: A dictionary of updated properties.
    """
    try:
        # Validate the relationship instance type
        if not isinstance(
            relationship, graph_db_models.AsyncPreviousVersionRel
        ):
            raise ValueError(
                "Expected instance of AsyncPreviousVersionRel, got"
                f" {type(relationship).__name__}"
            )

        # Compute or retrieve properties
        if not target_node or not target_node.node_id:
            raise ValueError(
                "Target node with valid node_id is required for previous"
                " version relationship"
            )

        version_difference = await calculate_version_difference(
            source_node, target_node
        )
        changes_summary = await generate_changes_summary(
            source_node, target_node
        )

        properties = {
            "version_difference": (
                version_difference or ""
            ),  # Optional, default to empty string
            "changes_summary": (
                changes_summary or ""
            ),  # Optional, default to empty string
            "previous_version_id": (
                target_node.node_id
            ),  # Required, from target node
        }

        # Update the relationship properties
        for key, value in properties.items():
            setattr(relationship, key, value)

        # Save the updated relationship
        await relationship.save()
        logging.info(
            "Updated properties for PreviousVersion relationship:"
            f" {relationship}"
        )

        return properties

    except Exception as e:
        logging.error(
            "Error updating PreviousVersion properties between"
            f" {source_node.node_id} and {target_node.node_id}: {str(e)}"
        )
        raise


async def set_previous_message_properties(
    relationship: graph_db_models.AsyncPreviousMessageRel,
    source_node: AsyncStructuredNode,
    target_node: AsyncStructuredNode,
) -> Dict[str, str]:
    """
    Set properties for an existing PreviousMessage relationship instance.

    Args:
        relationship (AsyncPreviousMessageRel): The relationship instance to update.
        source_node (AsyncStructuredNode): The source node in the relationship.
        target_node (AsyncStructuredNode): The target node in the relationship.

    Returns:
        Dict[str, str]: A dictionary of updated properties.
    """
    try:
        # Validate the relationship instance type
        if not isinstance(
            relationship, graph_db_models.AsyncPreviousMessageRel
        ):
            raise ValueError(
                "Expected instance of AsyncPreviousMessageRel, got"
                f" {type(relationship).__name__}"
            )

        # Compute or retrieve properties
        if not target_node or not target_node.node_id:
            raise ValueError(
                "Target node with valid node_id is required for previous"
                " message relationship"
            )

        properties = {
            "previous_id": target_node.node_id,  # Required, from target node
        }

        # Update the relationship properties
        for key, value in properties.items():
            setattr(relationship, key, value)

        # Save the updated relationship
        await relationship.save()
        logging.info(
            "Updated properties for PreviousMessage relationship:"
            f" {relationship}"
        )

        return properties

    except Exception as e:
        logging.error(
            "Error updating PreviousMessage properties between"
            f" {source_node.node_id} and {target_node.node_id}: {str(e)}"
        )
        raise


async def set_references_properties(
    relationship: graph_db_models.AsyncReferencesRel,
    source_node: AsyncStructuredNode,
    target_node: AsyncStructuredNode,
) -> Dict[str, str]:
    """
    Set properties for an existing References relationship instance.

    Args:
        relationship (AsyncStructuredRel): The relationship instance to update.
        source_node (AsyncStructuredNode): The source node in the relationship.
        target_node (AsyncStructuredNode): The target node in the relationship.

    Returns:
        Dict[str, str]: A dictionary of updated properties.
    """
    try:
        # Validate the relationship instance type
        if not isinstance(relationship, graph_db_models.AsyncReferencesRel):
            raise ValueError(
                "Expected instance of AsyncReferencesRel, got"
                f" {type(relationship).__name__}"
            )

        # Compute or retrieve properties
        relevance_score = await calculate_relevance_score(
            source_node, target_node
        )
        context = await extract_context(source_node, target_node)
        cited_section = await extract_cited_section(source_node, target_node)

        properties: Dict[str, str] = {
            "relevance_score": max(
                0, min(100, relevance_score or 0)
            ),  # Ensure between 0-100
            "context": context or "",  # Optional, default to empty string
            "cited_section": (
                cited_section or ""
            ),  # Optional, default to empty string
        }

        # Update the relationship properties
        for key, value in properties.items():
            setattr(relationship, key, value)

        # Save the updated relationship
        await relationship.save()
        logging.info(f"Updated properties for relationship: {relationship}")

        return properties

    except Exception as e:
        logging.error(
            "Error updating references properties between"
            f" {source_node.node_id} and {target_node.node_id}: {str(e)}"
        )
        raise


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
        rel_type = graph_db_models.RELATIONSHIP_MAPPING[source_node_type][
            rel_name
        ][1]
        relationship.relationship_type = rel_type
    else:
        # Default to the rel_name if not found in RELATIONSHIP_MAPPING
        relationship.relationship_type = rel_name.upper()


# END RELATIONSHIP SETTER FUNCTIONS ==============================================


# =================== BEGIN PIPELINE HELPERS ================


async def build_relationships_in_batches(
    nodes_dict: Dict[str, List[graph_db_models.AsyncStructuredNode]],
    batch_size: int = 1000,
    checkpoint_file: str = "relationship_checkpoint.json",
) -> None:
    """
    Build relationships between nodes in batches with checkpointing and progress tracking.

    Args:
        nodes_dict: Dictionary mapping node types to lists of nodes
        batch_size: Number of relationships to process in each batch
        checkpoint_file: File to store progress information
    """
    logging.info("Starting relationship building process with batching")
    tracker = RelationshipTracker(batch_size=batch_size)

    # Load checkpoint if exists
    processed_pairs = set()
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                processed_pairs = set(tuple(pair) for pair in json.load(f))
            logging.info(
                f"Loaded {len(processed_pairs)} processed pairs from"
                " checkpoint"
            )
        except Exception as e:
            logging.warning(f"Error loading checkpoint file: {str(e)}")

    try:
        # Get all possible relationship types for the node types
        relationship_types = set()
        for node_type in nodes_dict.keys():
            if node_type in graph_db_models.RELATIONSHIP_MAPPING:
                for _, rel_info in graph_db_models.RELATIONSHIP_MAPPING[
                    node_type
                ].items():
                    relationship_types.add(
                        rel_info[1]
                    )  # Add the relationship type string

        # Initialize relationship tracking
        all_node_ids = [
            node.node_id for nodes in nodes_dict.values() for node in nodes
        ]
        await tracker.fetch_existing_relationships(
            list(nodes_dict.keys()), all_node_ids, list(relationship_types)
        )
        logging.info("tracker attempted to fetch existing relationships")
        # Pre-process MSRCPost nodes if present
        if "MSRCPost" in nodes_dict:
            nodes_dict["MSRCPost"] = await sort_and_update_msrc_nodes(
                nodes_dict["MSRCPost"]
            )

        # Process each node type in batches
        total_pairs = sum(len(nodes) for nodes in nodes_dict.values())
        with tqdm(total=total_pairs, desc="Building relationships") as pbar:
            for node_type_str, nodes in nodes_dict.items():
                if node_type_str not in graph_db_models.RELATIONSHIP_MAPPING:
                    logging.warning(
                        "No relationship mappings found for node type:"
                        f" {node_type_str}"
                    )
                    continue
                logging.info(f"Processing node type: {node_type_str}")
                # Process nodes in batches
                for i in range(0, len(nodes), batch_size):
                    batch = nodes[i : i + batch_size]
                    try:
                        await _process_node_batch(
                            node_type_str=node_type_str,
                            nodes=batch,
                            nodes_dict=nodes_dict,
                            tracker=tracker,
                            processed_pairs=processed_pairs,
                            checkpoint_file=checkpoint_file,
                        )
                    except Exception as e:
                        logging.error(
                            f"Error processing batch for {node_type_str}:"
                            f" {str(e)}"
                        )
                        # Save checkpoint before re-raising
                        _save_checkpoint(processed_pairs, checkpoint_file)
                        raise

                    pbar.update(len(batch))

        # Clean up checkpoint file after successful completion
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

    except Exception as e:
        error_msg = f"Error building relationships: {str(e)}"
        logging.error(error_msg)
        raise ValueError(error_msg) from e


async def _process_node_batch(
    node_type_str: str,
    nodes: List[AsyncStructuredNode],
    nodes_dict: Dict[str, List[AsyncStructuredNode]],
    tracker: RelationshipTracker,
    processed_pairs: Set[Tuple[str, str]],
    checkpoint_file: str,
) -> None:
    """Process a batch of nodes for relationship building."""
    for node in tqdm(nodes, desc=f"Processing {node_type_str} relationships"):
        for rel_name, rel_info in graph_db_models.RELATIONSHIP_MAPPING[
            node_type_str
        ].items():
            target_model, rel_type, rel_class = rel_info
            target_nodes = nodes_dict.get(target_model, [])
            logging.debug(
                f"Processing relationship between {node.node_id} and"
                f" {target_model}"
            )
            for target_node in target_nodes:
                # Skip if this pair has already been processed
                pair_key = (node.node_id, target_node.node_id)
                if pair_key in processed_pairs:
                    continue
                logging.debug(
                    f"Processing relationship between {node.node_id} and"
                    f" {target_node.node_id}"
                )
                try:
                    await _process_single_relationship(
                        source_node=node,
                        target_node=target_node,
                        rel_name=rel_name,
                        rel_info=rel_info,
                        rel_type=rel_type,
                        rel_class=rel_class,
                        tracker=tracker,
                    )

                    # Mark as processed and update checkpoint
                    processed_pairs.add(pair_key)
                    if len(processed_pairs) % 1000 == 0:  # Periodic checkpoint
                        _save_checkpoint(processed_pairs, checkpoint_file)

                except Exception as e:
                    logging.error(
                        "Error processing relationship between"
                        f" {node.node_id} and {target_node.node_id}: {str(e)}"
                    )
                    raise


async def _process_single_relationship(
    source_node: AsyncStructuredNode,
    target_node: AsyncStructuredNode,
    rel_name: str,
    rel_info: Any,
    rel_type: str,
    rel_class: Type[AsyncStructuredRel],
    tracker: RelationshipTracker,
) -> None:
    """Process a single relationship between two nodes."""
    if not await should_be_related(source_node, target_node, rel_info):
        return

    if tracker.relationship_exists(source_node, rel_type, target_node):
        logging.debug(
            f"Skipping existing relationship: {source_node.node_id} "
            f"-[{rel_type}]-> {target_node.node_id}"
        )
        return

    relationship = getattr(source_node, rel_name)
    await _create_and_configure_relationship(
        relationship=relationship,
        source_node=source_node,
        target_node=target_node,
        rel_class=rel_class,
        rel_type=rel_type,
    )


def _save_checkpoint(
    processed_pairs: Set[Tuple[str, str]], checkpoint_file: str
) -> None:
    """Save the current progress to a checkpoint file."""
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump([list(pair) for pair in processed_pairs], f)
    except Exception as e:
        logging.error(f"Error saving checkpoint: {str(e)}")


# =================== END PIPELINE HELPERS ================

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
            RELATIONSHIP_TYPE_MAPPING[key] = (
                rel_name,
                target_model,
                rel_class,
            )

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
                f"Target node type mismatch: expected {target_model}, got"
                f" {target_node.__class__.__name__}"
            )
            continue

        relationship = getattr(source_node, rel_name)
        tracker = RelationshipTracker()
        # Check if the relationship already exists
        if not tracker.relationship_exists(source_node, rel_type, target_node):
            # print(
            #     f"Creating new relationship: {node.node_label} -{rel_type}-> {target_node.node_label}"
            # )
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
                # await set_symptom_cause_fix_properties(
                #     rel_instance, source_node, target_node
                # )
                pass
            elif rel_class == graph_db_models.AsyncPreviousVersionRel:
                await set_previous_version_properties(
                    rel_instance, source_node, target_node
                )
            elif rel_class == graph_db_models.AsyncPreviousMessageRel:
                await set_previous_message_properties(
                    rel_instance, source_node, target_node
                )
            elif rel_class == graph_db_models.AsyncAffectsProductRel:
                await set_affects_product_properties(
                    rel_instance, source_node, target_node
                )
            elif rel_class == graph_db_models.AsyncHasUpdatePackageRel:
                await set_has_update_package_properties(
                    rel_instance, source_node, target_node
                )
            elif rel_class == graph_db_models.AsyncReferencesRel:
                await set_references_properties(
                    rel_instance, source_node, target_node
                )

            print(
                f"Created relationship: {source_id} -[{rel_type}]->"
                f" {target_id}"
            )


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
            "product_architecture": "x64",
            "product_version": "NV",
            "description": "Windows 11 Version 22H2 for x64-based Systems",
            "node_label": "windows_11 x64 22H2",
        }

        print(
            "\nAttempting to retrieve a ProductBuild node using get_or_none()"
            " with test_properties:"
        )
        print(f"Test properties: {test_properties}")

        print("\nTesting direct access using ProductBuild.nodes.get_or_none()")
        try:
            node = await graph_db_models.ProductBuild.nodes.get_or_none(
                node_id=test_properties["node_id"]
            )
            if node:
                print(
                    "Node retrieved successfully (direct):",
                    node.__properties__,
                )
            else:
                print("No node found (direct).")
        except Exception as e:
            print(f"Error during direct test: {e}")

        # Test case 2: Using ProductBuildService to access the model and get_or_none()
        print(
            "\nTesting access via"
            " ProductBuildService.model.nodes.get_or_none()"
        )
        try:
            # Here we're calling the internal model's nodes.get_or_none() via the service instance
            node = await product_build_service.model.nodes.get_or_none(
                node_id=test_properties["node_id"]
            )
            if node:
                print(
                    "Node retrieved successfully (via service):",
                    node.__properties__,
                )
            else:
                print("No node found (via service).")
        except Exception as e:
            print(f"Error during service test: {e}")

        # Further test with more complex unique property combinations if needed
        # You can adapt this part according to the unique properties in your model
        print(
            "\nAttempting to retrieve a ProductBuild node using get_or_none()"
            " with product_name, product_version, and product_architecture:"
        )
        node = await graph_db_models.ProductBuild.nodes.get_or_none(
            product_name=test_properties["product_name"],
            product_version=test_properties["product_version"],
            product_architecture=test_properties["product_architecture"],
        )
        if node:
            print(
                f"Node retrieved successfully with multiple properties: {node}"
            )
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
    graph_db_uri = (
        f"{credentials.protocol}://{credentials.host}:{credentials.port}"
    )
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
        "ProductService created. Handles database interactions on behalf of"
        " the node instance"
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
        print("Product created")
        # Set build numbers (example data)
        if product:
            product.set_build_numbers(
                [[10, 0, 19041, 450], [10, 0, 19042, 450]]
            )
            print(f"Build Numbers: {product.get_build_numbers()}")
            await product.save()
    else:
        print(f"Error: {message}")
        product = await product_service.get(
            "86b28b3e-c61e-4fa1-8445-158dd8dc06c6"
        )

    print(f"type: {type(product)}\n{product}")

    #     product_build_data = {
    #         "node_id": "fdbb56d3-d1aa-6819-59bd-dc326d0b64c0",
    #         "product": "Windows 11 Version 22H2 for x64-based Systems",
    #         "product_name": "windows_11",
    #         "product_architecture": "x64",
    #         "product_version": "22H2",
    #         "node_label": "ProductBuild [10,0,19045,4046]",
    #         "build_number": [10, 0, 19045, 4046],
    #         "cve_id": "CVE-2024-38027",
    #         "kb_id": "kb5040442",
    #         "published": datetime.now(),
    #         "article_url": "https://support.microsoft.com/help/5034763",
    #         "cve_url": "https://msrc.microsoft.com/update-guide/vulnerability/CVE-2024-21406",
    #         "impact_type": "spoofing",
    #         "severity_type": "important",
    #         "post_type": "Solution provided",
    #         "attack_complexity": "low",
    #         "attack_vector": "network",
    #         "exploit_code_maturity": "Proof of Concept",
    #         "exploitability": "Functional",
    #         "post_id": "CVE-2024-38027",
    #         "summary": "Sample Summary",
    #         "build_numbers": [[10, 0, 22631, 3880], [10, 0, 22621, 3880]],
    #         "text": """ \
    # Security Vulnerability
    # Released: 9 Jul 2024
    # Assigning CNA:
    # Microsoft
    # CVE-2024-38027
    # Impact: Denial of Service
    # Max Severity: Important
    # Weakness:
    # CWE-400: Uncontrolled Resource Consumption
    # CVSS Source:
    # Microsoft
    # CVSS:3.1 6.5 / 5.7
    # Base score metrics: 6.5 / Temporal score metrics: 5.7
    # Base score metrics: 6.5 / Temporal score metrics: 5.7
    # Base score metrics
    # (8)
    # Attack Vector
    # This metric reflects the context by which vulnerability exploitation is possible. The Base Score increases the more remote (logically, and physically) an attacker can be in order to exploit the vulnerable component.
    # Adjacent
    # The vulnerable component is bound to the network stack, but the attack is limited at the protocol level to a logically adjacent topology. This can mean an attack must be launched from the same shared physical (e.g., Bluetooth or IEEE 802.11) or logical (e.g., local IP subnet) network, or from within a secure or otherwise limited administrative domain (e.g., MPLS, secure VPN to an administrative network zone)
    # Attack Complexity
    # This metric describes the conditions beyond the attacker's control that must exist in order to exploit the vulnerability. Such conditions may require the collection of more information about the target or computational exceptions. The assessment of this metric excludes any requirements for user interaction in order to exploit the vulnerability. If a specific configuration is required for an attack to succeed, the Base metrics should be scored assuming the vulnerable component is in that configuration.
    # Low
    # Specialized access conditions or extenuating circumstances do not exist. An attacker can expect repeatable success against the vulnerable component.

    # Privileges Required
    # This metric describes the level of privileges an attacker must possess before successfully exploiting the vulnerability.
    # None
    # The attacker is unauthorized prior to attack, and therefore does not require any access to settings or files to carry out an attack.
    # User Interaction
    # This metric captures the requirement for a user, other than the attacker, to participate in the successful compromise the vulnerable component. This metric determines whether the vulnerability can be exploited solely at the will of the attacker, or whether a separate user (or user-initiated process) must participate in some manner.
    # None
    # The vulnerable system can be exploited without any interaction from any user.
    #          """,
    #     }

    # msrc_service = MSRCPostService(db_manager)
    # msrc_post, message, status = await msrc_service.create(**msrc_post_data)
    # print(f"{status}: {message}")
    # if msrc_post:
    #     print(f"MSRC Post: {msrc_post}")

    # else:
    #     msrc_post = await msrc_service.get(
    #         "6f5403e0-d9b1-804a-67db-fa99ba5f7b44"
    #     )

    # create multiple relationships
    # existing_product_rel = await msrc_post.affects_products.relationship(
    #     product
    # )

    # if not existing_product_rel:
    #     await msrc_post.affects_products.connect(
    #         product,
    #         {
    #             "relationship_type": "AFFECTS_PRODUCT",
    #         },
    #     )
    # else:
    #     print(
    #         "Product relationship already exists between MSRCPost and Product."
    #     )

    # Check if the KBArticle relationship exists between MSRCPost and KBArticle
    # existing_kb_rel = await msrc_post.has_kb_articles.relationship(kb_article)

    # if not existing_kb_rel:
    #     await msrc_post.has_kb_articles.connect(
    #         kb_article,
    #         {
    #             "relationship_type": "HAS_KB",
    #         },
    #     )
    # else:
    #     print("KBArticle relationship already exists between MSRCPost and KBArticle.")

    # ====================================
    # update package service test
    # ====================================
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
                "install_resources_text": (
                    "Restart behavior: Can request restart May request user"
                    " input: No"
                ),
            },
            {
                "package_name": "windows_10",
                "package_version": "22H2",
                "package_architecture": "x64",
                "update_type": "Cumulative",
                "file_size": "200MB",
                "install_resources_text": (
                    "Restart behavior: Can request restart May request user"
                    " input: No"
                ),
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
    # existing_update_rel_msrc = (
    #     await msrc_post.has_update_packages.relationship(update_package)
    # )

    # if not existing_update_rel_msrc:
    #     await msrc_post.has_update_packages.connect(
    #         update_package, {"relationship_type": "HAS_UPDATE_PACKAGE"}
    #     )
    # else:
    #     print(
    #         "Update package relationship already exists between MSRCPost and UpdatePackage."
    #     )

    # Check if the UpdatePackage relationship exists between ProductBuild and UpdatePackage
    # existing_update_rel_product_build = (
    #     await product_build.has_update_packages.relationship(update_package)
    # )

    # if not existing_update_rel_product_build:
    #     await product_build.has_update_packages.connect(
    #         update_package, {"relationship_type": "HAS_UPDATE_PACKAGE"}
    #     )
    # else:
    #     print(
    #         "Update package relationship already exists between ProductBuild and UpdatePackage."
    #     )

    # Check if the UpdatePackage relationship exists between KBArticle and UpdatePackage
    # existing_update_rel_kb = await kb_article.has_update_packages.relationship(
    #     update_package
    # )

    # if not existing_update_rel_kb:
    #     await kb_article.has_update_packages.connect(
    #         update_package, {"relationship_type": "HAS_UPDATE_PACKAGE"}
    #     )
    # else:
    #     print(
    #         "Update package relationship already exists between KBArticle and UpdatePackage."
    #     )

    # ====================================
    # Symptom Service test
    # ====================================
    symptom_service = SymptomService(db_manager)

    symptom_data = {
        "node_id": "ElevationOfPrivilegeVulnerability",
        "description": (
            "Unsafe default configurations for LDAP channel binding and LDAP"
            " signing exist on Active Directory domain controllers that let"
            " LDAP clients communicate with them without enforcing LDAP"
            " channel binding and LDAP signing. This can open Active Directory"
            " domain controllers to an elevation of privilege vulnerability."
        ),
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
        symptom = await symptom_service.get(
            "ElevationOfPrivilegeVulnerability"
        )

    # Check if the Symptom relationship exists between MSRCPost and Symptom
    # existing_symptom_rel_msrc = await msrc_post.has_symptoms.relationship(
    #     symptom
    # )

    # if not existing_symptom_rel_msrc:
    #     await msrc_post.has_symptoms.connect(
    #         symptom, {"relationship_type": "HAS_SYMPTOM"}
    #     )
    # else:
    #     print(
    #         "Symptom relationship already exists between MSRCPost and Symptom."
    #     )

    # Check if the Symptom relationship exists between KBArticle and Symptom
    # existing_symptom_rel_kbarticle = (
    #     await kb_article.has_symptoms.relationship(symptom)
    # )

    # if not existing_symptom_rel_kbarticle:
    #     await kb_article.has_symptoms.connect(
    #         symptom, {"relationship_type": "HAS_SYMPTOM"}
    #     )
    # else:
    #     print("Symptom relationship already exists between KBArticle and Symptom.")

    # Check if the Symptom relationship exists between Product and Symptom
    existing_product_rel_symptom = await symptom.affects_products.relationship(
        product
    )

    if not existing_product_rel_symptom:
        await symptom.affects_products.connect(
            product, {"relationship_type": "AFFECTS_PRODUCT"}
        )
    else:
        print(
            "Symptom relationship already exists between Product and Symptom."
        )

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
    # existing_cause_rel_msrc = await msrc_post.has_causes.relationship(cause)

    # if not existing_cause_rel_msrc:
    #     await msrc_post.has_causes.connect(
    #         cause, {"relationship_type": "HAS_CAUSE"}
    #     )
    # else:
    #     print("Cause relationship already exists between MSRCPost and Cause.")

    # Check if the Cause relationship exists between KBArticle and Cause
    # existing_cause_rel_kbarticle = await kb_article.has_causes.relationship(cause)

    # if not existing_cause_rel_kbarticle:
    #     await kb_article.has_causes.connect(cause, {"relationship_type": "HAS_CAUSE"})
    # else:
    #     print("Cause relationship already exists between KBArticle and Cause.")

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
    # existing_fix_rel_msrc = await msrc_post.has_fixes.relationship(fix)

    # if not existing_fix_rel_msrc:
    #     await msrc_post.has_fixes.connect(
    #         fix, {"relationship_type": "HAS_FIX"}
    #     )
    # else:
    #     print("Fix relationship already exists between MSRCPost and Fix.")

    # Check if the Fix relationship exists between KBArticle and Fix
    # existing_fix_rel_kbarticle = await kb_article.has_fixes.relationship(fix)

    # if not existing_fix_rel_kbarticle:
    #     await kb_article.has_fixes.connect(fix, {"relationship_type": "HAS_FIX"})
    # else:
    #     print("Fix relationship already exists between KBArticle and Fix.")

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
    # existing_faq_rel_msrc = await msrc_post.has_faqs.relationship(faq)

    # if not existing_faq_rel_msrc:
    #     await msrc_post.has_faqs.connect(faq, {"relationship_type": "HAS_FAQ"})
    # else:
    #     print("FAQ relationship already exists between MSRCPost and FAQ.")

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
    # existing_tool_rel_msrc = await msrc_post.has_tools.relationship(tool)

    # if not existing_tool_rel_msrc:
    #     await msrc_post.has_tools.connect(
    #         tool, {"relationship_type": "HAS_TOOL"}
    #     )
    # else:
    #     print("Tool relationship already exists between MSRCPost and Tool.")

    # Check if the Tool relationship exists between KBArticle and Tool
    # existing_tool_rel_kb = await kb_article.has_tools.relationship(tool)

    # if not existing_tool_rel_kb:
    #     await kb_article.has_tools.connect(tool, {"relationship_type": "HAS_TOOL"})
    # else:
    #     print("Tool relationship already exists between KBArticle and Tool.")

    # ====================================
    # Patch Service test
    # ====================================
    patch_management_post_service = PatchManagementPostService(db_manager)

    patch_management_post_data = {
        "node_id": "db8232be-16a0-aebe-7856-e6a79300149b",
        "receivedDateTime": "2024-08-16T08:54:23+00:00",
        "published": datetime.strptime(
            "2024-08-16 00:00:00", "%Y-%m-%d %H:%M:%S"
        ),
        "topic": "",
        "subject": "Re: [patchmanagement] Windows 10 Security Patch Issues",
        "text": (
            "Hello team, we're seeing issues after applying the latest Windows"
            " 10 security patch..."
        ),
        "post_type": "Problem Statement",
        "conversation_link": "https://groups.google.com/d/msgid/patchmanagement/112233445566778899aabbccddeeff00",
        "cve_mentions": "CVE-2024-12345",
        "keywords": [
            "server 2008 r2 machine",
            "patch",
            "may contain copyright material",
            "third party",
        ],
        "noun_chunks": [
            "these patches",
            "Server 2008 R2 machines",
            "the environment",
        ],
        "metadata": {
            "collection": "patch_management",
        },
        "embedding": [],
        "node_label": "PatchManagement",
    }
    patch_management_post, message, status = (
        await patch_management_post_service.create(
            **patch_management_post_data
        )
    )
    print(f"{status}: {message}")

    if patch_management_post:
        print(f"Created PatchManagementPost: {patch_management_post}")
    else:
        patch_management_post = await patch_management_post_service.get(
            "db8232be-16a0-aebe-7856-e6a79300149b"
        )

    # Symptom relationship
    existing_symptom_rel = (
        await patch_management_post.has_symptoms.relationship(symptom)
    )
    if not existing_symptom_rel:
        await patch_management_post.has_symptoms.connect(
            symptom, {"relationship_type": "HAS_SYMPTOM"}
        )
    else:
        print("Symptom relationship already exists.")

    # Cause relationship
    existing_cause_rel = await patch_management_post.has_causes.relationship(
        cause
    )
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
    # existing_kb_rel = await patch_management_post.references_kb_articles.relationship(
    #     kb_article
    # )
    # if not existing_kb_rel:
    #     await patch_management_post.references_kb_articles.connect(
    #         kb_article, {"relationship_type": "REFERENCES"}
    #     )
    # else:
    #     print("KBArticle relationship already exists.")

    # MSRCPost relationship
    # existing_msrc_rel = (
    #     await patch_management_post.references_msrc_posts.relationship(
    #         msrc_post
    #     )
    # )
    # if not existing_msrc_rel:
    #     await patch_management_post.references_msrc_posts.connect(
    #         msrc_post, {"relationship_type": "REFERENCES"}
    #     )
    # else:
    #     print("MSRCPost relationship already exists.")

    # Product relationship
    existing_product_rel = (
        await patch_management_post.affects_products.relationship(product)
    )
    if not existing_product_rel:
        await patch_management_post.affects_products.connect(
            product, {"relationship_type": "AFFECTS_PRODUCT"}
        )
    else:
        print("Product relationship already exists.")

    # Tool relationship
    existing_tool_rel = await patch_management_post.has_tools.relationship(
        tool
    )
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
