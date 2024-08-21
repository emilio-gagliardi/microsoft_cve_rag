# Purpose: Manage graph database operations
# Inputs: Graph queries
# Outputs: Query results
# Dependencies: None (external graph database library)

# services/graph_db_service.py
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
print(sys.path)
from typing import Type, TypeVar, Generic, List, Dict, Optional, Any
from neomodel import db, AsyncStructuredNode
import uuid
from application.app_utils import get_app_config, get_graph_db_credentials
from application.core.models import graph_db_models

settings = get_app_config()
graph_db_settings = settings["GRAPHDB_CONFIG"]
credentials = get_graph_db_credentials()

# Assuming you have the following keys in your .env or config

username = credentials["GRAPH_DATABASE_USERNAME"]
password = credentials["GRAPH_DATABASE_PASSWORD"]
host = credentials["GRAPH_DATABASE_HOST"]
port = credentials["GRAPH_DATABASE_PORT"]
protocol = credentials["GRAPH_DATABASE_PROTOCOL"]
db_uri = f"{protocol}://{username}:{password}@{host}:{port}"
db.set_connection(db_uri)

# BaseService definition

T = TypeVar("T", bound=AsyncStructuredNode)


class BaseService(Generic[T]):
    def __init__(self, model: Type[T]):
        """
        Initialize the service with a specific Neomodel node class.

        :param model: The Neomodel node class to manage.
        """
        self.model = model

    async def create(self, **properties) -> T:
        """
        Create a new node instance.

        :param properties: Key-value pairs of properties to set on the node.
        :return: The created node instance.
        """
        node = self.model(**properties)
        await node.save()
        return node

    async def get(self, node_id: uuid.UUID) -> Optional[T]:
        """
        Retrieve a node by its UUID.

        :param node_id: The UUID of the node to retrieve.
        :return: The node instance if found, otherwise None.
        """
        return await self.model.nodes.get_or_none(id=node_id)

    async def update(self, node_id: uuid.UUID, **properties) -> Optional[T]:
        """
        Update an existing node's properties.

        :param node_id: The UUID of the node to update.
        :param properties: Key-value pairs of properties to update.
        :return: The updated node instance if found, otherwise None.
        """
        node = await self.get(node_id)
        if node:
            for key, value in properties.items():
                setattr(node, key, value)
            await node.save()
        return node

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

    async def create_or_update(self, node_id: Optional[uuid.UUID], **properties) -> T:
        """
        Create a new node or update an existing node based on the UUID.

        :param node_id: The UUID of the node to update, or None to create a new node.
        :param properties: Key-value pairs of properties to set or update on the node.
        :return: The created or updated node instance.
        """
        if node_id:
            node = await self.update(node_id, **properties)
            if node:
                return node
        return await self.create(**properties)

    async def get_or_create(self, **properties) -> T:
        """
        Retrieve a node with matching properties, or create it if it doesn't exist.

        :param properties: Key-value pairs of properties to match or set.
        :return: The existing or newly created node instance.
        """
        node = await self.model.nodes.get_or_none(**properties)
        if node:
            return node
        return await self.create(**properties)

    async def cypher(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute an asynchronous Cypher query.

        :param query: The Cypher query to execute.
        :param params: Optional parameters to pass to the query.
        :return: A list of dictionaries representing the query results.
        """
        results, _ = await db.cypher_query(query, params)
        return [dict(record) for record in results]

    async def get_by_property(self, **properties) -> List[T]:
        """
        Retrieve nodes by matching properties.

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


class MSRCPostService(BaseService[graph_db_models.MSRCPost]):
    def __init__(self):
        super().__init__(graph_db_models.MSRCPost)


class ProductService(BaseService[graph_db_models.Product]):
    def __init__(self):
        super().__init__(graph_db_models.Product)


class ProductBuildService(BaseService[graph_db_models.ProductBuild]):
    def __init__(self):
        super().__init__(graph_db_models.ProductBuild)


class SymptomService(BaseService[graph_db_models.Symptom]):
    def __init__(self):
        super().__init__(graph_db_models.Symptom)


class CauseService(BaseService[graph_db_models.Cause]):
    def __init__(self):
        super().__init__(graph_db_models.Cause)


class FixService(BaseService[graph_db_models.Fix]):
    def __init__(self):
        super().__init__(graph_db_models.Fix)


class FAQService(BaseService[graph_db_models.FAQ]):
    def __init__(self):
        super().__init__(graph_db_models.FAQ)


class ToolService(BaseService[graph_db_models.Tool]):
    def __init__(self):
        super().__init__(graph_db_models.Tool)


class KBArticleService(BaseService[graph_db_models.KBArticle]):
    def __init__(self):
        super().__init__(graph_db_models.KBArticle)


class UpdatePackageService(BaseService[graph_db_models.UpdatePackage]):
    def __init__(self):
        super().__init__(graph_db_models.UpdatePackage)


class PatchManagementEmailService(BaseService[graph_db_models.PatchManagementEmail]):
    def __init__(self):
        super().__init__(graph_db_models.PatchManagementEmail)


msrc_service = MSRCPostService()
