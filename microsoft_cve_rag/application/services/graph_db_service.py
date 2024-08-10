# Purpose: Manage graph database operations
# Inputs: Graph queries
# Outputs: Query results
# Dependencies: None (external graph database library)

# services/graph_db_service.py

from application.core.models import GraphNode
from neo4j import GraphDatabase
from application.app_utils import get_graph_db_credentials
from application.config import DEFAULT_EMBEDDING_CONFIG


class GraphDBService:
    def __init__(self):
        credentials = get_graph_db_credentials()
        self.driver = GraphDatabase.driver(
            credentials.uri, auth=(credentials.username, credentials.password)
        )
        self.embedding_config = DEFAULT_EMBEDDING_CONFIG

    def close(self):
        self.driver.close()

    def create_node(self, node: GraphNode):
        with self.driver.session() as session:
            result = session.write_transaction(self._create_node, node)
            return result

    def get_node(self, node_id: str):
        with self.driver.session() as session:
            result = session.read_transaction(self._get_node, node_id)
            return result

    def update_node(self, node_id: str, node: GraphNode):
        with self.driver.session() as session:
            result = session.write_transaction(self._update_node, node_id, node)
            return result

    def delete_node(self, node_id: str):
        with self.driver.session() as session:
            result = session.write_transaction(self._delete_node, node_id)
            return result

    def query(self, cypher: str, parameters: dict):
        with self.driver.session() as session:
            result = session.read_transaction(self._query, cypher, parameters)
            return result

    @staticmethod
    def _create_node(tx, node: GraphNode):
        query = """
        CREATE (n:Node {id: $id, embedding: $embedding, metadata: $metadata, relationships: $relationships, text: $text, class_name: $class_name, created_at: $created_at, updated_at: $updated_at})
        RETURN id(n) as id
        """
        result = tx.run(
            query,
            id=str(node.id_),
            embedding=node.embedding,
            metadata=node.metadata.dict(),
            relationships=node.relationships,
            text=node.text,
            class_name=node.class_name,
            created_at=node.created_at,
            updated_at=node.updated_at,
        )
        return result.single()["id"]

    @staticmethod
    def _get_node(tx, node_id: str):
        query = "MATCH (n:Node {id: $id}) RETURN n"
        result = tx.run(query, id=node_id)
        return result.single()

    @staticmethod
    def _update_node(tx, node_id: str, node: GraphNode):
        query = """
        MATCH (n:Node {id: $id})
        SET n += {embedding: $embedding, metadata: $metadata, relationships: $relationships, text: $text, class_name: $class_name, updated_at: $updated_at}
        RETURN n
        """
        result = tx.run(
            query,
            id=node_id,
            embedding=node.embedding,
            metadata=node.metadata.dict(),
            relationships=node.relationships,
            text=node.text,
            class_name=node.class_name,
            updated_at=node.updated_at,
        )
        return result.single()

    @staticmethod
    def _delete_node(tx, node_id: str):
        query = "MATCH (n:Node {id: $id}) DELETE n"
        result = tx.run(query, id=node_id)
        return result.single()

    @staticmethod
    def _query(tx, cypher: str, parameters: dict):
        result = tx.run(cypher, parameters)
        return [record for record in result]
