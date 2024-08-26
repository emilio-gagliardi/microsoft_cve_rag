# Purpose: Manage graph database operations
# Inputs: Graph queries
# Outputs: Query results
# Dependencies: None (external graph database library)

# services/graph_db_service.py
# import os
# import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# print(sys.path)

from typing import (
    Type,
    TypeVar,
    Generic,
    List,
    Dict,
    Optional,
    Any,
    Tuple,
    Union,
)
from neomodel import (
    AsyncStructuredNode,
    UniqueProperty,
    DoesNotExist,
    DeflateError,
    db,
)
import uuid
import logging
from application.app_utils import get_app_config, get_graph_db_credentials
from application.core.models import graph_db_models

# import asyncio  # required for testing
from neomodel import config as NeomodelConfig  # required by AsyncDatabase
from neomodel.async_.core import AsyncDatabase  # required for db CRUD
from neo4j import GraphDatabase  # required for constraints check

settings = get_app_config()
graph_db_settings = settings["GRAPHDB_CONFIG"]
credentials = get_graph_db_credentials()
logger = logging.getLogger(__name__)

username = credentials.username
password = credentials.password
host = credentials.host
port = credentials.port
protocol = credentials.protocol
db_uri = f"{protocol}://{username}:{password}@{host}:{port}"


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
            await node.save()
            logger.debug(f"Created {self.model.__name__}\n{node}")
            return node, "Node created successfully", 201
        except UniqueProperty as e:
            logger.warning(
                f"Duplicate {self.model.__name__} node. Skipping create...\n{str(e)}"
            )
            return None, f"Duplicate entry: {str(e)}", 409
        except DeflateError as e:
            logger.error(
                f"Choice constraint violation in {self.model.__name__}node: {node}\n{str(e)}"
            )
            return None, f"Choice constraint violation: {str(e)}", 422
        except Exception as e:
            logger.error(f"Error creating {self.model.__name__} node: {node}\n{str(e)}")
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

        # try:
        node = await self.get(node_id)

        if node:
            for key, value in properties.items():
                setattr(node, key, value)
            await node.save()
            response["message"] = "Successfully updated node."
        else:
            response["status"] = "error"
            response["message"] = "No node found"
        # except ValueError as ve:
        #     response["status"] = "error"
        #     response["message"] = f"Value error in properties: {str(ve)}"
        # except Exception as e:
        #     response["status"] = "error"
        #     response["message"] = f"Exception returned by the graph db: {str(e)}"
        #     traceback.print_exc()  # Print the exception traceback

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
        node = await self.model.nodes.get_or_none(**properties)
        if node:
            return node, "Existing node retrieved", 200
        return await self.create(**properties)

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

    async def bulk_create(
        self, items: List[Dict[str, Any]]
    ) -> List[Tuple[Union[Any, None], str, int]]:
        results = []
        async with self.db_manager:
            with db.transaction:
                try:
                    for item in items:
                        node, message, code = await self.create(**item)
                        if code == 201:
                            results.append(node)
                        else:
                            print(f"bulk_create: {code} - {message}\n{item}")

                except Exception as e:
                    logger.error(f"Error during bulk creation: {str(e)}")
                    return [(None, f"Error during bulk creation: {str(e)}", 500)]
        if results:
            print(f"num bulk create: {len(results)} of type: {type(results[0])}")
        return results


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


# Function to check if a constraint exists in the graph database
def graph_db_constraint_exists(session, constraint_name):
    result = session.run("SHOW CONSTRAINTS")
    constraints = [record["name"] for record in result if "name" in record]
    return constraint_name in constraints


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

    return {
        "status": (
            "success"
            if all(
                status["status"] in ["created", "exists"]
                for status in constraints_status
            )
            else "partial_success"
        ),
        "message": "Constraints checked and applied.",
        "constraints_status": constraints_status,
    }


# from neomodel import install_all_labels

# async def graph_db_setup_schema():
#     # unclear if this is needed still
#     NeomodelConfig.DATABASE_URL = 'bolt://neo4j:password@localhost:7687'
#     await install_all_labels()
#     print("Schema setup complete.")


# How to convert from neo4j Node instances to
# Neomodel AsynchStructuredNode instances
async def convert_nodes_to_products(node_results):
    products = []
    for result in node_results:
        node = result[0]  # Assuming each result is a list with one Node
        product = graph_db_models.Product.inflate(node)
        products.append(product)
    return products


async def inflate_nodes(node_results):
    inflated_objects = []
    for result in node_results:
        node = result[0]  # Assuming each result is a list with one Node
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


async def main():
    from datetime import datetime

    install_labels(graph_db_models.Product)
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

    # print("Starting main()")
    # asyncio.run(main())
    # print("Finished main()")
    print("This module doesn't run on its own.")
