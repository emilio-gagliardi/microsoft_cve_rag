# Purpose: Manage document operations
# Inputs: Document data
# Outputs: Processed documents
# Dependencies: None

# import os
# import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# print(sys.path)

from bson import ObjectId
from application.core.models import Document
from pymongo import MongoClient
from pymongo.errors import (
    PyMongoError,
    ConnectionFailure,
    OperationFailure,
    ConfigurationError,
)
from application.app_utils import get_documents_db_credentials
from application.core.schemas.document_schemas import DocumentRecordBase
from datetime import datetime
from typing import List, Dict


class DocumentService:
    def __init__(
        self, db_name: str = "report_docstore", collection_name: str = "docstore"
    ):
        """
        Initialize the DocumentService with database and collection names.

        Args:
            db_name (str): Name of the database.
            collection_name (str): Name of the collection.
        """
        credentials = get_documents_db_credentials()
        self.uri = credentials.uri
        self.client = MongoClient(credentials.uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def create_document(self, document: Document) -> str:
        """
        Create a single document in the collection.

        Args:
            document (Document): Document to be created.

        Returns:
            str: ID of the created document.
        """
        document_dict = document.model_dump()
        if "id_" in document_dict:
            document_dict["id_"] = str(document_dict["id_"])
        if "metadata" in document_dict and "id" in document_dict["metadata"]:
            document_dict["metadata"]["id"] = str(document_dict["metadata"]["id"])

        try:
            result = self.collection.insert_one(document_dict)
            return str(result.inserted_id)
        except ConnectionFailure as e:
            print(f"Connection to MongoDB failed: {e}")
            raise
        except OperationFailure as e:
            print(f"Operation failed: {e}")
            raise
        except ConfigurationError as e:
            print(f"Configuration error: {e}")
            raise
        except PyMongoError as e:
            print(f"General MongoDB error: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    def get_document(self, document_id: str) -> dict:
        """
        Retrieve a single document from the collection by its ID.

        Args:
            document_id (str): ID of the document to be retrieved.

        Returns:
            dict: Retrieved document.
        """
        query = {}
        if ObjectId.is_valid(document_id):
            query["_id"] = ObjectId(document_id)
        else:
            query["id_"] = document_id

        try:
            document = self.collection.find_one(query)
            if document is None:
                print(f"Document with ID {document_id} not found.")
            return document
        except ConnectionFailure as e:
            print(f"Connection to MongoDB failed: {e}")
            raise
        except OperationFailure as e:
            print(f"Operation failed: {e}")
            raise
        except ConfigurationError as e:
            print(f"Configuration error: {e}")
            raise
        except PyMongoError as e:
            print(f"General MongoDB error: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    def update_document(self, document_id: str, document: Document) -> int:
        """
        Update a single document in the collection by its ID.

        Args:
            document_id (str): ID of the document to be updated.
            document (Document): Updated document data.

        Returns:
            int: Number of documents updated.
        """
        query = {}
        if ObjectId.is_valid(document_id):
            query["_id"] = ObjectId(document_id)
        else:
            query["id_"] = document_id

        document_dict = document.model_dump()
        if "id_" in document_dict:
            document_dict["id_"] = str(document_dict["id_"])
        if "metadata" in document_dict and "id" in document_dict["metadata"]:
            document_dict["metadata"]["id"] = str(document_dict["metadata"]["id"])

        try:
            result = self.collection.update_one(query, {"$set": document_dict})
            return result.modified_count
        except ConnectionFailure as e:
            print(f"Connection to MongoDB failed: {e}")
            raise
        except OperationFailure as e:
            print(f"Operation failed: {e}")
            raise
        except ConfigurationError as e:
            print(f"Configuration error: {e}")
            raise
        except PyMongoError as e:
            print(f"General MongoDB error: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    def delete_document(self, document_id: str) -> int:
        """
        Delete a single document from the collection by its ID.

        Args:
            document_id (str): ID of the document to be deleted.

        Returns:
            int: Number of documents deleted.
        """
        query = {}
        if ObjectId.is_valid(document_id):
            query["_id"] = ObjectId(document_id)
        else:
            query["id_"] = document_id

        try:
            result = self.collection.delete_one(query)
            return result.deleted_count
        except ConnectionFailure as e:
            print(f"Connection to MongoDB failed: {e}")
            return []
        except OperationFailure as e:
            print(f"Operation failed: {e}")
            return []
        except ConfigurationError as e:
            print(f"Configuration error: {e}")
            return []
        except PyMongoError as e:
            print(f"General MongoDB error: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []

    def query_documents(self, query: dict) -> List[dict]:
        """
        Query documents in the collection based on a filter.

        Args:
            query (dict): Filter to match documents. Example:
                {
                    "metadata.title": "Sample Document",
                    "metadata.severity_type": "High"
                }

        Returns:
            List[dict]: List of matched documents.
        """
        if not isinstance(query, dict):
            raise ValueError("Query must be a dictionary")

        try:
            results = list(self.collection.find(query))
            return results
        except ConnectionFailure as e:
            print(f"Connection to MongoDB failed: {e}")
            return []
        except OperationFailure as e:
            print(f"Operation failed: {e}")
            return []
        except ConfigurationError as e:
            print(f"Configuration error: {e}")
            return []
        except PyMongoError as e:
            print(f"General MongoDB error: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []

    def create_documents(self, documents: List[Document]) -> List[str]:
        """
        Create multiple documents in the collection.

        Args:
            documents (List[Document]): List of documents to be created.

        Returns:
            List[str]: List of IDs of the created documents.
        """
        try:
            result = self.collection.insert_many(
                [doc.model_dump() for doc in documents]
            )
            return result.inserted_ids
        except ConnectionFailure as e:
            print(f"Connection to MongoDB failed: {e}")
            return []
        except OperationFailure as e:
            print(f"Operation failed: {e}")
            return []
        except ConfigurationError as e:
            print(f"Configuration error: {e}")
            return []
        except PyMongoError as e:
            print(f"General MongoDB error: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []

    def update_documents(self, filter: dict, update: dict) -> int:
        """
        Update multiple documents in the collection based on a filter.

        Args:
            filter (dict): Filter to match documents to be updated.
            update (dict): Update operations to be applied to the matched documents.

        Returns:
            int: Number of documents updated.
        """
        try:
            result = self.collection.update_many(filter, {"$set": update})
            return result.modified_count
        except ConnectionFailure as e:
            print(f"Connection to MongoDB failed: {e}")
            return []
        except OperationFailure as e:
            print(f"Operation failed: {e}")
            return []
        except ConfigurationError as e:
            print(f"Configuration error: {e}")
            return []
        except PyMongoError as e:
            print(f"General MongoDB error: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []

    def delete_documents(self, filter: dict) -> int:
        """
        Delete multiple documents in the collection based on a filter.

        Args:
            filter (dict): Filter to match documents to be deleted.

        Returns:
            int: Number of documents deleted.
        """
        try:
            result = self.collection.delete_many(filter)
            return result.deleted_count
        except ConnectionFailure as e:
            print(f"Connection to MongoDB failed: {e}")
            return []
        except OperationFailure as e:
            print(f"Operation failed: {e}")
            return []
        except ConfigurationError as e:
            print(f"Configuration error: {e}")
            return []
        except PyMongoError as e:
            print(f"General MongoDB error: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []

    def aggregate_documents(self, pipeline: List[dict]) -> List[dict]:
        """
        Execute an aggregation pipeline on the documents collection.

        Args:
            pipeline (List[dict]): List of aggregation stages.
                Example:
                    pipeline = [
                        {"$match": {"metadata.severity_type": "High"}},
                        {"$group": {"_id": "$metadata.products", "count": {"$sum": 1}}},
                        {"$sort": {"count": -1}}
                    ]

        Returns:
            List[dict]: Result of the aggregation pipeline.
                Example:
                    [
                        {"_id": "Product A", "count": 10},
                        {"_id": "Product B", "count": 5}
                    ]
        """
        if not isinstance(pipeline, list):
            raise ValueError("Pipeline must be a list")

        if not all(isinstance(item, dict) for item in pipeline):
            raise ValueError("All items in the pipeline must be dictionaries")

        try:
            result = list(self.collection.aggregate(pipeline))
            return result
        except ConnectionFailure as e:
            print(f"Connection to MongoDB failed: {e}")
            return []
        except OperationFailure as e:
            print(f"Operation failed: {e}")
            return []
        except ConfigurationError as e:
            print(f"Configuration error: {e}")
            return []
        except PyMongoError as e:
            print(f"General MongoDB error: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []

    def _describe(self):
        """
        Print all instance properties of the DocumentService.
        """
        properties = {
            "uri": self.uri,
            "client": self.client,
            "db": self.db,
            "collection": self.collection,
        }
        for key, value in properties.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    service = DocumentService()

    # Create a document
    document_data = {
        "metadata": {
            "title": "Sample Microsoft CVE document",
            "description": "This is a sample description to mimick a typical microsoft article.",
            "products": ["Windows 10 99H9", "Windows 10 98H8"],
            "severity_type": "High",
            "published": datetime.now(),
            "collection": "test_data",
        },
        "text": "This is the text content of the document.",
        "embedding": [0.1] * 1024,  # Assuming 1024-length embedding
    }
    document = DocumentRecordBase(**document_data)
    created_id = service.create_document(document)
    print(f"Created document ID: {created_id}")
    fetched_document = service.get_document(created_id)
    print(f"Fetched document: {fetched_document}")
    # Get the created document
    fetch_id = "3fb82ead-5114-45a7-b2de-11bf31a9c0d5"
    fetched_document = service.get_document(fetch_id)
    print(f"Fetched document: {fetched_document}")

    # Update the document
    document_data["text"] = "This is an updated sample document."
    updated_document = DocumentRecordBase(**document_data)
    update_count = service.update_document(created_id, updated_document)
    print(f"Number of documents updated: {update_count}")

    # Fetch the updated document
    updated_fetched_document = service.get_document(created_id)
    print(f"Updated fetched document: {updated_fetched_document}")

    # Query documents
    query_result = service.query_documents(
        {"metadata.title": "Sample Microsoft CVE document"}
    )
    print(f"Query result: {query_result}")

    # Delete the document
    delete_count = service.delete_document(created_id)
    print(f"Number of documents deleted: {delete_count}")

    # Verify deletion
    deleted_fetched_document = service.get_document(created_id)
    print(f"Document after deletion (should be None): {deleted_fetched_document}")
