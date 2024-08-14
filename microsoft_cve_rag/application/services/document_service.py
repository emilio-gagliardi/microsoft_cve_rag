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

from datetime import datetime
from typing import List, Dict, Any


def preprocess_pipeline(pipeline):
    def process_value(value):
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.rstrip("Z"))
            except ValueError:
                return value
        elif isinstance(value, dict):
            return {k: process_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [process_value(v) for v in value]
        return value

    return [process_value(stage) for stage in pipeline]


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

        # print(f"Document Service received:\n{document}")
        update_dict = {}
        document_dict = document.model_dump(exclude_unset=True)
        for key, value in document_dict.items():
            if key == "metadata" and isinstance(value, dict):
                for meta_key, meta_value in value.items():
                    if meta_value is not None:
                        update_dict[f"metadata.{meta_key}"] = meta_value
            elif value is not None:
                update_dict[key] = value

        # Convert ObjectId to string
        if "id_" in update_dict:
            update_dict["id_"] = str(update_dict["id_"])
        if "metadata.id" in update_dict:
            update_dict["metadata.id"] = str(update_dict["metadata.id"])

        # print(f"Passing to pymongo dict: {update_dict}")

        try:
            result = self.collection.update_one(query, {"$set": update_dict})
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
        print(f"deleting document: {document_id}")
        try:
            result = self.collection.delete_one(query)
            return result.deleted_count
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
            print(f"Error deleting document: {e}")
            raise

    def query_documents(
        self, query: dict, page: int = 1, page_size: int = 10
    ) -> Dict[str, Any]:
        """
        Query documents in the collection based on a filter with pagination.

        Args:
            query (dict): Filter to match documents.
            page (int): Page number for pagination. Default is 1.
            page_size (int): Number of documents per page. Default is 10.

        Returns:
            Dict[str, Any]: Dictionary containing results and total count.
        """
        if not isinstance(query, dict):
            raise ValueError("Query must be a dictionary")

        try:
            total_count = self.collection.count_documents(query)
            skip = (page - 1) * page_size
            cursor = self.collection.find(query).skip(skip).limit(page_size)
            results = list(cursor)
            return {"results": results, "total_count": total_count}
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
        # print("Begin preprocessing...")
        processed_pipeline = preprocess_pipeline(pipeline)
        try:
            result = list(self.collection.aggregate(processed_pipeline))
            # if result:
            #     print(f"len: {len(result)} item: {result[0]}")
            return result
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
