# Purpose: Manage document operations
# Inputs: Document data
# Outputs: Processed documents
# Dependencies: None

# import os
# import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# print(sys.path)
import os
from bson import ObjectId
from application.core.models.basic_models import Document
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import (
    PyMongoError,
    ConnectionFailure,
    OperationFailure,
    ConfigurationError,
)
from application.app_utils import get_documents_db_credentials, setup_logger
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# Get the logging level from the environment variable, default to INFO
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
# Convert the string to a logging level
log_level = getattr(logging, log_level, logging.INFO)

logger = setup_logger(__name__, level=log_level)

def preprocess_pipeline(pipeline):
    """
    Preprocesses a MongoDB aggregation pipeline by converting ISO format date strings ending with 'Z' into datetime objects.

    This function recursively processes each value in the pipeline to ensure that date strings are properly converted to datetime objects.
    Other types (e.g., dictionaries, lists) are recursively processed as well to ensure all nested date strings are handled.
    
    The preprocess_pipeline function is designed to convert ISO format date strings (like "2024-08-01T00:00:00Z") into Python datetime objects. If your MongoDB aggregation pipeline contains such date strings, you might need to preprocess them depending on how your MongoDB Python driver (like pymongo) expects the data.

    Args:
        pipeline (list): A list representing a MongoDB aggregation pipeline.

    Returns:
        list: The preprocessed aggregation pipeline with datetime objects replacing ISO date strings.
    """
    # Helper function to process each value in the pipeline
    def process_value(value):
        if isinstance(value, str):
            try:
                # Convert ISO format date strings ending with 'Z' into datetime objects
                return datetime.fromisoformat(value.rstrip("Z"))
            except ValueError:
                # If the string cannot be converted, return it as is
                return value
        elif isinstance(value, dict):
            # Recursively process dictionary values
            return {k: process_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            # Recursively process list elements
            return [process_value(v) for v in value]
        # Return the value as is if it is not a string, dictionary, or list
        return value

    # Apply the helper function to each stage in the pipeline
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

    def update_document(self, document_id: str, document: Document, exclude_unset: bool = True) -> int:
        """
        Update a single document in the collection by its ID.
        
        Args:
            document_id (str): The ID of the document to update
            document (Document): The document containing the updates
            exclude_unset (bool): If True, only include fields that were explicitly set.
                                If False, include all fields including None values.
        """
        query = {}
        if ObjectId.is_valid(document_id):
            query["_id"] = ObjectId(document_id)
        else:
            query["id_"] = document_id

        # Convert document to dict - let caller decide what fields to include
        update_dict = document.model_dump(exclude_unset=exclude_unset)
        
        # Flatten metadata fields for MongoDB dot notation
        metadata = update_dict.pop('metadata', {})
        for meta_key, meta_value in metadata.items():
            update_dict[f"metadata.{meta_key}"] = meta_value

        try:
            print(f"$set: {update_dict}")
            result = self.collection.update_one(query, {"$set": update_dict})

            return result.modified_count
        except Exception as e:
            logger.error(f"Error updating document: {e}")
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
        self,
        query: dict,
        page: int = 1,
        page_size: int = 999,
        max_records: int = None,
        sort: Optional[List[Tuple[str, int]]] = None,
        projection: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query documents in the collection based on a filter with pagination.

        Args:
            query (dict): Filter to match documents.
            page (int): Page number for pagination. Default is 1.
            page_size (int): Number of documents per page. Default is 10.
            sort (Optional[List[Tuple[str, int]]]): Sort order for the results. Default is None.
            projection (Optional[Dict[str, Any]]): Projection dictionary to specify included/excluded fields.

        Returns:
            Dict[str, Any]: Dictionary containing 'results', 'total_count', 'limit'.
        """
        if not isinstance(query, dict):
            raise ValueError("Query must be a dictionary")
        # print(
        #     f"DocumentService received: page: {page} page_size: {page_size} max_records: {max_records}"
        # )
        try:
            # total_count is the number of records in the collection defined by the query, not the total number of documents in the collection.
            total_count = self.collection.count_documents(query)
            # Calculate the effective limit considering pagination and max_records
            skip = (page - 1) * page_size
            effective_limit = page_size
            # Determine the limit to apply to the query
            if max_records is not None:
                effective_limit = min(page_size, max_records - skip)
                if effective_limit <= 0:
                    return {"results": [], "total_count": total_count}

            cursor = (
                self.collection.find(query, projection=projection)
                .skip(skip)
                .limit(effective_limit)
            )
            if sort:
                cursor = cursor.sort(sort)
            results = list(cursor)
            return {
                "results": results,
                "total_count": total_count,
                "limit": max_records,
            }

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
    
    def test_update(self, document_id: str):
        """Test function to verify document updates"""
        # Check document before update
        before = self.collection.find_one({"id_": document_id})
        logger.info(f"Document before update: {before}")
        
        # Perform a simple test update
        test_update = {"$set": {"test_field": "test_value"}}
        result = self.collection.update_one({"id_": document_id}, test_update)
        logger.info(f"Test update result - matched: {result.matched_count}, modified: {result.modified_count}")
        
        # Check document after update
        after = self.collection.find_one({"id_": document_id})
        logger.info(f"Document after update: {after}")
        
        return before, after
    
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
    # document = DocumentRecordBase(**document_data)
    # created_id = service.create_document(document)
    # print(f"Created document ID: {created_id}")
    # fetched_document = service.get_document(created_id)
    # print(f"Fetched document: {fetched_document}")
    # Get the created document
    # fetch_id = "3fb82ead-5114-45a7-b2de-11bf31a9c0d5"
    # fetched_document = service.get_document(fetch_id)
    # print(f"Fetched document: {fetched_document}")

    # Update the document
    # document_data["text"] = "This is an updated sample document."
    # updated_document = DocumentRecordBase(**document_data)
    # update_count = service.update_document(created_id, updated_document)
    # print(f"Number of documents updated: {update_count}")

    # Fetch the updated document
    # updated_fetched_document = service.get_document(created_id)
    # print(f"Updated fetched document: {updated_fetched_document}")

    # Query documents
    # query_result = service.query_documents(
    #     {"metadata.title": "Sample Microsoft CVE document"}
    # )
    # print(f"Query result: {query_result}")

    # Delete the document
    # delete_count = service.delete_document(created_id)
    # print(f"Number of documents deleted: {delete_count}")

    # Verify deletion
    # deleted_fetched_document = service.get_document(created_id)
    # print(f"Document after deletion (should be None): {deleted_fetched_document}")
