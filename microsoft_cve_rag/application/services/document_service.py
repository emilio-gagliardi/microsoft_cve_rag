# Purpose: Manage document operations
# Inputs: Document data
# Outputs: Processed documents
# Dependencies: None

from typing import List, Dict
from application.core.models import Document
from pymongo import MongoClient
from application.app_utils import get_documents_db_credentials
from application.core.schemas.document_schemas import DocumentRecordBase
from datetime import datetime


class DocumentService:
    def __init__(self, db_name="report_docstore", collection_name="docstore"):
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
        result = self.collection.insert_one(document.model_dump())
        return result.inserted_id

    def get_document(self, document_id: str) -> dict:
        """
        Retrieve a single document from the collection by its ID.

        Args:
            document_id (str): ID of the document to be retrieved.

        Returns:
            dict: Retrieved document.
        """
        return self.collection.find_one({"_id": document_id})

    def update_document(self, document_id: str, document: Document) -> int:
        """
        Update a single document in the collection by its ID.

        Args:
            document_id (str): ID of the document to be updated.
            document (Document): Updated document data.

        Returns:
            int: Number of documents updated.
        """
        result = self.collection.update_one(
            {"_id": document_id}, {"$set": document.model_dump()}
        )
        return result.modified_count

    def delete_document(self, document_id: str) -> int:
        """
        Delete a single document from the collection by its ID.

        Args:
            document_id (str): ID of the document to be deleted.

        Returns:
            int: Number of documents deleted.
        """
        result = self.collection.delete_one({"_id": document_id})
        return result.deleted_count

    def query_documents(self, query: dict) -> List[dict]:
        """
        Query documents in the collection based on a filter.

        Args:
            query (dict): Filter to match documents.

        Returns:
            List[dict]: List of matched documents.
        """
        return list(self.collection.find(query))


if __name__ == "__main__":
    service = DocumentService()

    # Create a document
    document_data = {
        "metadata": {
            "title": "Sample Document",
            "description": "This is a sample document.",
            "products": ["Product A", "Product B"],
            "severity_type": "High",
            "published": {"date": datetime.now()},
        },
        "text": "This is the text content of the document.",
        "embedding": [0.1, 0.2, 0.3] * 171,  # Assuming 512-length embedding
    }
    document = DocumentRecordBase(**document_data)
    created_id = service.create_document(document)
    print(f"Created document ID: {created_id}")

    # Get the created document
    fetched_document = service.get_document(created_id)
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
    query_result = service.query_documents({"metadata.title": "Sample Document"})
    print(f"Query result: {query_result}")

    # Delete the document
    delete_count = service.delete_document(created_id)
    print(f"Number of documents deleted: {delete_count}")

    # Verify deletion
    deleted_fetched_document = service.get_document(created_id)
    print(f"Document after deletion (should be None): {deleted_fetched_document}")
