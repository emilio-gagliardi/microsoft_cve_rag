# Purpose: Extract data from various sources
# Inputs: Source configurations
# Outputs: Raw data
# Dependencies: None
import hashlib
import json
import logging

# import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from application.services.document_service import DocumentService
from pymongo import ASCENDING


def extract_from_mongo(
    db_name: str,
    collection_name: str,
    query: Dict[str, Any],
    max_records: Optional[int] = None,
    sort: Optional[List[Tuple[str, int]]] = None,
    projection: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extracts documents from a MongoDB collection based on the provided query.

    Args:
        db_name (str): The name of the database.
        collection_name (str): The name of the collection.
        query (Dict[str, Any]): The query to filter documents.
        max_records (Optional[int]): The maximum number of records to return. Defaults to None.
        sort (Optional[List[Tuple[str, int]]]): Sort order for the results. Default is None.
        projection (Optional[Dict[str, Any]]): Projection dictionary to specify included/excluded fields.

    Returns:
       Dict[str, Any]: A dictionary with the following keys:
           - results: The list of documents matching the query.
           - total_count: The total number of documents matching the query.
           - limit: The maximum number of documents returned.
    """
    document_service = DocumentService(db_name, collection_name)

    return document_service.query_documents(
        query, max_records=max_records, sort=sort, projection=projection
    )


def patch_fe_extractor(
    start_date: datetime, end_date: datetime, max_records: int = 10
) -> List[Dict[str, Any]]:
    """Extract patch management documents for feature engineering.

    This extractor ignores thread/link logic and is separate from the main pipeline
    extractor.

    Args:
        start_date (datetime): The start date of the time range to extract documents.
        end_date (datetime): The end date of the time range to extract documents.
        max_records (int, optional): Maximum number of records to extract.
            Defaults to 10.

    Returns:
        List[Dict[str, Any]]: List of extracted patch management documents.
    """
    db_name = "report_docstore"
    collection_name = "docstore"

    # Initialize document service at the start and keep reference
    document_service = DocumentService(db_name, collection_name)

    query = {
        "metadata.collection": "patch_management",
        "metadata.published": {"$gte": start_date, "$lt": end_date},
    }

    projection = {
        "_id": 0,
        "metadata.added_to_vector_store": 0,
        "metadata.added_to_summary_index": 0,
        "metadata.added_to_graph_store": 0,
        "metadata.build_numbers": 0,
        "relationships": 0,
        "start_char_idx": 0,
        "end_char_idx": 0,
        "text_template": 0,
        "metadata_template": 0,
        "metadata_seperator": 0,
        "class_name": 0,
        "hash": 0,
    }

    # Pull the docs from Mongo using the service instance
    mongo_result = document_service.query_documents(
        query, max_records=max_records, sort=None, projection=projection
    )
    patch_docs_unsorted = mongo_result["results"]

    # Convert to datetime objects and ensure chronological order
    patch_docs_unsorted = [
        convert_received_date_time(doc) for doc in patch_docs_unsorted
    ]
    patch_docs_sorted = sorted(
        patch_docs_unsorted, key=lambda x: x["metadata"]["receivedDateTime"]
    )

    return patch_docs_sorted


# get products
def extract_products(max_records: int = 10) -> List[Dict[str, Any]]:
    """Extract Microsoft product documents.

    Extracts Microsoft product documents from the "report_docstore" MongoDB database.

    Note:
        DO NOT MODIFY this function.

    Args:
        max_records (int, optional): Maximum number of records to extract.
            Defaults to 10.

    Returns:
        List[Dict[str, Any]]: List of extracted product documents.
    """
    db_name = "report_docstore"
    collection_name = "microsoft_products"
    query = {
        "product_name": {"$nin": ["edge_ext"]},
        "product_version": {"$in": ["21H2", "22H2", "23H2", "24H2", ""]},
        "product_architecture": {
            "$in": ["32-bit_systems", "x64-based_systems", ""]
        },
    }
    max_records = max_records
    sort = [
        ("product_name", ASCENDING),
        ("product_architecture", ASCENDING),
        ("product_version", ASCENDING),
    ]
    projection = {
        "_id": 0,
        "hash": 0,
    }
    product_docs = extract_from_mongo(
        db_name, collection_name, query, max_records, sort, projection
    )["results"]

    return product_docs

    # get product_builds


def extract_product_builds(
    start_date: datetime,
    end_date: datetime,
    max_records: int = 10,
) -> List[Dict[str, Any]]:
    """Extract Microsoft product build documents.

    Extracts Microsoft product build documents from the "report_docstore" MongoDB
    database.

    Note:
        DO NOT MODIFY this function.

    Args:
        start_date (datetime): The start date of the time range to extract documents.
        end_date (datetime): The end date of the time range to extract documents.
        max_records (int, optional): Maximum number of records to extract.
            Defaults to 10.

    Returns:
        List[Dict[str, Any]]: List of extracted product build documents.
    """
    db_name = "report_docstore"
    collection_name = "microsoft_product_builds"
    query = {
        "published": {"$gte": start_date, "$lt": end_date},
        "product_version": {"$in": ["21H2", "22H2", "23H2", "24H2", ""]},
        "product_architecture": {
            "$in": ["32-bit_systems", "x64-based_systems", ""]
        },
        "product_name": {
            "$in": [
                "windows_10",
                "windows_11",
                "microsoft_edge",
                "microsoft_edge_(chromium-based)",
            ]
        },
    }
    max_records = max_records
    sort = [
        ("product_name", ASCENDING),
        ("product_architecture", ASCENDING),
        ("product_version", ASCENDING),
        ("cve_id", ASCENDING),
    ]
    projection = {
        "_id": 0,
        "hash": 0,
        "summary": 0,
        "article_url": 0,
        "cve_url": 0,
    }
    product_build_docs = extract_from_mongo(
        db_name, collection_name, query, max_records, sort, projection
    )["results"]

    return product_build_docs


def generate_kb_id(document):
    """Generate a unique KB ID for a document.

    Creates a SHA-256 hash of the document's JSON representation. The hash is then
    truncated and formatted to resemble a UUID.

    Args:
        document: The document to generate an ID for.

    Returns:
        str: The generated KB ID.
    """
    # Convert the document to a JSON string
    document_json = json.dumps(document, sort_keys=True, default=str)

    # Create a SHA-256 hash of the JSON string
    hash_object = hashlib.sha256(document_json.encode("utf-8"))
    full_hash = hash_object.hexdigest()

    # Truncate the hash to the length of a UUID (36 characters including hyphens)
    truncated_hash = full_hash[:36]

    # Insert hyphens to match the UUID format
    unique_id = f"{truncated_hash[:8]}-{truncated_hash[8:12]}-{truncated_hash[12:16]}-{truncated_hash[16:20]}-{truncated_hash[20:]}"

    return unique_id


def process_kb_article(article):
    """
    Process a single KB article document.

    Args:
        article (dict): The KB article document to process.

    Returns:
        dict: The processed KB article document with a new "build_number" field and an "id" field if it did not exist.

    Notes:
        - If the "kb_id" field is None, the "build_number" field is set to an empty tuple.
        - If the "kb_id" field is not None, the "build_number" field is set to a tuple of integers parsed from the "kb_id" string.
        - If the "id" field does not exist, it is generated using the `generate_kb_id` function.
    """
    if article["kb_id"] is None:
        article["build_number"] = ()
    else:
        article["build_number"] = tuple(map(int, article["kb_id"].split(".")))
    if "id" not in article:
        article["id"] = generate_kb_id(article)
    return article


def convert_build_number(article):
    """
    Convert the build number from a tuple of integers to a list of integers.

    Args:
        article (dict): The KB article document to process.

    Returns:
        dict: The processed KB article document with a new "build_number" field.

    Notes:
        - The "build_number" field is a tuple of integers parsed from the "kb_id" string.
        - The "build_number" field is converted to a list of integers.
    """
    article["build_number"] = list(article["build_number"])
    return article


def process_kb_articles(articles):
    """
    Process a list of KB article documents.

    Args:
        articles (list): A list of KB article documents to process.

    Returns:
        list: A list of processed KB article documents.

    Notes:
        - The function processes each article in the list by calling the `process_kb_article` function.
        - The function creates tuples from the original data and generates new ID hashes.
        - The function sorts the list by the new tupled build_number.
        - The function converts the tuple back to a list of ints.
    """
    # Step 1: Create tuples from the original data and generate new ID hash
    articles = list(map(process_kb_article, articles))

    # Step 2: Sort the list by the new tupled build_number
    articles.sort(key=lambda x: x["build_number"])

    # Step 3: Convert the tuple back to a list of ints
    articles = list(map(convert_build_number, articles))

    return articles


def extract_kb_articles(
    start_date: datetime, end_date: datetime, max_records: int = 10
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract Microsoft KB articles.

    Extracts Microsoft KB articles from the "report_docstore" MongoDB database.

    Note:
        DO NOT MODIFY this function.

    Args:
        start_date (datetime): The start date of the time range to extract documents.
        end_date (datetime): The end date of the time range to extract documents.
        max_records (int, optional): Maximum number of records to extract.
            Defaults to 10.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: A tuple of two lists, the first containing Windows-based KB articles and the second containing Edge-based KB articles.

    Notes:
        - Windows-based KB articles are extracted from the "microsoft_kb_articles" collection and filtered based on "cve_id" being None.
        - The resulting documents are sorted by "kb_id" and projected to exclude "_id" and "cve_id" fields.
        - The extracted documents are then matched with documents from the "docstore" collection based on "post_id" matching the "kb_id" from the "microsoft_kb_articles" collection.
        - The matched documents have their "text" and "title" fields added to the "microsoft_kb_articles" documents.
        - Edge-based KB articles are extracted from the "microsoft_kb_articles" collection and filtered based on "cve_id" being None and "product_name" being "edge_ext".
        - The resulting documents are sorted by "kb_id" and projected to exclude "_id" and "cve_id" fields.
        - The extracted documents are then matched with documents from the "docstore" collection based on "post_id" matching the "kb_id" from the "microsoft_kb_articles" collection.
        - The matched documents have their "text" and "title" fields added to the "microsoft_kb_articles" documents.
    """
    db_name = "report_docstore"
    collection_name = "microsoft_kb_articles"

    # equals null is for windows kb articles
    query = {
        "published": {"$gte": start_date, "$lte": end_date},
        "kb_id": {"$regex": "^[0-9]+$", "$options": "i"}
    }
    projection = {"_id": 0, "cve_id": 0}
    max_records = max_records
    sort = [
        ("kb_id", ASCENDING),
    ]
    kb_article_docs_windows_results = extract_from_mongo(
        db_name, collection_name, query, max_records, sort, projection
    )
    print(
        "windows kb articles count:"
        f" {kb_article_docs_windows_results['total_count']}"
    )
    if kb_article_docs_windows_results["results"]:
        # extract windows 10 and windows 11 docs from docstore to match with docs from microsoft_kb_articles.
        # This is to extract the text and title from the docstore docs and pass them to the microsoft_kb_articles docs.
        # get the unique 'kb_id's for windows-based KB articles
        unique_kb_ids_windows = set(
            kb_article["kb_id"]
            for kb_article in kb_article_docs_windows_results["results"]
        )
        print(
            "windows unique kb"
            f" ids:\n{', '.join(['kb' + kb_id for kb_id in unique_kb_ids_windows])}"
        )
        collection_name = "docstore"
        query = {
            "$or": [
                {"metadata.post_id": {"$regex": kb_id}}
                for kb_id in list(unique_kb_ids_windows)
            ]
        }
        projection = {
            "_id": 0,
            "start_char_idx": 0,
            "end_char_idx": 0,
            "text_template": 0,
            "metadata_template": 0,
            "metadata_seperator": 0,
            "class_name": 0,
        }
        max_records = None  # we want all matching documents
        windows_docs = extract_from_mongo(
            db_name, collection_name, query, max_records
        )["results"]

        for kb_article in kb_article_docs_windows_results["results"]:
            kb_id = kb_article["kb_id"]
            for doc in windows_docs:
                if kb_id in doc["metadata"]["post_id"]:
                    kb_article["title"] = doc["metadata"]["title"]
                    kb_article["text"] = doc["text"]

    # For edge-based KB articles
    # These are synthetic in that microsoft doesn't publish kbs for edge issues, they point to the stable notes/security notes. I'm building these for symmetry with windows-based kb articles
    db_name = "report_docstore"
    collection_name = "microsoft_kb_articles"
    pipeline_stable_channel = [
        {
            "$match": {
                "published": {"$gte": start_date, "$lte": end_date},
                "kb_id": {"$not": {"$regex": "^[0-9]+$", "$options": "i"}}
            }
        },
        {
            "$lookup": {
                "from": "docstore",
                "let": {
                    "build_number_str": {
                        "$reduce": {
                            "input": "$build_number",
                            "initialValue": "",
                            "in": {
                                "$concat": [
                                    "$$value",
                                    {
                                        "$cond": [
                                            {"$eq": ["$$value", ""]},
                                            "",
                                            ".",
                                        ]
                                    },
                                    {"$toString": "$$this"},
                                ]
                            },
                        }
                    }
                },
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {
                                        "$regexMatch": {
                                            "input": "$metadata.subject",
                                            "regex": {
                                                "$concat": [
                                                    ".*",
                                                    "$$build_number_str",
                                                    ".*",
                                                ]
                                            },
                                        }
                                    },
                                    {
                                        "$eq": [
                                            "$metadata.collection",
                                            "stable_channel_notes",
                                        ]
                                    },
                                ]
                            }
                        }
                    },
                    {"$project": {"text": 1, "metadata.subject": 1}},
                ],
                "as": "docstore_data",
            }
        },
        {"$match": {"docstore_data": {"$ne": []}}},
        {"$unwind": "$docstore_data"},
        {
            "$group": {
                "_id": "$build_number",
                "id": {"$first": "$id"},
                "kb_id": {
                    "$first": {
                        "$concat": [
                            {
                                "$toString": {
                                    "$arrayElemAt": ["$build_number", 0]
                                }
                            },
                            ".",
                            {
                                "$toString": {
                                    "$arrayElemAt": ["$build_number", 1]
                                }
                            },
                            ".",
                            {
                                "$toString": {
                                    "$arrayElemAt": ["$build_number", 2]
                                }
                            },
                            ".",
                            {
                                "$toString": {
                                    "$arrayElemAt": ["$build_number", 3]
                                }
                            },
                        ]
                    }
                },
                "cve_ids": {"$push": "$cve_id"},
                "published": {"$first": "$published"},
                "product_build_id": {"$first": "$product_build_id"},
                "article_url": {"$first": "$article_url"},
                "docstore_data": {"$first": "$docstore_data"},
            }
        },
        {
            "$addFields": {
                "text": "$docstore_data.text",
                "title": "$docstore_data.metadata.subject",
            }
        },
        {"$project": {"docstore_data": 0, "_id": 0}},
    ]
    pipeline_security_channel = [
        {
            "$match": {
                "cve_id": {"$ne": None},
                "published": {"$gte": start_date, "$lte": end_date},
            }
        },
        {
            "$lookup": {
                "from": "docstore",
                "let": {
                    "build_number_str": {
                        "$reduce": {
                            "input": "$build_number",
                            "initialValue": "",
                            "in": {
                                "$concat": [
                                    "$$value",
                                    {
                                        "$cond": [
                                            {"$eq": ["$$value", ""]},
                                            "",
                                            ".",
                                        ]
                                    },
                                    {"$toString": "$$this"},
                                ]
                            },
                        }
                    }
                },
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {
                                        "$regexMatch": {
                                            "input": "$metadata.subject",
                                            "regex": {
                                                "$concat": [
                                                    ".*",
                                                    "$$build_number_str",
                                                    ".*",
                                                ]
                                            },
                                        }
                                    },
                                    {
                                        "$eq": [
                                            "$metadata.collection",
                                            "security_update_notes",
                                        ]
                                    },
                                ]
                            }
                        }
                    },
                    {"$project": {"text": 1, "metadata.subject": 1}},
                ],
                "as": "docstore_data",
            }
        },
        {"$match": {"docstore_data": {"$ne": []}}},
        {"$unwind": "$docstore_data"},
        {
            "$group": {
                "_id": "$build_number",
                "id": {"$first": "$id"},
                "kb_id": {
                    "$first": {
                        "$concat": [
                            {
                                "$toString": {
                                    "$arrayElemAt": ["$build_number", 0]
                                }
                            },
                            ".",
                            {
                                "$toString": {
                                    "$arrayElemAt": ["$build_number", 1]
                                }
                            },
                            ".",
                            {
                                "$toString": {
                                    "$arrayElemAt": ["$build_number", 2]
                                }
                            },
                            ".",
                            {
                                "$toString": {
                                    "$arrayElemAt": ["$build_number", 3]
                                }
                            },
                        ]
                    }
                },
                "cve_ids": {"$push": "$cve_id"},
                "published": {"$first": "$published"},
                "product_build_id": {"$first": "$product_build_id"},
                "article_url": {"$first": "$article_url"},
                "docstore_data": {"$first": "$docstore_data"},
            }
        },
        {
            "$addFields": {
                "text": "$docstore_data.text",
                "title": "$docstore_data.metadata.subject",
            }
        },
        {"$project": {"docstore_data": 0, "_id": 0}},
    ]
    document_service = DocumentService(db_name, collection_name)

    kb_article_docs_edge_stable = document_service.aggregate_documents(
        pipeline_stable_channel
    )
    kb_article_docs_edge_security = document_service.aggregate_documents(
        pipeline_security_channel
    )

    # documents_with_ids = map(lambda doc: {**doc, 'id': generate_unique_id(doc)}, documents)
    if kb_article_docs_edge_stable:
        kb_article_docs_edge_stable = process_kb_articles(
            kb_article_docs_edge_stable
        )
    if kb_article_docs_edge_security:
        kb_article_docs_edge_security = process_kb_articles(
            kb_article_docs_edge_security
        )

    kb_article_docs_edge_combined = (
        kb_article_docs_edge_stable + kb_article_docs_edge_security
    )

    return (
        kb_article_docs_windows_results["results"],
        kb_article_docs_edge_combined,
    )


# get update_packages
def extract_update_packages(
    start_date: datetime, end_date: datetime, max_records: int = 10
) -> List[Dict[str, Any]]:
    """Extract Microsoft update package documents.

    Extracts Microsoft update package documents from the "report_docstore" MongoDB
    database.

    Note:
        DO NOT MODIFY this function.

    Args:
        start_date (datetime): The start date of the time range to extract documents.
        end_date (datetime): The end date of the time range to extract documents.
        max_records (int, optional): Maximum number of records to extract.
            Defaults to 10.

    Returns:
        List[Dict[str, Any]]: A list of update package documents.

    Notes:
        - Update packages are filtered based on published date and matched to product builds.
        - Resulting documents are grouped by package URL and sorted by build number.
        - The projection excludes "downloadable_packages.install_resources_html" and "_id" fields.
    """
    db_name = "report_docstore"
    collection_name = "microsoft_update_packages"

    pipeline = [
        {"$match": {"published": {"$gte": start_date, "$lte": end_date}}},
        {
            "$lookup": {
                "from": "microsoft_product_builds",
                "localField": "product_build_id",
                "foreignField": "product_build_id",
                "as": "matched_builds",
            }
        },
        {
            "$match": {
                "matched_builds": {
                    "$elemMatch": {
                        "product_version": {
                            "$in": ["21H2", "22H2", "23H2", "24H2", ""]
                        },
                        "product_architecture": {
                            "$in": ["32-bit_systems", "x64-based_systems", ""]
                        },
                    }
                }
            }
        },
        {
            "$project": {
                "downloadable_packages.install_resources_html": 0,
            }
        },
        {
            "$project": {
                "_id": 0,
                "product_build_id": 1,
                "published": 1,
                "build_number": 1,
                "package_type": 1,
                "package_url": 1,
                "id": 1,
                "downloadable_packages": 1,
            }
        },
        {
            "$group": {
                "_id": "$package_url",
                "product_build_ids": {"$push": "$product_build_id"},
                "first_doc": {"$first": "$$ROOT"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "product_build_ids": 1,
                "published": "$first_doc.published",
                "build_number": "$first_doc.build_number",
                "package_type": "$first_doc.package_type",
                "package_url": "$first_doc.package_url",
                "id": "$first_doc.id",
                "downloadable_packages": "$first_doc.downloadable_packages",
            }
        },
    ]
    document_service = DocumentService(db_name, collection_name)
    update_packages_docs = document_service.aggregate_documents(pipeline)
    for article in update_packages_docs:
        article["build_number"] = tuple(article["build_number"])

    update_packages_docs.sort(key=lambda x: x["build_number"])
    for article in update_packages_docs:
        article["build_number"] = list(article["build_number"])

    return update_packages_docs


def extract_msrc_posts(
    start_date: datetime, end_date: datetime, max_records: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Extract MSRC post documents.

    Extracts MSRC post documents from the "report_docstore" MongoDB database.

    Note:
        DO NOT MODIFY this function.

    Args:
        start_date (datetime): The start date of the time range to extract documents.
        end_date (datetime): The end date of the time range to extract documents.
        max_records (Optional[int], optional): Maximum number of records to extract.
            Defaults to None.

    Returns:
        List[Dict[str, Any]]: A list of MSRC post documents.

    Notes:
        - MSRC posts are filtered based on collection name, published date, and product build IDs.
        - Resulting documents have the following fields excluded: _id, added_to_vector_store, added_to_summary_index, added_to_graph_store, relationships,
            start_char_idx, end_char_idx, text_template, metadata_template, metadata_seperator, class_name, and hash.
        - The kb_ids field is initialized to an empty list if it does not exist in a document.
    """
    db_name = "report_docstore"
    collection_name = "docstore"
    query = {
        "metadata.collection": "msrc_security_update",
        "metadata.published": {"$gte": start_date, "$lt": end_date},
        # "metadata.product_build_ids": {"$exists": True, "$not": {"$size": 0}},
    }
    max_records = max_records
    projection = {
        "_id": 0,
        "metadata.added_to_vector_store": 0,
        "metadata.added_to_summary_index": 0,
        "metadata.added_to_graph_store": 0,
        "relationships": 0,
        "start_char_idx": 0,
        "end_char_idx": 0,
        "text_template": 0,
        "metadata_template": 0,
        "metadata_seperator": 0,
        "class_name": 0,
        "hash": 0,
    }
    msrc_docs = extract_from_mongo(
        db_name, collection_name, query, max_records, None, projection
    )["results"]
    for msrc in msrc_docs:
        msrc.setdefault("kb_ids", [])

    return msrc_docs


def convert_received_date_time(doc):
    """Convert the received date time from a string to a datetime object.

    Args:
        doc (dict): The document to process.

    Returns:
        dict: The processed document with the converted received date time.
    """
    try:
        received_date_time = doc["metadata"]["receivedDateTime"]

        if isinstance(received_date_time, datetime):
            # Ensure timezone awareness
            if received_date_time.tzinfo is None:
                received_date_time = received_date_time.replace(
                    tzinfo=timezone.utc
                )
            doc["metadata"]["receivedDateTime"] = received_date_time
            return doc

        doc["metadata"]["receivedDateTime"] = datetime.fromisoformat(
            str(received_date_time).rstrip('Z')
        ).replace(tzinfo=timezone.utc)
        logging.debug(
            "Converted receivedDateTime to:"
            f" {doc['metadata']['receivedDateTime']}"
        )
    except KeyError:
        logging.warning("receivedDateTime missing in metadata")
        pass
    except ValueError as e:
        logging.error(f"Failed to convert receivedDateTime: {str(e)}")
        pass
    return doc


def extract_patch_posts(
    start_date: datetime,
    end_date: datetime,
    max_records: int = 10,
) -> List[Dict[str, Any]]:
    """
    Extracts patch management posts from the "report_docstore" MongoDB database.

    Args:
        start_date (datetime): The start date of the time range to extract documents.
        end_date (datetime): The end date of the time range to extract documents.
        max_records (int, optional): The maximum number of records to return. Defaults to 10.

    Returns:
        List[Dict[str, Any]]: A list of patch management post documents.

    Notes:
        - Patch management posts are filtered based on collection name and published date
        - Documents are sorted by receivedDateTime
        - Historical document lookup and thread linking is handled by the transformer
    """
    db_name = "report_docstore"
    collection_name = "docstore"

    # Primary query: date-based extraction
    query = {
        "metadata.collection": "patch_management",
        "metadata.published": {"$gte": start_date, "$lt": end_date},
    }

    projection = {
        "_id": 0,
        "metadata.added_to_vector_store": 0,
        "metadata.added_to_summary_index": 0,
        "metadata.added_to_graph_store": 0,
        "metadata.build_numbers": 0,
        "relationships": 0,
        "start_char_idx": 0,
        "end_char_idx": 0,
        "text_template": 0,
        "metadata_template": 0,
        "metadata_seperator": 0,
        "class_name": 0,
        "hash": 0,
    }

    # Extract documents within date range
    docs = extract_from_mongo(
        db_name, collection_name, query, max_records, None, projection
    )["results"]

    # Convert receivedDateTime and sort
    docs = [convert_received_date_time(doc) for doc in docs]
    docs = sorted(docs, key=lambda x: x["metadata"]["receivedDateTime"])
    # print out the metadata.id for each document, add some text

    return docs
