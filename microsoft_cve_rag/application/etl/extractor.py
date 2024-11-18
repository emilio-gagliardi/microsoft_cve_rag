# Purpose: Extract data from various sources
# Inputs: Source configurations
# Outputs: Raw data
# Dependencies: None
from typing import Any, Dict, Optional, List, Tuple
from application.services.document_service import DocumentService
from pymongo import ASCENDING, DESCENDING
from bson import ObjectId
import uuid
from datetime import datetime
import hashlib
import json


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


# get products
def extract_products(max_records=10):
    db_name = "report_docstore"
    collection_name = "microsoft_products"
    query = {
        "product_name": {"$nin": ["edge_ext"]},
        "product_version": {"$in": ["21H2", "22H2", "23H2", "24H2", ""]},
        "product_architecture": {"$in": ["32-bit_systems", "x64-based_systems", ""]},
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
    print(f"Total Products: {len(product_docs)}")
    return product_docs

    # get product_builds


def extract_product_builds(start_date, end_date, max_records=10):
    db_name = "report_docstore"
    collection_name = "microsoft_product_builds"
    query = {
        "published": {"$gte": start_date, "$lt": end_date},
        "product_version": {"$in": ["21H2", "22H2", "23H2", "24H2", ""]},
        "product_architecture": {"$in": ["32-bit_systems", "x64-based_systems", ""]},
        "product_name": {
            "$in": [
                "windows_10",
                "windows_11",
                "microsoft_edge_(chromium-based)_extended_stable",
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
    print(f"Total Product Builds: {len(product_build_docs)}")
    return product_build_docs


def generate_kb_id(document):
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
    if article["kb_id"] is None:
        article["build_number"] = ()
    else:
        article["build_number"] = tuple(map(int, article["kb_id"].split(".")))
    article["id"] = generate_kb_id(article)
    return article


def convert_build_number(article):
    article["build_number"] = list(article["build_number"])
    return article


def process_kb_articles(articles):
    # Step 1: Create tuples from the original data and generate new ID hash
    articles = list(map(process_kb_article, articles))

    # Step 2: Sort the list by the new tupled build_number
    articles.sort(key=lambda x: x["build_number"])

    # Step 3: Convert the tuple back to a list of ints
    articles = list(map(convert_build_number, articles))

    return articles


def extract_kb_articles(start_date, end_date, max_records=10):
    # # get kb articles
    # # NOTE. This required merging data from two collections. microsoft_kb_articles contains everything but the text. Search docstore of windows 10/11 edge documents with matching KB_ID

    # remove kbs without product_builds
    db_name = "report_docstore"
    collection_name = "microsoft_kb_articles"
    # pipeline = [
    #     {
    #         "$lookup": {
    #             "from": "microsoft_product_builds",
    #             "localField": "product_build_id",
    #             "foreignField": "product_build_id",
    #             "as": "matched_builds",
    #         }
    #     },
    #     {"$match": {"matched_builds": {"$size": 0}}},
    #     {"$project": {"_id": 1}},
    # ]

    # document_service = DocumentService(db_name, collection_name)
    # results = document_service.aggregate_documents(pipeline)
    # ids_to_delete = [doc["_id"] for doc in results]
    # print(f"num kbs to remove: {len(ids_to_delete)}")
    # delete_count = document_service.delete_documents({"_id": {"$in": ids_to_delete}})
    # print(f"delete count: {delete_count}")
    db_name = "report_docstore"
    collection_name = "microsoft_kb_articles"
    # equals null is for windows kb articles
    query = {
        "published": {"$gte": start_date, "$lt": end_date},
        "cve_id": {"$eq": None},
    }
    projection = {"_id": 0, "cve_id": 0}
    max_records = max_records
    sort = [
        ("kb_id", ASCENDING),
    ]
    kb_article_docs_windows = extract_from_mongo(
        db_name, collection_name, query, max_records, sort, projection
    )["results"]

    if kb_article_docs_windows:
        # extract windows 10 and windows 11 docs from docstore to match with docs from microsoft_kb_articles.
        # This is to extract the text and title from the docstore docs and pass them to the microsoft_kb_articles docs.
        # get the unique 'kb_id's for windows-based KB articles
        unique_kb_ids_windows = set(
            kb_article["kb_id"] for kb_article in kb_article_docs_windows
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
        windows_docs = extract_from_mongo(db_name, collection_name, query, max_records)[
            "results"
        ]

        for kb_article in kb_article_docs_windows:
            kb_id = kb_article["kb_id"]
            for doc in windows_docs:
                if kb_id in doc["metadata"]["post_id"]:
                    kb_article["title"] = doc["metadata"]["title"]
                    kb_article["text"] = doc["text"]

    print(f"Total Windows-based KB articles: {len(kb_article_docs_windows)}")

    # For edge-based KB articles
    # These are synthetic in that microsoft doesn't publish kbs for edge issues, they point to the stable notes/security notes. I'm building these for symmetry with windows-based kb articles
    db_name = "report_docstore"
    collection_name = "microsoft_kb_articles"
    pipeline_stable_channel = [
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
                                    {"$cond": [{"$eq": ["$$value", ""]}, "", "."]},
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
                "kb_id": {
                    "$first": {
                        "$concat": [
                            {"$toString": {"$arrayElemAt": ["$build_number", 0]}},
                            ".",
                            {"$toString": {"$arrayElemAt": ["$build_number", 1]}},
                            ".",
                            {"$toString": {"$arrayElemAt": ["$build_number", 2]}},
                            ".",
                            {"$toString": {"$arrayElemAt": ["$build_number", 3]}},
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
                                    {"$cond": [{"$eq": ["$$value", ""]}, "", "."]},
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
                "kb_id": {
                    "$first": {
                        "$concat": [
                            {"$toString": {"$arrayElemAt": ["$build_number", 0]}},
                            ".",
                            {"$toString": {"$arrayElemAt": ["$build_number", 1]}},
                            ".",
                            {"$toString": {"$arrayElemAt": ["$build_number", 2]}},
                            ".",
                            {"$toString": {"$arrayElemAt": ["$build_number", 3]}},
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
        kb_article_docs_edge_stable = process_kb_articles(kb_article_docs_edge_stable)
    if kb_article_docs_edge_security:
        kb_article_docs_edge_security = process_kb_articles(
            kb_article_docs_edge_security
        )

    kb_article_docs_edge_combined = (
        kb_article_docs_edge_stable + kb_article_docs_edge_security
    )
    print(f"Total Edge-based KB articles: {len(kb_article_docs_edge_combined)}")
    return kb_article_docs_windows, kb_article_docs_edge_combined


# get update_packages
def extract_update_packages(start_date, end_date, max_records=10):
    # db_name = "report_docstore"
    # collection_name = "microsoft_update_packages"
    # pipeline = [
    #     {
    #         "$lookup": {
    #             "from": "microsoft_product_builds",
    #             "localField": "product_build_id",
    #             "foreignField": "product_build_id",
    #             "as": "matched_builds",
    #         }
    #     },
    #     {"$match": {"matched_builds": {"$size": 0}}},
    #     {"$project": {"_id": 0, "id": 1}},
    # ]

    # # Assuming you have a DocumentService class with an aggregate_documents method
    # document_service = DocumentService(db_name, collection_name)
    # results = document_service.aggregate_documents(pipeline)
    # ids_to_delete = [doc["id"] for doc in results]
    # print(f"num update_packages to remove: {len(ids_to_delete)}\n{ids_to_delete}")

    # delete_count = document_service.delete_documents({"id": {"$in": ids_to_delete}})
    # print(f"delete count: {delete_count}")

    db_name = "report_docstore"
    collection_name = "microsoft_update_packages"
    # query = {
    #     "published": {"$gte": start_date, "$lt": end_date},
    # }
    # projection = {"_id": 0, "downloadable_packages.install_resources_html": 0}
    # max_records = max_records
    # update_packages_docs = extract_from_mongo(
    #     db_name, collection_name, query, max_records, None, projection
    # )["results"]
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
    # for item in update_packages_docs:
    #     print(item)
    print(f"Total Update Packages: {len(update_packages_docs)}")
    return update_packages_docs


def extract_msrc_posts(start_date, end_date, max_records=None):
    # get msrc posts
    #
    db_name = "report_docstore"
    collection_name = "docstore"
    query = {
        "metadata.collection": "msrc_security_update",
        "metadata.published": {"$gte": start_date, "$lt": end_date},
        "metadata.product_build_ids": {"$exists": True, "$not": {"$size": 0}},
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

    print(f"Total MSRC Posts: {len(msrc_docs)}")
    return msrc_docs


def convert_received_date_time(doc):
    try:
        received_date_time_str = doc["metadata"]["receivedDateTime"]
        doc["metadata"]["receivedDateTime"] = datetime.fromisoformat(
            received_date_time_str
        )
    except KeyError:
        # Handle case where 'receivedDateTime' might be missing
        pass
    except ValueError:
        # Handle case where date string is not in the correct format
        pass
    return doc


def extract_patch_posts(start_date, end_date, max_records=10):
    # get msrc posts
    #
    db_name = "report_docstore"
    collection_name = "docstore"
    query = {
        "metadata.collection": "patch_management",
        "metadata.published": {"$gte": start_date, "$lt": end_date},
    }
    max_records = max_records
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
    patch_docs_unsorted = extract_from_mongo(
        db_name, collection_name, query, max_records, None, projection
    )["results"]
    patch_docs_unsorted = [
        convert_received_date_time(doc) for doc in patch_docs_unsorted
    ]
    patch_docs_sorted = sorted(
        patch_docs_unsorted, key=lambda x: x["metadata"]["receivedDateTime"]
    )
    print(f"Total Patch Posts: {len(patch_docs_sorted)}")

    return patch_docs_sorted
