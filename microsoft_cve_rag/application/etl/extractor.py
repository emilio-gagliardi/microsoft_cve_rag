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
def extract_products():
    db_name = "report_docstore"
    collection_name = "microsoft_products"
    query = {
        "product_version": {"$in": ["21H2", "22H2", "23H2", "24H2", ""]},
        "product_architecture": {"$in": ["32-bit_systems", "x64-based_systems", ""]},
    }
    max_records = None
    sort = [
        ("product_name", ASCENDING),
        ("product_architecture", ASCENDING),
        ("product_version", ASCENDING),
    ]
    product_docs = extract_from_mongo(
        db_name, collection_name, query, max_records, sort
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
    }
    max_records = max_records
    sort = [
        ("product_name", ASCENDING),
        ("product_architecture", ASCENDING),
        ("product_version", ASCENDING),
        ("cve_id", ASCENDING),
    ]
    product_build_docs = extract_from_mongo(
        db_name, collection_name, query, max_records, sort
    )["results"]
    print(f"Total Product Builds: {len(product_build_docs)}")
    return product_build_docs


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
    query = {
        "published": {"$gte": start_date, "$lt": end_date},
        "cve_id": None,
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
            "excluded_embed_metadata_keys": 0,
            "excluded_llm_metadata_keys": 0,
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

    for article in kb_article_docs_edge_stable:
        article["build_number"] = tuple(map(int, article["kb_id"].split(".")))
    kb_article_docs_edge_stable.sort(key=lambda x: x["build_number"])
    for article in kb_article_docs_edge_stable:
        article["build_number"] = list(article["build_number"])

    for article in kb_article_docs_edge_security:
        article["build_number"] = tuple(map(int, article["kb_id"].split(".")))
        article["id"] = str(uuid.uuid4())
    kb_article_docs_edge_security.sort(key=lambda x: x["build_number"])
    for article in kb_article_docs_edge_security:
        article["build_number"] = list(article["build_number"])

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
    ]
    document_service = DocumentService(db_name, collection_name)
    update_packages_docs = document_service.aggregate_documents(pipeline)
    for article in update_packages_docs:
        article["build_number"] = tuple(article["build_number"])

    update_packages_docs.sort(key=lambda x: x["build_number"])
    for article in update_packages_docs:
        article["build_number"] = list(article["build_number"])
    print(f"Total Update Packages: {len(update_packages_docs)}")
    return update_packages_docs


def extract_msrc_posts(start_date, end_date, max_records=10):
    # get msrc posts
    #
    db_name = "report_docstore"
    collection_name = "docstore"
    query = {
        "metadata.collection": "msrc_security_update",
        "metadata.published": {"$gte": start_date, "$lt": end_date},
    }
    max_records = max_records
    projection = {
        "_id": 0,
        "metadata.added_to_vector_store": 0,
        "metadata.added_to_summary_index": 0,
        "metadata.added_to_graph_store": 0,
        "excluded_embed_metadata_keys": 0,
        "excluded_llm_metadata_keys": 0,
        "relationships": 0,
        "start_char_idx": 0,
        "end_char_idx": 0,
        "text_template": 0,
        "metadata_template": 0,
        "metadata_seperator": 0,
        "class_name": 0,
    }
    msrc_docs = extract_from_mongo(
        db_name, collection_name, query, max_records, None, projection
    )["results"]
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
        "excluded_embed_metadata_keys": 0,
        "excluded_llm_metadata_keys": 0,
        "relationships": 0,
        "start_char_idx": 0,
        "end_char_idx": 0,
        "text_template": 0,
        "metadata_template": 0,
        "metadata_seperator": 0,
        "class_name": 0,
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
    return patch_docs_sorted
