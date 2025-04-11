import asyncio
import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, DefaultDict, Set
from application.services.document_service import DocumentService
from pydantic import BaseModel
from collections import defaultdict
import pandas as pd
import numpy as np
import marvin
from marvin.ai.text import generate_llm_response
import markdown
# from markdown.extensions.codehilite import CodeHiliteExtension
# from markdown.extensions.fenced_code import FencedCodeExtension
from markupsafe import Markup  # Important for Jinja rendering
from bs4 import BeautifulSoup
import re
from openai import AuthenticationError, APIConnectionError
import logging

logging.getLogger(__name__)

marvin.settings.openai.chat.completions.model = "gpt-4o"
marvin_restructure_prompt = r"""
**Objective:** Extract key information from the provided Microsoft KB Article markdown, using its structure (headings, lists) as a primary guide. Output the information in a precise JSON format suitable for technical review. Prioritize direct extraction and fidelity to the source text. Minimize summarization or rephrasing.

**Instructions:**

Parse the following Microsoft KB Article markdown and structure the extracted information into a JSON object matching this schema:

```json
{{
    "doc_id": "string",                 // The unique document ID provided.
    "report_title": "string",           // The KB article title provided.
    "report_os_builds": ["string"],     // List of applicable OS Build numbers AND Windows Versions.
    "report_new_features": ["string"],  // List of key new features or improvements described.
    "report_bug_fixes": ["string"],     // List of key bug fixes or resolved vulnerabilities described.
    "report_known_issues_workarounds": ["string"], // List of known issues, preserving source details.
    "summary": ""                      // MUST BE AN EMPTY STRING.
}}
```
Detailed Extraction Rules:

1. doc_id: Use the doc_id value provided below.
2. report_title: Use the kb_article_title value provided below.
3. report_os_builds:
    - Extract OS Build numbers (e.g., "19044.4894") primarily from the kb_article_title.
    - Extract the list of Windows versions directly from the text under the 'Applies To' heading in the markdown.
    - Combine both builds and versions into a single list of strings.
    - If no relevant information is found, return ["No Information Available"].
4. report_new_features:
    - Look for sections like 'Summary', 'Highlights', or 'Improvements'.
    - Extract sentences or bullet points describing new functionality, significant updates, or enhancements introduced by the KB. Focus on additions, not fixes.
    - Extract the text verbatim as much as possible.
    - Limit to a maximum of 10 distinct items.
    - If no relevant information is found, return ["No Information Available"].
5. report_bug_fixes:
    - Look for sections like 'Highlights', 'Improvements', or specific mentions of fixes/vulnerabilities.
    - Extract sentences or bullet points describing resolved issues, addressed problems, or security vulnerabilities fixed by the KB.
    - Extract the text verbatim as much as possible.
    - Limit to a maximum of 10 distinct items.
    - If no relevant information is found, return ["No Information Available"].
6. report_known_issues_workarounds:
    - Locate the text under the 'Known issues in this update' heading (or similarly named section).
    - If the text indicates no known issues (e.g., "We are currently not aware of any issues..."), return ["No Information Available"].
    - If issues ARE listed:
        - Extract the description of each distinct issue.
        - Crucially: If the source markdown uses sub-headings or labels like 'Symptom:', 'Workaround:', 'Applies To:' within the description of a specific issue, include those labels and their corresponding text literally as part of the string for that issue.
        - Do NOT invent structure like "Who/What/Why/How". Simply copy the relevant text for each issue.
        - Format each distinct known issue (including its Symptom/Workaround/AppliesTo details if present) as a single string item in the list.
        - Limit to a maximum of 10 distinct items.
        - If no relevant information is found, return ["No Information Available"].
7. summary:
    - MUST BE AN EMPTY STRING.

Input Data:

doc_id: {doc_id}
KB Article title:
{kb_article_title}

KB Article Markdown Text:
--- START MARKDOWN ---
{kb_article_text}
--- END MARKDOWN ---

Output: Return ONLY the valid JSON object. Do not include any explanations or text before or after the JSON.
    """


# BEGIN report extractors =============================
def extract_product_info_from_text(text: str | None) -> Dict[str, List[str]]:
    """Extract product information from KB article text content."""
    products = []
    build_numbers = []

    # Handle None or empty text
    if not text or not isinstance(text, str):
        return {
            'products': products,
            'build_numbers': build_numbers
        }

    # Check summary and scraped_markdown
    text = text.lower()

    # Extract Windows versions
    if 'windows 10' in text:
        products.append('Windows 10')
    if 'windows 11' in text:
        products.append('Windows 11')
    if 'windows server' in text:
        products.append('Windows Server')

    # Extract build numbers using regex
    build_pattern = r'(?:build|version)\s+(\d+\.\d+\.\d+)'
    matches = re.finditer(build_pattern, text)
    build_numbers.extend(match.group(1) for match in matches)

    return {
        'products': list(set(products)),
        'build_numbers': list(set(build_numbers))
    }


def extract_kb_articles_for_report(
    start_date: datetime,
    end_date: datetime,
    kb_service: DocumentService
) -> List[Dict[str, Any]]:
    """Extract KB articles within a date range, preparing them for downstream processing.

    This function extracts two groups of KB articles:
    1. Those linked to product builds via a lookup.
    2. Those only found in the main KB collection (without direct build links).

    It performs deduplication based on kb_id, prioritizing records with more complete data.
    Feature engineering (like extracting product info from text) has been moved out
    of this function.

    Args:
        start_date (datetime): Start date for KB article search.
        end_date (datetime): End date for KB article search.
        kb_service (DocumentService): Document service instance for KB articles.

    Returns:
        List[Dict[str, Any]]: List of KB articles ready for further processing.
                               Group 1 docs have 'products'/'build_numbers' from lookup.
                               Group 2 docs have 'products'/'build_numbers' as empty lists.

    NOTE: Handles duplicate kb records by sorting and deduplicating, prioritizing
          records with 'cve_ids' and other content fields.
    """
    # --- Group 1: KB articles linked to product builds ---
    # Define the aggregation pipeline to find KB articles and their associated products/builds
    pipeline = [
        # Match KB articles within the specified date range and with numeric kb_id
        {
            "$match": {
                "published": {"$gte": start_date, "$lte": end_date},
                "kb_id": {"$regex": "^[0-9]+$", "$options": "i"}
            }
        },
        # Perform a left join with the 'microsoft_product_builds' collection
        {
            "$lookup": {
                "from": "microsoft_product_builds",
                "let": {"kb_id": "$kb_id"},  # Define variable for use in sub-pipeline
                "pipeline": [
                    # Match documents in 'microsoft_product_builds' with the same kb_id
                    {
                        "$match": {
                            "$expr": {
                                "$eq": ["$kb_id", "$$kb_id"]  # Compare build's kb_id with article's kb_id
                            }
                        }
                    },
                    # Project only the necessary fields from the joined documents
                    {
                        "$project": {
                            "_id": 0,  # Exclude the default _id field
                            "product": 1,
                            "build_number": 1
                        }
                    }
                ],
                "as": "product_info"  # Name of the array field to add with the lookup results
            }
        },
        # Add new fields 'products' and 'build_numbers' derived from the 'product_info' lookup array
        {
            "$addFields": {
                # Create a unique list of product names from the lookup results
                "products": {
                    "$reduce": {
                        "input": "$product_info",
                        "initialValue": [],  # Start with an empty array
                        "in": {
                            "$setUnion": [  # Add the current product to the set (ensures uniqueness)
                                "$$value",  # Accumulator (the list being built)
                                ["$$this.product"]  # Current product name in an array
                            ]
                        }
                    }
                },
                # Create a unique list of build numbers from the lookup results
                "build_numbers": {
                    "$reduce": {
                        "input": "$product_info",
                        "initialValue": [],  # Start with an empty array
                        "in": {
                            "$setUnion": [  # Add the current build number to the set (ensures uniqueness)
                                "$$value",  # Accumulator (the list being built)
                                ["$$this.build_number"]  # Current build number in an array
                            ]
                        }
                    }
                },
                # Add a source field to identify how these documents were primarily matched (optional)
                "source": {"$literal": "product_builds"}
            }
        },
        # Project the final desired fields for Group 1 documents
        {
            "$project": {
                "_id": 0,  # Exclude MongoDB default ID
                "excluded_llm_metadata_keys": 0,  # Exclude specific metadata field if present
                "product_info": 0,  # Remove the temporary lookup result array
                # Keep all other fields from the original KB article + the added fields
            }
        }
    ]

    # Execute the aggregation pipeline for Group 1
    group1_docs = list(kb_service.collection.aggregate(pipeline))

    # --- Deduplication using Pandas ---
    # Convert Group 1 results to a DataFrame for efficient processing and deduplication
    # This is necessary because the initial query might return multiple versions of the same KB article
    if group1_docs:  # Proceed only if there are documents to process
        df = pd.DataFrame(group1_docs)

        # Normalize 'cve_ids' to be a list (empty if missing)
        df['cve_ids'] = df['cve_ids'].apply(
            lambda x: [] if (x is None or (isinstance(x, float) and np.isnan(x))) else x
        )
        # Create a scoring system to prioritize keeping the "best" version of duplicate KB articles
        # Higher scores are given to records with more valuable content (summary, CVEs, title, etc.)
        df['score'] = (
            (df['summary'].notna() & (df['summary'].str.strip() != '')).astype(int) * 8 +  # Highest priority: Non-empty summary
            (df['cve_ids'].apply(lambda x: isinstance(x, list) and len(x) > 0)).astype(int) * 4 +  # Second priority: Has CVE IDs (check list non-empty)
            (df['title'].notna() & (df['title'].str.strip() != '')).astype(int) * 2 +  # Third priority: Non-empty title
            (df['scraped_markdown'].notna() & (df['scraped_markdown'].str.strip() != '')).astype(int) +  # Fourth priority: Non-empty markdown
            (df['text'].notna() & (df['text'].str.strip() != '')).astype(int)  # Fifth priority: Non-empty text (fallback)
        )

        # Sort by kb_id and then by score (descending) to bring the highest-scored record for each kb_id to the top
        df = df.sort_values(['kb_id', 'score'], ascending=[True, False])
        # Group by kb_id and keep only the first (highest scored) record in each group
        df = df.groupby('kb_id').first().reset_index()

        # Drop the temporary scoring column
        df = df.drop(columns=['score'])
        # Convert the deduplicated DataFrame back to a list of dictionaries
        group1_docs = df.to_dict('records')
    else:
        # If group1_docs was empty initially, ensure it remains an empty list
        group1_docs = []

    # --- REMOVED Feature Engineering for Group 1 ---
    # The following loop that called extract_product_info_from_text has been removed.
    # This step should now happen in a separate, dedicated feature engineering function/pipeline.
    # for doc in group1_docs:
    #     text = f"{doc.get('summary', '')} {doc.get('scraped_markdown', '')}"
    #     text_product_info = extract_product_info_from_text(text)
    #     # Merge with existing products and build numbers
    #     doc["products"] = list(set(doc.get("products", []) + text_product_info["products"]))
    #     doc["build_numbers"] = list(set(doc.get("build_numbers", []) + text_product_info["build_numbers"]))
    # --- End Removed Section ---

    # Get the set of kb_ids found in Group 1 to exclude them from the Group 2 query
    group1_kb_ids = set(doc["kb_id"] for doc in group1_docs)

    # --- Group 2: KB articles only present in the main collection (not linked via product_builds) ---
    # Define the query to find KB articles within the date range that were NOT in Group 1
    query = {
        "published": {"$gte": start_date, "$lte": end_date},  # Match date range
        "$and": [
            {"kb_id": {"$regex": "^[0-9]+$", "$options": "i"}},  # Match numeric kb_id format
            {"kb_id": {"$nin": list(group1_kb_ids)}}  # Exclude kb_ids already processed in Group 1
        ]
    }

    # Define the projection for Group 2 documents (exclude unnecessary fields)
    projection = {
        "_id": 0,
        "excluded_llm_metadata_keys": 0
    }

    # Query the document service for Group 2 articles
    # Assuming a large enough page_size to get all results for simplicity here.
    # In production, pagination might be needed if the result set is very large.
    result = kb_service.query_documents(
        query=query,
        projection=projection,
        page=1,
        page_size=9999  # Use a large page size or implement proper pagination
    )

    # Extract the list of documents from the query result
    group2_docs = result.get("results", [])

    # --- REMOVED Feature Engineering for Group 2 ---
    # The following loop that called extract_product_info_from_text has been removed.
    # for doc in group2_docs:
    #     text = f"{doc.get('summary', '')} {doc.get('scraped_markdown', '')}"
    #     product_info = extract_product_info_from_text(text)
    #     doc["products"] = product_info["products"]
    #     doc["build_numbers"] = product_info["build_numbers"]
    # --- End Removed Section ---

    # --- ADDED: Ensure structural consistency for Group 2 ---
    # Iterate through Group 2 documents to add 'products' and 'build_numbers' fields
    # Initialize them as empty lists to match the structure of Group 1 documents,
    # even though these documents didn't go through the product_builds lookup.
    # This ensures that all documents returned by this function have these fields.
    for doc in group2_docs:
        doc["products"] = []        # Add 'products' field, initialized as an empty list
        doc["build_numbers"] = []   # Add 'build_numbers' field, initialized as an empty list
        # Optionally add a source marker if desired, similar to Group 1
        # doc["source"] = "kb_only"

    # --- Combine both groups ---
    # Concatenate the processed Group 1 list and the prepared Group 2 list
    all_docs = group1_docs + group2_docs

    # Return the combined list of documents
    return all_docs


def extract_cve_details_for_report(
    cve_ids: List[str],
    kb_ids: List[str],
    docstore_service: DocumentService
) -> List[Dict[str, Any]]:
    """Extract CVE details for a list of CVE IDs and KB IDs.

    Args:
        cve_ids (List[str]): List of CVE IDs from KB articles
        kb_ids (List[str]): List of KB IDs to find additional CVE references
        docstore_service (DocumentService): Document service instance for docstore

    Returns:
        List[Dict[str, Any]]: List of CVE details with associated KB IDs
    """
    projection = {
        "_id": 0,
        "metadata.id": 1,
        "metadata.revision": 1,
        "metadata.published": 1,
        "metadata.source": 1,
        "metadata.post_id": 1,
        "metadata.cve_category": 1,
        "metadata.adp_base_score_num": 1,
        "metadata.adp_base_score_rating": 1,
        "metadata.cna_base_score_num": 1,
        "metadata.cna_base_score_rating": 1,
        "metadata.nist_base_score_num": 1,
        "metadata.nist_base_score_rating": 1,
        "kb_ids": 1  # Add kb_ids to projection
    }

    # Query 1: Find CVEs by post_id (direct CVE references)
    query1 = {"metadata.post_id": {"$in": cve_ids}}
    result1 = docstore_service.query_documents(
        query=query1,
        projection=projection,
        page=1,
        page_size=999
    )

    kb_ids_with_prefix = [f"kb{kb_id}" for kb_id in kb_ids]
    # Query 2: Find CVEs that reference the KB IDs
    query2 = {"kb_ids": {"$in": kb_ids_with_prefix}}
    result2 = docstore_service.query_documents(
        query=query2,
        projection=projection,
        page=1,
        page_size=999
    )

    # Combine results and add source information
    all_results = []
    seen_post_ids = set()

    for doc in result1.get("results", []):
        post_id = doc.get("metadata", {}).get("post_id")
        if post_id and post_id not in seen_post_ids:
            doc["cve_source"] = "direct"  # CVE was directly referenced in KB
            all_results.append(doc)
            seen_post_ids.add(post_id)

    for doc in result2.get("results", []):
        post_id = doc.get("metadata", {}).get("post_id")
        if post_id and post_id not in seen_post_ids:
            doc["cve_source"] = "kb_reference"  # CVE was found via KB reference
            all_results.append(doc)
            seen_post_ids.add(post_id)

    return all_results


# END report extractors =============================


# BEGIN REPORT TRANSFORMERS ==========================
class JSONSanitizingEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling datetime objects and other special types."""
    def default(self, obj):
        try:
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, (set, frozenset)):
                return list(obj)
            elif hasattr(obj, 'tolist'):  # Handle numpy arrays
                return obj.tolist()
            elif hasattr(obj, '__dict__'):  # Handle objects with __dict__
                return obj.__dict__
            elif isinstance(obj, float) and (obj != obj):  # Check for NaN
                return None
            return str(obj)
        except Exception as e:
            logging.warning(f"Error serializing object {type(obj)}: {e}")
            return str(obj)


class CVEScore(BaseModel):
    """Model for CVE score data."""
    source: str
    score_num: Optional[float]
    score_rating: Optional[str]


def choose_score_for_kb_report(metadata: Dict[str, Any]) -> CVEScore:
    """Select the highest score between CNA, NIST, and ADP scores.

    Args:
        metadata (Dict[str, Any]): Document metadata containing score information

    Returns:
        CVEScore: Selected score with its source and rating
    """
    scores = [
        ("CNA", metadata.get("cna_base_score_num"), metadata.get("cna_base_score_rating")),
        ("NIST", metadata.get("nist_base_score_num"), metadata.get("nist_base_score_rating")),
        ("ADP", metadata.get("adp_base_score_num"), metadata.get("adp_base_score_rating"))
    ]

    # Convert score numbers to float if they exist
    valid_scores = []
    for src, num, rating in scores:
        try:
            num_float = float(num) if num is not None else None
            if num_float is not None:
                valid_scores.append((src, num_float, rating))
        except (ValueError, TypeError):
            continue

    if not valid_scores:
        return CVEScore(source="None", score_num=None, score_rating=None)

    # Sort by score descending, then by priority (CNA > NIST > ADP)
    sorted_scores = sorted(
        valid_scores,
        key=lambda x: (-x[1], ["CNA", "NIST", "ADP"].index(x[0]))
    )

    best = sorted_scores[0]
    return CVEScore(
        source=best[0],
        score_num=best[1],
        score_rating=best[2]
    )


def process_cve_data_for_kb_report(
    cve_details: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Process CVE details and create a lookup dictionary with KB associations.

    Args:
        cve_details (List[Dict[str, Any]]): List of CVE documents

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping CVE IDs to their details
            and includes a special __kb_to_cve_map key for KB-to-CVE relationships
    """
    cve_lookup = {}
    kb_to_cve_map = defaultdict(set)  # Track KB to CVE relationships

    for doc in cve_details:
        metadata = doc.get("metadata", {})
        if not metadata:
            continue

        score = choose_score_for_kb_report(metadata)
        post_id = metadata.get("post_id")

        if not post_id:
            continue
        cve_category = metadata.get("cve_category")
        if cve_category == "NC":
            cve_category = "No_Category_Specified"
        cve_info = {
            "id": metadata.get("id"),
            "post_id": post_id,
            "revision": metadata.get("revision"),
            "published": metadata.get("published"),
            "source": metadata.get("source"),
            "category": cve_category,
            "score": score.model_dump(),
            "cve_source": doc.get("cve_source", "No Source Available"),
            "referenced_kbs": doc.get("kb_ids", [])  # Get kb_ids from root level
        }

        cve_lookup[post_id] = cve_info

        # Build KB to CVE mapping from root-level kb_ids
        for kb_id in doc.get("kb_ids", []):
            kb_to_cve_map[kb_id].add(post_id)

    # Convert set to list for JSON serialization
    kb_cve_map_json = {
        kb_id: list(cve_ids)
        for kb_id, cve_ids in kb_to_cve_map.items()
    }

    # Add the mapping to the lookup
    cve_lookup["__kb_to_cve_map"] = kb_cve_map_json

    return cve_lookup


# Utility functions ============================


def get_markdown_snippet(markdown_text: Optional[str], headings_to_stop_at: List[str], max_words: int) -> str:
    """
    Extracts an initial snippet from markdown text.

    The snippet includes text from the beginning up to the first occurrence
    of any specified heading (case-insensitive) or up to max_words,
    whichever comes first.

    Args:
        markdown_text: The input markdown string.
        headings_to_stop_at: A list of heading strings that mark the end of the desired snippet.
        max_words: The maximum number of words to include in the snippet.

    Returns:
        The extracted markdown snippet as a string.
    """
    if not markdown_text or not isinstance(markdown_text, str):
        return ""

    min_heading_index = len(markdown_text)  # Default to end if no heading found

    # Find the earliest occurrence of any stop heading
    for heading in headings_to_stop_at:
        # Use regex search for case-insensitivity and flexibility
        # Escape the heading text in case it contains regex special characters
        try:
            # Adding optional leading/trailing whitespace/hashes for robustness
            pattern = r"^[#\s]*" + re.escape(heading) + r"[#\s]*$"  # Match heading on its own line potentially
            match_standalone = re.search(pattern, markdown_text, re.IGNORECASE | re.MULTILINE)

            # Fallback to simple search if standalone pattern fails
            match_simple = re.search(re.escape(heading), markdown_text, re.IGNORECASE)

            match = match_standalone if match_standalone else match_simple

            if match:
                # Found a heading, update the minimum index if this one occurs earlier
                min_heading_index = min(min_heading_index, match.start())
        except re.error as e:
            # Log regex error if the heading is complex and causes issues
            logging.warning(f"Regex error searching for heading '{heading}': {e}. Skipping this heading.")
            continue  # Skip to the next heading

    # Slice the markdown up to the first heading found
    content_before_heading = markdown_text[:min_heading_index].strip()

    # Apply the word limit to the potentially truncated content
    words = content_before_heading.split()
    snippet_words = words[:max_words]  # Take the first max_words

    return ' '.join(snippet_words)


def extract_build_from_text(text: Optional[str]) -> Set[str]:
    """Extract build numbers (like 19045.3803) from text content using regex.

    Args:
        text: The text string to search within.

    Returns:
        A set of unique build numbers found.
    """
    builds: Set[str] = set()
    if not text or not isinstance(text, str):
        return builds

    # Regex patterns to find build numbers (Major.Minor format)
    # Covers "OS Build(s) X.Y", "build X.Y", "Version: **OS Build(s) X.Y**" etc.
    # Added word boundaries (\b) to avoid partial matches within other numbers.
    build_patterns = [
        r'\bOS Build[s]?\s+(\d+\.\d+)(?:\s+and\s+(\d+\.\d+))?',
        r'\bbuild\s+(\d+\.\d+)',
        r'\bBuild[s]?\s+(\d+\.\d+)(?:\s+and\s+(\d+\.\d+))?',
        r'Version:\s+\*\*OS Build[s]?\s+(\d+\.\d+)(?:\s+and\s+(\d+\.\d+))?\*\*'
    ]

    for pattern in build_patterns:
        # Use re.finditer to catch all occurrences, including 'and' clauses
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Group 1 always contains the first build number
            if match.group(1):
                builds.add(match.group(1))
            # Group 2 contains the second build number if 'and' was present
            if match.lastindex == 2 and match.group(2):
                builds.add(match.group(2))

    # Also look for builds mentioned directly, e.g., within parentheses or lists
    # Example: (Build 19045.3803) or KB5031356 (OS Builds 19044.3570 and 19045.3570)
    # This simpler pattern catches X.Y formats that might be missed above
    standalone_build_pattern = r'\b(\d+\.\d+)\b'
    matches = re.finditer(standalone_build_pattern, text)
    for match in matches:
        # Very basic sanity check: avoid matching things like '10.0' if it's likely a version
        if '.' in match.group(1) and not match.group(1).endswith('.0'):
            builds.add(match.group(1))

    return builds


def extract_products_from_text(text: Optional[str]) -> Set[str]:
    """Extract potential product names (Windows versions, Server) from text.

    Args:
        text: The text string to search within.

    Returns:
        A set of unique product keywords found.
    """
    products: Set[str] = set()
    if not text or not isinstance(text, str):
        return products

    text_lower = text.lower()
    # Use word boundaries (\b) to ensure whole words/phrases are matched
    if re.search(r'\bwindows\s+11\b', text_lower):
        products.add('Windows 11')
    if re.search(r'\bwindows\s+10\b', text_lower):
        products.add('Windows 10')
    # Match "Windows Server" followed by year or general "Windows Server"
    if re.search(r'\bwindows\s+server(\s+\d{4})?\b', text_lower):
        products.add('Windows Server')
    # Add other specific versions if needed, e.g.:
    # if re.search(r'\bwindows\s+server\s+2022\b', text_lower):
    #     products.add('Windows Server 2022')

    return products


def consolidate_product_info(row: pd.Series) -> pd.Series:
    """
    Consolidates product and build information from initial data and text fields.

    Args:
        row: A Pandas Series representing a KB article document.

    Returns:
        A Pandas Series containing consolidated 'extracted_products' and
        'extracted_build_numbers' (as sets for uniqueness).
    """
    # Start with data potentially provided by the extraction function (Group 1 or empty lists)
    initial_products = set(row.get('products', []))
    initial_builds = set(row.get('build_numbers', []))

    # Define text fields to search within
    title = row.get('title', '') or ''  # Ensure empty string if None
    summary = row.get('summary', '') or ''  # Ensure empty string if None
    markdown = row.get('scraped_markdown', '') or ''  # Ensure empty string if None
    # --- Define headings that typically mark the end of introductory/applicability sections ---
    stop_headings = [
        "How to get this update",
        "Improvements",
        "Known issues",  # Regex handles case like "Known Issues"
        "Known Issues in this update",
        "File information",
        "References",
        "More Information",
        "Summary",  # Sometimes used as a heading within the doc
        "Highlights",  # Common in newer KBs
        "Issue details"
        # Add any other common section headers that signal the end of the relevant intro part
    ]
    max_snippet_words = 150  # Set the desired word limit

    markdown_snippet = get_markdown_snippet(markdown, stop_headings, max_snippet_words)
    summary_snippet = ' '.join(summary.split()[0:max_snippet_words])
    # Combine relevant text fields for searching
    # Prioritize title and summary, add markdown as fallback
    search_text = f"{title} {summary_snippet} {markdown_snippet}"  # Combine for comprehensive search

    # Extract products and builds from the combined text
    text_products = extract_products_from_text(search_text)
    text_builds = extract_build_from_text(search_text)

    # Combine initial data with text extractions
    # Using sets automatically handles deduplication
    consolidated_products = initial_products.union(text_products)
    consolidated_builds = initial_builds.union(text_builds)

    # Remove potential placeholder values like [0,0,0,0] if they came from initial data
    consolidated_builds.discard('0.0.0.0')  # Assuming build number format is always string here
    consolidated_builds = {b for b in consolidated_builds if b}  # Remove empty strings if any

    return pd.Series({
        'extracted_products': list(consolidated_products),  # Convert back to list for consistency
        'extracted_build_numbers': list(consolidated_builds)  # Convert back to list
    })


def process_product_info(row: pd.Series) -> Dict[str, Any]:
    """
    Process product information from a KB article row.

    Args:
        row (pd.Series): Row from the KB DataFrame containing product information

    Returns:
        Dict[str, Any]: Dictionary containing:
            - product_families (List[str]): List of product families
            - products (List[str]): List of products
            - build_numbers (List[str]): List of build numbers
            - is_windows_10 (bool): Whether article affects Windows 10
            - is_windows_11 (bool): Whether article affects Windows 11
            - is_server (bool): Whether article affects Windows Server
    """
    # Get products and build numbers with defensive defaults
    products = row.get('products', []) or []
    build_numbers = row.get('build_numbers', []) or []

    # Check if build numbers need to be extracted from text
    if not build_numbers or (len(build_numbers) == 4 and all(b == 0 for b in build_numbers)):
        markdown_text = row.get('scraped_markdown', '') or ''
        # Limit markdown text to first 200 words
        markdown_snippet = ' '.join(markdown_text.split()[:200])
        text_builds = extract_build_from_text(markdown_snippet)
        if text_builds:
            build_numbers = text_builds

    # Derive product families from products
    product_families = []
    if isinstance(products, list):
        products_str = ' '.join(str(p).lower() for p in products)
        if 'windows 10' in products_str:
            product_families.append('Windows 10')
        if 'windows 11' in products_str:
            product_families.append('Windows 11')
        if 'server' in products_str:
            product_families.append('Windows Server')

    # Default classifications
    is_windows_10 = 'Windows 10' in product_families
    is_windows_11 = 'Windows 11' in product_families
    is_server = 'Windows Server' in product_families

    # If no product info found, fall back to text analysis
    if not any([is_windows_10, is_windows_11, is_server]):
        text = row.get('text', '')
        text_classification = classify_os(text)
        is_windows_10 = text_classification.get('is_windows_10', False)
        is_windows_11 = text_classification.get('is_windows_11', False)
        is_server = text_classification.get('is_server', False)

        # Update product families based on text classification
        if is_windows_10:
            product_families.append('Windows 10')
        if is_windows_11:
            product_families.append('Windows 11')
        if is_server:
            product_families.append('Windows Server')

    return {
        'product_families': product_families,
        'products': products,
        'build_numbers': build_numbers,
        'is_windows_10': is_windows_10,
        'is_windows_11': is_windows_11,
        'is_server': is_server
    }


def get_all_cve_ids(row: pd.Series, kb_to_cve_map: Dict[str, List[str]]) -> List[str]:
    """
    Get both direct and indirect CVE references for a KB article.

    Combines CVE IDs directly listed in the KB article with those found through
    indirect references in the CVE-to-KB mapping.

    Args:
        row (pd.Series): Row from the KB DataFrame containing kb_id and cve_ids
        kb_to_cve_map (Dict[str, List[str]]): Mapping of KB IDs to CVE IDs

    Returns:
        List[str]: Combined and deduplicated list of CVE IDs
    """
    kb_id = str(row['kb_id'])  # Ensure kb_id is a string

    # Handle direct CVEs - if cve_ids is NaN or None, use empty list
    direct_cves = []
    if isinstance(row.get('cve_ids'), list):  # Only use if it's actually a list
        direct_cves = row['cve_ids']

    # Get indirect CVEs from the mapping, ensure we get a list back
    indirect_cves = kb_to_cve_map.get(f"kb{kb_id}", [])

    # Ensure both are lists before trying set operations
    direct_cves = list(direct_cves) if isinstance(direct_cves, (list, set)) else []
    indirect_cves = list(indirect_cves) if isinstance(indirect_cves, (list, set)) else []

    # Combine and deduplicate
    return list(set(direct_cves + indirect_cves))  # Union of both sets


def attach_cve_details(
    row: pd.Series,
    cve_lookup: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Create CVE details structure for a KB record."""
    try:
        if not isinstance(row.get('cve_ids'), list):
            return {}

        # Group CVEs by category
        category_groups: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        product_impact: DefaultDict[str, Dict[str, Any]] = defaultdict(
            lambda: {'total_cves': 0, 'max_score': 0.0, 'categories': defaultdict(list)}
        )

        for cve_id in row['cve_ids']:
            try:
                if not cve_id or cve_id not in cve_lookup:
                    continue

                cve_info = cve_lookup[cve_id]
                if not isinstance(cve_info, dict):
                    logging.warning(f"CVE info for {cve_id} is not a dict: {type(cve_info)}")
                    continue

                category = cve_info.get('category', 'Uncategorized')

                # Create flattened CVE detail for template
                try:
                    score_dict = cve_info.get('score', {})
                    if not isinstance(score_dict, dict):
                        logging.warning(f"Score for CVE {cve_id} is not a dict: {type(score_dict)}")
                        score_dict = {}

                    score_num = score_dict.get('score_num')
                    score_rating = score_dict.get('score_rating', 'Unknown')

                    cve_detail = {
                        'id': cve_id,
                        'post_id': cve_id,  # Required by template
                        'score': {  # Maintain nested structure for template
                            'score_num': score_num if isinstance(score_num, (int, float)) else 0.0,
                            'score_rating': score_rating
                        }
                    }
                except Exception as e:
                    logging.error(f"Error processing score for CVE {cve_id}: {e}")
                    logging.error(f"Score dict: {score_dict}")
                    continue

                # Add to category groups with minimal info
                try:
                    category_groups[category].append(cve_detail)
                except Exception as e:
                    logging.error(f"Error adding CVE {cve_id} to category {category}: {e}")
                    logging.error(f"CVE detail: {cve_detail}")
                    continue

                # Process product family impact
                try:
                    for family in row.get('product_families', []):
                        if not family:
                            continue

                        product_impact[family]['total_cves'] += 1
                        score = float(score_num) if isinstance(score_num, (int, float)) else 0.0
                        product_impact[family]['max_score'] = max(
                            product_impact[family]['max_score'],
                            score
                        )
                        product_impact[family]['categories'][category].append(cve_detail)
                except Exception as e:
                    logging.error(f"Error processing product family for CVE {cve_id}: {e}")
                    logging.error(f"Family: {family}, Score: {score_num}")
                    continue

            except Exception as e:
                logging.error(f"Error processing CVE {cve_id}: {e}")
                logging.error(f"CVE lookup info: {cve_lookup.get(cve_id)}")
                continue

        # Sort CVEs within each category by score
        try:
            def get_cve_score(cve: Dict[str, Any]) -> float:
                try:
                    return float(cve.get('score', {}).get('score_num', 0.0))
                except (TypeError, ValueError) as e:
                    logging.error(f"Error getting score from CVE: {cve}: {e}")
                    return 0.0

            # Sort all category groups
            for category in category_groups:
                try:
                    category_groups[category] = sorted(
                        category_groups[category],
                        key=get_cve_score,
                        reverse=True
                    )
                except Exception as e:
                    logging.error(f"Error sorting category {category}: {e}")
                    logging.error(f"CVEs in category: {category_groups[category]}")
                    continue

            # Sort product family CVEs
            for family in product_impact:
                for category in product_impact[family]['categories']:
                    try:
                        product_impact[family]['categories'][category] = sorted(
                            product_impact[family]['categories'][category],
                            key=get_cve_score,
                            reverse=True
                        )
                    except Exception as e:
                        logging.error(f"Error sorting family {family} category {category}: {e}")
                        logging.error(f"CVEs: {product_impact[family]['categories'][category]}")
                        continue
        except Exception as e:
            logging.error(f"Error during sorting operations: {e}")
            logging.error(f"Category keys: {list(category_groups.keys())}")
            logging.error(f"Product impact keys: {list(product_impact.keys())}")
            # Continue with unsorted data rather than failing

        # Final return with error handling
        try:
            categories_dict = dict(sorted(category_groups.items(), key=lambda x: str(x[0] or '')))
            product_impact_dict = dict(sorted(product_impact.items(), key=lambda x: str(x[0] or '')))

            return {
                'total_cves': len(row['cve_ids']),
                'categories': categories_dict,
                'product_impact': product_impact_dict
            }
        except Exception as e:
            logging.error(f"Error during final dictionary creation: {e}")
            logging.error(f"Category keys: {list(category_groups.keys())}")
            logging.error(f"Product impact keys: {list(product_impact.keys())}")
            return {
                'total_cves': len(row['cve_ids']),
                'categories': {},
                'product_impact': {}
            }

    except Exception as e:
        logging.error(f"Critical error in attach_cve_details: {e}")
        logging.error(f"Input row: {row}")
        return {
            'total_cves': 0,
            'categories': {},
            'product_impact': {}
        }


# Determine OS classification based on product info
def determine_os_classification(extracted_products: List[str]) -> str:
    """
    Determines the OS classification based on a list of extracted product names.

    Args:
        extracted_products: List of product names (e.g., "Windows 10", "Windows Server").

    Returns:
        A classification string: 'windows_10', 'windows_11', 'server', 'multi', or 'unknown'.
    """
    has_win10 = False
    has_win11 = False
    has_server = False

    for product in extracted_products:
        product_lower = str(product).lower()
        if 'windows 10' in product_lower:
            has_win10 = True
        if 'windows 11' in product_lower:
            has_win11 = True
        if 'server' in product_lower:  # Broad check for "server"
            has_server = True

    # Count distinct OS categories found
    categories = sum([has_win10, has_win11, has_server])

    if categories > 1:
        return 'multi'
    elif has_win10:
        return 'windows_10'
    elif has_win11:
        return 'windows_11'
    elif has_server:
        return 'server'
    else:
        # If no product info gave a clue, we classify as unknown here.
        # Text fallback could be added *here* if product info fails,
        # but relying on products first is cleaner.
        return 'unknown'


def format_build_numbers(builds: List[str]) -> List[str]:
    """
    Formats build numbers to keep the last two parts (e.g., 19045.3693)
    and handles potential LLM output variations.

    Args:
        builds: A list of string build numbers.

    Returns:
        A list of formatted, unique build numbers, or a default message.
    """
    formatted_builds_set: Set[str] = set()

    for build in builds:
        if not build or not isinstance(build, str):
            continue

        # Extract numeric parts using regex to handle variations like "OS Build X.Y"
        # This regex finds sequences like X.Y or X.Y.Z.W
        matches = re.findall(r'\b(\d+\.\d+(?:\.\d+)?(?:\.\d+)?)\b', build)

        for num_part in matches:
            parts = num_part.split('.')
            if len(parts) >= 2:
                # Take the last two significant parts for consistency
                # Assumes standard build format like 10.0.19045.3693 -> 19045.3693
                # Or directly 19045.3693
                major_build = parts[-2]
                minor_build = parts[-1]
                # Basic validation: Ensure they look like numbers and aren't just zero.
                if major_build.isdigit() and minor_build.isdigit() and (major_build != '0' or minor_build != '0'):
                    formatted_builds_set.add(f"{major_build}.{minor_build}")

    # Sort for consistent output order (optional but nice)
    sorted_builds = sorted(list(formatted_builds_set), key=lambda x: list(map(int, x.split('.'))), reverse=True)

    return sorted_builds if sorted_builds else ["Build number not available"]


# Fix list handling for report sections
def ensure_list(value):
    if value is None or (isinstance(value, list) and not value):
        return ["No Information Available"]
    if isinstance(value, str):
        # Only split if it's the "No Information Available" string being treated as chars
        if value == "No Information Available":
            return [value]
        return value.split('\n')  # Split on newlines for actual content
    if isinstance(value, list):
        return value
    return ["No Information Available"]


# Process build numbers using both LLM and regex extraction
def process_build_numbers(row):
    builds = set()  # Use set for automatic deduplication

    # First, get builds from LLM extraction
    llm_builds = row.get('report_os_builds', [])
    if isinstance(llm_builds, list):
        for build in llm_builds:
            if build != "No Information Available":
                # Extract numbers from potential text descriptions
                matches = re.finditer(r'(\d+\.\d+(?:\.\d+)?(?:\.\d+)?)', str(build))
                for match in matches:
                    builds.add(match.group(1))

    # Then extract from title and scraped_markdown using our regex
    if isinstance(row.get('title'), str):
        builds.update(extract_build_from_text(row['title']))
    if isinstance(row.get('scraped_markdown'), str):
        builds.update(extract_build_from_text(row['scraped_markdown']))

    # Clean and format build numbers
    formatted_builds = []
    for build in builds:
        parts = build.split('.')
        if len(parts) == 2:  # If we only have the last two numbers (e.g., 19045.3693)
            formatted_builds.append(build)  # Keep as is, frontend will handle display
        elif len(parts) == 4:  # Full build number
            if all(p == '0' for p in parts):  # [0,0,0,0] case
                continue
            formatted_builds.append('.'.join(parts[-2:]))  # Keep last two parts

    return formatted_builds if formatted_builds else ["Build number not available"]


# Ensure CVE details structure is complete and sanitized
def sanitize_cve_details(details):
    default_cve_details = {
        'total_cves': 0,
        'categories': {},
        'product_impact': {}
    }
    if not isinstance(details, dict):
        return default_cve_details

    # Ensure categories is a dict
    categories = details.get('categories', {})
    if not isinstance(categories, dict):
        categories = {}

    # Ensure each category has a list of CVEs
    sanitized_categories = {}
    for cat, cves in categories.items():
        if not isinstance(cves, list):
            continue
        sanitized_cves = []
        for cve in cves:
            if not isinstance(cve, dict):
                continue
            # Ensure score is properly structured
            score = cve.get('score', {})
            if not isinstance(score, dict):
                score = {'score_num': None, 'score_rating': 'Unknown'}
            sanitized_cves.append({
                'id': str(cve.get('id', '')),
                'post_id': str(cve.get('post_id', '')),
                'score': {
                    'score_num': score.get('score_num'),
                    'score_rating': str(score.get('score_rating', 'Unknown'))
                }
            })
        if sanitized_cves:
            sanitized_categories[str(cat)] = sanitized_cves

    # Ensure product_impact is a dict
    product_impact = details.get('product_impact', {})
    if not isinstance(product_impact, dict):
        product_impact = {}

    sanitized_impact = {}
    for product, impact in product_impact.items():
        if not isinstance(impact, dict):
            continue
        sanitized_impact[str(product)] = {
            'total_cves': int(impact.get('total_cves', 0)),
            'max_score': float(impact.get('max_score', 0.0)),
            'categories': sanitize_categories(impact.get('categories', {}))
        }

    return {
        'total_cves': int(details.get('total_cves', 0)),
        'categories': sanitized_categories,
        'product_impact': sanitized_impact
    }


def sanitize_categories(categories):
    if not isinstance(categories, dict):
        return {}
    sanitized = {}
    for cat, cves in categories.items():
        if not isinstance(cves, list):
            continue
        sanitized_cves = []
        for cve in cves:
            if not isinstance(cve, dict):
                continue
            score = cve.get('score', {})
            if not isinstance(score, dict):
                score = {'score_num': None, 'score_rating': 'Unknown'}
            sanitized_cves.append({
                'id': str(cve.get('id', '')),
                'post_id': str(cve.get('post_id', '')),
                'score': {
                    'score_num': score.get('score_num'),
                    'score_rating': str(score.get('score_rating', 'Unknown'))
                }
            })
        if sanitized_cves:
            sanitized[str(cat)] = sanitized_cves
    return sanitized


def clean_title(row):
    title = row.get('title', '')
    if isinstance(title, str):
        # Remove "- Microsoft Support" suffix only
        return re.sub(r'\s*-\s*Microsoft Support$', '', title).strip()
    return title


def validate_report_structure(response: Dict[str, Any]) -> bool:
    required_fields = {
        "doc_id", "report_title", "report_os_builds",
        "report_new_features", "report_bug_fixes",
        "report_known_issues_workarounds", "summary"
    }
    return all(field in response for field in required_fields)


def classify_os(text: str | None) -> Dict[str, bool]:
    """
    Classify the operating system based on the given text.

    Args:
        text (str | None): Text to analyze for OS classification

    Returns:
        Dict[str, bool]: Dictionary containing classification results:
            - is_windows_10 (bool): Whether text indicates Windows 10
            - is_windows_11 (bool): Whether text indicates Windows 11
            - is_server (bool): Whether text indicates Windows Server
    """
    # Return default values if text is None or empty
    if not text or not isinstance(text, str):
        return {
            "is_windows_10": False,
            "is_windows_11": False,
            "is_server": False
        }

    # Extract the first 200 words from the text
    words = text.lower().split()
    snippet = " ".join(words[:200])

    # Check for Windows 10 references
    is_windows_10 = bool(re.search(r"windows 10|windows10", snippet))

    # Check for Windows 11 references
    is_windows_11 = bool(re.search(r"windows 11|windows11", snippet))

    # Check for Windows Server references
    is_server = bool(re.search(r"windows server|server \d{4}", snippet))

    return {
        'is_windows_10': is_windows_10,
        'is_windows_11': is_windows_11,
        'is_server': is_server
    }


def get_os_from_text(text: str | None) -> List[str]:
    # Extract the first 200 words from the text.
    if not isinstance(text, str):
        return text
    words = text.split()
    snippet = " ".join(words[:200])
    found_os_results = []
    if re.search(r"windows 11|windows11", snippet.lower()):
        found_os_results.append("windows_11")
    if re.search(r"windows 10|windows10", snippet.lower()):
        found_os_results.append("windows_10")
    if re.search(r"windows server|server \d{4}", snippet.lower()):
        found_os_results.append("windows_server")
    return found_os_results


def extract_json_string(response_str: str) -> str:
    """
    Extracts the JSON string from a given LLM response that might be wrapped in markdown fences
    or contain extraneous text, returning only the content between the first '{' and the last '}'.

    Args:
        response_str (str): The raw response string from the LLM

    Returns:
        str: The extracted JSON string, or an empty string if no JSON is found
    """
    # Find the first '{' and the last '}'
    if not isinstance(response_str, str):
        return ""
    start_idx = response_str.find('{')
    end_idx = response_str.rfind('}')

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return response_str[start_idx:end_idx + 1]
    return ""


# Fix OS classification to properly detect 'multi' cases
# Deprecate
# def improved_os_classification(row):
#     # First check if we have both Windows 10 and Windows 11
#     has_win10 = False
#     has_win11 = False
#     has_server = False
#     logging.info(f"Classifying os for row: {row.to_dict()}")
#     # Check products array
#     products = row.get('products', [])
#     if isinstance(products, list):
#         for product in products:
#             product_str = str(product).lower()
#             if 'windows 10' in product_str:
#                 has_win10 = True
#             if 'windows 11' in product_str:
#                 has_win11 = True
#             if 'server' in product_str:
#                 has_server = True

#     # Check product_families
#     families = row.get('product_families', [])
#     if isinstance(families, list):
#         for family in families:
#             family_str = str(family).lower()
#             if 'windows 10' in family_str:
#                 has_win10 = True
#             if 'windows 11' in family_str:
#                 has_win11 = True
#             if 'server' in family_str:
#                 has_server = True

#     summary_os = get_os_from_text(row.get('summary', ''))

#     # Check is_* flags
#     if row.get('is_windows_10', False):
#         has_win10 = True
#     if row.get('is_windows_11', False):
#         has_win11 = True
#     if row.get('is_server', False):
#         has_server = True

#     # Determine classification
#     if (has_win10 and has_win11) or (has_win10 and has_server) or (has_win11 and has_server):
#         return 'multi'
#     elif has_win10:
#         return 'windows_10'
#     elif has_win11:
#         return 'windows_11'
#     elif has_server:
#         return 'server'

#     # Fallback to original function if no clear indicators
#     return determine_os_classification(row)


def sort_build_numbers(build_numbers: List[str]) -> List[str]:
    """Sort build numbers treating each component as an integer.

    Args:
        build_numbers: List of build numbers as strings (e.g., ["26100.712", "26100.1710"])

    Returns:
        List[str]: Sorted build numbers
    """
    def build_key(build: str) -> tuple:
        return tuple(int(x) for x in build.split('.'))

    return sorted(build_numbers, key=build_key)


# Comprehensive build number processing
# deprecate
# def comprehensive_build_processing(row):
#     """Process build numbers from all available sources.

#     Args:
#         row: DataFrame row

#     Returns:
#         List[str]: List of processed build numbers
#     """
#     all_builds = set()

#     # 1. Get build numbers from MongoDB (already processed earlier)
#     mongo_builds = row.get('build_numbers', [])
#     if isinstance(mongo_builds, list):
#         all_builds.update(str(b) for b in mongo_builds)

#     # 2. Get builds from LLM extraction in report_os_builds
#     llm_builds = row.get('report_os_builds', [])
#     if isinstance(llm_builds, list):
#         for build in llm_builds:
#             if build != "No Information Available":
#                 # Extract numbers from potential text descriptions
#                 matches = re.finditer(r'(\d+\.\d+(?:\.\d+)?(?:\.\d+)?)', str(build))
#                 for match in matches:
#                     all_builds.add(str(match.group(1)))

#     # 3. Extract from title and scraped_markdown using our regex
#     if isinstance(row.get('title'), str):
#         title_builds = extract_build_from_text(row['title'])
#         all_builds.update(str(b) for b in title_builds)
#     if isinstance(row.get('scraped_markdown'), str):
#         markdown_builds = extract_build_from_text(row['scraped_markdown'][:500])
#         all_builds.update(str(b) for b in markdown_builds)

#     # Clean and format build numbers
#     formatted_builds = []
#     for build in all_builds:
#         parts = str(build).split('.')
#         if len(parts) == 2:  # If we only have the last two numbers (e.g., 19045.3693)
#             formatted_builds.append(str(build))  # Keep as is, frontend will handle display
#         elif len(parts) == 4:  # Full build number
#             if all(p == '0' for p in parts):  # [0,0,0,0] case
#                 continue
#             formatted_builds.append('.'.join(parts[-2:]))  # Keep last two parts
#     formatted_builds = sort_build_numbers(formatted_builds) if formatted_builds else ["Build number not available"]
#     return formatted_builds

# import difflib


# def lint_raw_markdown(markdown_text: str, document_id: str = "unknown") -> bool:
#     """
#     Lint a markdown string by comparing it against the formatting
#     produced by mdformat. Any differences indicate potential issues.

#     Args:
#         markdown_text: The raw markdown string to lint.
#         document_id: An optional identifier for logging/reporting.

#     Returns:
#         bool: True if formatting issues were found (i.e. differences), False otherwise.
#     """
#     if not markdown_text or not isinstance(markdown_text, str):
#         logging.warning(f"[{document_id}] Received empty or invalid markdown.")
#         return False

#     try:
#         import mdformat
#     except ImportError:
#         logging.error("mdformat is not installed. Please run: pip install mdformat")
#         return True

#     # Format the markdown using mdformat's API
#     formatted_text = mdformat.text(markdown_text)

#     if formatted_text != markdown_text:
#         # Generate a diff between the original and formatted markdown
#         diff = difflib.unified_diff(
#             markdown_text.splitlines(),
#             formatted_text.splitlines(),
#             fromfile="original",
#             tofile="formatted",
#             lineterm=""
#         )
#         diff_str = "\n".join(diff)
#         print(f"\n--- Linting Issues for Document: {document_id} ---")
#         print(diff_str)
#         print("-" * 50)
#         return True
#     else:
#         print(f"mdformat: No issues found for Document: {document_id}.")
#         return False


def extract_code_block_languages(md_text: str) -> list:
    """
    Extract the language identifier from each fenced code block in the raw markdown.
    Returns a list of languages in order of appearance (defaulting to 'plaintext').
    """
    if not isinstance(md_text, str) or not md_text:
        logging.warning("extract_code_block_languages: Markdown input is not a string")
        return []
    # Pattern matches a fenced code block with an optional language tag.
    # The (?m) flag ensures ^ and $ match at the start and end of each line.
    pattern = re.compile(r'^```(?P<lang>\w+)?\n.*?^```', re.MULTILINE | re.DOTALL)
    languages = []
    for match in pattern.finditer(md_text):
        lang = match.group('lang')
        languages.append(lang.lower() if lang else 'plaintext')
    return languages


def apply_code_block_language_lookup(raw_md: str, html: str) -> str:
    """
    Replace <pre><code> blocks in the HTML output with div wrappers that include
    a header showing the language extracted from the raw markdown. Handles cases
    where codehilite might not add a class attribute.
    """
    if not isinstance(raw_md, str) or not raw_md:
        logging.warning("apply_code_block_language_lookup: Markdown input is not a string")
        return html
    if not isinstance(html, str) or not html:
        logging.warning("apply_code_block_language_lookup: HTML input is not a string")
        return html
    # Extract language info from the raw markdown.
    languages = extract_code_block_languages(raw_md)
    if not languages:
        logging.debug("apply_code_block_language_lookup: No languages extracted from markdown. No code block styling applied.")
        return html
    # --- Use BeautifulSoup ---
    soup = BeautifulSoup(html, 'html.parser')  # Or 'lxml' if installed

    # Find the code blocks - target the div codehilite generates
    # If codehilite failed completely, it might just be <pre><code>
    # Prioritize the div.codehilite structure first
    code_elements = soup.find_all('div', class_='codehilite')

    # Fallback: If no divs found, maybe codehilite didn't run or add the div? Look for plain <pre>
    # This might need refinement depending on how markdown lib behaves if codehilite fails entirely
    if not code_elements:
        pre_tags = soup.find_all('pre')
        # Filter pre tags that directly contain a code tag (basic check)
        code_elements = [pre for pre in pre_tags if pre.find('code', recursive=False)]
        if code_elements:
            logging.debug("Found <pre><code> blocks instead of div.codehilite.")
        else:
            logging.debug("No div.codehilite or pre>code blocks found by BeautifulSoup.")

    logging.debug(f"apply_code_block_language_lookup: Found {len(code_elements)} code elements using BeautifulSoup.")

    if len(code_elements) != len(languages):
        logging.warning(
            f"Mismatch between found code blocks ({len(code_elements)}) and extracted languages ({len(languages)}). "
            f"Styling might be incorrect."
        )
        # Decide how to handle mismatch: stop, style only matched count, etc.
        # For now, we'll style up to the minimum count.

    lang_class_map = {
        "powershell": "code-block-powershell", "batch": "code-block-batch", "cmd": "code-block-cmd",
        "shell": "code-block-shell", "registry": "code-block-registry", "reg": "code-block-registry",
        "plaintext": "code-block-text", "text": "code-block-text", "diff": "code-block-diff",
        "python": "code-block-python", "py": "code-block-python", "javascript": "code-block-javascript",
        "js": "code-block-javascript", "html": "code-block-html", "css": "code-block-css",
        "json": "code-block-json", "yaml": "code-block-yaml", "sql": "code-block-sql",
        "bash": "code-block-shell", "sh": "code-block-shell"
    }

    # Iterate through found code elements and apply the wrapper
    # Use min() to avoid index errors if counts mismatch
    num_to_process = min(len(code_elements), len(languages))
    for i in range(num_to_process):
        original_element = code_elements[i]
        lang = languages[i]
        logging.debug(f"Processing block {i} with language '{lang}'")

        # Determine the CSS class for the wrapper based on the *extracted* language
        wrapper_lang_class = lang_class_map.get(lang.lower(), "code-block-text")

        # Create the new wrapper structure
        new_wrapper = soup.new_tag('div', attrs={'class': f'code-block-wrapper {wrapper_lang_class}'})

        # Create the header
        header = soup.new_tag('div', attrs={'class': 'code-block-header'})
        header.string = lang.upper()  # Use the extracted language for the header
        new_wrapper.append(header)

        # --- Crucial: Move the *content* (the <pre> tag) into the new wrapper ---
        # Find the <pre> tag within the original element (div.codehilite or the <pre> itself)
        pre_tag = original_element.find('pre')
        if not pre_tag and original_element.name == 'pre':  # Handle the fallback case where we found <pre> directly
            pre_tag = original_element

        if pre_tag:
            # Detach the <pre> tag from its original parent
            pre_tag.extract()
            # Append the detached <pre> tag to our new wrapper
            new_wrapper.append(pre_tag)
            # Replace the original element (div.codehilite or pre) with the new wrapper
            original_element.replace_with(new_wrapper)
            logging.debug(f"  Replaced original element with new wrapper for block {i}")
        else:
            logging.warning(f"  Could not find <pre> tag inside element {i}. Skipping replacement.")

    # Return the modified HTML as a string
    return str(soup)


def add_target_blank_to_links(html_content: str) -> str:
    """
    Parses HTML content and adds target="_blank" and rel="noopener noreferrer"
    to all <a> tags.
    """
    if not html_content:
        return ""
    try:
        # Parse the HTML. Use 'lxml' if installed, otherwise default 'html.parser'.
        soup = BeautifulSoup(html_content, 'lxml' if 'lxml' in globals() else 'html.parser')

        # Find all <a> tags that have an href attribute (valid links)
        links = soup.find_all('a', href=True)

        for link in links:
            # Add or overwrite the target attribute
            link['target'] = '_blank'
            # Add or overwrite rel attribute for security best practice
            link['rel'] = 'noopener noreferrer'

        # Return the modified HTML as a string
        return str(soup)
    except Exception as e:
        logging.error(f"Error modifying HTML links: {e}", exc_info=True)
        return html_content  # Return original HTML on error


def preprocess_markdown(md_text: str) -> str:
    """
    Preprocesses markdown text generated by an LLM to fix common formatting
    issues, aiming for better compatibility with Markdown parsers.

    Focuses on:
    1. Normalizing newlines.
    2. Collapsing excessive blank lines.
    3. Ensuring *at least one* blank line separates major block elements
       (headings, lists, code fences) from preceding non-blank content,
       without inserting blank lines *within* lists or code blocks.
    4. Removing leading/trailing whitespace.
    """
    if not md_text or not isinstance(md_text, str):
        return ""

    # 1. Normalize Newlines (Consistent)
    text = md_text.replace('\r\n', '\n').replace('\r', '\n')

    # 2. Initial Whitespace Cleanup (Less Aggressive)
    #    - Remove leading/trailing whitespace from the whole text.
    #    - Collapse sequences of 3 or more newlines down to 2 (one blank line).
    #    This avoids the problematic line-by-line addition of blank lines.
    text = text.strip()
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Before Headings (#)
    text = re.sub(r'(?<=\S)\n(^(#{1,6}\s+.*))', r'\n\n\1', text, flags=re.M)

    return text.strip()  # Return final cleaned text


def markdown_to_html(md_text: str) -> Markup:
    if not md_text:
        return Markup("")

    # print("--- RAW MARKDOWN FROM MONGO ---")
    # print(md_text)
    # print("-----------------------------------------\n")
    # Preprocess markdown
    processed_md = preprocess_markdown(md_text)  # Or use raw md_text for isolated testing

    # print("--- MARKDOWN INPUT to markdown.markdown ---")
    # print(processed_md)
    # print("-----------------------------------------\n")
    active_extensions = ["tables", "fenced_code", "codehilite"]
    # Convert markdown to HTML
    html_output = markdown.markdown(
        processed_md,
        extensions=active_extensions,
        extension_configs={
            'codehilite': {
                'guess_lang': False,
                'noclasses': False,
                'use_pygments': True,
                'pygments_style': 'xcode'
            }
        }
    )

    # print("--- RAW HTML OUTPUT from markdown.markdown ---")
    # print(html_output)  # <<< INSPECT THIS OUTPUT
    # print("--------------------------------------------\n")

    # --- Wrap generated tables ---
    html_with_table_wrappers = re.sub(r'<table(.*?)>', r'<div class="table-container"><table\1>', html_output)
    html_with_table_wrappers = re.sub(r'</table>', r'</table></div>', html_with_table_wrappers)

    # --- Apply code block language lookup ---
    styled_html = apply_code_block_language_lookup(processed_md, html_with_table_wrappers)

    # --- Add target_blank to links ---
    styled_html = add_target_blank_to_links(styled_html)

    # print("--- Styled HTML Output ---")
    # print(styled_html)
    # print("-------------------------\n")

    return Markup(styled_html)


def merge_and_format_builds(row):
    """
    Merges build numbers from the formatted extraction ('build_numbers')
    and the LLM output ('report_os_builds'), then re-formats them consistently.
    """
    # Get both lists, ensuring they are actual lists using ensure_list
    builds_formatted = ensure_list(row.get('build_numbers', []))
    builds_llm = ensure_list(row.get('report_os_builds', []))

    # Define known placeholder values to ignore from each source
    placeholder_formatted = "Build number not available"
    # Add common LLM placeholders (case-insensitive check later)
    placeholders_llm_lower = {"no information available", "n/a", "none", "unknown", ""}

    # Use a set to automatically handle duplicates during collection
    combined_raw_set = set()

    # Add non-placeholder builds from the formatted list
    for build in builds_formatted:
        build_str = str(build).strip()  # Ensure string and strip whitespace
        if build_str and build_str != placeholder_formatted:
            combined_raw_set.add(build_str)

    # Add non-placeholder builds from the LLM list
    for build in builds_llm:
        build_str = str(build).strip()  # Ensure string and strip whitespace
        # Check against lowercased placeholders for robustness
        if build_str and build_str.lower() not in placeholders_llm_lower:
            # Add textual identifiers like '24H2' directly if needed,
            # otherwise rely on format_build_numbers to filter them if only numeric needed.
            combined_raw_set.add(build_str)

    # Apply the standard formatting function to the combined *raw* list
    # This ensures all builds (numeric or potentially textual like '24H2')
    # are processed by the formatter.
    final_builds = format_build_numbers(list(combined_raw_set))

    return final_builds


# ==============================================
# END UTILITY FUNCTIONS ========================
# ==============================================


async def transform_kb_data_for_kb_report(
    kb_articles: List[Dict[str, Any]],
    cve_lookup: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """Transforms raw KB article data for reporting, including product/build extraction and OS classification.

    Args:
        kb_articles: List of KB article dictionaries from the extractor function.
        cve_lookup: CVE lookup dictionary.

    Returns:
        A processed Pandas DataFrame ready for the report.
    """
    if not kb_articles:
        logging.warning("Received empty list of KB articles. Returning empty DataFrame.")
        return pd.DataFrame()

    logging.info(f"Starting transformation for {len(kb_articles)} KB articles.")
    kb_df = pd.DataFrame(kb_articles)

    # --- 1. Initial Data Cleaning ---
    # Replace infinite/NaN values - important before string operations
    kb_df = kb_df.replace([float('inf'), float('-inf')], float('nan'))  # Handle inf first
    # Fill NaN in object columns with empty string, numeric columns might need different handling if used
    for col in kb_df.select_dtypes(include=['object']).columns:
        kb_df[col] = kb_df[col].fillna('')
    # Ensure list types are present even if null/NaN initially
    list_cols = ['products', 'build_numbers', 'cve_ids']  # Add others if needed
    for col in list_cols:
        if col not in kb_df.columns:
            kb_df[col] = pd.Series([[] for _ in range(len(kb_df))], index=kb_df.index)
        else:
            kb_df[col] = kb_df[col].apply(lambda x: x if isinstance(x, list) else [])

    logging.debug("Step 1: Initial data cleaning complete.")

    # --- 2. Consolidate Product and Build Information ---
    # Apply the consolidation function to combine initial data + text extraction
    logging.debug("Step 2: Consolidating product and build information...")
    extracted_info = kb_df.apply(consolidate_product_info, axis=1)
    kb_df = pd.concat([kb_df, extracted_info], axis=1)
    # We now have 'extracted_products' and 'extracted_build_numbers' columns
    logging.debug("Product/Build consolidation complete.")
    # Log sample of extracted data
    if len(kb_df) > 0:
        logging.debug(f"Sample extracted products: {kb_df['extracted_products'].iloc[0]}")
        logging.debug(f"Sample extracted build numbers: {kb_df['extracted_build_numbers'].iloc[0]}")

    # --- 3. Determine OS Classification ---
    logging.debug("Step 3: Determining OS classification...")
    # Use the consolidated products to classify the OS
    kb_df['os_classification'] = kb_df['extracted_products'].apply(determine_os_classification)
    logging.debug("OS Classification Distribution:")
    logging.debug(kb_df['os_classification'].value_counts().to_dict())

    # --- 4. Process and Format Build Numbers ---
    logging.debug("Step 4: Formatting build numbers...")
    # Apply the formatting function to the consolidated build numbers
    # This becomes the definitive 'build_numbers' column for the report
    kb_df['build_numbers'] = kb_df['extracted_build_numbers'].apply(format_build_numbers)
    logging.debug("Build number formatting complete.")
    # Log sample formatted builds
    if len(kb_df) > 0:
        logging.debug(f"Sample formatted build numbers: {kb_df['build_numbers'].iloc[0]}")

    # --- 5. CVE Processing ---
    logging.debug("Step 5: Processing CVE information...")
    kb_to_cve_map = cve_lookup.pop('__kb_to_cve_map', {})
    # Ensure cve_ids uses the most complete list (consider if initial row['cve_ids'] should be merged here)
    kb_df['cve_ids'] = kb_df.apply(lambda row: get_all_cve_ids(row, kb_to_cve_map), axis=1)  # Assuming this function handles merging if needed
    kb_df['cve_details'] = kb_df.apply(lambda row: attach_cve_details(row, cve_lookup), axis=1)
    kb_df['cve_details'] = kb_df['cve_details'].apply(lambda x: sanitize_cve_details(x or {}))
    logging.debug("CVE processing complete.")

    # --- 6. Generate Report Structures (LLM/Async Tasks) ---
    logging.debug("Step 6: Generating report structures (async)...")
    tasks = []
    # Use a temporary unique ID if 'id' column is not reliable or missing
    if 'id' not in kb_df.columns:
        kb_df['temp_doc_id'] = [f"doc_{i}" for i in range(len(kb_df))]
        id_col = 'temp_doc_id'
    else:
        kb_df['id'] = kb_df['id'].astype(str)  # Ensure ID is string
        id_col = 'id'

    for _, row in kb_df.iterrows():
        # Ensure text fields used by LLM are strings
        title = str(row.get('title', ''))
        # Prefer scraped_markdown if available, fall back to summary, then text
        text = str(row.get('scraped_markdown', row.get('summary', row.get('text', ''))))
        doc_id = str(row[id_col])
        tasks.append(generate_kb_report_structure(title, text, doc_id))

    try:
        report_structures = await asyncio.gather(*tasks)
        # Handle cases where LLM might return None or errors
        valid_reports = [r for r in report_structures if r and isinstance(r, dict) and 'doc_id' in r]
        if len(valid_reports) != len(report_structures):
            logging.warning(f"Expected {len(report_structures)} report structures, but received {len(valid_reports)} valid ones.")
        if not valid_reports:
            logging.error("No valid report structures generated by LLM tasks.")
            # Decide how to proceed: return kb_df only, or raise error?
            # For now, create an empty report_df to avoid merge errors
            report_df = pd.DataFrame(columns=['doc_id'])  # Ensure doc_id column exists
        else:
            report_df = pd.DataFrame(valid_reports)

        logging.debug(f"Merging {len(kb_df)} KB articles with {len(report_df)} report structures.")
        # Merge KB data with LLM report structures
        final_df = pd.merge(
            kb_df,
            report_df,
            left_on=id_col,
            right_on='doc_id',
            how='left',  # Keep all KB articles, even if LLM fails for some
            suffixes=('_orig', '')  # Suffix for any overlapping columns from LLM
        )
        # Drop the temporary ID if used
        if id_col == 'temp_doc_id':
            final_df = final_df.drop(columns=['temp_doc_id'])
        # Ensure doc_id from report_df doesn't conflict if 'id' was original column
        if 'doc_id' in final_df.columns and id_col == 'id' and 'doc_id' != 'id':
            final_df = final_df.drop(columns=['doc_id'])

        logging.debug("Report structure merging complete.")

    except Exception as e:
        logging.exception("Error during async report generation or merging.")
        # Depending on requirements, either raise e or return the DataFrame without LLM data
        # For robustness, let's return the df processed so far, logging the error
        final_df = kb_df  # Fallback to kb_df if async part fails
        final_df['error_llm_generation'] = str(e)  # Add error column

    # --- 8. Final Formatting and Cleanup ---
    logging.debug("Step 8: Final formatting and cleanup...")

    # Ensure list fields from LLM are lists and handle missing ones
    list_fields_llm = {
        'report_new_features', 'report_bug_fixes', 'report_known_issues_workarounds', 'report_os_builds'
    }
    default_list_value = ["No Information Available"]

    for field in list_fields_llm:
        if field not in final_df.columns:
            # If LLM failed or didn't return the field, add it with default
            final_df[field] = pd.Series([default_list_value for _ in range(len(final_df))], index=final_df.index)
        else:
            # Ensure existing column contains lists, fill NaNs/None with default
            final_df[field] = final_df[field].apply(
                lambda x: processed_list if len(processed_list := ensure_list(x)) > 0 else default_list_value
            )

    # Apply the merging and formatting function to update 'report_os_builds'
    # This column will now hold the definitive, merged, and formatted list
    logging.debug("Merging and re-formatting build numbers from 'build_numbers' and 'report_os_builds'...")
    final_df['report_os_builds'] = final_df.apply(merge_and_format_builds, axis=1)

    # Generate HTML summary
    final_df['summary_html'] = final_df['summary_orig'].apply(markdown_to_html)

    # Clean title (handle potential _orig suffix from merge)
    final_df['title'] = final_df.apply(clean_title, axis=1)

    # Define final columns needed for the report
    final_columns = [
        'kb_id',
        'title',
        'published',
        'article_url',
        'os_classification',  # The result of our classification logic
        'build_numbers',      # Formatted build numbers from our extraction
        'cve_ids',
        'cve_details',
        'summary',           # Original summary text (if needed)
        'summary_html',      # HTML version of summary
        # Report structure fields (filled with defaults if LLM failed)
        'report_title',
        'report_os_builds',
        'report_new_features',
        'report_bug_fixes',
        'report_known_issues_workarounds',
        # Add any other essential original columns: 'source', 'tags', etc.
        'source',  # Keep track of original source
    ]

    # Select and reorder columns, dropping unnecessary ones
    # Create missing columns with default values if they don't exist
    for col in final_columns:
        if col not in final_df.columns:
            logging.warning(f"Column '{col}' not found in DataFrame. Adding with default value (None).")
            final_df[col] = None  # Or appropriate default (e.g., '' for string, [] for list)

    # Filter to keep only the final columns
    final_df = final_df[final_columns]

    logging.debug(f"Transformation complete. Final DataFrame shape: {final_df.shape}")
    logging.debug(f"Final columns: {list(final_df.columns)}")

    # Log a sample row for debugging final output
    if len(final_df) > 0:
        logging.debug("Sample final row:")
        # Convert Series to dict for cleaner logging
        logging.debug(final_df.iloc[0].to_dict())

    return final_df


async def generate_kb_report_structure(
    title: str,
    text: str,
    doc_id: str
) -> Dict[str, Any]:
    """Generate structured report data from KB article title and text. Core workflow function.

    Args:
        title (str): KB article title
        text (str): KB article text content
        doc_id (str): Unique document ID for cache file naming

    Returns:
        Dict[str, Any]: Structured report data
    """
    # Define cache directory
    cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "application",
        "data",
        "llm_cache",
        "kb_report_v1"
    )
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file path using the unique document ID
    cache_file = os.path.join(
        cache_dir,
        f"kb_report_restructured_{doc_id}.json"
    )

    # Check if cache exists
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_response = json.load(f)
                if validate_report_structure(cached_response):
                    return cached_response
                else:
                    print(f"Cached response for {doc_id} failed validation, regenerating...")
        except Exception as e:
            logging.warning(f"Error reading cache file {cache_file}: {e}")

    if pd.isna(text) or text is None or str(text).strip() == "":
        structured_response = {
            "doc_id": doc_id,
            "report_title": title,
            "report_os_builds": [],
            "report_new_features": ["No Information Available"],
            "report_bug_fixes": ["No Information Available"],
            "report_known_issues_workarounds": ["No Information Available"],
            "summary": ""
        }

        return structured_response
    else:
        model_kwargs = {
            "max_tokens": 2500,
            "temperature": 0.05,
            "response_format": {"type": "json_object"},
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        try:
            # Ensure the OpenAI API key is set
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

            # Attempt to generate the LLM response
            llm_response = await generate_llm_response(
                marvin_restructure_prompt.format(
                    kb_article_title=title,
                    kb_article_text=text,
                    doc_id=doc_id),
                model_kwargs=model_kwargs,
            )

            response_content = llm_response.response.choices[0].message.content
            # Cache the result
            structured_response = ""
            try:
                structured_response = json.loads(extract_json_string(response_content))
                if not validate_report_structure(structured_response):
                    raise ValueError(f"LLM response for {doc_id} failed validation")
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(structured_response, f, cls=JSONSanitizingEncoder, indent=2)
            except Exception as e:
                logging.warning(f"Error writing cache file {cache_file}: {e}")
            return structured_response

        except EnvironmentError as env_err:
            logging.error(f"Marvin Environment error: {env_err}")
            # Handle missing environment variable or other environment-related issues
            raise env_err
        except AuthenticationError as auth_err:
            logging.error(f"Marvin Authentication error: {auth_err}")
            # Handle issues related to API authentication
            raise auth_err
        except APIConnectionError as conn_err:
            logging.error(f"Marvin API connection error: {conn_err}")
            # Handle issues related to network connectivity or API server availability
            raise conn_err
        except Exception as e:
            logging.error(f"A Marvin error occurred: {e}")
            # Handle any other unforeseen exceptions
            raise e


def main():
    # from application.app_utils import (
    #     get_app_config,
    #     get_documents_db_credentials,
    # )
    """Test function to generate KB report with hardcoded date range."""
    from datetime import datetime
    import json
    from pprint import pprint
    from application.services.document_service import DocumentService

    # Initialize services
    kb_service = DocumentService(
        collection_name="microsoft_kb_articles",
        db_name="report_docstore"
    )

    # Hardcoded date range
    start_date = datetime(2024, 10, 1)
    end_date = datetime(2024, 10, 16)

    print(f"Extracting KB articles from {start_date} to {end_date}...")

    # Extract KB articles
    kb_articles = extract_kb_articles_for_report(start_date, end_date, kb_service)
    print(f"Found {len(kb_articles)} KB articles")

    # Extract CVE details
    cve_details = extract_cve_details_for_report(kb_service)
    print(f"Found {len(cve_details)} CVE details")

    # Transform KB data
    kb_df = transform_kb_data_for_kb_report(kb_articles, cve_details)
    print(f"Transformed data into {len(kb_df)} rows")

    # Save results to file for inspection
    output_file = "kb_report_test_output.json"
    with open(output_file, "w") as f:
        json.dump(kb_df.to_dict('records'), f, default=str, indent=2)

    print(f"Results saved to {output_file}")

    # Print sample of the first KB article
    if len(kb_df) > 0:
        print("\nSample KB article:")
        sample = kb_df.iloc[0]
        # Print only key fields to avoid overwhelming output
        sample_preview = {
            "kb_id": sample.get("kb_id"),
            "title": sample.get("title"),
            "published": sample.get("published"),
            "product_families": sample.get("product_families"),
            "total_cves": sample.get("cve_details", {}).get("total_cves", 0)
        }
        pprint(sample_preview)


if __name__ == "__main__":
    main()
