from typing import Any, Dict, List, Tuple
from neo4j import GraphDatabase
from neomodel import config as NeomodelConfig  # required by AsyncDatabase
from neomodel.async_.core import AsyncDatabase  # required for db CRUD
from datetime import datetime
import pandas as pd
from application.app_utils import (
    get_app_config,
    get_graph_db_credentials,
    get_openai_api_key,
)
from application.core.models import graph_db_models
from application.services.graph_db_service import (
    GraphDatabaseManager,
    ProductService,
    ProductBuildService,
    KBArticleService,
    UpdatePackageService,
    MSRCPostService,
    PatchManagementPostService,
    get_graph_db_uri,
    set_graph_db_uri,
)
import csv
import json
import uuid
import re
import openai
from openai import OpenAI
import asyncio

settings = get_app_config()
graph_credentials = get_graph_db_credentials()
openai.api_key = get_openai_api_key()
embedding_config = settings["EMBEDDING_CONFIG"]

graphdb_config = settings["GRAPHDB_CONFIG"]

db = AsyncDatabase()
set_graph_db_uri()
db_manager = GraphDatabaseManager(db)

mapping = {
    "labels": {
        "Symptom": "Symptom",
        "Cause": "Cause",
        "Fix": "Fix",
        "Tool": "Tool",
        "FAQ": "FAQ",
        "AffectedProduct": "Technology",
    },
    "properties": {
        "Symptom": {
            "id": "symptom_label",
            "description": "description",
        },
        "Cause": {
            "id": "node_id",
            "description": "description",
        },
        "Fix": {
            "id": "node_id",
            "description": "description",
        },
        "Tool": {
            "id": "node_id",
            "description": "description",
            "url": "source_url",
        },
        "FAQ": {
            "id": "node_id",
            "name": "node_label",
            "answer": "answer",
            "question": "question",
        },
        "AffectedProduct": {
            "id": "node_id",
            "name": "product_name",
            "version": "product_version",
        },
    },
    "relationships": {
        "HAS_SYMPTOM": "HAS_SYMPTOM",
        "HAS_CAUSE": "HAS_CAUSE",
        "HAS_FIX": "HAS_FIX",
        "HAS_TOOL": "HAS_TOOL",
        "HAS_FAQ": "HAS_FAQ",
        "AFFECTS_PRODUCT": "AFFECTS_PRODUCT",
    },
}

PRODUCT_NAME_MAPPING = {
    "chrome": "edge",
    "chromium": "edge",
    "chromiumbased": "edge",
    "microsoft edge": "edge",
    "microsoft edge chromiumbased": "edge",
    "edge": "edge",
}


async def load_v2_products() -> List[graph_db_models.Product]:
    # Fetch all Product nodes from the v2 database

    product_service = ProductService(db_manager)
    products = await product_service.model.nodes.all()
    return products


def normalize_name(product_name: str) -> str:
    """
    Normalize product names by:
    1. Converting to lowercase
    2. Removing special characters (parentheses, hyphens)
    3. Using a mapping to convert variations of product names to a canonical form ('edge')
    """
    # Remove special characters like parentheses and extra spaces
    normalized = re.sub(r"[^\w\s]", "", product_name.lower()).strip()
    print(f"Normalized product name: {normalized}")  # Optional: Debugging log
    return PRODUCT_NAME_MAPPING.get(normalized, normalized)


v1_to_v2_product_mapping = {}


async def process_row_with_ai(row: Dict[str, str], v2_products: List) -> Dict:
    client = OpenAI()
    # Create a string representation of v2 products for the prompt
    v2_product_specs = ", ".join(
        [
            f"{p.product_name} {p.product_version} {p.product_architecture}"
            for p in v2_products
        ]
    )

    prompt = f"""
Given the following data from a CSV row:
Labels: {row['labels']}
Properties: {row['props']}

And the following list of v2 products (format: product_name product_version product_architecture): {v2_product_specs}

Please analyze this data and return a JSON object with the following structure:
{{
    "new_label": "The appropriate label (Symptom, Cause, Fix, Tool, Product, Technology, or Ignore)",
    "new_props": {{
        "node_id": "Use the original id from the input",
        "name": "The normalized name",
        "version": "The version information, or null if not applicable",
        "architecture": "The architecture information, or null if not applicable",
        "build_number": "Product build number, or null if not applicable",
        "description": "Description if available",
        "url": "URL if it's a Tool, otherwise null",
        "original_id": "The original id from the input"
    }},
    "v2_product_mapping": "If this is a Product that maps to a v2 product, provide the exact v2 product specification (product_name product_version product_architecture), otherwise null"
}}

Rules:
1. Ignore (set new_label to "Ignore") for FAQ and PatchManagement nodes.
2. For AffectedProducts:
   - Try to map to the provided v2 products exactly, using all three attributes (product_name, product_version, product_architecture).
   - If it can't be mapped to a v2 product, use Technology as the new_label.
   - For Windows products, map to the appropriate Windows version, including version and architecture.
   - For the product Edge, use the correct format (e.g., "edge NV NA").
3. Normalize product names to match the v2 product specifications exactly when possible.
4. Handle cases where multiple products or versions are listed in a single field.
5. For Symptoms, Causes, Fixes, and Tools, keep their original labels.
6. For Symptoms, the v1 id also maps to the v2 property 'symptom_label'.
7. For Tools, include the URL in the new_props if available.
"""

    # Define a synchronous function to call OpenAI API
    def call_openai():
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use your desired model
            messages=[
                {
                    "role": "system",
                    "content": "You are a data transformation assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return response

    # Run the synchronous OpenAI call in a separate thread
    response = await asyncio.to_thread(call_openai)
    content = response.choices[0].message.content.strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from LLM response: {content}")
        raise e

    # Generate UUID for Symptoms
    if result["new_label"] == "Symptom":
        result["new_props"]["node_id"] = str(uuid.uuid4())
        result["new_props"]["symptom_label"] = result["new_props"].pop("original_id")
    else:
        result["new_props"]["node_id"] = result["new_props"]["original_id"]

    return result


async def transform_nodes(
    input_file: str, v2_products: List
) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    node_mapping = {}
    transformed_nodes = []

    with open(input_file, "r", encoding="utf-8-sig") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            # Handle potential BOM in column names
            if "ï»¿labels" in row:
                row["labels"] = row.pop("ï»¿labels")

            try:
                result = await process_row_with_ai(row, v2_products)

                if result["new_label"] != "Ignore":
                    transformed_nodes.append(
                        {
                            "labels": [result["new_label"]],
                            "properties": result["new_props"],
                        }
                    )

                    node_mapping[result["new_props"]["original_id"]] = {
                        "new_label": result["new_label"],
                        "new_id": result["new_props"]["node_id"],
                    }

                    # Track v1 to v2 product mappings
                    if result["v2_product_mapping"]:
                        v1_to_v2_product_mapping[result["new_props"]["original_id"]] = (
                            result["v2_product_mapping"]
                        )

            except Exception as e:
                print(f"Error processing row: {row}")
                print(f"Error message: {str(e)}")

    return node_mapping, v1_to_v2_product_mapping


# Insert non conforming v1 products as LegacyProducts
# await product_legacy_service.bulk_create(
#         products.to_dict(orient="records")
#     )

# Insert v1 Symptoms as Symptoms
# await symptom_service.bulk_create(
#         products.to_dict(orient="records")
#     )

# Insert v1 Causes as Causes
# await cause_service.bulk_create(
#         products.to_dict(orient="records")
#     )

# Insert v1 Causes as Causes
# await fix_service.bulk_create(
#         products.to_dict(orient="records")
#     )

# Insert v1 Causes as Causes
# await faq_service.bulk_create(
#         products.to_dict(orient="records")
#     )

# Insert v1 Causes as Causes
# await tool_service.bulk_create(
#         products.to_dict(orient="records")
#     )
# Here you would typically insert the transformed_nodes into your database
# For example: await graph_db_models.bulk_create_nodes(transformed_nodes)


async def transform_relationships(
    input_file: str,
    node_mapping: Dict[str, Dict],
    v1_to_v2_product_mapping: Dict[str, str],
) -> List[Dict]:
    transformed_relationships = []

    with open(input_file, "r", encoding="utf-8-sig") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            start_node = node_mapping.get(row["source_original_id"])
            end_node = node_mapping.get(row["target_original_id"])

            if start_node and end_node:
                # Check if the end node is a v1 product that was mapped to a v2 product
                if (
                    end_node["new_label"] == "Product"
                    and row["target_original_id"] in v1_to_v2_product_mapping
                ):
                    end_id = v1_to_v2_product_mapping[row["target_original_id"]]
                else:
                    end_id = end_node["new_id"]

                transformed_relationships.append(
                    {
                        "type": row["type"],
                        "start_id": start_node["new_id"],
                        "end_id": end_id,
                        "properties": json.loads(row["props"]) if row["props"] else {},
                    }
                )

    return transformed_relationships
