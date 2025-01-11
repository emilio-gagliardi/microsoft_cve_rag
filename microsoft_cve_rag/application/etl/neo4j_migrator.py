from typing import Any, Dict, List, Tuple, Set
import tiktoken  # Import tiktoken for token counting
from neo4j import GraphDatabase
from neomodel import config as NeomodelConfig  # required by AsyncDatabase
from neomodel.async_.core import AsyncDatabase  # required for db CRUD
from datetime import datetime
import time
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
    SymptomService,
    CauseService,
    FixService,
    ToolService,
    get_graph_db_uri,
    set_graph_db_uri,
)

from application.etl.ai_prompts import AIMigrationPrompts

import json
import uuid
import re
import openai
from openai import OpenAI
import asyncio
import hashlib
import os

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
        "MSRCSecurityUpdate": "MSRCPost",
        "PatchManagement": "PatchManagementPost",
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
        },
        "MSRCSecurityUpdate": {
            "mongo_id": "node_id",
        },
        "PatchManagement": {
            "id": "node_id",
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
    "chromium based": "edge",
    "microsoft edge": "edge",
    "microsoft edge chromiumbased": "edge",
    "edge": "edge",
}


async def load_v2_products() -> List[graph_db_models.Product]:
    # Fetch all Product nodes from the v2 database

    product_service = ProductService(db_manager)
    products = await product_service.model.nodes.all()
    return products


async def load_v2_msrc_posts() -> List[graph_db_models.MSRCPost]:
    # Fetch all MSRCPost nodes from the v2 database

    msrc_post_service = MSRCPostService(db_manager)
    posts = await msrc_post_service.model.nodes.all()
    return posts


async def load_v2_patch_management_posts() -> List[graph_db_models.PatchManagementPost]:
    # Fetch all PatchManagementPost nodes from the v2 database

    patch_management_post_service = PatchManagementPostService(db_manager)
    posts = await patch_management_post_service.model.nodes.all()
    return posts


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


# Helper function to count tokens and calculate costs
def count_tokens_and_costs(prompt: str, response: str) -> Tuple[int, int, float]:
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    input_tokens = len(enc.encode(prompt))
    output_tokens = len(enc.encode(response))
    input_cost = (input_tokens / 1_000_000) * 0.150
    output_cost = (output_tokens / 1_000_000) * 0.600
    return input_tokens, output_tokens, input_cost + output_cost


async def process_row_with_ai(
    row: Dict[str, Any], v2_nodes: List, node_type: str, token_counter: Dict[str, float]
) -> Dict:
    client = OpenAI()

    # Prepare the V2 specs based on node type (same as before)
    if node_type == "AffectedProduct":
        v2_specs = ", ".join(
            [
                f"({p.node_id}, {getattr(p, 'product_name', 'null')} {getattr(p, 'product_version', 'null')} {getattr(p, 'product_architecture', 'null')})"
                for p in v2_nodes
            ]
        )
    elif node_type == "MSRCSecurityUpdate":
        v2_specs = ", ".join(
            [
                f"({p.node_id}, {getattr(p, 'msrc_id', 'null')}, {getattr(p, 'title', 'null')}, {getattr(p, 'published', 'null')})"
                for p in v2_nodes
            ]
        )
    elif node_type == "PatchManagement":
        v2_specs = ", ".join([f"({p.node_id})" for p in v2_nodes])
    elif node_type == "FAQ":
        return None
    else:
        v2_specs = ""  # For types without V2 nodes

    if not v2_specs:
        v2_specs = "No Nodes of this type in V2 database. Create this node in V2."

    property_keys = set(row.keys()) - {"NodeId", "Labels"}
    props = {k: row[k] for k in property_keys}

    # Compute a unique key for the row
    def compute_row_key(row, node_type):
        data_to_hash = json.dumps({"node_type": node_type, "row": row}, sort_keys=True)
        row_hash = hashlib.sha256(data_to_hash.encode("utf-8")).hexdigest()
        return row_hash

    row_key = compute_row_key(row, node_type)
    data_directory = r"C:\Users\emili\PycharmProjects\microsoft_cve_rag\microsoft_cve_rag\application\data"
    cache_dir = os.path.join(data_directory, "llm_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file_path = os.path.join(cache_dir, f"{row_key}.json")

    # Check if result is already cached
    if os.path.exists(cache_file_path):
        # Load cached result
        def load_cached_result():
            with open(cache_file_path, "r", encoding="utf-8") as f:
                return json.load(f)

        result = await asyncio.to_thread(load_cached_result)

        # Ensure token counters are incremented by 0
        token_counter["input_tokens"] += 0
        token_counter["output_tokens"] += 0
        token_counter["total_cost"] += 0.0
        return result

    # Implementing the desired behavior (same as before)
    if node_type == "MSRCSecurityUpdate":
        v1_mongo_id = row.get("mongo_id")
        if v1_mongo_id:
            v2_node_id_set = {p.node_id for p in v2_nodes}
            if v1_mongo_id in v2_node_id_set:
                # Match found
                v1_original_id = row.get("id") or str(row.get("NodeId"))
                result = {
                    "action": "map_to_existing",
                    "v1_original_id": v1_original_id,
                    "v2_node_id": v1_mongo_id,  # V2 node_id matches V1 mongo_id
                    "new_label": None,
                    "new_props": {},
                }

                # Save the result to cache
                def save_result_to_cache():
                    with open(cache_file_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False)

                await asyncio.to_thread(save_result_to_cache)
                # Ensure token counters are incremented by 0
                token_counter["input_tokens"] += 0
                token_counter["output_tokens"] += 0
                token_counter["total_cost"] += 0.0
                return result

    elif node_type == "PatchManagement":
        v1_id = row.get("id")
        if v1_id:
            v2_node_id_set = {p.node_id for p in v2_nodes}
            if v1_id in v2_node_id_set:
                # Match found
                v1_original_id = v1_id
                result = {
                    "action": "map_to_existing",
                    "v1_original_id": v1_original_id,
                    "v2_node_id": v1_id,  # V2 node_id matches V1 id
                    "new_label": None,
                    "new_props": {},
                }

                # Save the result to cache
                def save_result_to_cache():
                    with open(cache_file_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False)

                await asyncio.to_thread(save_result_to_cache)
                # Ensure token counters are incremented by 0
                token_counter["input_tokens"] += 0
                token_counter["output_tokens"] += 0
                token_counter["total_cost"] += 0.0
                return result

    # If no match is found, or for other node types, proceed to call the LLM
    # Retrieve the prompt template
    prompt_template = AIMigrationPrompts.get_prompt(node_type)

    # Format the prompt with runtime variables
    prompt = prompt_template.format(
        labels=row.get("Labels", ""),
        props=json.dumps(props, ensure_ascii=False),
        v2_specs=v2_specs,
    )

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

    # Populate new_props and handle defaults (same as before)
    if result["new_label"] == "Symptom":
        result["v2_node_id"] = str(uuid.uuid4())
        if "v1_original_id" in result:
            result["new_props"]["symptom_label"] = result["v1_original_id"]
        else:
            print(
                f"Warning: v1_original_id missing in the response for symptom {result}"
            )
            result["new_props"]["symptom_label"] = None
    else:
        if result["action"] == "map_to_existing":
            # Use the v2_node_id provided by the LLM
            result["v2_node_id"] = result["v2_node_id"]
        elif result["action"] == "create_new":
            # Generate a new UUID for the node
            result["v2_node_id"] = str(uuid.uuid4())

    if result["new_props"]:
        # Ensure 'version' and 'architecture' have default values
        version = result["new_props"].get("version")
        architecture = result["new_props"].get("architecture")
        result["new_props"]["version"] = version if version else "NV"
        result["new_props"]["architecture"] = architecture if architecture else "NA"

        # Parse build_number into an array of ints if it's a string
        build_number = result["new_props"].get("build_number")
        if build_number and isinstance(build_number, str):
            numbers = re.findall(r"\d+", build_number)
            result["new_props"]["build_number"] = [int(num) for num in numbers]
        elif build_number is None:
            result["new_props"]["build_number"] = None

    # Count tokens and costs
    input_tokens, output_tokens, cost = count_tokens_and_costs(prompt, content)
    token_counter["input_tokens"] += input_tokens
    token_counter["output_tokens"] += output_tokens
    token_counter["total_cost"] += cost

    # Save the result to cache
    def save_result_to_cache():
        with open(cache_file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

    await asyncio.to_thread(save_result_to_cache)

    return result


async def transform_nodes(
    df_v1_nodes: pd.DataFrame, v2_nodes: List, node_type: str
) -> Tuple[List[Dict], Dict[str, str], Dict[str, str], List[str]]:
    nodes_to_create = []
    node_mapping = {}
    v1_to_v2_mapping = {}
    token_counter = {"input_tokens": 0, "output_tokens": 0, "total_cost": 0.0}
    processed_v1_ids = []

    # Convert DataFrame to records
    rows = df_v1_nodes.to_dict(orient="records")
    print(f"num rows: {len(rows)}\n")

    for row in rows:
        try:
            result = await process_row_with_ai(row, v2_nodes, node_type, token_counter)
            if result is None or result.get("action") == "ignore":
                continue

            v1_id = result["v1_original_id"]
            processed_v1_ids.append(v1_id)
            action = result["action"]

            if action == "map_to_existing":
                v2_node_id = result["v2_node_id"]
                node_mapping[v1_id] = v2_node_id
                v1_to_v2_mapping[v1_id] = v2_node_id
            elif action == "create_new":
                new_label = result["new_label"]
                new_props = result["new_props"]
                new_node_id = result["v2_node_id"]  # Use the generated v2_node_id
                new_props["node_id"] = new_node_id

                nodes_to_create.append(
                    {
                        "labels": [new_label],
                        "properties": new_props,
                    }
                )

                node_mapping[v1_id] = new_node_id
                if new_label in ["Product", "MSRCPost", "PatchManagementPost"]:
                    v1_to_v2_mapping[v1_id] = new_node_id

        except Exception as e:
            print(f"Error processing row: {row}")
            print(f"Error message: {str(e)}")

    print(f"Total input tokens: {token_counter['input_tokens']}")
    print(f"Total output tokens: {token_counter['output_tokens']}")
    print(f"Total cost: ${token_counter['total_cost']:.6f}")
    print("Completed processing nodes....\n")
    return nodes_to_create, node_mapping, v1_to_v2_mapping, processed_v1_ids


def process_json_file(file_path):
    with open(file_path, "r", encoding="utf-8-sig") as file:
        content = file.read()
    # print("Raw content from file:")
    # print(repr(content[:500]))
    fixed_content = fix_escape_chars(content)
    # print("Content after fixing escape characters:")
    # print(repr(fixed_content[:500]))
    try:
        data = json.loads(fixed_content)
        # print("Successfully parsed the file after fixing escape characters.")
        return data, None
    except json.JSONDecodeError as e:
        print(f"JSON parser failed even after fixing escape characters. Error: {e}")
        return None, str(e)


def fix_escape_chars(content):
    # Function to replace invalid escape sequences
    def replace_invalid_escapes(match):
        char = match.group(1)
        if char in 'bfnrt"\\/':
            return "\\" + char
        elif char == "u":
            return match.group(0)  # Keep valid unicode escapes
        else:
            return char  # Remove backslash from invalid escapes

    # Replace invalid escape sequences
    fixed_content = re.sub(r"\\(.)", replace_invalid_escapes, content)

    # Fix invalid Unicode escape sequences
    fixed_content = re.sub(
        r"\\u[0-9a-fA-F]{0,3}",
        lambda m: m.group(0) + "0" * (4 - len(m.group(0)[2:])),
        fixed_content,
    )

    return fixed_content


def identify_missing_nodes(
    relationships_file: str, processed_v1_ids: List[str]
) -> Set[str]:
    missing_nodes = set()

    # Read the JSON data
    data, error = process_json_file(relationships_file)
    if error:
        print(f"Error reading relationships file: {error}")
        return missing_nodes  # or handle the error as appropriate
    # print(f"type: {type(data)}")
    # print(f"processed_v1_ids: {processed_v1_ids}")
    for row in data:
        source_id = str(row["sourceId"])
        target_id = str(row["targetId"])

        if source_id not in processed_v1_ids:
            missing_nodes.add(source_id)
        if target_id not in processed_v1_ids:
            missing_nodes.add(target_id)

    return missing_nodes


async def transform_relationships(
    input_file: str,
    node_mapping: Dict[str, str],
    v1_to_v2_mapping: Dict[str, str],
    missing_nodes: Set[str],
) -> List[Dict]:
    transformed_relationships = []

    # Read the JSON data
    data, error = process_json_file(input_file)
    print(f"type: {type(data)}")
    # print(f"node_mapping: {node_mapping}")
    if error:
        print(f"Error reading relationships file: {error}")
        return transformed_relationships  # or handle the error as appropriate

    for row in data:
        # Use 'id' from 'sourceProperties' and 'targetProperties' as V1 IDs
        source_v1_id = row["sourceProperties"].get("id")
        target_v1_id = row["targetProperties"].get("id")

        if not source_v1_id:
            print(f"Error: Missing 'id' in sourceProperties for row {row}")
            continue
        if not target_v1_id:
            print(f"Error: Missing 'id' in targetProperties for row {row}")
            continue

        # Map V1 IDs to V2 IDs
        source_id = node_mapping.get(source_v1_id)
        target_id = node_mapping.get(target_v1_id)

        if not source_id:
            if source_v1_id in missing_nodes:
                # Skip relationships involving missing nodes
                continue
            else:
                print(f"Error: Missing mapping for source_v1_id {source_v1_id}")
                continue

        if not target_id:
            if target_v1_id in missing_nodes:
                # Skip relationships involving missing nodes
                continue
            else:
                print(f"Error: Missing mapping for target_v1_id {target_v1_id}")
                continue

        # If the target node is a V1 node mapped to a V2 node, use the mapped V2 node id
        if target_v1_id in v1_to_v2_mapping:
            target_id = v1_to_v2_mapping[target_v1_id]

        # Similarly, if the source node is mapped, update the source_id
        if source_v1_id in v1_to_v2_mapping:
            source_id = v1_to_v2_mapping[source_v1_id]

        # Get the relationship type and properties
        relationship_type = row["relationshipType"]
        relationship_properties = row.get("relationshipProperties", {})

        transformed_relationships.append(
            {
                "type": relationship_type,
                "source_id": source_id,
                "target_id": target_id,
                "properties": relationship_properties,
            }
        )
        print("relationship processed")
        # Optionally print or log the transformed relationship
        print(
            {
                "type": relationship_type,
                "source_id": source_id,
                "target_id": target_id,
                "properties": relationship_properties,
            }
        )
    return transformed_relationships
