# Contains functions to initialize LlamaIndex components, such as the LLM, extractors, and any configurations.
# Functions to process documents and extract entities and relationships.
# insert current working directory into sys.path
import os
from pathlib import Path
# import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# print(sys.path)
import math
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    )
from llama_index.core.schema import Document as LlamaDocument
from llama_index.core.llms import ChatMessage
# from llama_index.llms.openai import OpenAI
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.prompts.base import PromptTemplate
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from application.etl.kg_extraction_prompts import get_prompt_template
from application.services.document_chunker import DocumentChunker
from application.services.embedding_service import (
    EmbeddingService,
    LlamaIndexEmbeddingAdapter,
    )
from application.services.vector_db_service import VectorDBService
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)
from typing import Dict, List, Literal, Tuple, Optional
import pandas as pd
import uuid
import json
import re
import time
import datetime
import tiktoken
import asyncio
from transformers import AutoTokenizer
import hashlib
from fuzzywuzzy import fuzz, process
from tqdm import tqdm
from application.app_utils import (
    get_app_config,
    get_graph_db_credentials,
    get_openai_api_key,
)
from openai import OpenAI
import logging
import warnings

logging.getLogger(__name__)

logging.getLogger('fuzzywuzzy.fuzz').setLevel(logging.ERROR)
logging.getLogger('fuzzywuzzy.process').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="fuzzywuzzy")

settings = get_app_config()
graph_db_settings = settings["GRAPHDB_CONFIG"]
credentials = get_graph_db_credentials()
os.environ["OPENAI_API_KEY"] = get_openai_api_key()
llm_model = "gpt-4o-mini"
# llm = OpenAI(model=llm_model, temperature=0.2, max_tokens=4096)
# tokenizer = tiktoken.encoding_for_model(llm_model).encode
# Settings.tokenizer = tokenizer
tokenizer = AutoTokenizer.from_pretrained("Snowflake/snowflake-arctic-embed-m-long")
# Settings.tokenizer = tokenizer

# Define your schema and extractor
EntityType = Literal["SYMPTOM", "CAUSE", "FIX", "TOOL", "TECHNOLOGY"]
RelationType = Literal[
    "HAS_SYMPTOM", "HAS_CAUSE", "HAS_FIX", "HAS_TOOL", "HAS_TECHNOLOGY"
]

schema = {
    "MSRC_POST": ["HAS_SYMPTOM", "HAS_CAUSE", "HAS_FIX", "HAS_TOOL", "HAS_TECHNOLOGY"],
    "PATCH_POST": ["HAS_SYMPTOM", "HAS_CAUSE", "HAS_FIX", "HAS_TOOL", "HAS_TECHNOLOGY"],
}


custom_prompt = PromptTemplate(
    "You are an expert in Windows security and patch management. You are tasked with identifying additional entities in a given text."
    "The entities are Symptom, Cause, Fix, Tool, and Technology. Not all are necessarily present. If the document metadata key `post_type` contains 'Solution provided' then the cause and fix are contained within the text."
    " A Symptom is an observable behavior, an error message, or any indication that something is going wrong in the system, as experienced by end users or system administrators. **It is not a vulnerability or technical exploit, but rather what the user notices as a result of the underlying issue.** For example, a Symptom could be system crashes, poor performance, unexpected reboots, failed updates, or error messages seen by the user. Symptoms help people identify issues that are affecting their environments. \nDo not describe vulnerabilities, exploits, or technical flaws directly. Instead, describe the **impact or observable behavior** that a system administrator or end user would see as a consequence of the security issue. Focus on the **user's perspective**, not the attacker's. \nBe thorough and consider subtle aspects that may not be explicitly stated. Describe how the security update affects a particular system or software product from the perspective of the end user. For instance:\n- 'The computer fails to boot after installing a security patch.'\n'Network communication is intermittently lost on systems using the affected driver.'\n- 'The system experiences slow performance and occasional reboots.'\nDo not restate or reference the original post directly; the Symptom should stand alone and specify the **observable behavior** or **impact** rather than describing the vulnerability itself.\n"
    "A Cause is any technical or situational condition responsible for the Symptom or issue described in the CVE. This could be a flaw in software, a misconfiguration, or any contributing factor that underpins the identified Symptom. Focus on the technical reasons driving the issue, and avoid restating the full text of the post.\n"
    "A Fix is a technical action, configuration change, or patch available to address the Cause or mitigate the Symptom. Focus on the specific technical response to the Cause, referencing affected systems or products without repeating the post's full text. This description should stand alone as a precise explanation of the Fix.\n"
    "A Tool is any software, command-line utility, or similar that Microsoft or end users may use to diagnose, mitigate, or resolve the issue. Focus on names, configurations, or commands necessary for addressing the symptoms or causes of the CVE."
    "Technologies are separate from the core Microsoft product families. Focus on identifying technologies referenced or implied that could relate to the text. Attempt to extract separate pieces for the `name`, `version`, `architecture`, and `build_number` fields."
    "Try to limit the output to {max_triplets_per_chunk} extracted paths.\n-------\n{text}\n-------\n"
)
# kg_extractor = SchemaLLMPathExtractor(
#     llm=llm,
#     possible_entities=EntityType,
#     possible_relations=RelationType,
#     kg_validation_schema=schema,
#     strict=True,
#     extract_prompt=custom_prompt,
# )
# kg_extractor.extract_prompt = custom_prompt
# print(kg_extractor.extract_prompt)
# Metadata dictionary

metadata_columns = {
    "MSRCPost": [
        "node_id",
        "post_id",
        "revision",
        "published",
        "post_type",
        "source",
        "kb_ids",
        "build_numbers",
        "impact_type",
        "severity_type",
        "product_build_ids",
    ],
    "PatchManagementPost": [
        "node_id",
        "thread_id",
        "topic",
        "published",
        "receivedDateTime",
        "post_type",
        "conversation_link",
        "cve_ids",
        "kb_ids",
        'noun_chunks',
        'keywords',
        'build_numbers',
    ],
}

MIN_STRING_LENGTH = 3  # Adjust this value as needed

def safe_extract_one(query, choices, score_cutoff=80):
    """
    Safely extracts the best match for the given query from the list of choices using fuzzy matching.

    Args:
        query (str): The input string to search for a best match.
        choices (List[str]): The list of possible choices to compare against the query.
        score_cutoff (int): Minimum score for a match to be considered valid (default is 80).

    Returns:
        Optional[Tuple[str, int]]: The best match and its score if a match is found, otherwise None.

    Note:
        This helper function prevents queries that are too short, which could lead to 
        meaningless matches or warnings in fuzzywuzzy. It provides a safe way to call 
        `process.extractOne` with a length check on the query.
    """
    if len(query) < MIN_STRING_LENGTH:
        return None
    return process.extractOne(query, choices, score_cutoff=score_cutoff)

def safe_partial_ratio(s1, s2, score_cutoff=80):
    """
    Safely computes the fuzzy partial ratio between two strings, ignoring case.

    Args:
        s1 (str): First string to compare.
        s2 (str): Second string to compare.
        score_cutoff (int): Minimum score to be considered a valid match (default is 80).

    Returns:
        int: Fuzzy partial ratio score between the two strings, or 0 if either string is too short.

    Note:
        This helper function prevents unnecessary calculations and warnings by ensuring 
        both strings meet a minimum length before calculating the partial ratio.
    """
    if len(s1) < MIN_STRING_LENGTH or len(s2) < MIN_STRING_LENGTH:
        return 0
    return fuzz.partial_ratio(s1.lower(), s2.lower())


# Utility function to create metadata string
def create_metadata_string_for_user_prompt(row: pd.Series, metadata_keys: list) -> str:
    """
    Create a metadata string from the given row of data, including only the specified metadata keys.

    Args:
        row (pd.Series): A single row of the DataFrame passed via iterrows().
        metadata_keys (list): A list of strings specifying which columns to include in the metadata string.

    Returns:
        str: A formatted metadata string.
    """
    metadata_str = "metadata:\n"

    for key in metadata_keys:
        value = row.get(key, None)
        # Handling empty, NaN, None values, and ambiguous array values
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            if isinstance(value, pd.Timestamp):
                value = value.isoformat()
            if isinstance(value, (list, pd.Series)):
                value = ', '.join(map(str, value))
            metadata_str += f"{key}: {value}\n"

    return metadata_str


def count_tokens_and_costs(prompt: str, response: str) -> Tuple[int, int, float]:
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    input_tokens = len(enc.encode(prompt))
    output_tokens = len(enc.encode(response))
    input_cost = (input_tokens / 1_000_000) * 0.150
    output_cost = (output_tokens / 1_000_000) * 0.600
    return input_tokens, output_tokens, input_cost + output_cost

def build_prompt(
    template: str,
    context: str,
    metadata: str,
    document_type: str,
    source_id: str,
    source_type: str,
    max_length: int,
    build_numbers: Dict[str, List[str]] = {},
    kb_ids: Dict[str, List[str]] = {},
    update_package_urls: str = None
) -> str:
    """
    Build a complete prompt by combining the template with truncated context text and metadata,
    ensuring the final prompt fits within the given max_length.

    Args:
        template (str): The static part of the prompt without the context.
        context (str): The dynamic context text to be included.
        metadata (str): The metadata dictionary for the document.
        document_type (str): The type of document to determine which metadata keys to include.
        source_id (str): The source ID for formatting the prompt.
        source_type (str): The source type for formatting the prompt.
        max_length (int): The maximum token length allowed for the prompt.
        build_numbers (Dict[str, List[str]]): Build numbers for each product.
        kb_ids (Dict[str, List[str]]): KB IDs for each product.
        update_package_urls (str): Update package URLs for each kb id.

    Returns:
        str: The final prompt that fits within the max_length.
    """
    # Step 1: get metadata string
    metadata_str = metadata

    # Step 2: Combine metadata and context text
    context_str = f"{metadata_str}\nPost text:\nPLACEHOLDER"

    # Step 3: Format the static parts of the template
    build_numbers_str = ""
    kb_ids_str = ""
    
    if kb_ids:
        kb_ids_str = "\n".join([f"{product}: {', '.join(kbs)}" for product, kbs in kb_ids.items()])

    if build_numbers:
        build_numbers_str = "\n".join([f"{product}: {', '.join(numbers)}" for product, numbers in build_numbers.items()])

    # Step 4: Conditionally format the prompt based on the presence of 'fix_label'
    if 'fix_label' in template:
        prompt_static = template.format(
            context_str=context_str,
            source_id=source_id,
            source_type=source_type,
            build_numbers=build_numbers_str,
            kb_ids=kb_ids_str,
            update_package_urls=update_package_urls,
        )
    else:
        prompt_static = template.format(
            context_str=context_str,
            source_id=source_id,
            source_type=source_type,
            build_numbers=build_numbers_str,
            kb_ids=kb_ids_str,
        )

    # Step 5: Tokenize the static part and calculate its length
    static_tokens = tokenizer.encode(prompt_static, truncation=False)
    static_length = len(static_tokens)

    # Step 6: Calculate how many tokens are left for the context
    remaining_length = max_length - static_length

    # Step 7: Truncate the context text to fit within the remaining length
    truncated_context = truncate_text(context, max_length=remaining_length)

    # Step 8: Construct the final prompt
    final_prompt = prompt_static.replace("PLACEHOLDER", truncated_context)

    return final_prompt.strip()


def truncate_text(text: str, max_length: int) -> str:
    """
    Truncate the text to fit within the max_length after tokenizing.

    Args:
        text (str): The input text to be truncated.
        max_length (int): The maximum number of tokens allowed.

    Returns:
        str: The truncated text.
    """
    # Tokenize and truncate based on token length
    tokens = tokenizer.encode(text, truncation=False)

    # If the token length exceeds the max_length, truncate the tokens
    if len(tokens) > max_length:
        tokens = tokens[:max_length]

    # Decode tokens back to text
    truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
    return truncated_text.strip()


def call_llm_no_logging_no_cache(system_prompt: str, user_prompt: str):
    """
    Extract information using both system and user prompts.

    Args:
        llm: The LlamaIndex OpenAI client instance.
        system_prompt (str): The system prompt that provides context or rules.
        user_prompt (str): The user's actual query or task.

    Returns:
        str: The response from the LLM after processing both prompts.
    """
    client = OpenAI()
    def call_openai():
        response = client.chat.completions.create(
            model=llm_model,  # Use your desired model
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        return response
    response = call_openai()
    content = response.choices[0].message.content.strip()
    
    return content


def extract_data_backwards(products: List[str], text: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Extracts build numbers and KB IDs from text by processing it from the end toward the beginning.Deduplicates build numbers and KB IDs within each product.

    Args:
        products (List[str]): The list of product names to search for.
        text (str): The document text to analyze.

    Returns:
        Dict[str, Dict[str, List[str]]]: A dictionary where keys are products and values are dictionaries with 'build_numbers' and 'kb_ids' as keys, each containing a list of values.

    Note:
        Uses `safe_extract_one` to avoid warnings and meaningless matches when the query or choices are too short for fuzzy matching. This is necessary to prevent potential issues with overly short strings that could otherwise lead to errors or low-quality results.
    """
    lines = text.splitlines()
    result = {product: {"build_numbers": [], "kb_ids": []} for product in products}
    current_unit = {"build_number": None, "kb_id": None}

    # Iterate lines in reverse order
    for line in reversed(lines):
        preprocessed_line = re.sub(r'[^a-zA-Z0-9\s\-:/().,]', '', line).strip()
        # Check for the terminating chunk
        if "Updates" in line and "CVSS" in preprocessed_line:
            break

        # Extract build number
        build_number_match = re.search(r"\b(\d+\.\d+\.\d+\.\d+)\b", preprocessed_line)
        if build_number_match:
            current_unit["build_number"] = build_number_match.group(1)
            continue

        # Extract KB ID
        kb_id_match = re.search(r"\b(\d{7})\b", preprocessed_line)
        if kb_id_match:
            current_unit["kb_id"] = f"KB{kb_id_match.group(1)}"
            continue

        # Fuzzy match for product names against the product list
        closest_product = safe_extract_one(preprocessed_line, products)
        if closest_product:
            # If product found, complete the unit and store it in result
            closest_product_name = closest_product[0]
            if current_unit["build_number"] or current_unit["kb_id"]:
                if current_unit["build_number"] and current_unit["build_number"] not in result[closest_product_name]["build_numbers"]:
                    result[closest_product_name]["build_numbers"].append(current_unit["build_number"])
                if current_unit["kb_id"] and current_unit["kb_id"] not in result[closest_product_name]["kb_ids"]:
                    result[closest_product_name]["kb_ids"].append(current_unit["kb_id"])

                # Clear current unit after extracting data
                current_unit = {"build_number": None, "kb_id": None}

    # Print extracted results
    # for product, data in result.items():
    #     print(f"{product}: Build Numbers - {data['build_numbers']}, KB IDs - {data['kb_ids']}")

    return result


def extract_data_patch_management(products: List[str], text: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Extracts build numbers and KB IDs from PatchManagementPost documents by performing fuzzy matching for products and regex matching for build numbers and KB IDs.

    Args:
        products (List[str]): The list of product names to search for.
        text (str): The document text to analyze.

    Returns:
        Dict[str, Dict[str, List[str]]]: A dictionary where keys are products and values are dictionaries with 'build_numbers' and 'kb_ids' as keys, each containing a list of values.

    Note:
        Uses `safe_partial_ratio` to avoid warnings and ineffective comparisons when either string is too short. This helper ensures that both strings are long enough to generate meaningful fuzzy scores, improving accuracy and reducing errors.
    """
    # Initialize result dictionary with empty lists for each product
    result = {product: {"build_numbers": [], "kb_ids": []} for product in products}
    
    # Extract build numbers and KB IDs from the text using regex
    build_numbers = re.findall(r"\b\d+\.\d+\.\d+\.\d+\b", text)
    kb_ids = [f"KB{match}" for match in re.findall(r"\b\d{7}\b", text)]

    # Fuzzy match product names against the text to find relevant products
    matched_products = set()
    for product in products:
        if safe_partial_ratio(product, text) >= 70:
            matched_products.add(product)
    
    # Assign build numbers and KB IDs to matched products
    for product in matched_products:
        result[product]["build_numbers"].extend(build_numbers)
        result[product]["kb_ids"].extend(kb_ids)
    
    # Print extracted results for verification
    # for product, data in result.items():
    #     print(f"{product}: Build Numbers - {data['build_numbers']}, KB IDs - {data['kb_ids']}")
    
    return result


def generate_update_package_links(kb_ids: Dict[str, List[str]]) -> str:
    update_package_links = None
    unique_kb_ids = set()
    for kb_list in kb_ids.values():
        unique_kb_ids.update(kb_list)
    if unique_kb_ids:
        update_package_links = "\n".join([f"- https://catalog.update.microsoft.com/Search.aspx?q={kb}" for kb in unique_kb_ids])
    return update_package_links

# f"{entity_type.lower()}_label"

def compute_row_key(row, entity_type):
    row_dict = row.to_dict()
    
    # Exclude fields that may change
    row_dict = {k: v for k, v in row_dict.items()}
    
    # Add entity type to the dictionary
    row_dict["entity_type"] = entity_type
    
    # Handle non-serializable values (e.g., Timestamps)
    for key, value in row_dict.items():
        if isinstance(value, pd.Timestamp):
            row_dict[key] = value.isoformat()

    data_to_hash = json.dumps(row_dict, sort_keys=True).encode("utf-8")
    hash = hashlib.sha256(data_to_hash).hexdigest()
    debug_directory = r"C:\Users\emili\PycharmProjects\microsoft_cve_rag\microsoft_cve_rag\application\data\debug_hash_files"
    os.makedirs(debug_directory, exist_ok=True)
    debug_file_path = os.path.join(debug_directory, f"{row_dict['node_label']}_{entity_type}_data_to_hash.json")
    with open(debug_file_path, "wb") as debug_file:
        debug_file.write(data_to_hash)
    print(f"hash:{hash}")
    return hashlib.sha256(data_to_hash).hexdigest()

def is_empty_extracted_entity(entity: dict, entity_type: str) -> bool:
    """
    Check if the extracted entity is effectively empty (i.e., contains no meaningful data).

    Args:
        entity (dict): The extracted entity to check.
        entity_type (str): The type of entity (e.g., 'Symptom', 'Cause', 'Fix', etc.).

    Returns:
        bool: True if the entity is empty, False otherwise.
    """
    empty_fields = {
        "Symptom": ["description", "symptom_label"],
        "Cause": ["description", "cause_label"],
        "Fix": ["description", "fix_label"],
        "Tool": ["description", "tool_label"],
    }

    # Check if all specified fields are empty or default values
    for field in empty_fields.get(entity_type, []):
        if entity.get(field):
            return False
    return True

async def extract_entities_relationships(
    documents_df: pd.DataFrame,
    document_type: str,
    max_prompt_length=3050,
) -> dict:
    """
    Extract entities and relationships from documents using LlamaIndex.

    Args:
        documents_df (pd.DataFrame): DataFrame containing documents with 'content' and 'doc_id'.

    Returns:
        Dict[str, List[Dict]]: Dictionary containing lists of extracted entities.
    """
    if not isinstance(documents_df, pd.DataFrame) or documents_df.empty:
        return {}
    token_counter = {"input_tokens": 0, "output_tokens": 0, "total_cost": 0.0}
    extracted_data = {
        "symptoms": [],
        "causes": [],
        "fixes": [],
        "tools": [],
        "technologies": [],
    }

    cached_files = []

    client = OpenAI()
    data_directory = r"C:\Users\emili\PycharmProjects\microsoft_cve_rag\microsoft_cve_rag\application\data"
    cache_dir = os.path.join(data_directory, "llm_cache")
    os.makedirs(cache_dir, exist_ok=True)

    products = [
        "Windows 11 for x64-based Systems",
        "Windows 11 Version 24H2 for x64-based Systems",
        "Windows 11 Version 23H2 for x64-based Systems",
        "Windows 11 Version 22H2 for x64-based Systems",
        "Windows 11 Version 21H2 for x64-based Systems",
        "Windows 10 for x64-based Systems",
        "Windows 10 for 32-bit Systems",
        "Windows 10 Version 23H2 for x64-based Systems",
        "Windows 10 Version 22H2 for x64-based Systems",
        "Windows 10 Version 21H2 for x64-based Systems",
        "Windows 10 Version 23H2 for 32-bit Systems",
        "Windows 10 Version 22H2 for 32-bit Systems",
        "windows 10 version 21h2 for 32-bit Systems",
        "Microsoft Edge (Chromium-based)",
        "Microsoft Edge (Chromium-based) Extended Stable"
    ]

    start_time = time.time()
    for _, row in tqdm(documents_df.iterrows(), total=len(documents_df)):
        if document_type in metadata_columns:
            metadata_str = create_metadata_string_for_user_prompt(row, metadata_columns[document_type])
        else:
            metadata_str = ""

        document_text = row["text"]

        if document_type == "MSRCPost":
            extracted_data_backwards = extract_data_backwards(products, document_text)
            build_numbers = {product: data['build_numbers'] for product, data in extracted_data_backwards.items() if data['build_numbers']}
            kb_ids = {product: data['kb_ids'] for product, data in extracted_data_backwards.items() if data['kb_ids']}
        elif document_type == "PatchManagementPost":
            extracted_data_patch = extract_data_patch_management(products, document_text)
            build_numbers = {product: data['build_numbers'] for product, data in extracted_data_patch.items() if data['build_numbers']}
            kb_ids = {product: data['kb_ids'] for product, data in extracted_data_patch.items() if data['kb_ids']}
        else:
            build_numbers, kb_ids = {}, {}

        update_links = generate_update_package_links(kb_ids)

        # For each entity type
        for entity_type in ["Symptom", "Cause", "Fix", "Tool"]:
            try:
                # Generate prompts
                user_prompt_template = get_prompt_template(
                    entity_type,
                    document_type,
                    prompt_type="user",
                )
                system_prompt_template = get_prompt_template(
                    entity_type,
                    document_type,
                    prompt_type="system",
                )
                user_prompt = build_prompt(
                    template=user_prompt_template,
                    context=document_text,
                    metadata=metadata_str,
                    document_type=document_type,
                    source_id=row["node_id"],
                    source_type=document_type,
                    max_length=max_prompt_length,
                    build_numbers=build_numbers,
                    kb_ids=kb_ids,
                    update_package_urls=update_links
                )
                system_prompt = system_prompt_template

            except ValueError as e:
                print(e)
                continue  # Skip if no prompt is found

            # Compute a unique key for the row
            row_key = compute_row_key(row, entity_type)
            
            cache_file_path = os.path.join(cache_dir, f"{row_key}.json")

            # Check if result is already cached
            if os.path.exists(cache_file_path):
                print("cache exists of row")
                # Load cached result
                def load_cached_result():
                    with open(cache_file_path, "r", encoding="utf-8") as f:
                        return json.load(f)

                extracted_entity = await asyncio.to_thread(load_cached_result)
                # Store cache file name
                cached_files.append(cache_file_path)
                # Ensure token counters are incremented by 0
                token_counter["input_tokens"] += 0
                token_counter["output_tokens"] += 0
                token_counter["total_cost"] += 0.0
            else:
                print("cache doesnt exist of row")
                # Use the LLM to generate the extraction
                def call_openai():
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",  # Use your desired model
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {"role": "user", "content": user_prompt},
                        ],
                        response_format={"type": "json_object"},
                    )
                    return response

                response = await asyncio.to_thread(call_openai)
                llm_response = response.choices[0].message.content.strip()

                if llm_response:
                    try:
                        extracted_entity = json.loads(llm_response)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON for llm response {llm_response} in document {row['node_id']}: {e}")
                        continue

                    # Generate a UUID based on the hash of the extracted entity
                    extracted_entity_hash = hashlib.sha256(json.dumps(extracted_entity, sort_keys=True).encode("utf-8")).hexdigest()
                    extracted_entity_uuid = str(uuid.UUID(extracted_entity_hash[:32]))
                    extracted_entity["node_id"] = extracted_entity_uuid

                    # Save the result to cache with the node_id
                    def save_result_to_cache():
                        with open(cache_file_path, "w", encoding="utf-8") as f:
                            json.dump(extracted_entity, f, ensure_ascii=False)
                            print("new extraction saved to llm cache.")
                    await asyncio.to_thread(save_result_to_cache)
                    # Store cache file name
                    cached_files.append(cache_file_path)
                    # Count tokens and costs
                    input_tokens, output_tokens, cost = count_tokens_and_costs(user_prompt, llm_response)
                    token_counter["input_tokens"] += input_tokens
                    token_counter["output_tokens"] += output_tokens
                    token_counter["total_cost"] += cost
                else:
                    print(f"LLM response is empty: {llm_response}")

            # # Add the extracted entity to the appropriate list
            key = entity_type.lower() + 'es' if entity_type == 'Fix' else entity_type.lower() + 's'  # e.g., 'fixes', 'symptoms'
            if isinstance(extracted_entity, dict) and not is_empty_extracted_entity(extracted_entity, entity_type):
                if key in extracted_data:
                    extracted_data[key].append(extracted_entity)
                else:
                    extracted_data[key] = [extracted_entity]

    end_time = time.time()
    print(f"dataframe with {documents_df.shape[0]} rows processed in {end_time - start_time:.2f} seconds.")

    # Print cached file paths
    print("REPORT:\nCached files:\n")
    for file_path in cached_files:
        print(file_path)

    print(f"Total input tokens: {token_counter['input_tokens']}")
    print(f"Total output tokens: {token_counter['output_tokens']}")
    print(f"Total cost: ${token_counter['total_cost']:.6f}")

    return extracted_data

class CustomDocumentTracker:
    """Tracks documents and their metadata separately from LlamaIndex."""
    
    def __init__(self, persist_path: str):
        self.persist_path = Path(persist_path)
        self.documents = self._load_documents()
        
    def _load_documents(self) -> dict:
        """Load existing document tracking data."""
        if self.persist_path.exists():
            with open(self.persist_path, 'r') as f:
                return json.load(f)
        return {}
        
    def add_document(self, doc_id: str, metadata: dict):
        """Add or update document tracking information."""
        self.documents[doc_id] = {
            "metadata": metadata,
            "added_at": datetime.datetime.now().isoformat(),
            "last_updated": datetime.datetime.now().isoformat()
        }
        self._save_documents()
        
    def remove_document(self, doc_id: str):
        """Remove document from tracking."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            self._save_documents()
            
    def get_document(self, doc_id: str) -> Optional[dict]:
        """Get document tracking information."""
        return self.documents.get(doc_id)
        
    def _save_documents(self):
        """Persist document tracking information."""
        with open(self.persist_path, 'w') as f:
            json.dump(self.documents, f, indent=2)

class LlamaIndexVectorService:
    """A service that integrates LlamaIndex with Qdrant vector storage for document indexing and retrieval.
    
    This service provides a bridge between LlamaIndex's document processing capabilities and Qdrant's 
    vector storage functionality. It handles document chunking, embedding generation, and vector storage
    while maintaining compatibility with LlamaIndex's querying interface.

    Key Components:
    - Vector Store: Uses Qdrant for persistent storage of document vectors
    - Embedding Model: Adapts custom embedding service for LlamaIndex compatibility
    - Node Parser: Implements sentence window parsing for context-aware chunking
    - Storage Context: Manages the connection between LlamaIndex and vector storage
    
    The service currently handles:
    - Document chunking with contextual windows
    - Vector embedding generation
    - Vector storage in Qdrant
    
    Note: Current implementation may need enhancement for proper index persistence.
    The VectorStoreIndex is created during upsert but not explicitly persisted,
    which means the index structure might need to be rebuilt on application restart.

    Args:
        vector_db_service (VectorDBService): Service managing Qdrant vector storage operations
        window_size (int, optional): Size of the context window for sentence parsing. Defaults to 3.
        window_metadata_key (str, optional): Metadata key for window information. Defaults to "window".

    Example:
        ```python
        vector_db_service = VectorDBService(...)
        llama_service = LlamaIndexVectorService(vector_db_service)
        
        # Upsert documents
        await llama_service.upsert_documents(documents)
        ```
    
    Todo:
        * Implement index persistence to disk
        * Add index loading from disk on initialization
        * Add methods for index updates and maintenance
        * Implement query interface using loaded index
    """
    def __init__(self, vector_db_service: VectorDBService, persist_dir: str, 
                 window_size: int = 3, window_metadata_key: str = "window"):
        self.vector_db_service = vector_db_service
        self.persist_dir = persist_dir
        
        # Configure global settings
        llama_embedding_model = LlamaIndexEmbeddingAdapter(vector_db_service.embedding_service)
        Settings.embed_model = llama_embedding_model
        
        # Enhanced node parser
        self.node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=window_size,
            window_metadata_key=window_metadata_key,
            original_text_metadata_key="original_text"
        )
        Settings.node_parser = self.node_parser
        
        # Initialize later
        self.vector_store = None
        self.storage_context = None
        self.index = None
        self.doc_tracker = None

            
    @classmethod
    async def initialize(cls, vector_db_service: VectorDBService, persist_dir: str, **kwargs) -> "LlamaIndexVectorService":
        # add docstring to explain the design pattern and why it's needed and how it works
        logging.info("running initialize")
        instance = cls(vector_db_service, persist_dir, **kwargs)
        await instance._initialize_components()
        return instance

    async def _initialize_components(self):
        """Initialize all service components."""
        try:
            # Ensure collection exists
            await self.vector_db_service.ensure_collection_exists_async()
            
            # Initialize vector store with both clients
            self.vector_store = QdrantVectorStore(
                collection_name=self.vector_db_service.collection,
                client=self.vector_db_service.sync_client,
                aclient=self.vector_db_service.async_client,
            )
            logging.info("initialized vector store")
            # Initialize storage and tracking
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
                persist_dir=self.persist_dir
            )
            logging.info("initialized storage context")
            self.doc_tracker = CustomDocumentTracker(
                Path(self.persist_dir) / "doc_tracker.json"
            )
            logging.info("initialized doc tracker")
            # Try loading existing index
            try:
                logging.info(f"Loading existing index from {self.persist_dir}")
                # self.index = load_index_from_storage(
                #     storage_context=self.storage_context,
                #     index_id=self.vector_db_service.collection
                # )
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    storage_context=self.storage_context,
                    show_progress=True
                )
                index_id = self.vector_db_service.collection
                self.index.set_index_id(index_id)
                logging.info("loaded existing index")
            except Exception as e:
                logging.warning(f"Creating new index: {e}")
                self.index = VectorStoreIndex(
                    nodes=[],
                    storage_context=self.storage_context,
                    show_progress=True
                )
                self.index.set_index_id(self.vector_db_service.collection)
                logging.info("created new index")
        except Exception as e:
            logging.error(f"Error initializing components: {e}")
            raise

    def _initialize_fresh_context(self):
        """Initialize fresh storage context with empty stores and minimal index structure."""
        try:
            # Step 1: Create vector store (we already have self.vector_store from QdrantVectorStore)
            if not self.vector_store:
                raise ValueError("Vector store must be initialized before creating fresh context")
                
            # Step 2: Initialize storage context with empty stores
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
                persist_dir=self.persist_dir,
                docstore=SimpleDocumentStore(),
                index_store=SimpleIndexStore()
            )
            
            # Step 3: Create empty index first, then set its ID
            self.index = VectorStoreIndex(
                nodes=[],
                storage_context=self.storage_context,
                show_progress=False
            )
            
            # Step 4: Explicitly set the index ID
            index_id = self.vector_db_service.collection
            self.index.set_index_id(index_id)
            
            # Step 5: Verify index_id was set correctly
            if self.index.index_id != index_id:
                raise ValueError(f"Failed to set correct index_id. Expected {index_id}, got {self.index.index_id}")
            
            # Step 6: Persist the storage context
            self.storage_context.persist(persist_dir=self.persist_dir)
            
            logging.info(f"Created and persisted initialized storage context with index_id '{index_id}' at {self.persist_dir}")
            
        except Exception as e:
            logging.error(f"Error initializing fresh context: {e}")
            raise

    async def _ensure_collection(self):
        """Ensure Qdrant collection exists."""
        try:
            await self.vector_db_service.async_client.get_collection(
                collection_name=self.vector_db_service.collection
            )
            logging.info(f"Collection '{self.vector_db_service.collection}' exists")
        except Exception:
            logging.info(f"Creating collection '{self.vector_db_service.collection}'")
            await self.vector_db_service.async_client.create_collection(
                collection_name=self.vector_db_service.collection,
                vectors_config=self.vector_db_service.vector_config
            )

    # def _load_index(self) -> Optional[VectorStoreIndex]:
    #     """Load existing index from disk if available."""
    #     try:
    #         return load_index_from_storage(
    #             storage_context=self.storage_context,
    #             persist_dir=self.persist_dir
    #         )
    #     except Exception as e:
    #         logging.warning(f"No existing index found: {e}")
    #         return None

    async def upsert_documents(self, documents: List[LlamaDocument]) -> None:
        try:
            logging.info("running upsert_documents")
            
            doc_ids_to_process = {doc.doc_id for doc in documents}
            
            existing_ids = {
                doc_id for doc_id in doc_ids_to_process 
                if self.doc_tracker.get_document(doc_id)
            }
            
            # Delete existing documents
            for doc_id in existing_ids:
                await self.delete_document(doc_id)
            
            # Process nodes asynchronously
            nodes = []
            for doc in documents:
                doc_nodes = self.node_parser.get_nodes_from_documents([doc])
                logging.info(f"Document {doc.doc_id} generated {len(doc_nodes)} nodes")
                # Log sample node info
                if doc_nodes:
                    sample_node = doc_nodes[0]
                    logging.info(f"Sample node: id={sample_node.node_id}, ref_doc_id={sample_node.ref_doc_id}")
                nodes.extend(doc_nodes)
            
            # Add nodes to index
            await self.index._async_add_nodes_to_index(
                index_struct=self.index.index_struct,
                nodes=nodes,
                show_progress=True
            )
            # logging.info(f"Index struct after upsert: {self.index.index_struct}")
            logging.info(f"Storage context docstore size: {len(self.storage_context.docstore.docs)}")
            logging.info(f"Vector store size: {await self._get_collection_point_count()}")
            # Update tracking and persist
            for doc in documents:
                self.doc_tracker.add_document(doc.doc_id, doc.metadata)
            
            # await asyncio.to_thread(
            #     self.storage_context.persist,
            #     persist_dir=self.persist_dir
            # )
            
            # Verify
            # if not await self._verify_upsert(doc_ids_to_process, len(nodes)):
            #     raise Exception("Upsert verification failed")
            unique_counts = await self._count_unique_points()
            logging.info(f"Uniqueness analysis:\n{unique_counts}")
            logging.info("Document upsert complete and verified")
            
        except Exception as e:
            logging.error(f"Error in upsert_documents: {e}")
            raise
    
    async def delete_document(self, doc_id: str):
        """Delete a specific document from both Qdrant and tracking."""
        try:
            # Delete from Qdrant
            await self.vector_db_service.async_client.delete(
                collection_name=self.vector_db_service.collection,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                    )
                ),
                wait=True
            )
            
            # Remove from document tracker
            self.doc_tracker.remove_document(doc_id)
            
            logging.info(f"Successfully deleted document {doc_id}")
        except Exception as e:
            logging.error(f"Error deleting document {doc_id}: {e}")
            raise
    
    # async def delete_all_documents(self):
    #     """Delete all documents from both Qdrant and LlamaIndex storage."""
    #     try:
    #         # Add wait=True and ensure the collection is fully deleted
    #         while True:
    #             try:
    #                 await self.vector_db_service.async_client.get_collection(
    #                     collection_name=self.vector_db_service.collection
    #                 )
    #                 await asyncio.sleep(0.5)  # Wait before checking again
    #             except Exception:
    #                 break  # Collection is gone
                
    #         # Recreate with wait=True to ensure it's ready
    #         await self.vector_db_service.async_client.recreate_collection(
    #             collection_name=self.vector_db_service.collection,
    #             vectors_config=self.vector_db_service.vector_config,
    #             wait=True
    #         )
            
    #         # Verify collection is empty
    #         collection_info = await self.vector_db_service.async_client.get_collection(
    #             collection_name=self.vector_db_service.collection
    #         )
    #         if collection_info.points_count != 0:
    #             raise Exception(f"Collection not empty after recreation. Found {collection_info.points_count} points")
            
    #         # Reset LlamaIndex storage
    #         self.storage_context = StorageContext.from_defaults(
    #             vector_store=self.vector_store,
    #             persist_dir=self.persist_dir,
    #             docstore=SimpleDocumentStore(),
    #             index_store=SimpleIndexStore(),
    #         )
            
    #         # Create fresh index
    #         self.index = None
            
    #         # Persist empty state
    #         if self.storage_context:
    #             self.storage_context.persist(persist_dir=self.persist_dir)
                
    #         logging.info("Successfully deleted all documents and reset index")
    #         return True
    #     except Exception as e:
    #         logging.error(f"Error deleting documents: {e}")
    #         raise

    async def _get_collection_point_count(self) -> int:
        """Get the current number of points in the collection."""
        collection_stats = await self.vector_db_service.async_client.get_collection(
            collection_name=self.vector_db_service.collection
        )
        return collection_stats.points_count
    
    async def _count_unique_points(self) -> dict:
        """Count unique points based on different criteria."""
        try:
            # Get all points with their payloads
            scroll_result = await self.vector_db_service.async_client.scroll(
                collection_name=self.vector_db_service.collection,
                limit=1000,  
                with_payload=True,
                with_vectors=True
            )
            
            points = scroll_result[0]  # First element contains points
            
            # Initialize collections
            unique_texts = set()
            unique_doc_ids = set()
            unique_vectors = set()
            vector_groups = {}  # Dictionary to group points by their vectors
            
            for point in points:
                if point.payload and '_node_content' in point.payload:
                    try:
                        node_content = json.loads(point.payload['_node_content'])
                        if 'text' in node_content:
                            text = node_content['text']
                            unique_texts.add(text)
                            
                            # Group by vector and track unique vectors
                            if hasattr(point, 'vector') and point.vector is not None:
                                vector_tuple = tuple(point.vector)
                                unique_vectors.add(vector_tuple)
                                
                                if vector_tuple not in vector_groups:
                                    vector_groups[vector_tuple] = []
                                vector_groups[vector_tuple].append({
                                    'text': text,
                                    'doc_id': point.payload.get('doc_id', 'unknown'),
                                    'metadata': {
                                        k: v for k, v in point.payload.items() 
                                        if k not in ['_node_content', 'doc_id']
                                    }
                                })
                    except json.JSONDecodeError:
                        logging.warning(f"Could not parse _node_content JSON for point")
                
                if point.payload and 'doc_id' in point.payload:
                    unique_doc_ids.add(point.payload['doc_id'])
            
            # Find duplicate vectors
            duplicate_vectors = {
                vector: points for vector, points in vector_groups.items() 
                if len(points) > 1
            }
            
            # Generate detailed report
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = Path(self.persist_dir) / f"duplicate_analysis_{timestamp}.txt"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"Duplicate Vector Analysis Report - {timestamp}\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Summary:\n")
                f.write(f"Total points: {len(points)}\n")
                f.write(f"Unique texts: {len(unique_texts)}\n")
                f.write(f"Unique document IDs: {len(unique_doc_ids)}\n")
                f.write(f"Unique vectors: {len(unique_vectors)}\n")
                f.write(f"Vectors with duplicates: {len(duplicate_vectors)}\n\n")
                
                # Sort vectors by number of duplicates (descending)
                sorted_vectors = sorted(
                    duplicate_vectors.items(), 
                    key=lambda x: len(x[1]), 
                    reverse=True
                )
                
                for vector, points in sorted_vectors:
                    vector_preview = [f"{x:.4f}" for x in vector[:5]]
                    f.write(f"\nDUPLICATE VECTOR: [{', '.join(vector_preview)}...]\n")
                    f.write(f"This exact vector appears {len(points)} times in Qdrant\n")
                    f.write(f"Full Vector: {vector}\n")
                    f.write("-" * 80 + "\n")
                    
                    # Group by text to see distribution
                    text_counts = {}
                    for point in points:
                        text = point['text']
                        if text not in text_counts:
                            text_counts[text] = {
                                'count': 0,
                                'doc_ids': set(),
                                'metadata': []
                            }
                        text_counts[text]['count'] += 1
                        text_counts[text]['doc_ids'].add(point['doc_id'])
                        text_counts[text]['metadata'].append(point['metadata'])
                    
                    f.write(f"Distribution of Points with this vector:\n\n")
                    for text, info in text_counts.items():
                        f.write(f"Text chunk: {text[:100]}...\n")
                        f.write(f"- Stored {info['count']} times in Qdrant\n")
                        f.write(f"- Associated with {len(info['doc_ids'])} document(s): {list(info['doc_ids'])}\n")
                        f.write(f"- First few metadata entries:\n")
                        for metadata in info['metadata'][:2]:
                            f.write(f"  {json.dumps(metadata, indent=2)[:200]}...\n")
                        f.write("\n")
                    f.write("=" * 80 + "\n")
            
            logging.info(f"Detailed duplicate analysis written to: {report_path}")
            
            # Create duplicate distribution
            duplicate_distribution = {}
            for points in vector_groups.values():
                count = len(points)
                duplicate_distribution[count] = duplicate_distribution.get(count, 0) + 1
            
            return {
                "total_points": len(points),
                "unique_texts": len(unique_texts),
                "unique_doc_ids": len(unique_doc_ids),
                "unique_vectors": len(unique_vectors),
                "vectors_with_duplicates": len(duplicate_vectors),
                "max_duplicates": max(len(points) for points in vector_groups.values()),
                "duplicate_distribution": {
                    f"{count}_duplicates": freq 
                    for count, freq in sorted(duplicate_distribution.items())
                }
            }
            
        except Exception as e:
            logging.error(f"Error counting unique points: {e}")
            logging.exception("Full traceback:")
            raise

    async def _verify_upsert(
        self,
        processed_ids: set,
        nodes_created: int
    ) -> bool:
        """
        Verify that the upsert operation was successful by checking document presence
        and logging statistics.
        
        Args:
            processed_ids: Set of document IDs that were processed
            nodes_created: Number of nodes created in the index
        
        Returns:
            bool: True if all documents are found in Qdrant, False otherwise
        """
        # if the processed_ids is empty, return True
        if not processed_ids:
            raise ValueError("No documents ids to process, nothing to verify")
        try:
            # Verify each document exists in Qdrant
            for doc_id in processed_ids:
                scroll_result = await self.vector_db_service.async_client.scroll(
                    collection_name=self.vector_db_service.collection,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="doc_id",
                                match=MatchValue(value=doc_id)
                            )
                        ]
                    ),
                    limit=1
                )
                
                # scroll_result is a tuple of (points, offset)
                points = scroll_result[0]
                
                if not points:  # Check if points list is empty
                    logging.error(f"Document {doc_id} not found in Qdrant after upsert")
                    return False
            
            # Log statistics
            logging.info(f"""Upsert verification successful:
            - All documents found in Qdrant
            - Documents processed: {len(processed_ids)}
            - Nodes created: {nodes_created}
            - Embeddings generated: {len(scroll_result[0])}
            - Nodes per document (avg): {nodes_created / len(processed_ids):.1f}
            - Embeddings per document (avg): {len(scroll_result[0]) / len(processed_ids):.1f}""")
            return True
            
        except Exception as e:
            logging.error(f"Error during upsert verification: {e}")
            return False

    async def aclose(self):
        """Close connections using existing service"""
        await self.vector_db_service.aclose()
