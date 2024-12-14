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
from llama_index.core.schema import Document as LlamaDocument, NodeRelationship, RelatedNodeInfo
from llama_index.core.llms import ChatMessage
# from llama_index.llms.openai import OpenAI
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.vector_stores.types import BasePydanticVectorStore
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
    MatchAny,
    FilterSelector,
)
from typing import Dict, List, Literal, Tuple, Optional, Any
import pandas as pd
import numpy as np
import uuid
import json
import shutil
import tempfile
import re
import time
import datetime
import tiktoken
import asyncio
from transformers import AutoTokenizer
import hashlib
from fuzzywuzzy import fuzz, process
from tqdm import tqdm
import warnings
from dataclasses import dataclass
from application.app_utils import (
    get_app_config,
    get_graph_db_credentials,
    get_openai_api_key,
)
from openai import OpenAI, AsyncOpenAI
import logging
# import logging.config

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


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    input_cost_per_million: float
    output_cost_per_million: float

    def to_dict(self) -> Dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
            "input_cost_per_million": self.input_cost_per_million,
            "output_cost_per_million": self.output_cost_per_million
        }


class TokenCounter:
    def __init__(self, model_name: str = "gpt-4o-mini",
                 input_cost_per_million: float = 0.150,
                 output_cost_per_million: float = 0.600):
        self.model_name = model_name
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.encoder = tiktoken.encoding_for_model(model_name)

    def _calculate_cost(self, num_tokens: int, cost_per_million: float) -> float:
        """Internal method to calculate cost for a given number of tokens."""
        return (num_tokens / 1_000_000) * cost_per_million

    def from_token_counts(self, input_tokens: int, output_tokens: int) -> TokenUsage:
        """Create TokenUsage from raw token counts."""
        input_cost = self._calculate_cost(input_tokens, self.input_cost_per_million)
        output_cost = self._calculate_cost(output_tokens, self.output_cost_per_million)

        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
            input_cost_per_million=self.input_cost_per_million,
            output_cost_per_million=self.output_cost_per_million
        )

    def from_prompt_response(self, prompt: str, response: str) -> TokenUsage:
        """Create TokenUsage from prompt and response strings."""
        input_tokens = len(self.encoder.encode(prompt))
        output_tokens = len(self.encoder.encode(response))
        return self.from_token_counts(input_tokens, output_tokens)

    def empty_usage(self) -> TokenUsage:
        """Create an empty TokenUsage with zero tokens and costs."""
        return self.from_token_counts(0, 0)


# Initialize TokenCounter at the top level with default GPT-4 rates
token_counter_instance = TokenCounter(
    model_name="gpt-4o-mini",
    input_cost_per_million=0.150,
    output_cost_per_million=0.600
)


def load_llm_usage_data(filename):
    """
    Safely load LLM usage data from a JSON file.
    If the file is malformed, attempt to recover by truncating to the last complete JSON object.
    """
    if not os.path.exists(filename):
        logging.info(f"File {filename} does not exist. Initializing with an empty list.")
        return []

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.warning(f"JSONDecodeError: {e}. Attempting to recover {filename}.")
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = f.read()

            # Find the position of the last closing brace
            last_brace = data.rfind('}')
            if last_brace == -1:
                logging.error(f"No closing brace found in {filename}. Resetting to an empty list.")
                return []

            # Truncate the data to the last complete JSON object and close the array
            recovered_data = data[:last_brace+1] + ']'

            # Ensure the data starts with '[' to form a valid JSON array
            if not recovered_data.strip().startswith('['):
                logging.error(f"Recovered data does not start with '['. Resetting to an empty list.")
                return []

            # Attempt to parse the recovered JSON
            usage_data = json.loads(recovered_data)

            # Backup the corrupted file before overwriting
            backup_filename = f"{filename}.backup.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            shutil.copy(filename, backup_filename)
            logging.info(f"Backed up corrupted file to {backup_filename}.")

            return usage_data

        except Exception as recovery_error:
            logging.error(f"Failed to recover {filename}: {recovery_error}")
            return []


def update_llm_usage(usage):
    """
    Update the LLM usage tracking file with new token and cost information.
    Handles potential JSON corruption by attempting to recover the file.
    """
    usage_dir = r"C:\Users\emili\PycharmProjects\microsoft_cve_rag\microsoft_cve_rag\application\data\llm_usage"
    usage_file = os.path.join(usage_dir, "llm_usage.json")

    # Create directory if it doesn't exist
    os.makedirs(usage_dir, exist_ok=True)

    # Get current date info
    now = datetime.datetime.now()
    current_date = now.strftime("%Y-%m-%d %H:%M:%S")
    current_month = now.strftime("%Y-%m")

    # Safely load existing data
    usage_data = load_llm_usage_data(usage_file)

    # Create new usage record
    usage_record = {
        "full_date": current_date,
        "month": current_month,
        **usage.to_dict()  # Unpack all token and cost information
    }

    # Append the new record
    usage_data.append(usage_record)

    # Write updated data atomically
    try:
        # Use a temporary file to ensure atomicity
        with tempfile.NamedTemporaryFile('w', delete=False, dir=usage_dir, suffix='.tmp', encoding='utf-8') as tmp_file:
            json.dump(usage_data, tmp_file, indent=2)
            temp_name = tmp_file.name

        # Replace the original file with the temporary file
        shutil.move(temp_name, usage_file)
        logging.debug(f"Successfully updated {usage_file} with new usage record.")

    except Exception as e:
        logging.error(f"Failed to write to {usage_file}: {e}")
        # Clean up the temporary file if it exists
        if 'temp_name' in locals() and os.path.exists(temp_name):
            os.remove(temp_name)
        # Depending on requirements, you might want to re-raise the exception
        raise


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
    update_package_urls: str = None,
    post_type: str = ""
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
        post_type (str): The type of post, one of ["Information only", "Critical", "Problem statement", "Helpful Tool", "Solution provided"].

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


async def call_llm_no_logging_no_cache(system_prompt: str, user_prompt: str):
    """
    Extract information using both system and user prompts.

    Args:
        system_prompt (str): The system prompt to use
        user_prompt (str): The user prompt to use

    Returns:
        str: The response content
    """
    client = AsyncOpenAI()

    try:
        response = await client.chat.completions.create(
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
        return response.choices[0].message.content
    except Exception as e:
        if "insufficient credits" in str(e).lower():
            logging.error("Tried to process API call, insufficient credits available.")
        else:
            logging.error(f"Exception: {e}")
        raise


def extract_data_backwards(products: List[str], text: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Extracts build numbers and KB IDs from text by processing it from the end toward the beginning.Deduplicates build numbers and KB IDs within each product.

    Args:
        products (List[str]): The list of product names to search for.
        text (str): The document text to analyze.

    Returns:
        Dict[str, Dict[str, List[str]]]: A dictionary where keys are products and values are dictionaries with 'build_numbers' and 'kb_ids' as keys, each containing a list of values.

    Note:
        Uses `safe_extract_one` to avoid warnings and meaningless matches when the query or choices are too short. This is necessary to prevent potential issues with overly short strings that could otherwise lead to errors or low-quality results.
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


def get_entities_to_extract(post_type: str) -> list:
    """
    Determine which entities to extract based on post type.

    Args:
        post_type (str): One of ["Information only", "Critical", "Problem statement",
                        "Helpful Tool", "Solution provided"]

    Returns:
        list: List of entity types to extract
    """
    valid_post_types = {
        "Information only": [],
        "Conversational": [],
        "Problem statement": ["Symptom", "Tool"],
        "Helpful Tool": ["Tool"],
        "Helpful tool": ["Tool"],
        "Solution provided": ["Symptom", "Cause", "Fix", "Tool"]
    }

    if post_type not in valid_post_types:
        logging.warning(f"Unexpected post_type: {post_type}. Using default extraction.")
        return ["Symptom", "Cause", "Fix", "Tool"]

    return valid_post_types[post_type]


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
            post_type = row["post_type"]
        elif document_type == "PatchManagementPost":
            extracted_data_patch = extract_data_patch_management(products, document_text)
            build_numbers = {product: data['build_numbers'] for product, data in extracted_data_patch.items() if data['build_numbers']}
            kb_ids = {product: data['kb_ids'] for product, data in extracted_data_patch.items() if data['kb_ids']}
            post_type = row["post_type"]
        else:
            build_numbers, kb_ids = {}, {}
            post_type = ""

        update_links = generate_update_package_links(kb_ids)

        # Get the entities to extract based on post_type
        entities_to_extract = get_entities_to_extract(post_type)
        if not entities_to_extract:
            logging.info(f"Skipping extraction for document {row['node_id']} with post_type: {post_type}")
            continue
        # For each entity type
        for entity_type in entities_to_extract:
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
                    update_package_urls=update_links,
                    post_type=post_type
                )
                system_prompt = system_prompt_template

            except ValueError as e:
                logging.error(f"ValueError: {e}")
                continue  # Skip if no prompt is found

            # Compute cache file name using entity_type and source_id
            cache_file_name = f"{entity_type.lower()}_{row['node_id']}.json"
            cache_file_path = os.path.join(cache_dir, cache_file_name)

            # Check if result is already cached
            if os.path.exists(cache_file_path):
                try:
                    with open(cache_file_path, "r", encoding="utf-8") as f:
                        extracted_entity = json.load(f)
                    cached_files.append(cache_file_path)
                    # No tokens used when loading from cache
                    token_counter["input_tokens"] += 0
                    token_counter["output_tokens"] += 0
                    token_counter["total_cost"] += 0.0
                    update_llm_usage(token_counter_instance.empty_usage())

                    # Add to appropriate list and continue to next iteration
                    key = entity_type.lower() + 'es' if entity_type == 'Fix' else entity_type.lower() + 's'
                    if not is_empty_extracted_entity(extracted_entity, entity_type):
                        extracted_data[key].append(extracted_entity)
                    continue
                except json.JSONDecodeError as e:
                    logging.error(f"Error reading cache file {cache_file_path}: {e}")
                    # If cache file is corrupted, proceed with LLM extraction

            logging.debug(f"Extracting {entity_type} from document {row['node_id']}...")
            # Use the LLM to generate the extraction
            try:
                llm_response = await call_llm_no_logging_no_cache(system_prompt, user_prompt)

                if not llm_response:
                    logging.warning(f"LLM response is empty for document {row['node_id']}")
                    continue

                extracted_entity = json.loads(llm_response)
                # Generate a unique node_id for each sub-entity
                extracted_entity["node_id"] = str(uuid.uuid4())
                # Ensure source_id points to parent document
                extracted_entity["source_id"] = row["node_id"]
                # Add entity type
                extracted_entity["entity_type"] = entity_type

                # Add to the list of extracted entities
                extracted_entities = []
                extracted_entities.append(extracted_entity)

                # Log success
                logging.debug(
                    f"Successfully extracted {entity_type} from document {row['node_id']}"
                )

                # Count tokens and costs only on successful extraction
                usage = token_counter_instance.from_prompt_response(
                    system_prompt + user_prompt,
                    llm_response
                )
                token_counter["input_tokens"] += usage.input_tokens
                token_counter["output_tokens"] += usage.output_tokens
                token_counter["total_cost"] += usage.total_cost
                update_llm_usage(usage)

                # Add to appropriate list
                key = entity_type.lower() + 'es' if entity_type == 'Fix' else entity_type.lower() + 's'
                if not is_empty_extracted_entity(extracted_entity, entity_type):
                    # Write to cache file
                    try:
                        with open(cache_file_path, "w", encoding="utf-8") as f:
                            json.dump(extracted_entity, f, indent=2, cls=JSONSanitizingEncoder)
                        cached_files.append(cache_file_path)
                        logging.debug(f"Cached extraction result to {cache_file_path}")
                    except Exception as e:
                        logging.error(f"Error writing cache file {cache_file_path}: {e}")

                    extracted_data[key].append(extracted_entity)

            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON for llm response in document {row['node_id']}: {e}")
                continue
            except Exception as e:
                logging.error(f"Error during LLM extraction for document {row['node_id']}: {e}")
                continue
    end_time = time.time()
    logging.info(
        f"{documents_df.shape[0]} rows processed for extraction of {document_type} "
        f" in {end_time - start_time:.2f} seconds."
    )

    # Print cached file paths
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug("REPORT: Cached files:")
        for file_path in cached_files:
            logging.debug(f"  {file_path}")

    logging.info(f"Total input tokens: {token_counter['input_tokens']}")
    logging.info(f"Total output tokens: {token_counter['output_tokens']}")
    logging.info(f"Total cost: ${token_counter['total_cost']:.6f}")

    return extracted_data


class JSONSanitizingEncoder (json.JSONEncoder):
    """Custom JSON encoder for handling datetime objects and other special types."""
    def default(self, obj):
        try:
            if isinstance(obj, datetime.datetime):
                return obj.isoformat()
            elif isinstance(obj, (set, frozenset)):
                return list(obj)
            elif hasattr(obj, 'tolist'):  # Handle numpy arrays
                return obj.tolist()
            elif hasattr(obj, '__dict__'):  # Handle objects with __dict__
                return obj.__dict__
            elif isinstance(obj, float) and (obj != obj):  # Check for NaN
                return None
            return str(obj)  # Convert any other unserializable objects to strings
        except Exception as e:
            logging.warning(f"Error serializing object {type(obj)}: {e}")
            return str(obj)


class CustomDocumentTracker:
    """
    Tracks documents and their metadata with a hybrid memory-disk approach.

    Uses a time-based caching strategy where:
    - Recent documents (last 30 days) are kept in memory for fast access
    - Older documents are stored on disk and loaded only when needed
    - Periodic cleanup of old documents to prevent infinite growth

    Attributes:
        persist_path (Path): Path to the file storing the full catalog of documents
        cache_days (int): Number of days to cache documents in memory
        cache_dir (Path): Path to the directory storing individual document files
        recent_documents (dict): In-memory cache of recent documents
        last_accessed (dict): Dictionary tracking when each document was last accessed
    """

    def __init__(self, persist_path: str, cache_days: int = 30):
        self.persist_path = Path(persist_path)
        self.cache_days = cache_days
        self.cache_dir = self.persist_path.parent / "doc_tracker_cache"
        self.cache_dir.mkdir(exist_ok=True)
        # In-memory cache of recent documents
        self.recent_documents = {}
        # Track when documents were last accessed
        self.last_accessed = {}
        self._initialize_cache()

    def _should_cache_in_memory(self, document: dict) -> bool:
        """
        Determine if a document should be cached in memory based on its publish date.

        If the document is less than or equal to the cache_days threshold old, it will be cached in memory.
        If the document is older than the cache_days threshold, it will be stored on disk.

        Args:
            document (dict): Document to check

        Returns:
            bool: True if the document should be cached in memory, False otherwise
        """
        try:
            # Extract date from the document
            if "published" in document:
                date_val = document["published"]
            else:
                date_val = document["added_at"]

            # Handle different date formats
            if hasattr(date_val, 'to_pydatetime'):  # pandas built-in to convert Timestamp objects to python datetime
                doc_date = date_val.to_pydatetime()
            elif isinstance(date_val, str):
                # Handle string format
                date_str = date_val.split('T')[0]
                doc_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            else:
                raise ValueError(f"Unexpected date format: {type(date_val)}")

            days_old = (datetime.datetime.now() - doc_date).days
            logging.debug(f"Document date: {doc_date}, days old: {days_old}, cache threshold: {self.cache_days}")
            return days_old <= self.cache_days

        except Exception as e:
            logging.warning(f"Error determining cache status for document, defaulting to memory cache. Error: {str(e)}")
            logging.debug(f"Document data - published: {document.get('published')}, added_at: {document.get('added_at')}")
            return True

    def _initialize_cache(self):
        """
        Initialize cache with recent documents.

        This method loads documents from a persistent file and populates the cache
        with the most recent documents. It checks if a document should be cached
        in memory or on disk based on the self._should_cache_in_memory method.

        If the file does not exist or is corrupted, it will be recreated.

        Args:
            None

        Returns:
            None
        """
        try:
            if not self.persist_path.exists():
                return

            with open(self.persist_path, 'r') as f:
                all_docs = json.load(f)

            # Load documents into appropriate storage
            for doc_id, doc in all_docs.items():
                if self._should_cache_in_memory(doc):
                    logging.debug(f"Caching document {doc_id} in memory")
                    self.recent_documents[doc_id] = doc
                    self.last_accessed[doc_id] = datetime.datetime.now()
                else:
                    logging.debug(f"Storing document {doc_id} on disk")
                    self._write_to_disk(doc_id, doc)

        except Exception as e:
            logging.error(f"Error initializing cache: {e}")
            self._handle_corruption()

    async def add_document(self, doc_id: str, document: dict):
        """
        Add or update document tracking information.

        This method will:
        1. If the document exists, update it
        2. If the document has a source_id, remove any existing documents with the same source_id
        3. Store the document either in memory or on disk based on its age

        Args:
            doc_id (str): The ID of the document to add or update.
            document (dict): The document data.
        """
        current_time = datetime.datetime.now()

        # Check if document exists
        existing_doc = self.get_document(doc_id)
        if existing_doc:
            original_added_at = existing_doc.get("added_at")
        else:
            original_added_at = current_time.isoformat()

        # If this document has a source_id, remove any existing documents with the same source_id
        source_id = document.get('source_id')
        if source_id:
            # Find and remove any documents that share this source_id
            docs_to_remove = set()

            # Check memory cache
            for existing_id, existing_doc in self.recent_documents.items():
                if existing_doc.get('source_id') == source_id:
                    docs_to_remove.add(existing_id)

            # Check disk storage
            for file_path in self.cache_dir.glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        existing_doc = json.load(f)
                        if existing_doc.get('source_id') == source_id:
                            docs_to_remove.add(file_path.stem)
                except Exception as e:
                    logging.error(f"Error reading document {file_path}: {e}")

            # Remove all found documents
            for doc_id_to_remove in docs_to_remove:
                await self.remove_document(doc_id_to_remove)

        # Update document
        document_copy = document.copy()
        document_copy["added_at"] = original_added_at
        document_copy["last_updated"] = current_time.isoformat()

        # Store based on document age
        if self._should_cache_in_memory(document_copy):
            logging.debug(f"Adding document {doc_id} to memory cache")
            self.recent_documents[doc_id] = document_copy
            self.last_accessed[doc_id] = current_time
        else:
            logging.debug(f"Adding document {doc_id} to disk storage")
            self._write_to_disk(doc_id, document_copy)

        try:
            self._save_catalog()
        except Exception as e:
            logging.error(f"Error saving catalog for document {doc_id}: {e}")
            # Rollback the document addition
            if doc_id in self.recent_documents:
                self.recent_documents.pop(doc_id)
                self.last_accessed.pop(doc_id, None)
            doc_path = self.cache_dir / f"{doc_id}.json"
            if doc_path.exists():
                try:
                    doc_path.unlink()
                except Exception as del_e:
                    logging.error(f"Error during rollback of document {doc_id}: {del_e}")
            logging.info(f"Rolled back document {doc_id}")
            raise

    def _write_to_disk(self, doc_id: str, document: dict):
        """Write a document to individual file in cache directory.

        Write a document to disk storage. This method writes the document to
        a JSON file in the cache directory. The file is named after the
        document ID. If a file for this document already exists, it will be
        removed before writing the new one.
        """
        file_path = self.cache_dir / f"{doc_id}.json"
        # Remove existing file if it exists
        if file_path.exists():
            file_path.unlink()
        with open(file_path, 'w') as f:
            json.dump(document, f, cls=JSONSanitizingEncoder)

    def _read_from_disk(self, doc_id: str) -> Optional[dict]:
        """
        Read a document from disk if it exists.

        This method reads a document from disk storage. It looks for a JSON file
        in the cache directory with the name of the document ID. If the file
        exists, it is read and the document is returned as a dictionary.
        Otherwise, None is returned.

        Args:
            doc_id (str): The ID of the document to read.

        Returns:
            Optional[dict]: The document as a dictionary if it exists, otherwise None.
        """
        file_path = self.cache_dir / f"{doc_id}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None

    def get_document(self, doc_id: str) -> Optional[dict]:
        """
        Retrieve document tracking information by document ID.

        This method first checks the in-memory cache for the document. If the document
        is found, it updates the last accessed timestamp and returns the document.
        If the document is not in memory, it checks the disk storage. If the document
        is found on disk and is eligible to be cached in memory, it moves the document
        to the in-memory cache and updates the last accessed timestamp.

        Args:
            doc_id (str): The ID of the document to retrieve.

        Returns:
            Optional[dict]: The document tracking information if found, otherwise None.
        """
        if doc_id in self.recent_documents:
            self.last_accessed[doc_id] = datetime.datetime.now()
            return self.recent_documents[doc_id]

        document = self._read_from_disk(doc_id)
        if document:
            if self._should_cache_in_memory(document):
                logging.debug(f"Moving document {doc_id} to memory cache")
                self.recent_documents[doc_id] = document
                self.last_accessed[doc_id] = datetime.datetime.now()

        return document

    async def remove_document(self, doc_id: str):
        """
        Remove document from tracking.

        This method removes the document from in-memory tracking and disk storage if it exists.
        If the document is a parent document, it also removes all child documents that reference
        it through their source_id.

        Args:
            doc_id (str): The ID of the document to remove.
        """
        # Store original state for rollback if needed
        original_memory_docs = self.recent_documents.copy()
        original_last_accessed = self.last_accessed.copy()
        removed_files = []

        try:
            # First collect all documents to remove (both from memory and disk)
            docs_to_remove = set()
            docs_to_remove.add(doc_id)

            # Find child documents in memory
            for child_id, child_doc in self.recent_documents.items():
                if child_doc.get('source_id') == doc_id:
                    docs_to_remove.add(child_id)

            # Find child documents on disk
            for file_path in self.cache_dir.glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        child_doc = json.load(f)
                        if child_doc.get('source_id') == doc_id:
                            docs_to_remove.add(file_path.stem)
                except Exception as e:
                    logging.error(f"Error reading document {file_path}: {e}")

            # Remove all collected documents
            for doc_id_to_remove in docs_to_remove:
                # Remove from memory if present
                self.recent_documents.pop(doc_id_to_remove, None)
                self.last_accessed.pop(doc_id_to_remove, None)

                # Remove from disk if present
                file_path = self.cache_dir / f"{doc_id_to_remove}.json"
                if file_path.exists():
                    try:
                        # Keep track of removed files for potential rollback
                        with open(file_path, 'r') as f:
                            removed_files.append((file_path, json.load(f)))
                        file_path.unlink()
                    except Exception as e:
                        logging.error(f"Error removing file {file_path}: {e}")
                        raise

            try:
                self._save_catalog()
            except Exception as e:
                # If saving fails, we need to rollback all changes
                raise Exception(f"Failed to save catalog after document removal: {e}")

        except Exception as e:
            logging.error(f"Error during document removal {doc_id}: {e}")
            # Rollback memory state
            self.recent_documents = original_memory_docs
            self.last_accessed = original_last_accessed

            # Restore removed files
            for file_path, content in removed_files:
                try:
                    with open(file_path, 'w') as f:
                        json.dump(content, f, cls=JSONSanitizingEncoder)
                except Exception as restore_error:
                    logging.error(f"Error during rollback of file {file_path}: {restore_error}")

            logging.info(f"Rolled back removal of document {doc_id} and its children")
            raise

    def _save_catalog(self):
        """
        Save the complete catalog (both memory and disk documents).

        This method is used to persist the complete catalog of documents to disk.
        It first collects all documents from memory and disk, then writes them to
        a temporary file. If the write is successful, it atomically replaces the
        existing catalog file with the temporary file. If the write fails, it
        handles corruption by creating backups and attempting recovery.

        :raises: Exception if the save fails
        """
        temp_path = self.persist_path.with_suffix('.tmp')
        backup_path = self.persist_path.with_suffix('.bak')

        try:
            # Collect all documents (both in memory and on disk)
            all_docs = {}
            all_docs.update(self.recent_documents)

            # Add documents from disk storage
            for file_path in self.cache_dir.glob('*.json'):
                doc_id = file_path.stem
                if doc_id not in all_docs:  # Only add if not already in memory
                    doc = self._read_from_disk(doc_id)
                    if doc is not None:
                        all_docs[doc_id] = doc

            # Write to temporary file first
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(all_docs, f, cls=JSONSanitizingEncoder)

            # Create backup of existing file if it exists
            if self.persist_path.exists():
                try:
                    import shutil
                    shutil.copy2(self.persist_path, backup_path)
                except Exception as e:
                    logging.warning(f"Failed to create backup: {e}")

            # Close any potential open handles to the target file
            import gc
            gc.collect()

            # Attempt to replace the file
            try:
                if temp_path.exists():
                    if self.persist_path.exists():
                        self.persist_path.unlink()
                    temp_path.rename(self.persist_path)
            except PermissionError as e:
                logging.error(f"Permission error while replacing catalog file: {e}")
                # Try alternative approach using shutil
                try:
                    import shutil
                    shutil.move(str(temp_path), str(self.persist_path))
                except Exception as move_error:
                    logging.error(f"Failed to move file using shutil: {move_error}")
                    raise

        except Exception as e:
            logging.error(f"Error saving catalog: {e}")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            raise

        finally:
            # Cleanup temporary file if it still exists
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass

    async def _count_points_for_document(self, doc_id: str, is_source: bool = True) -> int:
        """Count points in vector store for a specific document."""
        try:
            # Build the appropriate filter based on is_source flag
            field_key = "metadata.source_id" if is_source else "doc_id"
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key=field_key,
                        match=MatchValue(value=doc_id)
                    )
                ]
            )

            # Count points using scroll
            points = await self.vector_db_service.async_client.scroll(
                collection_name=self.vector_db_service.collection,
                scroll_filter=filter_condition,
                limit=100,
                offset=None,
                with_payload=False,
                with_vectors=False
            )

            # Return count of points
            return len(points[0]) if points and points[0] else 0

        except Exception as e:
            logging.error(f"Error counting points for document {doc_id}: {e}")
            return 0

    def _get_doc_date(self, document: dict) -> datetime.datetime:
        """Helper method to extract date from document for age comparison."""
        try:
            # Try published date first
            if "published" in document:
                date_val = document["published"]
            else:
                date_val = document["added_at"]

            # Handle different date formats
            if hasattr(date_val, 'to_pydatetime'):  # pandas built-in to convert Timestamp objects to python datetime
                return date_val.to_pydatetime()
            elif isinstance(date_val, str):
                # Handle string format
                if 'T' in date_val:
                    date_str = date_val.split('T')[0]
                else:
                    date_str = date_val
                return datetime.datetime.strptime(date_str, "%Y-%m-%d")
            else:
                raise ValueError(f"Unexpected date format: {type(date_val)}")
        except Exception as e:
            logging.warning(f"Error parsing document date, using current time: {str(e)}")
            return datetime.datetime.now()

    def _handle_corruption(self):
        """Handle corrupted file scenarios by creating backups and attempting recovery.

        This method:
        1. Creates a timestamped backup of the corrupted file
        2. Attempts to recover data from the corrupted file
        3. Initializes a fresh tracker if recovery fails
        4. Logs all steps of the corruption handling process
        """
        if self.persist_path.exists():
            # Create timestamped backup
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.persist_path.with_suffix(f'.corrupted_{timestamp}')

            try:
                # Attempt to backup the corrupted file
                import shutil
                shutil.copy2(self.persist_path, backup_path)
                logging.warning(f"Backed up corrupted tracker file to {backup_path}")

                # Attempt to recover data from corrupted file
                try:
                    with open(self.persist_path, 'r') as f:
                        corrupted_data = json.load(f)

                    # If we can read the file, try to salvage any valid documents
                    recovered_docs = {}
                    for doc_id, doc in corrupted_data.items():
                        try:
                            if isinstance(doc, dict) and all(key in doc for key in ['content', 'metadata']):
                                recovered_docs[doc_id] = doc
                        except Exception:
                            continue

                    if recovered_docs:
                        logging.warning(f"Recovered {len(recovered_docs)} valid documents from corrupted file")
                        # Save recovered documents
                        with open(self.persist_path, 'w') as f:
                            json.dump(recovered_docs, f, cls=JSONSanitizingEncoder)
                        return

                except json.JSONDecodeError:
                    logging.error("Could not parse corrupted file as JSON")
                except Exception as recovery_err:
                    logging.error(f"Failed to recover data: {str(recovery_err)}")

                # If recovery failed, initialize fresh tracker
                logging.warning("Initializing fresh document tracker due to unrecoverable corruption")
                self.persist_path.unlink()
                self._initialize_cache()

            except Exception as backup_err:
                logging.error(f"Failed to create backup: {str(backup_err)}")
                # Even if backup fails, try to initialize fresh tracker
                self.persist_path.unlink()
                self._initialize_cache()

    def cleanup_old_documents(self, max_age_days: int = 365):
        """
        Remove documents older than the specified age from both memory and disk.

        This method iterates over all documents stored in memory and on disk, checking their
        'published' or 'added_at' date against a computed cutoff date. If a document is older
        than the specified maximum age in days, it is removed from the respective storage
        location. After cleanup, the catalog is saved to ensure consistency.

        Args:
            max_age_days (int): The maximum age of documents to retain, in days. Defaults to 365.
        """
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=max_age_days)
        logging.info(f"Cleaning up documents older than {cutoff_date}")

        # Clean memory cache
        docs_to_remove = []
        for doc_id, doc in self.recent_documents.items():
            if self._get_doc_date(doc) < cutoff_date:
                docs_to_remove.append(doc_id)

        for doc_id in docs_to_remove:
            del self.recent_documents[doc_id]
            if doc_id in self.last_accessed:
                del self.last_accessed[doc_id]

        # Clean disk storage and remove orphaned files
        existing_doc_ids = set()
        for doc_id in self.recent_documents.keys():
            existing_doc_ids.add(doc_id)

        # Check all files in cache directory
        for file_path in self.cache_dir.glob("*.json"):
            doc_id = file_path.stem
            try:
                # If file belongs to a document we know about
                if doc_id in existing_doc_ids:
                    continue

                # Try to read the document to check its age
                with open(file_path, 'r') as f:
                    doc = json.load(f)

                if self._get_doc_date(doc) < cutoff_date:
                    logging.debug(f"Removing old document file: {file_path}")
                    file_path.unlink()
            except Exception as e:
                logging.error(f"Error cleaning up document {doc_id}: {e}")

        self._save_catalog()

    def _initialize_fresh_context(self):
        """
        Initialize a fresh storage context with empty stores and minimal index structure.

        This method creates a new storage context and vector index from scratch, following these steps:
        1. Validates that vector store is already initialized
        2. Creates a storage context with empty document and index stores
        3. Creates an empty vector store index
        4. Sets and verifies the index ID using the collection name
        5. Persists the initialized storage context to disk

        This method is typically called when no existing index is found or when
        a fresh initialization is needed.

        Raises:
            ValueError: If vector store is not initialized or index ID verification fails
            Exception: If any other initialization step fails, with the specific error
                      logged before re-raising
        """
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
            logging.debug(f"Created and persisted initialized storage context with index_id '{index_id}' at {self.persist_dir}")

        except Exception as e:
            logging.error(f"Error initializing fresh context: {e}")
            raise

    async def _ensure_collection(self):
        """Ensure Qdrant collection exists."""
        try:
            # Check if collection exists
            await self.vector_db_service.async_client.get_collection(
                collection_name=self.vector_db_service.collection
            )
            logging.debug(f"Collection '{self.vector_db_service.collection}' exists")
        except Exception:
            # Create collection if it doesn't exist
            logging.warning(f"Creating collection '{self.vector_db_service.collection}'")
            await self.vector_db_service.async_client.create_collection(
                collection_name=self.vector_db_service.collection,
                vectors_config=self.vector_db_service.vector_config
            )

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
        if not processed_ids:
            raise ValueError("No documents ids to process, nothing to verify")

        try:
            # Get total points in collection
            collection_info = await self.vector_db_service.async_client.get_collection(
                collection_name=self.vector_db_service.collection
            )
            total_points = collection_info.points_count
            logging.debug(f"Total points in collection: {total_points}")
            logging.debug(f"Expected nodes: {nodes_created}")

            if total_points < nodes_created:
                logging.error(f"Point count mismatch: Found {total_points} points but created {nodes_created} nodes")
                return False

            # Verify each document exists in Qdrant
            for doc_id in processed_ids:
                logging.debug(f"Verifying document {doc_id}")
                scroll_result = await self.vector_db_service.async_client.scroll(
                    collection_name=self.vector_db_service.collection,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="metadata.source_id",
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
                else:
                    logging.debug(f"Found {len(points)} points for document {doc_id}")

            # Log statistics
            logging.debug(f"""Upsert verification successful:
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


async def track_dataframe_documents(
    df: pd.DataFrame,
    doc_tracker: CustomDocumentTracker,
    root_keys: List[str]
) -> None:
    """
    Process a DataFrame and track documents, maintaining specified root-level keys.

    For each row in the DataFrame:
      - Extracts root-level fields defined in `root_keys`.
      - Merges other fields into a metadata dictionary.
      - Updates existing metadata if keys overlap with columns in the row.

    Args:
        df (pd.DataFrame): Input DataFrame with rows to process.
        doc_tracker (CustomDocumentTracker): Document tracker instance to store processed documents.
        root_keys (List[str]): List of keys to maintain as root-level fields.

    Raises:
        Exception: Logs an error if adding a document fails.

    Example Row Transformation:
        Input Row:
        {
            "node_id": "1",
            "tool_label": "Tool A",
            "metadata": {"existing_key": "value1"},
            "extra_col1": "extra1"
        }
        Output Dictionary:
        {
            "node_id": "1",
            "tool_label": "Tool A",
            "metadata": {
                "existing_key": "value1",
                "extra_col1": "extra1"
            }
        }
    """
    # Replace NaN with None for clean JSON serialization
    df_clean = df.replace({np.nan: None})

    for _, row in df_clean.iterrows():
        # Convert the row to a dictionary
        doc_dict = row.to_dict()
        node_id = str(doc_dict.get('node_id'))
        if not node_id:
            continue

        # Initialize the document structure
        doc_data = {}
        metadata = {}

        # Process all fields in the row
        for key, value in doc_dict.items():
            if key in root_keys:
                # Assign root-level fields directly
                doc_data[key] = value
            elif key == "metadata":
                # Handle metadata field specially
                if isinstance(value, str):
                    try:
                        # Try to parse if it's a JSON string
                        parsed_metadata = json.loads(value)
                        if isinstance(parsed_metadata, dict):
                            # Remove any nested metadata key
                            parsed_metadata.pop('metadata', None)
                            metadata.update(parsed_metadata)
                    except json.JSONDecodeError:
                        # If not valid JSON, store as is
                        metadata[key] = value
                elif isinstance(value, dict):
                    # If it's already a dict, remove nested metadata and update
                    metadata_copy = value.copy()
                    metadata_copy.pop('metadata', None)
                    metadata.update(metadata_copy)
                else:
                    # For any other type, store as is
                    metadata[key] = value
            else:
                # Add non-root fields to metadata
                metadata[key] = value

        # Add the metadata to the document
        doc_data['metadata'] = metadata

        try:
            # Add the processed document to the tracker
            await doc_tracker.add_document(node_id, doc_data)
        except Exception as e:
            logging.error(f"Error tracking document {node_id}: {str(e)}")


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
        persist_dir (str): Directory for storing index and document tracking data
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
        self.embedding_service = vector_db_service.embedding_service
        self.persist_dir = persist_dir

        # Configure global settings
        llama_embedding_model = LlamaIndexEmbeddingAdapter(embedding_service=self.embedding_service)
        Settings.embed_model = llama_embedding_model

        # Enhanced node parser
        self.node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=window_size,
            window_metadata_key=window_metadata_key,
            original_text_metadata_key="original_text"
        )
        Settings.node_parser = self.node_parser

        # Initialize later
        self.vector_store: Optional[BasePydanticVectorStore] = None
        self.storage_context: Optional[StorageContext] = None
        self.index: Optional[VectorStoreIndex] = None
        self.doc_tracker: Optional[CustomDocumentTracker] = None

    @classmethod
    async def initialize(cls, vector_db_service: VectorDBService, persist_dir: str, **kwargs) -> "LlamaIndexVectorService":
        """
        Asynchronously initialize a new instance of LlamaIndexVectorService.

        This factory method creates and initializes a new instance with all necessary components.
        The two-step initialization (create instance + initialize components) allows for proper
        async setup of database connections and other I/O-bound resources.

        Args:
            vector_db_service (VectorDBService): Service for vector database operations
            persist_dir (str): Directory path where index and related files will be persisted
            **kwargs: Additional keyword arguments to pass to the constructor

        Returns:
            LlamaIndexVectorService: A fully initialized instance ready for use

        Raises:
            Exception: If component initialization fails
        """
        instance = cls(vector_db_service, persist_dir, **kwargs)
        await instance._initialize_components()
        return instance

    async def _initialize_components(self):
        """
        Initialize all required components for the LlamaIndex service.

        This method performs the following initialization steps:
        1. Ensures the vector database collection exists
        2. Initializes the Qdrant vector store with sync and async clients
        3. Sets up the storage context for persisting index data
        4. Initializes the document tracker for managing document metadata
        5. Either loads an existing vector store index or creates a new one

        The method is designed to be called only once during instance initialization
        and should not be called directly by users of the class.

        Raises:
            Exception: Logs an error if any component initialization fails, with the specific error
                    logged before re-raising
        """
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
            # TODO. IDE identifies an error
            # Argument of type "Path" cannot be assigned to parameter "persist_path" of type "str" in function "__init__"
            # "Path" is not assignable to "str"windsurfPyrightreportArgumentType
            # (variable) persist_dir: str
            self.doc_tracker = CustomDocumentTracker(
                Path(self.persist_dir) / "doc_tracker.json"
            )
            logging.info("initialized doc tracker")
            # Try loading existing index
            try:
                logging.info(f"Loading existing index from {self.persist_dir}")
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    storage_context=self.storage_context,
                    show_progress=True
                )
                index_id = self.vector_db_service.collection
                self.index.set_index_id(index_id)

            except Exception as e:
                logging.warning(f"No existing index found: {e}")
                self.index = VectorStoreIndex(
                    nodes=[],
                    storage_context=self.storage_context,
                    show_progress=True
                )
                self.index.set_index_id(self.vector_db_service.collection)
                logging.warning("created new index")
        except Exception as e:
            logging.error(f"Error initializing components: {e}")
            raise

    async def _delete_document_points(self, doc_id: str, is_source: bool = True) -> None:
        """Delete all points associated with a document, including child documents.

        Args:
            doc_id: The document ID to delete
            is_source: If True, delete points where doc_id matches source_id in metadata
                      If False, delete points where doc_id matches the point's doc_id
        """
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                logging.info(f"Deleting Points for document {doc_id} (attempt {retry_count + 1})")

                # Build filter condition based on is_source flag
                field_key = "metadata.source_id" if is_source else "doc_id"
                filter_condition = Filter(
                    must=[
                        FieldCondition(
                            key=field_key,
                            match=MatchValue(value=doc_id)
                        )
                    ]
                )

                # Delete points
                await self.vector_db_service.async_client.delete(
                    collection_name=self.vector_db_service.collection,
                    points_selector=FilterSelector(filter=filter_condition),
                    wait=True
                )

                # Verify deletion using our new count method
                count = await self._count_points_for_document(doc_id, is_source)
                if count == 0:
                    break
                logging.warning(f"Deletion verification failed for {doc_id}, {count} points remain")

            except Exception as e:
                logging.error(f"Error deleting points for document {doc_id} (attempt {retry_count + 1}): {e}")

            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(1)  # Wait before retry

        if retry_count == max_retries:
            raise Exception(f"Failed to delete points for document {doc_id} after {max_retries} attempts")

    async def upsert_documents(
        self,
        documents: List[LlamaDocument],
        verify_upsert: bool = False
    ) -> None:
        """
        Upsert documents to the vector store.

        This method:
        1. Checks document tracker for existing documents
        2. Deletes any existing Points for those documents from vector store (including child documents)
        3. Parses documents into nodes using the sentence window parser
        4. Generates embeddings for each node's text content
        5. Creates and upserts vector store points with embeddings and metadata
        6. Updates document tracker AFTER successful vector store update

        Args:
            documents: List of documents to upsert
            verify_upsert: If True, verify that all nodes were properly created in vector store (development only)
        """
        try:
            start_time = time.time()
            processed_doc_ids = set()  # Track successfully processed documents

            # Get list of documents that already exist in tracker
            existing_docs = []
            for doc in documents:
                if self.doc_tracker.get_document(doc.doc_id):
                    existing_docs.append(doc.doc_id)

            if existing_docs:
                logging.info(f"Found {len(existing_docs)} existing documents in tracker")
                # First delete points matching doc_id directly (1-to-1 deletion)
                for doc_id in existing_docs:
                    await self._delete_document_points(doc_id, is_source=False)

                # Then check for any remaining points with matching source_ids
                # This catches sub-document entities from previous runs that might have different node_ids
                for doc_id in existing_docs:
                    await self._delete_document_points(doc_id, is_source=True)

            # Convert documents to nodes using sentence window parser
            all_nodes = []
            doc_id_to_nodes = {}  # Track nodes per document

            with tqdm(total=len(documents), desc="Parsing documents", unit="doc") as pbar:
                for doc in documents:
                    # Each node will contain a sentence or window of sentences
                    nodes = self.node_parser.get_nodes_from_documents([doc])

                    # Add document level relationships
                    for i in range(len(nodes)):
                        current_node = nodes[i]

                        # Set NEXT relationship
                        if i < len(nodes) - 1:
                            current_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                                node_id=nodes[i + 1].node_id
                            )

                        # Set PREVIOUS relationship
                        if i > 0:
                            current_node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                                node_id=nodes[i - 1].node_id
                            )

                    all_nodes.extend(nodes)
                    doc_id_to_nodes[doc.doc_id] = nodes
                    pbar.update(1)

            nodes_created = len(all_nodes)
            logging.info(f"Created {nodes_created} nodes from {len(documents)} documents")

            # Generate embeddings and create points in batches
            batch_size = 50  # Process 50 nodes at a time to manage memory
            points = []

            with tqdm(total=len(all_nodes), desc="Generating embeddings", unit="node") as pbar:
                for i in range(0, len(all_nodes), batch_size):
                    batch = all_nodes[i:i + batch_size]

                    # Extract text content from nodes
                    texts = [node.get_content(metadata_mode="all") for node in batch]

                    try:
                        # Generate embeddings for the batch using the embedding service
                        embeddings = await self.embedding_service.generate_embeddings_async(texts)

                        # Create points with embeddings and metadata
                        batch_points = []
                        for node, embedding in zip(batch, embeddings):
                            relationships_serialized = {
                                rel_type.name: rel_info.node_id
                                for rel_type, rel_info in node.relationships.items()
                            }

                            payload = {
                                **node.metadata,
                                'relationships': relationships_serialized,
                            }

                            point = {
                                'id': node.node_id,
                                'payload': payload,
                                'vector': embedding,
                            }
                            batch_points.append(point)

                        points.extend(batch_points)
                        pbar.update(len(batch))

                    except Exception as e:
                        logging.error(f"Error processing batch: {e}")
                        raise

            # Upsert all points to the vector store with transaction-like behavior
            total_points = len(points)
            logging.info(f"Upserting {total_points} points to vector store")

            try:
                with tqdm(total=total_points, desc="Upserting points to vector store", unit="point") as pbar:
                    for i in range(0, total_points, batch_size):
                        batch = points[i:i + batch_size]
                        await self.vector_db_service.async_client.upsert(
                            collection_name=self.vector_db_service.collection,
                            points=batch,
                            wait=True
                        )
                        # Track which documents have been processed based on points in this batch
                        for point in batch:
                            doc_id = point['payload'].get('doc_id')
                            if doc_id:
                                processed_doc_ids.add(doc_id)
                        pbar.update(len(batch))

                # Only update document tracker after successful vector store update
                for doc in documents:
                    if doc.doc_id not in processed_doc_ids:
                        continue
                    try:
                        if self.doc_tracker is None:
                            raise ValueError("Document tracker is not initialized")

                        await self.doc_tracker.add_document(doc.doc_id, doc.metadata)
                        logging.debug(f"Updated document tracker for {doc.doc_id}")

                    except Exception as track_error:
                        logging.error(f"Error tracking document {doc.doc_id}: {track_error}")
                        # Continue processing other documents even if tracking fails for one
                        continue

            except Exception as e:
                logging.error(f"Error during vector store upsert: {e}")
                # Attempt rollback for processed documents
                for doc_id in processed_doc_ids:
                    try:
                        await self.delete_document(doc_id)
                        logging.info(f"Rolled back document {doc_id}")
                    except Exception as rollback_error:
                        logging.error(f"Error rolling back document {doc_id}: {rollback_error}")
                raise

            end_time = time.time()
            logging.info(f"Upsert completed in {end_time - start_time:.2f} seconds.")
            logging.info(f"Documents processed: {len(documents)}")

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
                        must=[
                            FieldCondition(
                                key="metadata.source_id",
                                match=MatchValue(value=doc_id)
                            )
                        ]
                    )
                ),
                wait=True
            )

            # Remove from document tracker
            self.doc_tracker.remove_document(doc_id)

            logging.debug(f"Successfully deleted document {doc_id}")
        except Exception as e:
            logging.error(f"Error deleting document {doc_id}: {e}")
            raise

    async def _get_collection_point_count(self) -> int:
        """Get the current number of points in the collection."""
        collection_stats = await self.vector_db_service.async_client.get_collection(
            collection_name=self.vector_db_service.collection
        )
        return collection_stats.points_count

    async def _count_unique_points(self) -> dict:
        """Count unique points based on different criteria."""
        try:
            # Get all points from collection
            points = await self.vector_db_service.async_client.scroll(
                collection_name=self.vector_db_service.collection,
                limit=10000,
                with_payload=True,
                with_vectors=True
            )

            if not points or not points[0]:  # Check if points is empty or first tuple element is empty
                return {
                    "total_points": 0,
                    "unique_vectors": 0,
                    "max_duplicates": 0,
                    "vector_groups": {}
                }

            # Extract points from scroll response
            points = points[0]  # First element contains points, second is next_page_offset

            # Group points by their vector (convert to tuple for hashability)
            vector_groups = {}
            for point in points:
                vector_tuple = tuple(point.vector)
                if vector_tuple not in vector_groups:
                    vector_groups[vector_tuple] = []
                vector_groups[vector_tuple].append(point)

            return {
                "total_points": len(points),
                "unique_vectors": len(vector_groups),
                "max_duplicates": max(len(points) for points in vector_groups.values()) if vector_groups else 0,
                "vector_groups": vector_groups
            }

        except Exception as e:
            logging.error(f"Error counting unique points: {e}")
            return {
                "total_points": 0,
                "unique_vectors": 0,
                "max_duplicates": 0,
                "vector_groups": {},
                "error": str(e)
            }

    async def aclose(self):
        """Close connections using existing service"""
        await self.vector_db_service.aclose()

    async def _count_points_for_document(self, doc_id: str, is_source: bool = True) -> int:
        """Count points in vector store for a specific document."""
        try:
            # Build the appropriate filter based on is_source flag
            field_key = "metadata.source_id" if is_source else "doc_id"
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key=field_key,
                        match=MatchValue(value=doc_id)
                    )
                ]
            )

            # Count points using scroll
            points = await self.vector_db_service.async_client.scroll(
                collection_name=self.vector_db_service.collection,
                scroll_filter=filter_condition,
                limit=100,
                offset=None,
                with_payload=False,
                with_vectors=False
            )

            # Return count of points
            return len(points[0]) if points and points[0] else 0

        except Exception as e:
            logging.error(f"Error counting points for document {doc_id}: {e}")
            return 0


def _truncate_for_logging(value: Any, max_length: int = 250) -> str:
    """Truncate long text values for logging purposes."""
    if not isinstance(value, str):
        value = str(value)
    if len(value) > max_length:
        return f"{value[:max_length]}... [truncated, total length: {len(value)}]"
    return value


def generate_cost_report():
    """Generate a report of LLM usage costs for the current run and cumulative costs."""
    usage_file = os.path.join(r"C:\Users\emili\PycharmProjects\microsoft_cve_rag\microsoft_cve_rag\application\data\llm_usage", "llm_usage.json")

    if not os.path.exists(usage_file):
        return "No usage data available."

    with open(usage_file, 'r') as f:
        usage_data = json.load(f)

    if not usage_data:
        return "No usage records found."

    # Get the latest run (last record)
    latest_run = usage_data[-1]

    # Calculate cumulative costs
    total_cumulative_cost = sum(record['total_cost'] for record in usage_data)
    total_cumulative_input_tokens = sum(record['input_tokens'] for record in usage_data)
    total_cumulative_output_tokens = sum(record['output_tokens'] for record in usage_data)

    # Generate report
    report = (
        f"\nCost Report for ETL Run\n"
        f"=====================\n"
        f"Current Run ({latest_run['full_date']}):\n"
        f"  Input Tokens: {latest_run['input_tokens']:,}\n"
        f"  Output Tokens: {latest_run['output_tokens']:,}\n"
        f"  Total Cost: ${latest_run['total_cost']:.4f}\n\n"
        f"Cumulative Statistics:\n"
        f"  Total Runs: {len(usage_data)}\n"
        f"  Total Input Tokens: {total_cumulative_input_tokens:,}\n"
        f"  Total Output Tokens: {total_cumulative_output_tokens:,}\n"
        f"  Total Cost to Date: ${total_cumulative_cost:.4f}\n"
    )

    logging.info(report)
    return report
