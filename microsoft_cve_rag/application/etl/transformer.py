# Purpose: Transform extracted data
# Inputs: Raw data
# Outputs: Transformed data
# Dependencies: None
import ast
import asyncio
import hashlib
import json
import logging
import os
import re
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
import time
from typing import Any, Dict, List, Optional, Union, Set, DefaultDict, Tuple
import marvin
from openai import AuthenticationError, APIConnectionError
# from neomodel import AsyncStructuredNode
import numpy as np
import pandas as pd
import spacy
from application.core.models.basic_models import Document
from microsoft_cve_rag.application.services.scraping_service import MicrosoftKbScraper
# from application.services.embedding_service import EmbeddingService
from application.etl.NVDDataExtractor import NVDDataExtractor, ScrapingParams
from fuzzywuzzy import fuzz, process
from llama_index.core import Document as LlamaDocument
from marvin.ai.text import generate_llm_response
from spacy.lang.en.stop_words import STOP_WORDS

# from microsoft_cve_rag.application.core.models import graph_db_models

marvin.settings.openai.chat.completions.model = "gpt-4o-mini"
# embedding_service = EmbeddingService.from_provider_name("fastembed")
logger = logging.getLogger(__name__)

# BEGIN PATCH FEATURE ENGINEERING HERLPERS ==================================


def _join_product_parts(row: pd.Series) -> str:
    """Join product parts into a full product string.

    Combines product name, version, and architecture into a single string,
    excluding 'NV' and 'NA' values.

    Args:
        row (pd.Series): DataFrame row containing product parts

    Returns:
        str: Space-separated string of valid product parts
    """
    parts = [
        part
        for part in [
            row["product_name"],
            row["product_version"],
            row["product_architecture"],
        ]
        if part not in ["NV", "NA"]
    ]
    return " ".join(parts)


def _join_product_name_version(row: pd.Series) -> str:
    """Join product name and version into a single string.

    Combines product name and version, excluding 'NV' values.

    Args:
        row (pd.Series): DataFrame row containing product information

    Returns:
        str: Space-separated string of product name and version
    """
    parts = [
        part
        for part in [row["product_name"], row["product_version"]]
        if part not in ["NV"]
    ]
    return " ".join(parts)


def _add_edge_terms(text: str, edge_terms: List[str]) -> str:
    """Add Edge-specific terms to a text field.

    Args:
        text (str): Original text to append terms to
        edge_terms (List[str]): List of Edge-specific terms to add

    Returns:
        str: Text with Edge-specific terms appended
    """
    return text + " " + " ".join(edge_terms)


def prepare_products(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare product information in DataFrame.

    Standardizes product names, creates full product strings, and adds
    Edge-specific search terms where applicable.

    Args:
        df (pd.DataFrame): DataFrame containing product information

    Returns:
        pd.DataFrame: DataFrame with prepared product information
    """
    df["product_name"] = df["product_name"].str.replace("_", " ")
    df["product_full"] = df.apply(_join_product_parts, axis=1)
    df["product_name_version"] = df.apply(_join_product_name_version, axis=1)

    # Add additional search terms for "edge"
    edge_terms = [
        "microsoft edge",
        "chromium",
        "chromiumbased",
        "chromium based",
    ]
    df.loc[df["product_name"] == "edge", "product_full"] = df.loc[
        df["product_name"] == "edge", "product_full"
    ].apply(lambda x: _add_edge_terms(x, edge_terms))
    df.loc[df["product_name"] == "edge", "product_name_version"] = df.loc[
        df["product_name"] == "edge", "product_name_version"
    ].apply(lambda x: _add_edge_terms(x, edge_terms))

    df = df.drop_duplicates(
        subset=["product_name", "product_version", "product_architecture",
                "product_full", "product_name_version"]
    )
    return df


def fuzzy_search(
    text: str,
    search_terms: List[str],
    threshold: int = 80
) -> List[str]:
    """Perform fuzzy string matching on text.

    Search for matches between text and search terms using partial ratio
    comparison.

    Args:
        text (str): Text to search within
        search_terms (List[str]): Terms to search for
        threshold (int, optional): Minimum match score. Defaults to 80.

    Returns:
        List[str]: List of matching terms above threshold
    """
    if pd.isna(text):
        return []

    matches = set()
    try:
        # First try to clean the text by removing non-alphanumeric chars
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        if not cleaned_text.strip():
            logging.debug(f"Text '{text}' contains no alphanumeric characters after cleaning")
            return []

        for term in search_terms:
            try:
                # Use cleaned text for comparison
                if fuzz.partial_ratio(cleaned_text.lower(), term.lower()) >= threshold:
                    matches.add(term)
            except (AttributeError, TypeError) as e:
                logging.warning(f"Error comparing term '{term}': {str(e)}")
                continue
            except Exception as e:
                logging.error(f"Unexpected error comparing term '{term}': {str(e)}")
                continue

    except (AttributeError, TypeError) as e:
        logging.warning(f"Error processing text '{text}': {str(e)}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error processing text '{text}': {str(e)}")
        return []

    return list(matches)


def fuzzy_search_column(
    column: pd.Series, products_df: pd.DataFrame, threshold: int = 80
) -> List[List[str]]:
    """Search for product mentions in a DataFrame column.

    Performs fuzzy matching against product names at different specificity
    levels.

    Args:
        column (pd.Series): Column to search within
        products_df (pd.DataFrame): DataFrame containing product information
        threshold (int, optional): Minimum match score. Defaults to 80.

    Returns:
        List[List[str]]: List of product mentions for each row
    """
    product_mentions = []
    for text in column:
        if isinstance(text, list):
            text = " ".join(text)
        matches = fuzzy_search(text, products_df["product_full"], threshold)
        if not matches:
            matches = fuzzy_search(
                text, products_df["product_name_version"], threshold
            )
        if not matches:
            matches = fuzzy_search(
                text, products_df["product_name"], threshold
            )
        product_mentions.append(matches)
    return product_mentions


def retain_most_specific(
    mentions: List[str], products_df: pd.DataFrame
) -> List[str]:
    """Filter product mentions to keep only the most specific versions.

    Retains mentions with the most detailed product information (name,
    version, architecture) when multiple versions exist.

    Args:
        mentions (List[str]): List of product mentions
        products_df (pd.DataFrame): DataFrame containing product information

    Returns:
        List[str]: Filtered list of most specific product mentions
    """
    specific_mentions = set()
    for mention in mentions:
        parts = mention.split()
        if len(parts) == 3:
            product_name, product_version, product_architecture = parts
            if product_version != "NV" and product_architecture != "NA":
                specific_mentions.add(mention)
        elif len(parts) == 2:
            product_name, product_version = parts
            if product_version != "NV":
                specific_mentions.add(mention)
        else:
            specific_mentions.add(mention)

    # Remove less specific mentions
    final_mentions = set()
    for mention in specific_mentions:
        if not any(
            mention in other for other in specific_mentions if other != mention
        ):
            final_mentions.add(mention)

    return list(final_mentions)


def convert_to_original_representation(mentions: List[str]) -> List[str]:
    """Convert product mentions back to original format.

    Replaces spaces with underscores to match original product naming.

    Args:
        mentions (List[str]): List of product mentions

    Returns:
        List[str]: Product mentions with underscores
    """
    if not mentions:
        return []
    return [mention.replace(" ", "_") for mention in mentions]


def construct_regex_pattern() -> str:
    """Construct regex pattern for build number extraction.

    Creates a pattern matching build numbers with up to 4 groups of digits,
    separated by dots.

    Returns:
        str: Regex pattern for build number matching
    """
    max_digits_per_group = (4, 4, 5, 5)
    pattern = (
        r"\b"
        + r"\.".join([
            r"\d{1," + str(max_digits) + r"}"
            for max_digits in max_digits_per_group
        ])
        + r"\b"
    )
    return pattern


# Function to extract build numbers from text
def extract_build_numbers(text: str, pattern: str) -> List[List[int]]:
    """Extract build numbers from text using regex pattern.

    Args:
        text (str): Text to extract build numbers from
        pattern (str): Regex pattern for matching build numbers

    Returns:
        List[List[int]]: List of build numbers as integer groups
    """
    matches = re.findall(pattern, text)
    build_numbers = [
        [int(part) for part in match.split(".")] for match in matches
    ]
    return build_numbers


def extract_windows_kbs(text: str) -> List[str]:
    """Extract Windows KB article references from text.

    Finds and standardizes KB article numbers to KB-XXXXXX format.

    Args:
        text (str): Text to extract KB references from

    Returns:
        List[str]: List of standardized KB article references
    """
    kb_pattern = r"(?i)KB[-\s]?\d{6,7}"
    raw_kb_matches = re.findall(kb_pattern, text)

    # Convert matches to uppercase and standardize format to "KB-123456"
    cleaned_kb_numbers = [
        match.upper().replace(" ", "").replace("KB", "") for match in raw_kb_matches
    ]
    standardized_kbs = [f"KB-{kb_number.strip('-')}" for kb_number in cleaned_kb_numbers]
    return list(set(standardized_kbs))


def extract_edge_kbs(row: pd.Series) -> List[str]:
    """Extract Edge KB references from build numbers.

    Creates KB references for Edge products based on build numbers.

    Args:
        row (pd.Series): DataFrame row containing product and build information

    Returns:
        List[str]: List of Edge KB references
    """
    # logger.info(f"Row types: { {k: type(v).__name__ for k, v in row.items()} }")
    # Get product_mentions more carefully
    try:
        product_mentions = row.get("product_mentions")
        # pm_type = type(product_mentions).__name__
        # pm_value = str(product_mentions) if product_mentions is not None else None
        # logger.info(f"Product mentions type: {pm_type}")
        # logger.info(f"Product mentions value (as string): {pm_value}")
    except Exception as e:
        logger.error(f"Error inspecting product_mentions: {str(e)}")
        return []
    edge_kbs = []

    if isinstance(product_mentions, (pd.Series, np.ndarray)):
        product_mentions = product_mentions.tolist()
    build_numbers = row.get("build_numbers")

    # Debug the actual type and value
    # logger.info(f"Type of product_mentions: {type(product_mentions)}")
    # logger.info(f"Product mentions value: {product_mentions}")
    if (product_mentions is not None
            and "edge" in product_mentions
            and build_numbers is not None):
        logger.info(f"Found Edge product with build_numbers: {build_numbers}")
        for build_number in build_numbers:
            if build_number:  # Additional check for non-empty build number
                logger.info(f"Processing build_number: {build_number}")
                build_str = ".".join(map(str, build_number))
                edge_kbs.append(f"KB-{build_str}")
    return list(set(edge_kbs))


async def update_email_records(emails_df, document_service):
    """
    Update documents in the database with information from an email dataframe.

    Args:
        emails_df (pd.DataFrame): DataFrame containing email data to update documents with.
        document_service (DocumentService): Service for interacting with the document database.

    Returns:
        None
    """
    total_processed = 0
    total_updated = 0
    not_updated_docs = []

    for index, row in emails_df.iterrows():
        document_id = row["node_id"]
        total_processed += 1
        try:
            document = Document(
                id_=document_id,
                product_mentions=row["product_mentions"],
                build_numbers=row["build_numbers"],
                kb_mentions=row["kb_mentions"],
                metadata={"id": document_id},
            )

            updated = document_service.update_document(document_id, document)
            if updated:
                total_updated += 1
            else:
                not_updated_docs.append(document_id)
                logging.debug(f"Document not updated: {document_id}")
        except Exception as e:
            not_updated_docs.append(document_id)
            logging.error(
                f"Exception {e}. Data input: doc id:"
                f" {document_id}\n{row['product_mentions']}\n{row['build_numbers']}\n{row['kb_mentions']}"
            )

    logging.info(
        f"Processed {total_processed} documents, updated"
        f" {total_updated} documents"
    )
    if not_updated_docs:
        logging.debug(
            f"Documents not updated: {', '.join(map(str, not_updated_docs))}"
        )


def normalize_mongo_kb_id(
    kb_id_input: Union[str, List[str]],
) -> Union[str, List[str]]:
    """Normalize the 'kb_id' field from MongoDB documents.

    Checks if the input is a string and converts it to a list if necessary.
    Removes any 'kb' or 'KB' prefix and ensures the 'kb_id' is in the format
    KB-XXXXXX or KB-XXX.XXX.XXX.XXX.

    Args:
        kb_id_input (Union[str, List[str]]): The 'kb_id' field from MongoDB documents

    Returns:
        Union[str, List[str]]: Normalized 'kb_id' field as string or list

    Notes:
        - Returns single item if list has only one element
        - Preserves list structure if multiple items present
    """
    if kb_id_input is None:
        return []

    if isinstance(kb_id_input, str):
        kb_id_list = [kb_id_input]
    else:
        kb_id_list = kb_id_input

    # Function to normalize a single kb_id
    def normalize_kb_id(kb_id: str) -> str:
        """
        Normalize a single Microsoft KB article ID.

        The function takes a string representing a Microsoft KB article ID, removes
        any 'kb' or 'KB' prefix, and ensures the returned string is in the format
        KB-XXXXXX or KB-XXX.XXX.XXX.XXX.

        Args:
            kb_id (str): A string representing a Microsoft KB article ID

        Returns:
            str: Normalized Microsoft KB article ID
        """
        if kb_id is None:
            return None
        # Remove any 'kb' prefix
        kb_id = kb_id.replace("kb", "").replace("KB", "")
        # Ensure the kb_id is in the format KB-XXXXXX or KB-XXX.XXX.XXX.XXX
        return f"KB-{kb_id}"

    # Extract substrings and replace the total strings
    processed_list = [
        normalize_kb_id(s.split("_")[0]) for s in kb_id_list if s is not None
    ]

    # Remove duplicates by converting the list to a set and back to a list
    processed_list = list(set(processed_list))
    processed_list.sort(reverse=True)
    return processed_list


def validate_and_adjust_columns(df, master_columns):
    # Get the current columns in the DataFrame
    current_columns = df.columns

    # Find columns that are in the master list but not in the DataFrame
    missing_columns = set(master_columns) - set(current_columns)

    # Find columns that are in the DataFrame but not in the master list
    extra_columns = set(current_columns) - set(master_columns)

    # Add missing columns to the DataFrame with NaN values
    for col in missing_columns:
        df[col] = pd.NA

    # Drop extra columns from the DataFrame
    df = df.drop(columns=extra_columns)

    # Reorder the columns to match the master list
    df = df[master_columns]

    return df


def custom_json_serializer(obj):
    """Custom JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def _map_product_name(x: str, mapping: Dict[str, str]) -> str:
    """Map product names using a predefined mapping dictionary.

    Args:
        x (str): Product name to map
        mapping (Dict[str, str]): Dictionary of product name mappings

    Returns:
        str: Mapped product name or original if no mapping exists
    """
    return mapping.get(x, x)


def _map_architecture(x: str, mapping: Dict[str, str]) -> str:
    """Map architecture names using a predefined mapping dictionary.

    Args:
        x (str): Architecture name to map
        mapping (Dict[str, str]): Dictionary of architecture mappings

    Returns:
        str: Mapped architecture name or original if no mapping exists
    """
    return mapping.get(x, x)


def _handle_single_item_list(x: Any) -> Any:
    """Extract single item from a list if it's a single-item list.

    Args:
        x (Any): Input value to process

    Returns:
        Any: Single item if input was single-item list, otherwise original input
    """
    return x[0] if isinstance(x, list) and len(x) == 1 else x


def _map_impact_type(x: str, mapping: Dict[str, str]) -> str:
    """Map impact type using a predefined mapping dictionary.

    Args:
        x (str): Impact type to map
        mapping (Dict[str, str]): Dictionary of impact type mappings

    Returns:
        str: Mapped impact type or original if no mapping exists
    """
    return mapping.get(x, x)


def _create_excluded_metadata_keys(x: Union[List, Any]) -> List[str]:
    """Create a list of excluded metadata keys.

    Combines existing keys with standard ones to create a comprehensive
    exclusion list for metadata processing.

    Args:
        x (Union[List, Any]): Existing excluded keys

    Returns:
        List[str]: Combined list of excluded metadata keys
    """
    base_keys = {
        "node_id",
        "cve_ids",
        "build_number",
        "node_label",
        "product_build_id",
        "product_build_ids",
    }
    existing_keys = set(x if isinstance(x, list) else [])
    return list(existing_keys | base_keys)


def _create_kb_catalog_url(kb_id: str) -> str:
    """Create a URL for the Microsoft Update Catalog.

    Args:
        kb_id (str): KB article ID

    Returns:
        str: Microsoft Update Catalog URL for the KB article
    """
    return (
        f"https://catalog.update.microsoft.com/Search.aspx?q={kb_id.replace('-', '')}"
        if pd.notna(kb_id)
        else ""
    )


def _get_first_label(labels: Any) -> str:
    """Get the first label from a list of labels.

    Args:
        labels (Any): Label or list of labels

    Returns:
        Any: First label if input is list, otherwise original input
    """
    return labels[0] if isinstance(labels, list) else labels


def _map_package_type(x: str, mapping: Dict[str, str]) -> str:
    """Map package type using a predefined mapping dictionary."""
    return mapping.get(x, x)


def _get_metadata_field(metadata: Dict[str, Any], field: str) -> Any:
    """Retrieve a specific field from the metadata dictionary.

    Args:
        metadata (Dict[str, Any]): Metadata dictionary
        field (str): Field name to retrieve

    Returns:
        Any: Field value if found, None otherwise
    """
    return metadata.get(field) if metadata else None


def _sort_kb_ids(kb_ids: Union[List[str], str]) -> Union[List[str], str]:
    """Sort KB IDs in reverse order if input is a list, otherwise return as is."""
    if isinstance(kb_ids, list):
        return sorted(kb_ids, reverse=True)
    return kb_ids


def _check_doc_processed(metadata: Dict[str, Any]) -> bool:
    """Check if a document has been processed based on its metadata."""
    return bool(
        metadata.get("etl_processing_status", {}).get(
            "document_processed", False
        )
    )


def _create_excluded_update_metadata_keys(x: List[str]) -> List[str]:
    """Create a list of excluded metadata keys specific to update packages."""
    base_keys = {
        "node_id",
        "node_label",
        "package_type",
        "published",
        "downloadable_packages",
    }
    existing_keys = set(x if isinstance(x, list) else [])
    return list(existing_keys | base_keys)


def transform_products(products: List[Dict[str, Any]]) -> pd.DataFrame:
    # clean up document dict from mongo to align with data models
    if products:
        df = pd.DataFrame(products, columns=list(products[0].keys()))

        # Apply the process_kb_id function to the 'kb_id' column
        df["kb_id"] = df["kb_id"].apply(normalize_mongo_kb_id)
        mapping = {"32-bit_systems": "x86", "x64-based_systems": "x64"}
        mapping_names = {
            "microsoft_edge_(chromium-based)": "edge",
            "microsoft_edge_(chromium-based)_extended_stable": "edge_ext",
        }
        df["product_name"] = df["product_name"].apply(
            lambda x: _map_product_name(x, mapping_names)
        )
        df["product_architecture"] = df["product_architecture"].apply(
            lambda x: _map_architecture(x, mapping)
        )
        df["product_architecture"] = df["product_architecture"].replace(
            "", "NA"
        )
        df["product_version"] = df["product_version"].replace("", "NV")
        df["node_label"] = "Product"
        df["published"] = pd.to_datetime(df["published"])
        df = df.rename(
            columns={
                "cve_id": "cve_ids",
                "kb_id": "kb_ids",
                "build_number": "build_numbers",
            }
        )
        print(f"Total Products transformed: {df.shape[0]}")
        # print(f"Columns: {df.columns}")
        return df
    else:
        print("No products to transform.")

    return None


def transform_product_builds(
    product_builds: List[Dict[str, Any]],
) -> pd.DataFrame:
    # clean up document dict from mongo to align with data models
    if product_builds:
        df = pd.DataFrame(
            product_builds, columns=list(product_builds[0].keys())
        )

        # Apply the process_kb_id function to the 'kb_id' column
        df["kb_id"] = df["kb_id"].apply(normalize_mongo_kb_id)
        df["kb_id"] = df["kb_id"].apply(_handle_single_item_list)
        # unclear why I added the next line. Neither the mongo doc nor the neo4j node have a `kb_ids` field.
        # df["kb_ids"] = df["kb_ids"].apply(lambda x: sorted(x, reverse=True))

        mapping_architectures = {
            "32-bit_systems": "x86",
            "x64-based_systems": "x64",
        }
        mapping_names = {
            "microsoft_edge_(chromium-based)": "edge",
            "microsoft_edge_(chromium-based)_extended_stable": "edge_ext",
        }
        mapping_impacts = {
            "information_disclosure": "disclosure",
            "elevation_of_privilege": "privilege_elevation",
        }
        df["product_architecture"] = df["product_architecture"].apply(
            lambda x: _map_architecture(x, mapping_architectures)
        )
        df["product_architecture"] = df["product_architecture"].replace(
            "", "NA"
        )
        df["product_version"] = df["product_version"].replace("", "NV")
        df["product_name"] = df["product_name"].apply(
            lambda x: _map_product_name(x, mapping_names)
        )
        df["impact_type"] = df["impact_type"].str.lower().str.replace(" ", "_")
        df["impact_type"] = df["impact_type"].apply(
            lambda x: _map_impact_type(x, mapping_impacts)
        )
        df["impact_type"] = df["impact_type"].fillna("NIT")
        df["severity_type"] = df["severity_type"].str.lower()
        df["severity_type"] = df["severity_type"].fillna("NST")
        df["node_label"] = "ProductBuild"
        df["published"] = pd.to_datetime(df["published"])
        # Convert the build_number lists to a string representation
        # df["build_number_str"] = df["build_number"].apply(lambda x: str(x))
        # Now drop duplicates based on all relevant columns
        df_unique = df.drop_duplicates(
            subset=[
                "product_name",
                "product_version",
                "product_architecture",
                "cve_id",
                "product_build_id",
            ]
        )
        # Drop the helper string column if no longer needed
        df_unique = df_unique.drop(
            columns=[
                "product",
            ]
        )
        print(f"Total Product Builds transformed: {df.shape[0]}")
        # print(f"Columns: {df.columns}")
        return df_unique
    else:
        print("No product builds to transform.")

    return None


async def async_generate_summary(text: str) -> Optional[str]:
    # Handle None, empty string, and pandas NA values
    if pd.isna(text) or text is None or str(text).strip() == "":
        return None

    marvin_summary_prompt = r"""
        Generate a highly technical summary of the following Microsoft KB Article text. This summary is intended for advanced system administrators and IT professionals specializing in modern device management with Intune MDM, Entra ID, Windows 365, and Azure.

        General Guidelines

        - Provide an engaging, sentence-based overview paragraph followed by a detailed technical breakdown with sub sections. Adhere to the structure below. Assume the reader has a strong background in patching and Microsoft device management. Maintain honesty and precision by sticking to the KB article's content, avoiding general or outdated advice unless specified in the article.
        - Use the subheadings outlined below to structure the summary, aligning with the KB article's content.
        - Format all commands, error codes, and configurations using triple-backtick Markdown code blocks with appropriate language identifiers.
        - Do not add a 'closing' or 'in conclusion' section or thoughts or any text following the last section presented below.

        Formatting Guidelines
        - **Commands and Scripts**: Use triple-backtick Markdown code blocks with language identifiers from this list only.
        - Include commands exactly as they appear in the KB article. Choose the identifier based on the command's context:
            - **PowerShell** (for PowerShell scripts and cmdlets):
                ```powershell
                Set-ItemProperty -Path "HKLM:\Software\Policies\Microsoft\Windows\Installer" -Name "DisableLUAInRepair" -Value 1
                ```
            - **Batch** (for Windows Command Prompt commands, e.g., DISM, wusa, net):
                ```batch
                DISM /online /get-packages
                ```
            - **Bash** (for Azure CLI, Intune terminal, or Unix-like commands):
                ```bash
                az intune policy set --policyName "DOCacheHost" --value "<Your MCCC Endpoint>"
                ```
            - **Text** (for error codes, logs, or non-executable text):
                ```text
                0x80070520
                ```
            - **Ini** (for registry keys or key-value configuration):
                ```ini
                [HKEY_LOCAL_MACHINE\SOFTWARE\Policies\Microsoft\Windows\Installer]
                DisableLUAInRepair=1
                ```
        - **Rules**:
            - Insert blank lines before and after lists and headings and tables
            - Four spaces per nested level indicates proper indentation for lists
            - Critical: multi-line content must align precisely within its list item. add a line break before code fences in nested lists, but do not indent code fences, code fences start at col 0.
                1. First level:
                    1.1. Second level:

                ```powershell
                dir
                ```

            - Use the GitHub-style (pipe-delimited) table format for tables and left align text
                The second line |:---|:---| sets column alignment:
                - :--- => left-aligned
            - Match the source text exactly — do not paraphrase commands.
            - For registry keys, format as INI-style (section header `[path]` followed by `key=value`).
            - If unsure of the language, default to `text`.
        - IMPORTANT: You must ensure the markdown is formatted perfectly so that it can be passed to markdown.markdown() and use codehilite and fenced_code extensions.

        Additional Notes
        - Highlight security risks, elevated-risk exploits, or known issues clearly.
        - Exclude file information (e.g., file lists) from the summary.
        - Include links to any mentioned KBs, CVEs or resources for easier navigation.
        - **CRITICAL Link Formatting Rule:** Format ALL hyperlinks strictly using standard Markdown: `[Link Text](URL)`.
        - The scraping service is generating malformed URLs in the markdown.
        - Extract only the final, canonical URL for the `URL` part (e.g., `https://msrc.microsoft.com/...` or `https://learn.microsoft.com/...`).
        - **DO NOT** include any wrapper angle brackets like `<...>` around the URL inside the parentheses `()`.
        - **DO NOT** prepend redundant paths like `https://support.microsoft.com/en-us/topic/` to the actual target URL within the parentheses `()`.
            - Correct Example: [CVE-2022-38042](https://msrc.microsoft.com/update-guide/vulnerability/CVE-2022-38042)
            - Correct Example: [NetJoinDomain](https://docs.microsoft.com/windows/win32/api/lmjoin/nf-lmjoin-netjoindomain)
            - Correct Example: [KB5044277](https://support.microsoft.com/help/5044277)

        <summary structure>
        ### Overview
        - Include:
        - The primary purpose of this KB article (e.g., security update, quality improvement).
        - The specific operating systems and versions affected in a bullet list.
        - Any critical security issues, vulnerabilities, or bugs addressed.
        - Search for two sections in the source text: 'Summary' and '*servicing stack update*' (e.g., 'Windows 11 servicing stack update (KB5043938) - 22000.3196') and incorporate the content from those sections into the 'Overview' section.
        - Avoid general advice on patch management.
        - Limit to 6 full sentences maximum.
        - Do not add sub sections to the Overview section.

        Example: 'KB5040427 is a critical security update released on July 9, 2024, targeting:\n
        - Windows 10 Enterprise LTSC 2021\n
        - IoT Enterprise LTSC 2021\n
        - Windows 10 version 22H2 (builds 19044.4651 and 19045.4651)\n
        The update addresses significant security vulnerabilities in RADIUS protocol (MD5 collisions) and implements enhanced BitLocker Secure Boot validation profiles. This combined SSU/LCU package requires specific prerequisites based on deployment method: KB5014032 for offline imaging or KB5005260 for WSUS deployment. The update includes SSU KB5039336 (builds 19044.4585 and 19045.4585) for improved update servicing capabilities.'

        ### Scope of Impact
        - Optional. Do not output this section header if it isn't in the source text.
        - Some KB articles provide a section on scope of impact, if it exists, present the high level points here.
        - Do not fabricate or add commentary that isn't in the source text.

        ### Available Mitigations
        - Optional. Do not output this section header if it isn't in the source text.
        - Some KB articles provide a section on available mitigations, if it exists, present the high level points here.
        - Do not fabricate or add commentary that isn't in the source text.

        ### Technical Breakdown
        - Generate a detailed response at least as long as this prompt (~1200 tokens) for KB articles up to 3000 words, including all improvements, bug fixes, and vulnerabilities from the 'Improvements' section with exhaustive technical details and examples where applicable. For articles exceeding 5000 words, scale down the summary length proportionally to remain concise yet comprehensive.
        - You may add sub sections here to better align with the source text.

        #### Vulnerabilities, Improvements and Bug Fixes
        - List all improvements, bug fixes, and vulnerabilities from the 'Improvements' section in bullet points with brief descriptions.
        - Prioritize high-impact items (e.g., security vulnerabilities, critical bugs) first.
        - Clearly state whether each vulnerability is high or low impact. Example: 'This update addresses high-impact vulnerabilities in Windows Installer and RADIUS protocol related to MD5 collision exploits. Full details in the July 2024 Security Updates.'
        - Clearly state whether the item is a: ['vulnerability', 'bug fix', 'improvement']
        - Include actionable details (e.g., affected components, configurations, downloads, scripts).
        - Maximum 10 items.

        ##### Vulnerabilities
        Example:
        1. (Impact: **Critical**) Vulnerability in Windows Installer: An identified security vulnerability in the Windows Installer that could allow an attacker to execute arbitrary code. Microsoft is working on a fix.

        ##### Improvements
        Example:
        1. (Impact: **High**) Improvement to OOBE Process: Enhances the overall user experience during the initial setup of Windows 11 by streamlining processes and ensuring smoother transitions.

        ##### Bug Fixes
        Example:
        1. (Impact: **Medium**) Fix for DHCP Option 235 Discovery Issue: Addresses a problem where the DHCP client might not be able to discover the DHCP server after installing this update.

        #### Known Issues and Workarounds
        - Detail any known issues and workarounds from the KB article in markdown table notation.
        - Do not fabricate or make up Known Issues if they don't exist, not all KB articles have Known Issues.
        - Specify the fix method (e.g., Autopatch, MDM, Group Policy, registry modification, Intune command) if provided, noting WSUS deprecation and preference for Autopatch/Intune where mentioned.
        - Example Output format:

            | Issue | Applies To | Symptoms | Workarounds |
            |:---|:---|:---|:---|
            | **Citrix** | All Users | Unable to complete installation of January 2023... | This issue is addressed in [KB5053643](https://support.microsoft.com/en-us/kb/5053643). |
            | **System Guard Runtime Monitor Broker service** | Enterprise users | Triggers unexpected alerts after certain updates... | Disable the service or... |
            | **Another Issue** | etc. | - Error Code: ```text 0x80070520``` |                                |

        #### Installation Process and Prerequisites

        ##### Prerequisites
        - Provide step-by-step prerequisites from the KB article, often found under a heading like 'Before you install this update'.

        ##### Installation Steps
        - Provide numbered steps to install the update

        ##### Installation Channels
        - Search for the section detailing how to install the update (often titled 'Install this update' or 'How to get this update'). Identify the different release channels mentioned (e.g., Windows Update, WSUS, Microsoft Update Catalog, Windows Update for Business). Extract the availability status (e.g., "Yes", "No") and the corresponding description or next steps for each channel. Present this information **exclusively** as a Markdown table with the columns: 'Channel', 'Availability', and 'Details'. **Do not** use bullet points or paragraphs for this channel information. If the source text uses odd formatting like repeating headers, consolidate the information logically per channel into single rows in the table.
        - Example Output Format:

            | Channel | Availability | Details |
            |:---|:---|:---|
            | **Windows Update** | Yes | None. This update will be downloaded and installed automatically from Windows Update and Microsoft Update. |
            | **Windows Update for Business** | Yes | None. This update will be downloaded and installed automatically from Windows Update for Business in accordance with configured policies. |
            | **Microsoft Update Catalog** | Yes | To get the standalone package for this update, go to the [Microsoft Update Catalog](https://...) website. |
            | **Windows Server Update Services (WSUS)** | Yes | This update will automatically sync with WSUS if you configure **Products and Classifications** as follows... (See WSUS Config below) |

        ##### Uninstallation Instructions
        - Extract any uninstallation instructions, often found under a heading like 'If you want to remove the LCU'. Use standard paragraphs and format any commands using triple-backtick code blocks with the correct language identifier (e.g., ```cmd ... ``` or ```powershell ... ```).
        </summary structure>

        KB Article text below
        ===
        {kb_article_text}
        ===
        """
    model_kwargs = {
        "max_tokens": 3250,
        "temperature": 0.1,
        "top_p": 1.0,
        "presence_penalty": 0.5,
        "frequency_penalty": 0.3,
    }
    try:
        # Ensure the OpenAI API key is set
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

        # Attempt to generate the LLM response
        response = await generate_llm_response(
            marvin_summary_prompt.format(kb_article_text=text),
            model_kwargs=model_kwargs,
        )
        return response.response.choices[0].message.content

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


async def scrape_kb_article_content(url: str, scraper: MicrosoftKbScraper) -> str:
    """Scrape content from a KB article URL.

    Args:
        url (str): URL of the KB article to scrape
        scraper (MicrosoftKbScraper): Instance of the KB article scraper

    Returns:
        str: Markdown content of the KB article, or empty string if scraping fails
    """
    if not url or not isinstance(url, str):
        logging.warning(f"Invalid URL provided: {url}")
        return ""

    try:
        # Use the scraper to get the article content
        await scraper.scrape_kb_article(url)

        # Extract markdown content from the result
        markdown = scraper.get_markdown()
        if markdown:
            return markdown
        else:
            logging.warning(f"No markdown content found for URL: {url}")
            return ""
    except Exception as e:
        logging.error(f"Error scraping KB article {url}: {str(e)}")
        return ""


async def retry_blocked_urls(
    scraper: MicrosoftKbScraper,
    blocked_urls: List[Tuple[int, str]],
    crawl_results: List[Any],
    attempt: int = 1,
    max_attempts: int = 3
) -> None:
    """Recursively retry blocked URLs with increasing delays.

    Args:
        scraper: The KB article scraper instance
        blocked_urls: List of tuples containing (original_index, url)
        crawl_results: List to update with successful results
        attempt: Current attempt number (1-based)
        max_attempts: Maximum number of retry attempts
    """
    if not blocked_urls or attempt > max_attempts:
        return

    delay_ranges = {
        1: (180, 200),  # 3-3.3 minutes
        2: (300, 330),  # 5-5.5 minutes
        3: (600, 600)   # 10 minutes
    }
    min_delay, max_delay = delay_ranges[attempt]

    still_blocked = []
    batch_to_save = []
    logging.info(f"Retry attempt {attempt}/{max_attempts} for {len(blocked_urls)} blocked URLs")

    for i, (original_idx, url) in enumerate(blocked_urls):
        logging.info(f"Retrying blocked URL {i+1}/{len(blocked_urls)}: {url}")

        delay = random.uniform(min_delay, max_delay)
        logging.info(f"Adding delay of {delay:.2f} seconds")
        start_time = time.perf_counter()
        await asyncio.sleep(delay)
        end_time = time.perf_counter()
        logging.info(f"Actual delay time: {(end_time - start_time):.1f} seconds")

        result = await scraper.scrape_kb_article(url)
        if result and hasattr(result, "status_code") and result.status_code == 403:
            logging.warning(f"URL still blocked (403) on attempt {attempt}: {url}")
            still_blocked.append((original_idx, url))
            if attempt == max_attempts:
                crawl_results[original_idx] = None
        else:
            logging.info(f"Successfully retrieved URL on attempt {attempt}: {url}")
            crawl_results[original_idx] = result
            batch_to_save.append(result)

        # Save successful results in batches of 3
        if len(batch_to_save) >= 3:
            await scraper.save_kb_bulk_results(batch_to_save)
            batch_to_save = []

    # Save any remaining successful results
    if batch_to_save:
        await scraper.save_kb_bulk_results(batch_to_save)

    # Recursively retry remaining blocked URLs
    if still_blocked:
        await retry_blocked_urls(
            scraper=scraper,
            blocked_urls=still_blocked,
            crawl_results=crawl_results,
            attempt=attempt + 1,
            max_attempts=max_attempts
        )


async def scrape_kb_articles(urls: pd.Series) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Scrape content from multiple KB article URLs with robust rate limiting.

    Uses single URL processing with manual delays between requests to prevent
    server blocking. Saves results using save_kb_bulk_results. Includes retry
    logic for URLs that return 403 status code.

    Args:
        urls (pd.Series): Series of URLs to scrape

    Returns:
        Tuple[List[str], List[Dict[str, Any]]]: Lists of markdown content and
            structured JSON data in the same order as input URLs
    """
    # Filter out None/empty URLs
    valid_urls = [url for url in urls if url]

    if not valid_urls:
        logging.warning("No valid URLs provided for scraping")
        return [], []

    try:
        # Create a single scraper instance for all URLs
        scraper = MicrosoftKbScraper(extraction_method="llm")

        # Process URLs one at a time with delays between requests
        crawl_results = []
        blocked_urls = []  # Track URLs that return 403 status code

        for i, url in enumerate(valid_urls):
            logging.info(f"Processing URL {i+1}/{len(valid_urls)}: {url}")

            # Add a delay between requests (35-50 seconds)
            if i > 0:
                delay = random.uniform(60, 70)
                logging.info(f"Adding delay of {delay:.2f} seconds")
                await asyncio.sleep(delay)

            # Process single URL
            result = await scraper.scrape_kb_article(url)

            # Check if the request was blocked (status code 403)
            if result and hasattr(result, "status_code") and result.status_code == 403:
                logging.warning(f"URL was blocked (403): {url}")
                blocked_urls.append((i, url))
                crawl_results.append(None)
            else:
                crawl_results.append(result if result else None)

            # Save results in batches of 3 to maintain existing file organization
            if (i + 1) % 3 == 0 or i == len(valid_urls) - 1:
                batch_results = [r for r in crawl_results[-3:] if r is not None]
                if batch_results:
                    summary_path = await scraper.save_kb_bulk_results(batch_results)
                    logging.info(f"Batch results saved to {summary_path}")

        # Retry blocked URLs with longer delays if any were blocked
        if blocked_urls:
            await retry_blocked_urls(
                scraper=scraper,
                blocked_urls=blocked_urls,
                crawl_results=crawl_results
            )

        # Process results into markdown and JSON lists
        markdown_list = []
        json_list = []

        for result in crawl_results:
            if result and hasattr(result, "success") and result.success and hasattr(result, "html") and result.html:
                # Extract markdown
                markdown = result.markdown if hasattr(result, "markdown") else ""

                # Extract and process JSON content
                extracted_content = scraper._parse_json_content(result.extracted_content) if hasattr(result, "extracted_content") else None
                if isinstance(extracted_content, list) and extracted_content:
                    # If it's a list, take the last non-empty dict
                    valid_contents = [c for c in extracted_content if c and isinstance(c, dict)]
                    json_content = valid_contents[-1] if valid_contents else {}
                elif isinstance(extracted_content, dict):
                    json_content = extracted_content
                else:
                    json_content = {}
            else:
                markdown = ""
                json_content = {}

            markdown_list.append(markdown)
            json_list.append(json_content)

        # Map results back to original URL order
        final_markdown = []
        final_json = []
        for url in urls:
            if url in valid_urls:
                idx = valid_urls.index(url)
                final_markdown.append(markdown_list[idx])
                final_json.append(json_list[idx])
            else:
                final_markdown.append("")
                final_json.append({})

        return final_markdown, final_json

    except Exception as e:
        logging.exception(f"Error in KB article scraping: {str(e)}")
        # Return empty results for all URLs in case of failure
        return ["" for _ in urls], [{} for _ in urls]


def is_valid_kb_json(json_data: Dict[str, Any]) -> bool:
    """Check if the KB article JSON has the required keys and valid content."""
    required_keys = {"title", "url", "os_builds"}
    if not isinstance(json_data, dict):
        return False
    return all(key in json_data for key in required_keys) and json_data != {"raw": "None"}


def get_text_for_summary(row: pd.Series) -> str:
    """Get the best available text content for summary generation.

    Precedence order:
    1. scraped_markdown (if available and non-empty)
    2. scraped_json (if valid KB JSON)
    3. raw text as fallback
    """
    if isinstance(row["scraped_markdown"], str) and row["scraped_markdown"].strip():
        return row["scraped_markdown"]
    elif isinstance(row["scraped_json"], dict) and is_valid_kb_json(row["scraped_json"]):
        return json.dumps(row["scraped_json"])
    return row["text"]


# Wrapper to handle async calls in apply
async def generate_summaries(texts: pd.Series) -> List[str]:
    tasks = [async_generate_summary(text) for text in texts]
    return await asyncio.gather(*tasks)


def transform_kb_articles(
    kb_articles_windows: List[Dict[str, Any]],
    kb_articles_edge: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Transform KB articles data into a pandas DataFrame."""
    master_columns = [
        "id",
        "kb_id",
        "title",
        "text",
        "published",
        "product_build_id",
        "product_build_ids",
        "build_number",
        "cve_ids",
        "excluded_embed_metadata_keys",
        "excluded_llm_metadata_keys",
        "node_label",
        "article_url",
        "summary",
        "scraped_markdown",
        "scraped_json",
        "update_package_url",
    ]
    dtypes = {
        'id': 'str',
        'kb_id': 'str',
        'title': 'str',
        'text': 'str',
        'published': 'datetime64[ns]',
        'product_build_id': 'str',
        'product_build_ids': 'object',
        'build_number': 'object',
        'cve_ids': 'object',
        'excluded_embed_metadata_keys': 'object',
        'excluded_llm_metadata_keys': 'object',
        'node_label': 'str',
        'article_url': 'str',
        'summary': 'str',
        'scraped_markdown': 'str',
        'scraped_json': 'object',
        'update_package_url': 'str',
    }

    # Process Windows KB articles (this branch of logic)
    if kb_articles_windows:
        df_windows = pd.DataFrame(kb_articles_windows)

        # Ensure all master columns exist, adding missing ones with NaN
        for col in master_columns:
            if col not in df_windows.columns:
                df_windows[col] = pd.NA

        df_windows = df_windows[master_columns].copy()
        # Convert published to datetime early for sorting, handle errors
        df_windows['published'] = pd.to_datetime(df_windows['published'], errors='coerce')

        # --- Start: Modified Duplicate Handling ---
        # Define criteria for "better" records: non-empty summary, markdown, cve_ids
        # Handle potential variations of empty (NaN, None, '', [])
        df_windows['has_summary'] = df_windows["summary"].notna() & df_windows["summary"].astype(str).str.strip().ne("")
        df_windows['has_markdown'] = df_windows["scraped_markdown"].notna() & df_windows["scraped_markdown"].astype(str).str.strip().ne("")
        # Check if cve_ids is notna and (if list, not empty; otherwise assume non-empty if notna)
        df_windows['has_cve_ids'] = df_windows["cve_ids"].notna() & df_windows["cve_ids"].apply(lambda x: bool(x) if isinstance(x, list) else True)

        # Sort by kb_id, then by preference criteria (non-empty first), then by published date (newest first)
        df_windows.sort_values(
            by=['kb_id', 'has_summary', 'has_markdown', 'has_cve_ids', 'published'],
            ascending=[True, False, False, False, False],  # True first for bools means ascending=False
            inplace=True,
            na_position='last'  # Keep NaNs in published at the end
        )

        # Keep the first (best) record for each kb_id
        df_windows = df_windows.drop_duplicates(subset=['kb_id'], keep='first')

        # Drop temporary sorting columns
        df_windows = df_windows.drop(columns=['has_summary', 'has_markdown', 'has_cve_ids'])
        # --- End: Modified Duplicate Handling ---

        df_windows["kb_id"] = df_windows["kb_id"].apply(normalize_mongo_kb_id)
        df_windows["kb_id"] = df_windows["kb_id"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x
        )

        # Re-apply validation after potential type changes from normalization
        df_windows = validate_and_adjust_columns(df_windows, master_columns)
        df_windows["node_label"] = "KBArticle"
        # Ensure published is datetime again after potential modifications
        df_windows["published"] = pd.to_datetime(df_windows["published"], errors='coerce')
        df_windows["excluded_embed_metadata_keys"] = [
            [] for _ in range(len(df_windows))
        ]
        df_windows["excluded_embed_metadata_keys"] = df_windows[
            "excluded_embed_metadata_keys"
        ].apply(_create_excluded_metadata_keys)

        # Initialize metadata with etl_processing_status
        df_windows["metadata"] = [
            {
                "etl_processing_status": {
                    "document_processed": True,
                    "entities_extracted": False,
                    "graph_prepared": False,
                    "vector_prepared": False,
                    "last_processed_at": None,
                    "processing_version": "1.0",
                }
            }
            for _ in range(len(df_windows))
        ]

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # --- Start: Modified Scraping and Summary Generation Logic ---

        # 1. Identify rows that NEED scraping (scraped_markdown is NaN and URL exists)
        needs_scraping_mask = (
            df_windows['scraped_markdown'].isna()
            & df_windows['article_url'].notna()
            & df_windows['article_url'].astype(str).str.strip().ne('')
        )
        df_to_scrape = df_windows[needs_scraping_mask]

        if not df_to_scrape.empty:
            logging.info(f"Scraping content for {df_to_scrape.shape[0]} Windows-based KBs with missing markdown.")
            urls_to_scrape = df_to_scrape["article_url"].tolist()

            # Call the scrape_kb_articles function
            scraped_markdown_list, scraped_jsons_list = loop.run_until_complete(scrape_kb_articles(urls_to_scrape))

            # Create Series with the results, indexed correctly to match df_to_scrape
            markdown_series = pd.Series(scraped_markdown_list, index=df_to_scrape.index)
            json_series = pd.Series(scraped_jsons_list, index=df_to_scrape.index, dtype='object')  # Ensure object dtype for json

            # Update the main DataFrame using the index
            # Use update() to fill NaNs based on index - more robust than direct assignment if lengths differ
            df_windows['scraped_markdown'].update(markdown_series)
            df_windows['scraped_json'].update(json_series)
            logging.info(f"Finished scraping for {len(markdown_series)} articles.")
        else:
            logging.info("No Windows-based KBs required scraping (markdown already present or no URL).")

        # 2. Identify rows that NEED summary generation (summary is NaN or empty string)
        needs_summary_mask = (
            df_windows["summary"].isna()
            | df_windows["summary"].fillna("").astype(str).str.strip().eq("")
        )
        df_to_summarize = df_windows[needs_summary_mask]

        if not df_to_summarize.empty:
            logging.info(f"Generating summaries for {df_to_summarize.shape[0]} Windows-based KBs with missing summaries.")

            # Use potentially updated scraped_markdown or original text
            texts_for_summary = df_to_summarize.apply(get_text_for_summary, axis=1).tolist()

            # Generate summaries
            generated_summaries = loop.run_until_complete(
                generate_summaries(texts_for_summary)
            )

            # Create Series with generated summaries, indexed correctly
            summary_series = pd.Series(generated_summaries, index=df_to_summarize.index)

            # Update the main DataFrame's summary column
            # Using .loc ensures we only update the rows that needed summarizing
            df_windows.loc[needs_summary_mask, 'summary'] = summary_series
            logging.info(f"Finished generating summaries for {len(summary_series)} articles.")

        else:
            logging.info("No Windows-based KBs required summary generation.")

        # --- End: Modified Scraping and Summary Generation Logic ---

        # Add update package URL for all Windows KB articles after processing
        df_windows["update_package_url"] = df_windows["kb_id"].apply(_create_kb_catalog_url)

        # Final sort before finishing the Windows section (optional, but good practice)
        df_windows.sort_values(by="kb_id", ascending=True, inplace=True)

        print(f"Total Windows-based KBs transformed: {df_windows.shape[0]}")

    else:
        # df_windows = pd.DataFrame(columns=master_columns) # Keep this commented if raising error
        print("No Windows KB articles to transform.")
        # Consider if raising an error is still desired, or just return empty/edge data
        # raise ValueError("No Windows KB articles to transform.")  # Kept as per original logic

    # Process Edge KB articles (logic unchanged)
    if kb_articles_edge:
        df_edge = pd.DataFrame(
            kb_articles_edge  # Assume columns are consistent or handled later
        )
        # Ensure necessary columns exist for processing, add missing ones
        for col in ['kb_id', 'cve_ids', 'published', 'id', 'title', 'text', 'article_url']:  # Add minimum needed? Or all master? Add 'id' at least
            if col not in df_edge.columns:
                df_edge[col] = pd.NA

        df_edge["kb_id"] = df_edge["kb_id"].apply(normalize_mongo_kb_id)
        df_edge["kb_id"] = df_edge["kb_id"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) == 1 else ""
        )
        # Simplified Edge deduplication - kept as is
        df_edge = df_edge.sort_values(
            by="cve_ids",
            key=lambda s, _=None: s.isna()  # Sorts NaNs last
        )
        df_edge = df_edge.drop_duplicates(subset=["kb_id"], keep="first")

        # Ensure all master columns exist before final processing
        df_edge = validate_and_adjust_columns(df_edge, master_columns)  # This should handle adding missing master cols

        df_edge["node_label"] = "KBArticle"
        df_edge["published"] = pd.to_datetime(df_edge["published"], errors='coerce')
        df_edge["excluded_embed_metadata_keys"] = [
            [] for _ in range(len(df_edge))
        ]
        df_edge["excluded_embed_metadata_keys"] = df_edge[
            "excluded_embed_metadata_keys"
        ].apply(_create_excluded_metadata_keys)

        # Initialize metadata with etl_processing_status
        df_edge["metadata"] = [
            {
                "etl_processing_status": {
                    "document_processed": True,
                    "entities_extracted": False,
                    "graph_prepared": False,
                    "vector_prepared": False,
                    "last_processed_at": None,
                    "processing_version": "1.0",
                }
            }
            for _ in range(len(df_edge))
        ]

        # Initialize summary and update_package_url for Edge (as per original logic)
        df_edge["summary"] = "" if "summary" not in df_edge.columns else df_edge["summary"].fillna("")
        df_edge["update_package_url"] = "" if "update_package_url" not in df_edge.columns else df_edge["update_package_url"].fillna("")

        df_edge.sort_values(by="kb_id", ascending=True, inplace=True)
        print(f"Total Edge-based KBs transformed: {df_edge.shape[0]}")

    else:
        df_edge = pd.DataFrame(columns=master_columns)
        print("No Edge-based KB articles to transform.")

    # Combine Windows and Edge KB articles (logic unchanged)
    dfs_to_concat = []
    # Ensure dataframes exist and have columns before applying dtypes/concatenating
    if 'df_windows' in locals() and not df_windows.empty:
        logging.info("Applying dtypes to Windows DataFrame")
        for col, dtype in dtypes.items():
            if col in df_windows.columns:
                try:
                    # Handle object dtype specifically for lists/dicts if needed
                    if dtype == 'object' and isinstance(df_windows[col].iloc[0], (list, dict)):
                        continue  # Avoid trying to astype complex objects directly if already correct
                    df_windows[col] = df_windows[col].astype(dtype)
                except (ValueError, TypeError, pd.errors.IntCastingNaNError) as e:
                    logging.warning(f"Warning: Could not convert column {col} to {dtype} in Windows DF: {e}. Trying astype(str).astype(dtype) for safety.")
                    try:
                        # Attempt conversion via string for robustness (e.g., float -> Int64 with NA)
                        if dtype == 'str':
                            df_windows[col] = df_windows[col].astype(str)
                        # Add other specific safe conversions if needed
                    except Exception as e_inner:
                        logging.error(f"Secondary conversion attempt failed for {col}: {e_inner}")

        dfs_to_concat.append(df_windows)

    if 'df_edge' in locals() and not df_edge.empty:
        logging.info("Applying dtypes to Edge DataFrame")
        for col, dtype in dtypes.items():
            if col in df_edge.columns:
                try:
                    if dtype == 'object' and isinstance(df_edge[col].iloc[0], (list, dict)):
                        continue
                    df_edge[col] = df_edge[col].astype(dtype)
                except (ValueError, TypeError, pd.errors.IntCastingNaNError) as e:
                    logging.warning(f"Warning: Could not convert column {col} to {dtype} in Edge DF: {e}. Trying astype(str).astype(dtype) for safety.")
                    try:
                        if dtype == 'str':
                            df_edge[col] = df_edge[col].astype(str)
                    except Exception as e_inner:
                        logging.error(f"Secondary conversion attempt failed for {col}: {e_inner}")

        dfs_to_concat.append(df_edge)

    # Only concatenate if we have DataFrames to combine
    if dfs_to_concat:
        kb_articles_combined_df = pd.concat(
            dfs_to_concat, axis=0, ignore_index=True, copy=True  # copy=True can prevent SettingWithCopyWarning downstream
        )

        # Rename 'id' to 'node_id' AFTER concatenation
        if 'id' in kb_articles_combined_df.columns:
            kb_articles_combined_df = kb_articles_combined_df.rename(
                columns={"id": "node_id"}
            )
        else:
            logging.warning("Column 'id' not found for renaming to 'node_id'.")
            # Add node_id if critical? Or assume validate_and_adjust_columns handled it?
            if 'node_id' not in kb_articles_combined_df.columns:
                kb_articles_combined_df['node_id'] = pd.NA  # Or generate UUIDs?

        # Final duplicate check (kept as is from original code) - Ensure columns exist
        # final_dedup_cols = ["build_number", "kb_id", "published", "product_build_id"]
        # if all(col in kb_articles_combined_df.columns for col in final_dedup_cols):
        #     # Convert build_number to tuple for comparison (if it's a list)
        #     kb_articles_combined_df["build_number_tuple"] = (
        #         kb_articles_combined_df["build_number"].apply(
        #             lambda x: tuple(x) if isinstance(x, list) else x
        #         )
        #     )

        #     # Drop duplicates keeping first occurrence
        #     kb_articles_combined_df = kb_articles_combined_df.drop_duplicates(
        #         subset=[
        #             "build_number_tuple",
        #             "kb_id",
        #             "published",
        #             "product_build_id",
        #         ],
        #         keep="first",
        #     )

        #     # Remove the temporary tuple column
        #     kb_articles_combined_df = kb_articles_combined_df.drop(
        #         columns=["build_number_tuple"]
        #     )
        # else:
        #     missing_cols = [col for col in final_dedup_cols if col not in kb_articles_combined_df.columns]
        #     logging.warning(f"Skipping final deduplication step due to missing columns: {missing_cols}")

        print(
            "Total KB articles transformed:"
            f" {kb_articles_combined_df.shape[0]}"
        )
        return kb_articles_combined_df
    else:
        print("No KB articles transformed (neither Windows nor Edge data processed).")
        # Return empty DataFrame matching expected columns, using node_id
        final_columns = [col if col != 'id' else 'node_id' for col in master_columns]
        return pd.DataFrame(columns=final_columns)


def process_downloadable_packages(
    packages: Union[str, List[Dict[str, Any]], None],
) -> str:
    if packages is None or (
        isinstance(packages, str) and packages.strip() == ""
    ):
        return json.dumps([], default=custom_json_serializer)

    if isinstance(packages, str):
        try:
            packages = json.loads(packages)
        except json.JSONDecodeError:
            return json.dumps([], default=custom_json_serializer)

    if not isinstance(packages, list):
        return json.dumps([], default=custom_json_serializer)

    for package in packages:
        if isinstance(package, dict):
            for key, value in package.items():
                if isinstance(value, datetime):
                    package[key] = value.isoformat()

    return json.dumps(packages, default=custom_json_serializer)


def transform_update_packages(
    update_packages: List[Dict[str, Any]],
) -> pd.DataFrame:
    mapping_types = {"security_hotpatch_update": "security_hotpatch"}
    # clean up document dict from mongo to align with data models
    if update_packages:
        df = pd.DataFrame(
            update_packages, columns=list(update_packages[0].keys())
        )
        df["package_type"] = (
            df["package_type"].str.lower().str.replace(" ", "_")
        )
        df["package_type"] = df["package_type"].apply(
            lambda x: _map_package_type(x, mapping_types)
        )
        # df["downloadable_packages"] = df["downloadable_packages"].apply(
        #     process_downloadable_packages
        # )
        df["node_label"] = "UpdatePackage"
        df["published"] = pd.to_datetime(df["published"])
        df["excluded_embed_metadata_keys"] = [[] for _ in range(len(df))]
        df["excluded_embed_metadata_keys"] = df[
            "excluded_embed_metadata_keys"
        ].apply(_create_excluded_update_metadata_keys)
        df = df.rename(columns={"id": "node_id"})
        # print(df["downloadable_packages"])
        print(f"Total Update Packages transformed: {df.shape[0]}")
        # print(f"Columns: {df.columns}")
        return df
    else:
        print("No Update Packages to transform.")
        return pd.DataFrame(
            columns=[
                "node_id",
                "package_type",
                "node_label",
                "published",
                "excluded_embed_metadata_keys",
                "downloadable_packages",
            ]
        )


def convert_to_list(value):
    if isinstance(value, str):
        return [value]
    elif isinstance(value, list):
        return value


def make_json_safe_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make metadata dictionary JSON-safe while preserving dictionary structure.
    Converts any non-JSON-serializable values to appropriate JSON-safe formats.

    Args:
        metadata: Dictionary containing metadata fields

    Returns:
        Dictionary with JSON-safe values but still in dictionary format
    """
    if not isinstance(metadata, dict):
        return metadata

    safe_metadata = {}
    for key, value in metadata.items():
        try:
            if isinstance(value, dict):
                # Recursively process nested dictionaries
                safe_metadata[key] = make_json_safe_metadata(value)
            elif isinstance(value, (datetime, np.datetime64)):
                # Convert datetime to ISO format string
                safe_metadata[key] = value.isoformat()
            elif isinstance(value, (list, tuple)):
                # Handle lists/tuples
                safe_metadata[key] = [
                    (
                        make_json_safe_metadata(item)
                        if isinstance(item, dict)
                        else item
                    )
                    for item in value
                ]
            elif isinstance(
                value, (float, np.float32, np.float64)
            ) and np.isnan(value):
                # Handle NaN values
                safe_metadata[key] = None
            elif isinstance(value, (np.int64, np.int32)):
                # Convert numpy integers to Python integers
                safe_metadata[key] = int(value)
            elif isinstance(value, (np.bool_)):
                # Convert numpy booleans to Python booleans
                safe_metadata[key] = bool(value)
            else:
                # Try json serialization to verify it's safe
                json.dumps(value)
                safe_metadata[key] = value
        except (TypeError, OverflowError, ValueError):
            # If value can't be JSON serialized, convert to string
            safe_metadata[key] = str(value)

    return safe_metadata


def remove_generic_text(text, threshold=80, max_match_length=500):

    # initial_char_count = len(text)
    problematic_pattern = (
        r"This metric describes the conditions beyond the attacker's control"
        r" that must exist in order to exploit the vulnerability. Such"
        r" conditions may require the collection of more information about the"
        r" target or computational exceptions. The assessment of this metric"
        r" excludes any requirements for user interaction in order to exploit"
        r" the vulnerability. If a specific configuration is required for an"
        r" attack to succeed, the Base metrics should be scored assuming the"
        r" vulnerable component is in that configuration."
    )
    icon_pattern = r"[^\w\s]+\s+Subscribe\s+RSS\s+PowerShell\s+[^\w\s]+\s+API"
    generic_text_patterns = [
        (
            r"This metric reflects the context by which vulnerability"
            r" exploitation is possible. The Base Score increases the more"
            r" remote \(logically, and physically\) an attacker can be in"
            r" order to exploit the vulnerable component."
        ),
        problematic_pattern,
        (
            r"This metric describes the level of privileges an attacker must"
            r" possess before successfully exploiting the vulnerability."
        ),
        (
            r"This metric captures the requirement for a user, other than the"
            r" attacker, to participate in the successful compromise the"
            r" vulnerable component. This metric determines whether the"
            r" vulnerability can be exploited solely at the will of the"
            r" attacker, or whether a separate user \(or user-initiated"
            r" process\) must participate in some manner."
        ),
        (
            r"Does a successful attack impact a component other than the"
            r" vulnerable component\? If so, the Base Score increases and the"
            r" Confidentiality, Integrity and Authentication metrics should be"
            r" scored relative to the impacted component."
        ),
        (
            r"This metric measures the impact to the confidentiality of the"
            r" information resources managed by a software component due to a"
            r" successfully exploited vulnerability. Confidentiality refers to"
            r" limiting information access and disclosure to only authorized"
            r" users, as well as preventing access by, or disclosure to,"
            r" unauthorized ones."
        ),
        (
            r"This metric measures the impact to integrity of a successfully"
            r" exploited vulnerability. Integrity refers to the"
            r" trustworthiness and veracity of information."
        ),
        (
            r"This metric measures the impact to the availability of the"
            r" impacted component resulting from a successfully exploited"
            r" vulnerability. It refers to the loss of availability of the"
            r" impacted component itself, such as a networked service \(e.g.,"
            r" web, database, email\). Since availability refers to the"
            r" accessibility of information resources, attacks that consume"
            r" network bandwidth, processor cycles, or disk space all impact"
            r" the availability of an impacted component."
        ),
        (
            r"This metric measures the likelihood of the vulnerability being"
            r" attacked, and is typically based on the current state of"
            r" exploit techniques, public availability of exploit code, or"
            r" active, 'in-the-wild' exploitation."
        ),
        (
            r"The Remediation Level of a vulnerability is an important factor"
            r" for prioritization. The typical vulnerability is unpatched when"
            r" initially published. Workarounds or hotfixes may offer interim"
            r" remediation until an official patch or upgrade is issued. Each"
            r" of these respective stages adjusts the temporal score"
            r" downwards, reflecting the decreasing urgency as remediation"
            r" becomes final."
        ),
        (
            r"This metric measures the degree of confidence in the existence"
            r" of the vulnerability and the credibility of the known technical"
            r" details. Sometimes only the existence of vulnerabilities are"
            r" publicized, but without specific details. For example, an"
            r" impact may be recognized as undesirable, but the root cause may"
            r" not be known. The vulnerability may later be corroborated by"
            r" research which suggests where the vulnerability may lie, though"
            r" the research may not be certain. Finally, a vulnerability may"
            r" be confirmed through acknowledgement by the author or vendor of"
            r" the affected technology. The urgency of a vulnerability is"
            r" higher when a vulnerability is known to exist with certainty."
            r" This metric also suggests the level of technical knowledge"
            r" available to would-be attackers."
        ),
        r"New\s+On this page\s+\ue70d",
        icon_pattern,
    ]
    patterns_found = 0
    missing_patterns = []
    modified_text = text
    # Attempt to remove exact matches with regex
    for pattern in generic_text_patterns:
        if re.search(pattern, modified_text):
            modified_text = re.sub(pattern, "", modified_text)
            patterns_found += 1
            # print(f"Pattern found and removed with regex: {pattern[:30]}... | modified_text len: {len(modified_text)}")
        else:
            missing_patterns.append(pattern)
            # print(f"Pattern not found with regex: {pattern[:30]}...")

    # Attempt fuzzy matching for missing patterns
    if missing_patterns:
        for pattern in missing_patterns:
            if pattern == problematic_pattern:
                # Try matching smaller segments of the problematic pattern
                segments = problematic_pattern.split(". ")
                for segment in segments:
                    best_match, score = process.extractOne(
                        segment, [modified_text], scorer=fuzz.partial_ratio
                    )

                    # Only replace if the score is high, the match length is reasonable, and it’s not too broad
                    if (
                        score >= threshold
                        and len(best_match) < max_match_length
                        and len(best_match) < len(modified_text) * 0.5
                    ):
                        modified_text = modified_text.replace(best_match, "")
                        patterns_found += 1
                        # print(f"Pattern segment removed using fuzzy matching: {segment[:30]}... (Score: {score}) | modified_text len: {len(modified_text)}")
                    else:
                        # print(f"Pattern segment skipped in fuzzy matching due to length or low score: {segment[:30]}... (Score: {score})")
                        pass

            elif (
                pattern == icon_pattern
            ):  # Fuzzy matching for the icon pattern
                if re.search(icon_pattern, modified_text):
                    modified_text = re.sub(icon_pattern, "", modified_text)
                    patterns_found += 1
                    # print(f"Icon pattern found and removed with regex | modified_text len: {len(modified_text)}")
                else:
                    # Fallback to fuzzy matching if regex fails
                    # print("Icon pattern not matched by regex; attempting fuzzy matching for each segment.")
                    segments = ["Subscribe", "RSS", "PowerShell", "API"]
                    for segment in segments:
                        best_match, score = process.extractOne(
                            segment, [modified_text], scorer=fuzz.partial_ratio
                        )
                        # Only replace if the match score is high and the length is reasonable
                        if (
                            score >= threshold
                            and len(best_match) < max_match_length
                        ):
                            modified_text = modified_text.replace(
                                best_match, ""
                            )
                            patterns_found += 1
                            # print(f"Icon pattern segment removed using fuzzy matching: {segment} (Score: {score}) | modified_text len: {len(modified_text)}")
                        else:
                            # print(f"Icon pattern segment skipped in fuzzy matching due to length or low score: {segment} (Score: {score})")
                            pass

    final_char_count = len(modified_text)
    logging.debug(f"Final character count: {final_char_count}")
    # Return the modified text and a summary of patterns found
    return (
        modified_text.strip(),
        patterns_found,
        len(generic_text_patterns) - patterns_found,
    )


def extract_cve_category_from_description(description: str) -> str:
    """
    Extract CVE category from NVD description text with enhanced pattern matching.
    Returns the most severe/specific category if multiple are found.
    """
    # Handle None, NA, and empty strings
    if pd.isna(description) or description is None or description == "":
        return "NC"

    description = str(description).lower()

    # Define keywords with variations for each category
    category_patterns = {
        "rce": [
            "remote code execution",
            "remote execution",
            "arbitrary code execution",
            "code execution",
            "command execution",
        ],
        "privilege_elevation": [
            "elevation of privilege",
            "privilege elevation",
            "escalation of privilege",
            "privilege escalation",
        ],
        "dos": [
            "denial of service",
            "denial-of-service",
            "service denial",
            "resource exhaustion",
        ],
        "disclosure": [
            "information disclosure",
            "information leak",
            "data disclosure",
            "memory leak",
            "sensitive information",
        ],
        "tampering": [
            "tampering",
            "data manipulation",
            "unauthorized modification",
        ],
        "spoofing": ["spoofing", "impersonation", "authentication bypass"],
        "feature_bypass": [
            "security feature bypass",
            "security bypass",
            "protection bypass",
        ],
        "availability": ["availability", "system crash", "system hang"],
        "mitm": [
            "man in the middle",
            "man in the middle attack",
            "mitm",
            "middle man attack",
            "eavesdropping attack",
            "interception attack",
        ],
    }

    # Priority order for categories (most severe/specific first)
    priority_order = [
        "rce",
        "privilege_elevation",
        "disclosure",
        "dos",
        "tampering",
        "spoofing",
        "feature_bypass",
        "availability",
        "mitm",
    ]

    # Find all matching categories
    found_categories = set()
    for category, patterns in category_patterns.items():
        if any(pattern in description for pattern in patterns):
            found_categories.add(category)

    # Return highest priority category if multiple found
    for category in priority_order:
        if category in found_categories:
            return category

    return "NC"


# Begin MSRC transformer =====================================


def _prepare_base_dataframe(
    msrc_posts: List[Dict[str, Any]], metadata_fields_to_move: List[str]
) -> pd.DataFrame:
    """Prepare the initial dataframe from MSRC posts."""
    df = pd.DataFrame(msrc_posts, columns=list(msrc_posts[0].keys()))
    for field in metadata_fields_to_move:
        df[field] = df["metadata"].apply(
            lambda x, field=field: x.get(field, None)
        )
    return df


def _apply_common_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """Apply transformations common to both new and pre-processed records."""
    df = df.rename(columns={"id_": "node_id", "impact_type": "cve_category"})

    # Try direct assignment instead of fillna
    # df.loc[df["cve_category"].isna(), "cve_category"] = "NC"
    cve_category_choices = {
        "Tampering": "tampering",
        "Spoofing": "spoofing",
        "Availability": "availability",
        "Elevation of Privilege": "privilege_elevation",
        "Denial of Service": "denial_of_service",
        "Information Disclosure": "disclosure",
        "Remote Code Execution": "remote_code_execution",
        "Security Feature Bypass": "feature_bypass",
        "No Category": "NC",
        "None": "none",
    }
    # Map display values to their corresponding keys
    # For example, "Information Disclosure" -> "disclosure"
    df["cve_category"] = (
        df["cve_category"].map(cve_category_choices).fillna("NC")
    )
    df["severity_type"] = df["severity_type"].str.lower()
    df["severity_type"] = df["severity_type"].fillna("NST")
    # df["metadata"] = df["metadata"].apply(make_json_safe_metadata)
    df["kb_ids"] = df["kb_ids"].apply(normalize_mongo_kb_id)
    df["kb_ids"] = df["kb_ids"].apply(lambda x: sorted(x, reverse=True))
    df["product_build_ids"] = df["product_build_ids"].apply(convert_to_list)
    df["node_label"] = "MSRCPost"
    df["published"] = pd.to_datetime(df["published"])
    df["excluded_embed_metadata_keys"] = [[] for _ in range(len(df))]
    df["excluded_embed_metadata_keys"] = df[
        "excluded_embed_metadata_keys"
    ].apply(
        lambda x: list(
            set(x if isinstance(x, list) else [])
            | {
                "source",
                "description",
                "product_build_ids",
                "kb_ids",
                "build_numbers",
                "node_label",
                "patterns_found",
                "patterns_missing",
            }
        )
    )
    df[["text", "patterns_found", "patterns_missing"]] = df["text"].apply(
        lambda x: pd.Series(remove_generic_text(x))
    )
    df = df.drop(["patterns_found", "patterns_missing"], axis=1)

    return df


def _extract_nvd_properties(metadata_dict: Dict) -> Dict[str, Any]:
    """Extract NVD properties from metadata dictionary."""
    nvd_properties = {}
    if not isinstance(metadata_dict, dict):
        return nvd_properties

    # Base properties that get prefixed
    base_properties = [
        "attack_complexity",
        "attack_vector",
        "availability",
        "base_score",
        "base_score_num",
        "base_score_rating",
        "confidentiality",
        "exploitability_score",
        "impact_score",
        "integrity",
        "privileges_required",
        "scope",
        "user_interaction",
        "vector",
    ]

    # Add prefixed properties (nist_, cna_, adp_)
    for prefix in ["nist_", "cna_", "adp_"]:
        nvd_properties.update({
            f"{prefix}{prop}": metadata_dict.get(f"{prefix}{prop}")
            for prop in base_properties
        })

    # Add non-prefixed properties
    nvd_properties.update({
        "nvd_description": metadata_dict.get("nvd_description"),
        "nvd_published_date": metadata_dict.get("nvd_published_date"),
        "cwe_id": metadata_dict.get("cwe_id"),
        "cwe_name": metadata_dict.get("cwe_name"),
        "cwe_source": metadata_dict.get("cwe_source"),
        "cwe_url": metadata_dict.get("cwe_url"),
    })

    return nvd_properties


def transform_msrc_posts(
    msrc_posts: List[Dict[str, Any]], process_all: bool = False
) -> Optional[pd.DataFrame]:
    """
    Transforms MSRC posts, scrapes NVD data, and integrates it,
    handling both new and pre-processed records efficiently.
    """
    logging.info(f"Starting MSRC transform. Process All: {process_all}")
    # metadata_fields_to_move remains the same

    if not msrc_posts:
        logging.warning("No MSRC Posts provided to transform.")
        return None

    # 1. Create initial DataFrame and ensure metadata structure
    # (Keep your existing logic for _prepare_base_dataframe and metadata init)
    metadata_fields_to_move = [
        "revision", "title", "description", "source", "severity_type",
        "post_type", "post_id", "summary", "build_numbers", "published",
        "product_build_ids", "collection", "impact_type",
    ]
    try:
        # Assuming _prepare_base_dataframe handles moving fields and returns df
        df = _prepare_base_dataframe(msrc_posts, metadata_fields_to_move)
        if 'post_id' not in df.columns:
            logging.error("Initial DataFrame missing required 'post_id' column after _prepare_base_dataframe.")
            return None
    except Exception as e:
        logging.error(f"Error during _prepare_base_dataframe: {e}", exc_info=True)
        return None

    # Ensure metadata column exists and is dict-like
    if 'metadata' not in df.columns:
        df['metadata'] = [{} for _ in range(len(df))]
    else:
        df['metadata'] = df['metadata'].apply(lambda x: x if isinstance(x, dict) else {})
        # Ensure etl_processing_status exists within metadata
        df['metadata'] = df['metadata'].apply(
            lambda x: {**x, 'etl_processing_status': x.get('etl_processing_status', {})}
        )

    # 2. Identify records needing NVD extraction
    if process_all:
        # If processing all, mark all rows for NVD extraction
        indices_to_scrape = df.index
        logging.info(f"Processing all {len(df)} records as requested.")
    else:
        # If not processing all, identify only those not marked as 'nvd_extracted'
        def needs_nvd_extraction(meta):
            # Process if metadata invalid
            if not isinstance(meta, dict): return True  # noqa: E701
            status = meta.get('etl_processing_status', {})
            # Process if status isn't a dict or nvd_extracted is not True
            return not (isinstance(status, dict) and status.get('nvd_extracted') is True)

        needs_nvd_mask = df['metadata'].apply(needs_nvd_extraction)
        indices_to_scrape = df[needs_nvd_mask].index
        logging.info(f"Identified {len(indices_to_scrape)} records needing NVD extraction (process_all=False).")

    # Store indices of records originally identified as new/needing processing
    # This helps determine the final return value when process_all=False
    original_new_indices = indices_to_scrape if not process_all else df.index

    # 3. Apply initial transformations (only if needed, maybe part of _prepare?)
    # Assuming _apply_common_transformations should happen regardless of NVD status
    try:
        df = _apply_common_transformations(df)
    except Exception as e:
        logging.error(f"Error during _apply_common_transformations: {e}", exc_info=True)
        # Decide whether to continue or return

    # --- NVD Extraction Section ---
    enriched_subset = pd.DataFrame()
    nvd_extractor = None

    # Only proceed if there are records identified for scraping
    if not indices_to_scrape.empty:
        # Create subset DF to pass to extractor
        records_to_scrape_df = df.loc[indices_to_scrape].copy()

        # Set up NVD extraction parameters
        num_cves = len(records_to_scrape_df)
        # Use a sensible default target time per CVE
        scraping_params = ScrapingParams.from_target_time(
            num_cves=num_cves, target_time_per_cve=5.0
        )
        estimated_minutes = scraping_params.estimate_total_time(num_cves) / 60
        logging.info(
            f"Estimated NVD processing time for {num_cves} records:"
            f" {estimated_minutes:.1f} minutes (approx)"
        )
        all_nvd_props = NVDDataExtractor.get_all_possible_columns()
        logging.debug(f"Requesting NVD Extractor to extract properties: {all_nvd_props}")
        # Instantiate the NVD Extractor
        # No need to specify properties_to_extract, class handles it
        nvd_extractor = NVDDataExtractor(
            properties_to_extract=all_nvd_props,
            max_records=None,
            scraping_params=scraping_params,
            headless=True,
            window_size=(1240, 1080),
            show_progress=True,
        )

        try:
            # Call augment_dataframe on the SUBSET of records needing scraping
            enriched_subset = nvd_extractor.augment_dataframe(
                df=records_to_scrape_df,
                url_column="post_id",
                batch_size=50,
            )
            logging.info(f"NVD extraction completed. Received {len(enriched_subset)} enriched records.")

            # *** CRITICAL STEP: Merge enriched data back into the main DataFrame ***
            if not enriched_subset.empty:
                missing_cols = [col for col in enriched_subset.columns if col not in df.columns]
                if missing_cols:
                    logging.debug(f"Initializing {len(missing_cols)} missing columns in main 'df' before update: {missing_cols}")
                    for col in missing_cols:
                        # Initialize with None or an appropriate dtype's null value
                        if pd.api.types.is_numeric_dtype(enriched_subset[col]):
                            df[col] = pd.NA  # Use pandas NA for nullable numerics
                        elif pd.api.types.is_datetime64_any_dtype(enriched_subset[col]):
                            df[col] = pd.NaT
                        else:
                            df[col] = None
                # Use update() to overwrite values in df based on index from enriched_subset
                logging.info(f"Updating main DataFrame with {len(enriched_subset)} enriched rows.")
                df.update(enriched_subset)
                logging.debug("Main DataFrame update complete.")
                if 'nvd_description' in df.columns:
                    logging.debug(f"Column 'nvd_description' successfully updated/added to main 'df'. Sample value: {df['nvd_description'].iloc[0]}")
                else:
                    logging.error("Column 'nvd_description' STILL MISSING after df.update(). This is unexpected.")
            else:
                logging.warning("NVD augment_dataframe returned empty results. No updates applied to main DataFrame.")

        except Exception as e:
            logging.error(f"Error during NVD data extraction: {e}", exc_info=True)
            # Decide if you want to re-raise or just log and continue
            # raise # Option to stop pipeline on NVD failure
        finally:
            if nvd_extractor:
                nvd_extractor.cleanup()  # Ensure cleanup happens

    else:
        logging.info("No records required NVD scraping.")

    # --- Post-NVD Processing (Applied to the main DataFrame 'df') ---
    # Only apply to rows that were intended for scraping (indices_to_scrape)
    # and successfully enriched (present in enriched_subset index)

    if not enriched_subset.empty:  # Check if any enrichment actually happened
        enriched_indices = enriched_subset.index

        logging.info(f"Performing post-NVD processing on {len(enriched_indices)} rows.")

        # Example: Update CVE category (ensure column exists first)
        if 'cve_category' not in df.columns:
            df['cve_category'] = 'Undetermined'
        if 'nvd_description' not in df.columns:
            logging.warning("Cannot update CVE category: 'nvd_description' column not found in DataFrame.")
        else:
            # Apply only to the rows that were enriched
            category_update_mask = df.index.isin(enriched_indices) & df['cve_category'].isin(['NC', '', 'Undetermined', None])
            df.loc[category_update_mask, 'cve_category'] = (
                df.loc[category_update_mask, 'nvd_description'].fillna('').apply(
                    extract_cve_category_from_description
                )
            )
            logging.info(f"Attempted CVE category update for {category_update_mask.sum()} enriched rows.")

        # Example: Update processing status in metadata
        current_time_iso = datetime.now(timezone.utc).isoformat()

        def update_status(metadata_dict):
            # Ensure input is a mutable dictionary
            if not isinstance(metadata_dict, dict):
                # Handle case where metadata might not be a dict (though earlier steps try to ensure it is)
                logging.warning(f"Metadata is not a dict: {type(metadata_dict)}. Cannot update status.")
                return {'etl_processing_status': {'nvd_extracted': True, 'last_processed_at': current_time_iso}}

            # Make a copy to avoid modifying the original dict directly if it's shared
            # meta_copy = metadata_dict.copy() # Shallow copy ok if etl_processing_status is top-level

            status = metadata_dict.get('etl_processing_status', {})
            if not isinstance(status, dict):
                status = {}  # Initialize if not a dict

            # Update the status dict
            status.update({
                'nvd_extracted': True,
                'last_processed_at': current_time_iso
            })

            # Put the updated status back into the main metadata dict
            metadata_dict['etl_processing_status'] = status
            return metadata_dict

        # Apply the status update ONLY to the rows that were successfully enriched
        df.loc[enriched_indices, 'metadata'] = df.loc[enriched_indices, 'metadata'].apply(update_status)
        logging.info(f"Updated metadata status for {len(enriched_indices)} enriched rows.")

        # Example: Clean up temporary or unwanted columns if they exist
        if 'impact_type' in df.columns:
            df.loc[enriched_indices, 'impact_type'] = None  # Or df.drop('impact_type', ...)
            logging.info("Cleaned up 'impact_type' column for enriched rows.")
        # Add similar cleanup for other columns if needed

    else:
        logging.info("Skipping post-NVD processing as no records were enriched.")

    # 4. Final Return Logic
    df.sort_values(by="post_id", ascending=True, inplace=True)  # Sort final DF

    if process_all:
        # Return the full updated DataFrame
        logging.info(f"Returning full transformed DataFrame with {len(df)} records (process_all=True).")
        return df
    else:
        # Return only the records that were originally identified as needing processing
        # Ensure these records reflect the updates made during enrichment
        result_df = df.loc[original_new_indices].copy()  # Select updated rows from main df
        logging.info(f"Returning {len(result_df)} newly processed/updated records (process_all=False).")
        if result_df.empty:
            return None  # Return None if no new records were processed
        return result_df


# End MSRC transformer ========================================

# Start Patch Transformer =====================================


nlp = spacy.load("en_core_web_lg")


def construct_natural_subject(noun_chunks, keywords, timestamp=None):
    """
    Construct a natural-sounding subject from noun chunks and keywords using spaCy.
    Format: <best_candidate>_<product_mention>_<timestamp>
    """
    # First convert string representations of lists to actual lists if needed
    if pd.isna(noun_chunks):
        noun_chunks = []
    elif isinstance(noun_chunks, str):
        try:
            # Handle string representation of list
            noun_chunks = ast.literal_eval(noun_chunks)
        except (ValueError, SyntaxError):
            noun_chunks = []

    if pd.isna(keywords):
        keywords = []
    elif isinstance(keywords, str):
        try:
            # Handle string representation of list
            keywords = ast.literal_eval(keywords)
        except (ValueError, SyntaxError):
            keywords = []

    # If we have no data, return empty string
    if not noun_chunks and not keywords:
        return ""

    # List of products we're watching for, from general to specific
    product_patterns = [
        # Windows 11
        (r"windows\s*11\b", "Windows 11"),
        (r"windows\s*11.*?24h2\b", "Windows 11 Version 24H2"),
        (r"windows\s*11.*?23h2\b", "Windows 11 Version 23H2"),
        (r"windows\s*11.*?22h2\b", "Windows 11 Version 22H2"),
        (r"windows\s*11.*?21h2\b", "Windows 11 Version 21H2"),
        (
            r"windows\s*11.*?(x64|64|64[\s-]*bit)\b",
            "Windows 11 for x64-based Systems",
        ),
        (
            r"windows\s*11.*?24h2.*?(x64|64|64[\s-]*bit)\b",
            "Windows 11 Version 24H2 for x64-based Systems",
        ),
        (
            r"windows\s*11.*?23h2.*?(x64|64|64[\s-]*bit)\b",
            "Windows 11 Version 23H2 for x64-based Systems",
        ),
        (
            r"windows\s*11.*?22h2.*?(x64|64|64[\s-]*bit)\b",
            "Windows 11 Version 22H2 for x64-based Systems",
        ),
        (
            r"windows\s*11.*?21h2.*?(x64|64|64[\s-]*bit)\b",
            "Windows 11 Version 21H2 for x64-based Systems",
        ),
        # Windows 10
        (r"windows\s*10\b", "Windows 10"),
        (r"windows\s*10.*?23h2\b", "Windows 10 Version 23H2"),
        (r"windows\s*10.*?22h2\b", "Windows 10 Version 22H2"),
        (r"windows\s*10.*?21h2\b", "Windows 10 Version 21H2"),
        (
            r"windows\s*10.*?(x64|64|64[\s-]*bit)\b",
            "Windows 10 for x64-based Systems",
        ),
        (
            r"windows\s*10.*?(32|32[\s-]*bit)\b",
            "Windows 10 for 32-bit Systems",
        ),
        (
            r"windows\s*10.*?23h2.*?(x64|64|64[\s-]*bit)\b",
            "Windows 10 Version 23H2 for x64-based Systems",
        ),
        (
            r"windows\s*10.*?22h2.*?(x64|64|64[\s-]*bit)\b",
            "Windows 10 Version 22H2 for x64-based Systems",
        ),
        (
            r"windows\s*10.*?21h2.*?(x64|64|64[\s-]*bit)\b",
            "Windows 10 Version 21H2 for x64-based Systems",
        ),
        (
            r"windows\s*10.*?23h2.*?(32|32[\s-]*bit)\b",
            "Windows 10 Version 23H2 for 32-bit Systems",
        ),
        (
            r"windows\s*10.*?22h2.*?(32|32[\s-]*bit)\b",
            "Windows 10 Version 22H2 for 32-bit Systems",
        ),
        (
            r"windows\s*10.*?21h2.*?(32|32[\s-]*bit)\b",
            "Windows 10 Version 21H2 for 32-bit Systems",
        ),
        # Edge
        (r"(?:microsoft\s*)?edge\b", "Microsoft Edge"),
        (
            r"(?:microsoft\s*)?edge.*?chromium",
            "Microsoft Edge (Chromium-based)",
        ),
        (
            r"(?:microsoft\s*)?edge.*?extended",
            "Microsoft Edge (Chromium-based) Extended Stable",
        ),
    ]

    def score_candidate_subject(text):
        words = text.lower().split()
        score = 0

        # Penalize product mentions to avoid them in initial selection
        text_lower = text.lower()
        for pattern, _ in product_patterns:
            if re.search(pattern, text_lower):
                score -= 30  # Significant penalty for product mentions

        # Score based on length (but not too much)
        score += min(len(words), 8) * 5

        # Bonus for action words/verbs
        action_words = {
            # High-importance terms (score ~20)
            'vulnerability': 20,  # signals security exposure
            'exploit': 20,  # signals active or potential exploitation
            'unsupported': 20,  # indicates environment is no longer supported
            'escalation': 20,  # signals a major issue has been escalated
            'critical': 20,  # indicates severity is very high
            'noncompliant': 20,  # variant/spelling related to "non compliant"
            'compliance': (
                20
            ),  # already in your dictionary, but ensure spelled variants are covered
            # Medium-high terms (score ~15)
            # Use for important actions or states that often necessitate urgent attention.
            'rollback': 15,  # signals rolling back patches/updates
            'uninstall': 15,  # indicates removing problematic updates
            'regression': 15,  # indicates the update caused something to break
            'dependency': 15,  # blocking or required component
            'deadline': 15,  # a crucial timeline factor
            'enable': 15,  # action that might be part of config changes
            'disable': 15,  # likewise, can cause compliance issues
            'retry': 15,  # re-attempting an install or deploy
            'priority': 15,  # indicates urgency or triage
            # Medium terms (score ~10)
            # Use for words that highlight a problem or state but may not be catastrophic.
            'conflict': 10,  # signals potential library/dll/app conflict
            'blocked': 10,  # signals an install/deploy step is halted
            'outage': 10,  # can be partial or total downtime
            'downtime': 10,  # indicates system unavailability
            'stuck': (
                10
            ),  # in-progress patching or installation that fails to complete
            'error': 10,  # already in your dictionary
            'fail': 10,  # already in your dictionary
            'issue': 10,  # already in your dictionary
            'stability': 10,  # signals reliability concerns
            'performance': 10,  # signals performance impacts
            'incompatibility': (
                10
            ),  # indicates mismatch in versions or dependencies
            'corrupt': 10,  # already in your dictionary
            # Lower terms (score ~5)
            # Use for words that might matter but aren’t always show-stoppers.
            'delay': 5,  # non-critical postpone
            'skip': 5,  # skipping a patch might be less urgent than failing
            'unattended': 5,  # a mode of deployment
            'interactive': 5,  # a mode of deployment
            'schedule': 5,  # might matter, but not necessarily urgent
            'notify': 5,  # indicates communication step
            'log': 5,  # logging or log files
            'backup': 5,  # not always urgent, but relevant to patch strategy
        }

        for action, bonus in action_words.items():
            if action in text_lower:
                score += bonus

        # Penalty for very generic phrases
        generic_words = {'the', 'a', 'an', 'this', 'that', 'these', 'those'}
        if all(w in generic_words for w in words):
            score -= 30

        return score

    # Score all candidates
    candidates = []
    if noun_chunks:
        candidates.extend(
            (chunk, score_candidate_subject(chunk)) for chunk in noun_chunks
        )
    if keywords:
        candidates.extend((kw, score_candidate_subject(kw)) for kw in keywords)

    # Sort by score and take the best one
    if not candidates:
        return ""

    best_candidate = max(candidates, key=lambda x: x[1])[0]

    # Scan both noun_chunks and keywords for product mentions
    def find_product_mention(text_list):
        if not text_list:
            return None

        def get_specificity_score(product_name):
            """Calculate how specific a product name is based on its components"""
            # Count meaningful components like version, architecture, etc.
            components = 0
            product_lower = product_name.lower()
            if "version" in product_lower:
                components += 1
            if any(arch in product_lower for arch in ["x64", "32-bit"]):
                components += 1
            if any(
                ver in product_lower
                for ver in ["21h2", "22h2", "23h2", "24h2"]
            ):
                components += 1
            if "extended" in product_lower:
                components += 1
            if "chromium" in product_lower:
                components += 1
            # Base product adds 1 (windows 10, windows 11, edge)
            components += 1
            return components

        found_product = None
        max_specificity = -1

        for text in text_list:
            text_lower = text.lower()
            for pattern, product_name in product_patterns:
                if re.search(pattern, text_lower):
                    specificity = get_specificity_score(product_name)
                    if found_product is None or specificity > max_specificity:
                        found_product = product_name
                        max_specificity = specificity
        return found_product

    # First check noun_chunks as they're likely to be more precise
    product_mention = find_product_mention(noun_chunks)

    # If no product found in noun_chunks, check keywords
    if not product_mention:
        product_mention = find_product_mention(keywords)

    # Format timestamp
    ts = ""
    if timestamp:
        try:
            # Try to parse the timestamp if it's a string
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = timestamp
            ts = dt.strftime("%Y%m%d_%H%M%S")
        except (ValueError, AttributeError):
            pass

    # Construct final subject
    parts = []

    # Add best candidate (main action/issue)
    parts.append(best_candidate.strip())

    # Add product if found
    if product_mention:
        parts.append(product_mention)

    # Add timestamp if available
    if ts:
        parts.append(ts)

    # Join with underscores
    return "_".join(parts)


def normalize_subject(
    subject, text=None, noun_chunks=None, keywords=None, timestamp=None
):
    """
    Normalize a subject string, or generate one from metadata if cleaning removes all content.
    Prioritizes the raw subject while still cleaning it up for better indexing.

    Args:
        subject: Original subject string
        text: Optional text content to use if subject is empty
        noun_chunks: Optional pre-extracted noun chunks
        keywords: Optional pre-extracted keywords
        timestamp: Optional ISO 8601 timestamp string. If not provided, current time will be used.
    """

    def get_fallback_subject():
        # Use provided timestamp or current time
        ts = timestamp or datetime.now().isoformat()
        # Remove timezone and subsecond precision
        ts = ts.split('.')[0].split('+')[0].split('-07:00')[0]
        return f"patch_management_thread_{ts}"

    if not subject:
        # Handle truly empty subjects (rare case)
        constructed = construct_natural_subject(
            noun_chunks or [], keywords or [], timestamp
        )
        if constructed:
            logging.warning(
                f"Generated subject for empty input: {constructed}"
            )
            subject = constructed
        else:
            logging.error("No content available to generate subject")
            return get_fallback_subject()

    # Initial cleanup of standard patterns
    patterns_to_remove = [
        r"\[patchmanagement\]",
        r"\[External\]",
        r"\[EXTERNAL\]",
        r"🟣",  # Specifically remove this emoji
    ]

    # Remove standard patterns
    for pattern in patterns_to_remove:
        subject = re.sub(pattern, "", subject, flags=re.IGNORECASE)

    # Handle email prefixes - keep replacing until no more changes
    prefix_pattern = r"(?i:^|\s+)(RE|FW|FWD|AW)\s*[_:]?\s+"
    prev_subject = None
    while prev_subject != subject:
        prev_subject = subject
        subject = re.sub(prefix_pattern, " ", subject, flags=re.IGNORECASE)
        subject = subject.strip()

    # Remove any remaining emojis or special symbols
    subject = re.sub(r"[^\w\s]", "", subject).strip()

    # If cleaning removed all content, try to construct from metadata
    if not subject or subject.isspace():
        logging.warning(
            "Subject empty after cleaning, attempting reconstruction"
        )
        constructed = construct_natural_subject(
            noun_chunks or [], keywords or [], timestamp
        )
        if constructed:
            logging.info(f"Reconstructed subject from metadata: {constructed}")
            subject = constructed
        else:
            logging.warning("Could not reconstruct subject from metadata")
            return get_fallback_subject()

    # Lowercase and tokenize
    words = subject.lower().split()

    # Remove stop words
    words = [word for word in words if word not in STOP_WORDS]

    # If we have less than 3 meaningful words, try to enhance with constructed subject
    if len(words) < 3:
        constructed = construct_natural_subject(
            noun_chunks or [], keywords or [], timestamp
        )
        if constructed:
            # Take the constructed subject but remove timestamp parts
            constructed_parts = [
                part
                for part in constructed.split('_')
                if not part.isdigit() and not re.match(r'\d{8}', part)
            ]
            if constructed_parts:
                # Skip the first two parts (original subject) and any timestamp parts
                additional_parts = constructed_parts[2:]
                if additional_parts:
                    words.extend(additional_parts)
                    logging.info(
                        "Enhanced short subject with constructed parts:"
                        f" {words}"
                    )

    if not words:
        return get_fallback_subject()

    # Join with underscores for final format
    return "_".join(words)


def generate_thread_id(subject):
    # Normalize the subject to get a meaningful base for the thread_id
    normalized_subject = normalize_subject(subject)

    # Generate a hash to ensure uniqueness
    unique_hash = hashlib.md5(normalized_subject.encode()).hexdigest()[:6]

    # Combine normalized subject and unique hash
    thread_id = f"{normalized_subject[:50]}_{unique_hash}"
    return thread_id


def group_emails(df, similarity_threshold=90):
    groups = defaultdict(list)
    for idx, row in df.iterrows():
        subject = row["metadata"].get("subject", "")
        text = row.get("text")  # DO NOT MODIFY
        noun_chunks = row.get("metadata", {}).get(
            "evaluated_noun_chunks", []
        )  # DO NOT MODIFY
        keywords = row.get("metadata", {}).get(
            "evaluated_keywords", []
        )  # DO NOT MODIFY
        timestamp = row.get("metadata", {}).get(
            "receivedDateTime"
        )  # DO NOT MODIFY
        normalized_subj = normalize_subject(
            subject,
            text=text,
            noun_chunks=noun_chunks,
            keywords=keywords,
            timestamp=timestamp,
        )
        matched = False
        for key in groups:
            if fuzz.ratio(normalized_subj, key) >= similarity_threshold:
                groups[key].append(idx)
                matched = True
                break
        if not matched:
            groups[normalized_subj].append(idx)
    return groups


def process_emails(df):
    grouped_emails = group_emails(df)

    # Create new columns for thread_id, previous_id, and next_id
    df["thread_id"] = None
    df["previous_id"] = None
    df["next_id"] = None

    # Assign thread_ids and previous/next ids
    for group in grouped_emails.values():
        sorted_group = sorted(
            group, key=lambda x: df.loc[x, "metadata"]["receivedDateTime"]
        )

        # Generate a unique thread_id based on the first email in the group
        thread_id = generate_thread_id(
            df.loc[sorted_group[0], "metadata"]["subject"]
        )

        for i, idx in enumerate(sorted_group):
            df.at[idx, "thread_id"] = thread_id

            if i > 0:
                previous_id = df.loc[sorted_group[i - 1], "node_id"]

                # Ensure `previous_id` is not equal to the current node's `node_id`
                if previous_id != df.loc[idx, "node_id"]:
                    df.at[idx, "previous_id"] = previous_id

            if i < len(sorted_group) - 1:
                df.at[idx, "next_id"] = df.loc[sorted_group[i + 1], "node_id"]

    return df


def print_threads(df):
    # Group the DataFrame by thread_id
    grouped = df.groupby("thread_id")

    for thread_id, thread_df in grouped:
        print(f"\n{'='*50}")
        print(f"Thread ID: {thread_id}")
        print(f"{'='*50}")

        # Sort the thread by receivedDateTime
        thread_df["sortDate"] = thread_df["receivedDateTime"]
        thread_df = thread_df.sort_values(by="sortDate")

        for _, email in thread_df.iterrows():
            print(f"\nEmail ID: {email['node_id']}")
            print(f"Subject: {email['subject']}")
            print(f"Received: {email['receivedDateTime']}")
            print(f"\nBuild Numbers: {email['build_numbers']}")
            print(f"\nKBs: {email['kb_ids']}")
            # print(f"Previous Email: {email['previous_id'] or 'None'}")
            # print(f"Next Email: {email['next_id'] or 'None'}")
            print(f"\nContent Preview: {str(email['text'])[:100]}...")
            print(f"{'-'*50}")


def remove_metadata_fields(
    metadata: Dict[str, Any], fields_to_remove: List[str]
) -> Dict[str, Any]:
    """Removes the specified fields from the metadata dictionary."""
    for field in fields_to_remove:
        metadata.pop(field, None)  # Safely remove the key if it exists
    return metadata


def group_by_normalized_subject(
    df: pd.DataFrame, subject_col: str = "subject", threshold: int = 90
) -> Dict[str, List[int]]:
    """
    Groups DataFrame indices by subject similarity.
    For rows with same stripped subject (e.g. "Re: [patchmanagement]"), ensures they get the same
    synthesized subject if stripping removes all content.

    Args:
        df (pd.DataFrame): Dataframe to group
        subject_col (str): Column name of the DataFrame containing subject text
        threshold (int): Minimum ratio for fuzzy matching

    Returns:
        Dict[str, List[int]]: A dictionary where the keys are normalized subjects and the values are lists of indices in the DataFrame
    """
    # First pass: Group by raw subject to find rows that need the same synthesized subject
    raw_subject_groups = defaultdict(list)
    for idx, row in df.iterrows():
        subject = row[subject_col]
        if pd.isna(subject):
            subject = ""
        raw_subject_groups[subject.lower()].append(idx)

    # Process each raw subject group
    # groups = defaultdict(list)
    processed_subjects = {}  # Maps raw subject to normalized subject

    # First normalize all raw subjects
    for raw_subject, indices in raw_subject_groups.items():
        if raw_subject not in processed_subjects:
            # Get data from first row in group to normalize subject
            row = df.iloc[indices[0]]
            text = row.get("text", "")  # DO NOT MODIFY
            noun_chunks = row.get("noun_chunks", [])  # DO NOT MODIFY
            keywords = row.get("keywords", [])  # DO NOT MODIFY
            timestamp = row.get("receivedDateTime")  # DO NOT MODIFY

            normalized_subj = normalize_subject(
                raw_subject,
                text=text,
                noun_chunks=noun_chunks,
                keywords=keywords,
                timestamp=timestamp,
            )
            processed_subjects[raw_subject] = normalized_subj

    # Now group by normalized subjects with fuzzy matching
    normalized_groups = defaultdict(list)
    for raw_subject, indices in raw_subject_groups.items():
        normalized_subj = processed_subjects[raw_subject]

        # Try to match with existing groups
        matched = False
        best_match = None
        best_ratio = -1

        for key in normalized_groups:
            ratio = fuzz.ratio(normalized_subj.lower(), key.lower())
            if ratio >= threshold and ratio > best_ratio:
                best_ratio = ratio
                best_match = key
                matched = True

        if matched:
            normalized_groups[best_match].extend(indices)
        else:
            normalized_groups[normalized_subj].extend(indices)

    return normalized_groups


_historical_docs_cache = {}


def find_previous_thread_doc(
    normalized_subject: str,
    before_date: datetime,
    collection: str = "docstore",
    lookback_days: int = 30,
) -> Optional[Dict]:
    """
    Find the most recent historical document in a thread before a given date.

    Args:
        normalized_subject: The normalized subject to match
        before_date: Find documents before this date
        collection: MongoDB collection name to query
        lookback_days: Number of days to look back for historical documents

    Returns:
        The most recent matching document, or None if none found
    """
    global _historical_docs_cache
    from application.services.document_service import DocumentService

    # Initialize document service
    document_service = DocumentService(
        db_name="report_docstore", collection_name=collection
    )

    # Create cache key using date only (since published dates are at midnight)
    cache_key = f"{collection}_{before_date.date()}"

    if cache_key not in _historical_docs_cache:
        # Calculate the lookback window
        query_end = before_date
        lookback_start = query_end - timedelta(days=lookback_days)

        # Query for processed documents within the lookback window
        query = {
            "metadata.collection": "patch_management",
            "metadata.published": {"$lt": query_end, "$gte": lookback_start},
            "metadata.etl_processing_status.document_processed": True,
        }
        logging.debug(
            "Querying MongoDB for historical documents:\n"
            f"  End date: {query_end}\n"
            f"  Start date: {lookback_start}\n"
            f"  Query: {query}"
        )

        result = document_service.query_documents(
            query=query,
            sort=[("metadata.subject", 1), ("metadata.receivedDateTime", 1)],
        )

        # Pre-process and sort docs by normalized subject for binary search
        processed_docs = []
        for doc in result["results"]:
            doc_subject = doc.get("metadata", {}).get(
                "subject", ""
            )  # DO NOT MODIFY
            doc_text = doc.get("text", "")  # DO NOT MODIFY
            doc_noun_chunks = doc.get("metadata", {}).get(
                "evaluated_noun_chunks", None
            )
            doc_noun_chunks = (
                [] if pd.isna(doc_noun_chunks) else doc_noun_chunks
            )
            doc_keywords = doc.get("metadata", {}).get(
                "evaluated_keywords", None
            )
            doc_keywords = [] if pd.isna(doc_keywords) else doc_keywords
            doc_received_date = doc.get("metadata", {}).get(
                "receivedDateTime", ""
            )  # DO NOT MODIFY

            # Normalize the subject before using it as a key
            norm_subject = normalize_subject(
                doc_subject,
                text=doc_text,
                noun_chunks=doc_noun_chunks,
                keywords=doc_keywords,
                timestamp=doc_received_date,
            )
            processed_docs.append((norm_subject, doc))

        # Sort by normalized subject AND receivedDateTime
        processed_docs.sort(
            key=lambda x: (
                x[0],  # normalized subject
                datetime.fromisoformat(
                    x[1]["metadata"].get("receivedDateTime", "")
                ),  # preserves timezone
            )
        )

        # Log the sorted documents to verify order
        # logging.info("Sorted documents:")
        # for norm_subj, doc in processed_docs:
        #     logging.info(f"  Normalized: {norm_subj}")
        #     logging.info(f"  Original: {doc['metadata'].get('subject', '')}")
        #     logging.info(f"  ID: {doc.get('id_', '')}")
        #     logging.info(f"  Date: {doc['metadata'].get('receivedDateTime', '')}")
        #     logging.info("---")

        _historical_docs_cache[cache_key] = processed_docs
        logging.info(
            f"Cached {len(processed_docs)} historical documents for"
            f" {cache_key}"
        )

        # for norm_subj, doc in _historical_docs_cache[cache_key]:
        #     logging.info(f"  Subject: {norm_subj}")
        #     logging.info(f"  Original: {doc['metadata'].get('subject', '')}")
        #     logging.info(f"  ID: {doc.get('id_', '')}")
        #     logging.info(f"  Date: {doc['metadata'].get('receivedDateTime', '')}")
        #     logging.info("---")

    # Constants for fuzzy matching thresholds
    HIGH_MATCH_THRESHOLD = 90
    FALLBACK_MATCH_THRESHOLD = 85

    # Group-based search
    docs = _historical_docs_cache[cache_key]
    if not docs:
        return None

    # Handle empty normalized_subject
    if not normalized_subject:
        return None

    first_char = normalized_subject[0]

    # Find a good starting point based on first character
    start_idx = 0
    for i, (subj, _) in enumerate(docs):
        if not subj:  # Skip empty subjects
            continue
        if subj[0] >= first_char:
            start_idx = i
            break

    # Search through subject groups
    current_idx = start_idx
    best_matches = []
    best_ratio = 0

    while current_idx < len(docs):
        current_subject, current_doc = docs[current_idx]
        match_ratio = fuzz.ratio(
            normalized_subject.lower(), current_subject.lower()
        )

        # First try to find matches above HIGH_MATCH_THRESHOLD
        if match_ratio >= HIGH_MATCH_THRESHOLD:
            print(f"\n{'='*50}\nFound high match: {match_ratio}\n{'='*50}")
            if match_ratio > best_ratio:
                best_ratio = match_ratio
                best_matches = []

            # Collect all docs in this group
            group_idx = current_idx
            while (
                group_idx < len(docs) and docs[group_idx][0] == current_subject
            ):
                if match_ratio == best_ratio:
                    best_matches.append((match_ratio, docs[group_idx][1]))
                group_idx += 1

            # Skip to next group
            current_idx = group_idx

        # If no high-confidence match, try fallback threshold
        elif match_ratio >= FALLBACK_MATCH_THRESHOLD:
            # Only collect fallback matches if we haven't found any high-confidence matches
            if best_ratio < HIGH_MATCH_THRESHOLD:
                if match_ratio > best_ratio:
                    best_ratio = match_ratio
                    best_matches = []

                # Collect all docs in this group
                group_idx = current_idx
                while (
                    group_idx < len(docs)
                    and docs[group_idx][0] == current_subject
                ):
                    if match_ratio == best_ratio:
                        best_matches.append((match_ratio, docs[group_idx][1]))
                    group_idx += 1

                # Skip to next group
                current_idx = group_idx
            else:
                # Skip this group since we already have high-confidence matches
                while (
                    current_idx < len(docs)
                    and docs[current_idx][0] == current_subject
                ):
                    current_idx += 1

        else:
            # Move to next different subject
            current_subject = docs[current_idx][0]
            while (
                current_idx < len(docs)
                and docs[current_idx][0] == current_subject
            ):
                current_idx += 1

            # If we've moved past subjects starting with a higher first letter, we can stop
            if (
                current_idx < len(docs)
                and docs[current_idx][0][0] > first_char
            ):
                break

    # Return latest matching document if any meet threshold
    if best_matches:
        # First try to find the latest document with next_id=None
        latest_doc = None
        latest_date = None

        for ratio, doc in best_matches:
            if doc["metadata"].get("next_id") is None:
                doc_date = datetime.fromisoformat(
                    doc["metadata"].get("receivedDateTime", "")
                )
                if latest_date is None or doc_date > latest_date:
                    latest_date = doc_date
                    latest_doc = (ratio, doc)

        # If no end-of-thread doc found, take the latest by date
        if latest_doc is None:
            logging.warning(
                "No end-of-thread document found, using latest by date"
            )
            latest_doc = max(
                best_matches,
                key=lambda x: datetime.fromisoformat(
                    x[1]["metadata"].get("receivedDateTime", "")
                ),
            )

        # logging.info(
        #     f"Selected latest match:\n"
        #     f"  Ratio: {latest_doc[0]}\n"
        #     f"  ID: {latest_doc[1]['id_']}\n"
        #     f"  Subject: {latest_doc[1]['metadata'].get('subject', '')}\n"
        #     f"  Date: {latest_doc[1]['metadata'].get('receivedDateTime')}\n"
        #     f"  Next ID: {latest_doc[1]['metadata'].get('next_id')}"
        # )
        return latest_doc[1]

    logging.info(f"No match found above threshold {FALLBACK_MATCH_THRESHOLD}")
    return None


def _write_debug_output(final_df: pd.DataFrame, debug: bool = False) -> None:
    """
    Write detailed debug output for patch post transformations.

    Args:
        final_df: The final transformed DataFrame
        debug: If True, generates and writes debug output
    """
    if not debug:
        return

    try:
        # Convert DataFrame to detailed markdown format
        output_lines = ["# Patch Posts Debug Output\n"]

        # Group documents by normalized subject for better organization
        subject_groups = {}
        for _, row in final_df.iterrows():
            thread_id = row["metadata"].get("thread_id")
            if thread_id not in subject_groups:
                subject_groups[thread_id] = []
            # Convert row to dict and ensure metadata is a dict
            row_dict = row.to_dict()
            if isinstance(row_dict["metadata"], str):
                try:
                    row_dict["metadata"] = json.loads(row_dict["metadata"])
                except json.JSONDecodeError:
                    row_dict["metadata"] = {}
            elif not isinstance(row_dict["metadata"], dict):
                row_dict["metadata"] = {}
            subject_groups[thread_id].append(row_dict)

        # Output each thread group
        for thread_id, docs in subject_groups.items():
            output_lines.append(f"\n## Thread Group: {thread_id}\n")

            # Sort docs by receivedDateTime
            docs.sort(key=lambda x: x["receivedDateTime"])

            # Output thread statistics
            historical_count = sum(
                1
                for doc in docs
                if doc["metadata"].get("is_historical", False)
            )
            output_lines.append(f"- Total Documents: {len(docs)}")
            output_lines.append(f"- Historical Documents: {historical_count}")
            output_lines.append(
                f"- In-batch Documents: {len(docs) - historical_count}\n"
            )

            # Output document details in chronological order
            output_lines.append("### Documents (Chronological Order)\n")
            for doc in docs:
                metadata = doc["metadata"]
                is_historical = metadata.get("is_historical", False)
                doc_type = "Historical" if is_historical else "In-batch"

                output_lines.append("```")
                output_lines.append(f"Document Type: {doc_type}")
                output_lines.append(f"Node ID: {doc['node_id']}")
                output_lines.append(f"Subject: {doc['subject']}")
                output_lines.append(f"Received: {doc['receivedDateTime']}")
                output_lines.append(
                    "Published:"
                    f" {doc['published'].isoformat() if pd.notnull(doc['published']) else 'None'}"
                )
                output_lines.append(
                    f"Thread ID: {metadata.get('thread_id', 'None')}"
                )
                output_lines.append(
                    f"Previous ID: {metadata.get('previous_id', 'None')}"
                )
                output_lines.append(
                    f"Next ID: {metadata.get('next_id', 'None')}"
                )
                output_lines.append(
                    f"Collection: {metadata.get('collection', 'None')}"
                )
                output_lines.append(
                    f"Is Historical: {metadata.get('is_historical', False)}"
                )
                output_lines.append("```\n")

        # Write to markdown file
        output_path = "C:/Users/emili/PycharmProjects/microsoft_cve_rag/microsoft_cve_rag/application/data/debug/patch_posts_debug.md"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))

        logging.debug(f"Debug output written to {output_path}")
    except Exception as e:
        logging.error(f"Error writing debug output: {str(e)}")


def transform_patch_posts_v2(
    patch_posts: List[Dict[str, Any]],
    process_all: bool = False,
    debug: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Transform patch management posts with thread management and historical document integration.

    This transformer handles both new and preprocessed documents, maintaining thread continuity
    and proper chronological ordering. It can process either all documents or only threads
    containing new documents.

    Workflow:
    1. Load and Prepare:
       - Convert all documents to DataFrame format
       - Extract standard fields from metadata to columns
       - Normalize column names and data types

    2. Thread Processing:
       - Group documents by normalized subject
       - For each group:
         a. Find historical documents from the last 30 days
         b. Sort all documents chronologically by receivedDateTime
         c. Link documents in chronological order:
            - Set thread_id (preserve existing or generate new)
            - Set previous and next IDs based on position in sorted list and document type
         d. Preserve historical document linking:
            - Maintain existing thread_id and previous_id
            - Update next_id of last historical to first in-batch
            - Update previous_id of first in-batch to last historical

    3. Document Status:
       - Preserve existing etl_processing_status for processed documents
       - Only update status for new documents:
         * Set document_processed = True
         * Initialize other status fields as False
         * Add timestamp and version

    Args:
        patch_posts: List of patch management post dictionaries
        process_all: If True, process and return all documents.
                    If False, only process threads containing at least one new document.
        debug: If True, generates debug output

    Returns:
        pd.DataFrame: Transformed documents with proper thread management and metadata.
                     Includes columns for node_id, metadata, and standard fields.

    Note:
        Historical documents are fetched from MongoDB and must have been previously processed
        (document_processed=True in their etl_processing_status).
    """
    if not patch_posts:
        print("No patch posts to transform.")
        return None

    df = pd.DataFrame(patch_posts)

    # Debug rows where text is NaN
    nan_text_rows = df[df['text'].isna()]
    if not nan_text_rows.empty:
        logging.debug(f"Found {len(nan_text_rows)} rows with NaN text values:")

    # Standard fields to move from metadata
    metadata_fields_to_move = [
        "receivedDateTime",
        "published",
        "subject",
        "conversation_link",
        "cve_mentions",
        "evaluated_noun_chunks",
        "post_type",
        "evaluated_keywords",
        "tags",
        "collection",
    ]

    # Move needed fields from metadata into top-level columns if not already present
    for field in metadata_fields_to_move:
        if field not in df.columns:
            df[field] = df["metadata"].apply(
                lambda x, field=field: x.get(field, None)
            )

    df = df.rename(
        columns={
            "id_": "node_id",
            "evaluated_noun_chunks": "noun_chunks",
            "evaluated_keywords": "keywords",
            "cve_mentions": "cve_ids",
            "kb_mentions": "kb_ids",
        }
    )
    # Sort KB IDs in reverse if present
    if "kb_ids" in df.columns:
        df["kb_ids"] = df["kb_ids"].apply(_sort_kb_ids)

    # Each doc is PatchManagementPost
    df["node_label"] = "PatchManagementPost"
    df["verification_status"] = "unverified"
    # Flag records that are new vs. preprocessed
    df["is_processed"] = df["metadata"].apply(_check_doc_processed)

    # Group all docs by subject similarity regardless of process_all
    groups = group_by_normalized_subject(df, subject_col="subject")

    if process_all:
        # Process everything but still maintain grouping
        df_to_process = df
        logging.info(f"Processing all {len(df)} docs with thread grouping.")
    else:
        # If there are no new docs, there's nothing to do
        unprocessed_df = df[~df["is_processed"]]
        if unprocessed_df.empty:
            logging.info(
                "No new documents found; returning None unless you want to"
                " re-process everything."
            )
            return None

        # We only want to unify threads that contain at least one new doc
        new_indices = set(unprocessed_df.index)

        # We'll gather all docs from the groups that have any new doc
        indices_to_include = []
        for norm_subj, idx_list in groups.items():
            if any(i in new_indices for i in idx_list):
                indices_to_include.extend(idx_list)

        df_to_process = df.loc[indices_to_include].copy()
        logging.info(
            f"Processing {len(df_to_process)} docs from threads containing new"
            " docs."
        )

    # Now we can process the documents in df_to_process
    # We'll use the same groups we already calculated
    process_groups = groups

    # We'll store updated rows in a list, then concat them
    processed_rows = []

    # Filter process_groups to only include indices that exist in df_to_process
    process_groups = {
        norm_subj: [idx for idx in idx_list if idx in df_to_process.index]
        for norm_subj, idx_list in groups.items()
        if any(idx in df_to_process.index for idx in idx_list)
    }

    for norm_subj, idx_list in process_groups.items():
        if not idx_list:  # Skip if no valid indices for this group
            continue

        group_df = df_to_process.loc[idx_list].copy()  # Create explicit copy

        # Get earliest date from group
        earliest_date = pd.to_datetime(group_df["receivedDateTime"]).min()
        # logging.info(f"\nReceived dates for group:")
        # for idx, row in group_df.iterrows():
        #     logging.info(f"  Node ID: {row['node_id']}")
        #     logging.info(f"  Received: {pd.to_datetime(row['receivedDateTime'])}")
        #     logging.info(f"  Subject: {row.get('metadata', {}).get('subject', 'NO_SUBJECT')}")
        #     logging.info("---")

        # Query for the most recent historical document before our earliest
        historical_doc = find_previous_thread_doc(
            normalized_subject=norm_subj,
            before_date=earliest_date,
            collection="docstore",
        )

        # Pause and show status
        # input("\nPress Enter to continue to next subject group...")
        # fmt: off
        if (
            historical_doc
            and historical_doc["id_"] not in group_df["node_id"].values
        ):
            # Get the first in-batch document's node_id for linking
            first_batch_node_id = group_df.iloc[0]["node_id"]

            # Convert historical doc to DataFrame row with same schema
            historical_df = pd.DataFrame(
                [
                    {
                        "node_id": historical_doc["id_"],
                        "text": historical_doc["text"],
                        "metadata": {
                            **historical_doc["metadata"],
                            "is_historical": True,  # Mark as historical for tracking
                            "next_id": (
                                first_batch_node_id
                            ),  # Link to first in-batch doc
                        },
                        "subject": historical_doc["metadata"].get("subject", ""),
                        "receivedDateTime": datetime.fromisoformat(
                            str(historical_doc["metadata"].get("receivedDateTime", ""))
                        ),
                        "published": historical_doc["metadata"].get("published", ""),
                        "conversation_link": historical_doc["metadata"].get(
                            "conversation_link", ""
                        ),
                        "cve_mentions": historical_doc["metadata"].get(
                            "cve_mentions", []
                        ),
                        "noun_chunks": historical_doc["metadata"].get(
                            "evaluated_noun_chunks", []
                        ),
                        "post_type": historical_doc["metadata"].get("post_type", ""),
                        "keywords": historical_doc["metadata"].get(
                            "evaluated_keywords", []
                        ),
                        "tags": historical_doc["metadata"].get("tags", []),
                        "collection": historical_doc["metadata"].get(
                            "collection", "patch_management"
                        ),
                        "node_label": "PatchManagementPost",
                        "kb_ids": historical_doc.get("kb_mentions", []),
                        "product_mentions": historical_doc.get("product_mentions"),
                        "build_numbers": historical_doc.get("build_numbers"),
                        "cve_ids": historical_doc["metadata"].get("cve_mentions", []),
                        "previous_id": historical_doc["metadata"].get(
                            "previous_id", None
                        ),
                        "next_id": (
                            first_batch_node_id
                        ),  # Set next_id to first in-batch doc
                        "verification_status": historical_doc["metadata"].get(
                            "verification_status", "unverified"
                        ),
                    }
                ]
            )

            logging.info(
                f"Found historical document for subject '{norm_subj}' (ID:"
                f" {historical_doc['id_']}, receivedDateTime:"
                f" {historical_doc['metadata'].get('receivedDateTime')})"
            )

            # Create a new DataFrame with the updated first row
            updated_rows = []

            # Update first row with historical link
            first_row = group_df.iloc[0].copy()
            first_row_metadata = first_row["metadata"].copy()
            first_row_metadata["previous_id"] = historical_doc["id_"]
            first_row["metadata"] = first_row_metadata
            first_row["previous_id"] = historical_doc["id_"]
            updated_rows.append(first_row)

            # Add remaining rows unchanged
            if len(group_df) > 1:
                for idx in range(1, len(group_df)):
                    updated_rows.append(group_df.iloc[idx].copy())

            # Create DataFrame ensuring index is preserved
            updated_group_df = pd.DataFrame(
                updated_rows, index=[r.name for r in updated_rows]
            )

            # Combine historical with current group and sort chronologically
            group_df = pd.concat(
                [historical_df, updated_group_df], ignore_index=True
            )
        elif historical_doc:
            logging.info(
                f"Skipping historical document for subject '{norm_subj}' - "
                f"ID {historical_doc['id_']} already exists in current group"
            )
        # fmt: on
        # Sort by receivedDateTime - already datetime objects
        group_df = group_df.sort_values("receivedDateTime")
        node_ids = group_df["node_id"].tolist()

        # Get thread_id from historical doc if it exists, otherwise generate new one
        thread_id = None
        if historical_doc:
            thread_id = historical_doc["metadata"].get("thread_id")
        if not thread_id:
            thread_id = norm_subj

        group_rows = []
        # Update the linking between documents
        for i, idx in enumerate(group_df.index):
            row = group_df.loc[idx].copy()  # Create explicit copy of the row
            m = row["metadata"].copy()
            current_id = row["node_id"]

            # Set thread_id for all documents in group
            m["thread_id"] = thread_id

            # Set previous and next IDs based on position in sorted list and document type
            next_id = node_ids[i + 1] if i < len(node_ids) - 1 else None
            # Don't link to self
            if next_id == current_id:
                next_id = None

            if m.get("is_historical", False):
                # For historical docs, preserve their existing previous_id and only update next_id
                previous_id = m.get("previous_id")  # Keep existing previous_id
                if previous_id == current_id:  # Don't link to self
                    previous_id = None
                m["next_id"] = next_id
            else:
                # For in-batch docs, set both previous and next
                previous_id = node_ids[i - 1] if i > 0 else None
                # Don't link to self
                if previous_id == current_id:
                    previous_id = None
                m["previous_id"] = previous_id
                m["next_id"] = next_id

            # Update processing status only for new documents
            if not m.get("etl_processing_status", {}).get(
                "document_processed", False
            ):
                current_time = datetime.now(timezone.utc).isoformat()
                m["etl_processing_status"] = {
                    "document_processed": True,
                    "entities_extracted": False,
                    "graph_prepared": False,
                    "vector_prepared": False,
                    "last_processed_at": current_time,
                    "processing_version": "1.0",
                }

            row["metadata"] = m

            # Extract fields from metadata to top-level columns
            row["thread_id"] = thread_id  # Use thread_id directly
            row["post_type"] = m.get("post_type", "")
            # row["product_mentions"] = m.get("product_mentions", [])
            # row["build_numbers"] = m.get("build_numbers", [])
            row["published"] = m.get("published", "")
            row["previous_id"] = previous_id  # Use previous_id directly
            row["next_id"] = next_id  # Use next_id directly

            group_rows.append(row)

        # Create DataFrame for this group and append to processed_rows
        if group_rows:
            group_df = pd.DataFrame(group_rows)
            processed_rows.append(group_df)
            logging.info(
                f"Processed group with subject '{norm_subj}' -"
                f" {len(group_rows)} documents"
            )

    # Combine all processed groups
    if processed_rows:
        # Filter out empty DataFrames before concatenation
        non_empty_rows = [df for df in processed_rows if not df.empty]
        final_df = pd.concat(non_empty_rows, axis=0, ignore_index=True)
        logging.info(
            f"Transformed {len(final_df)} patch posts (including historical"
            " thread docs)"
        )
        if debug:
            _write_debug_output(final_df, debug)
    # final_df["metadata"] = final_df["metadata"].apply(make_json_safe_metadata)
    return final_df


# End Feature Engineering Patch Transformer ===================================


def expand_version_architectures(mention: str, products_df: pd.DataFrame) -> List[str]:
    """Expand product mentions to include architecture variants.

    Args:
        mention: Product mention to expand
        products_df: DataFrame with product information

    Returns:
        List of full product names with architectures
    """
    mention = mention.replace(' ', '_').lower()

    # Extract version components
    version_match = re.search(
        r'(\d{4}[a-z]?|\d{2}h\d)',
        mention,
        flags=re.IGNORECASE
    )

    if version_match:
        version = version_match.group().lower()
        base = mention.split(version)[0].strip('_')
        pattern = f"{base}_{version}"

        # Find architecture variants
        arch_matches = products_df[
            products_df["product_full"].str.contains(pattern, case=False)
        ]
        if not arch_matches.empty:
            return arch_matches["product_full"].tolist()

    # Fallback to basic architecture expansion
    if any(arch in mention for arch in ['x86', 'x64']):
        return [mention]

    base_matches = products_df[
        products_df["product_name_version"].str.lower() == mention
    ]
    return base_matches["product_full"].tolist() if not base_matches.empty else [mention]


def normalize_product_mentions(
    mentions: List[str], products_df: pd.DataFrame
) -> List[str]:
    """Normalize product mentions to consistent format.

    Args:
        mentions: List of product mentions to normalize
        products_df: DataFrame containing product information

    Returns:
        List of normalized product mentions
    """
    if not mentions:
        return []
    expanded = set()
    for m in mentions:
        expanded.update(expand_version_architectures(m, products_df))

    # Build product hierarchy
    product_tree = defaultdict(set)
    for product in expanded:
        parts = product.split('_')
        current = []
        for part in parts:
            current.append(part)
            product_tree['_'.join(current)].add(product)

    # Keep only most specific leaves
    return [
        p for p in expanded
        if not any(
            p != child and child.startswith(p + '_')
            for child in expanded
        )
    ]


def patch_fe_transformer(
    docs: List[Dict[str, Any]], products_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Transform the extracted patch documents to populate product mentions,
    build numbers, and KB references. Ignores thread_id/previous_id/next_id logic.
    Returns a DataFrame ready for updates.
    """
    metadata_fields_to_move = [
        "receivedDateTime",
        "published",
        "subject",
        "conversation_link",
        "cve_mentions",
        "evaluated_noun_chunks",
        "post_type",
        "evaluated_keywords",
        "tags",
        "collection",
    ]

    if not docs:
        print("No patch posts to feature engineer.")
        return None
    # Convert docs to DataFrame for easier manipulation
    patch_posts_df = pd.DataFrame(docs)
    # Extract metadata fields
    for field in metadata_fields_to_move:
        if field not in patch_posts_df.columns:
            patch_posts_df[field] = patch_posts_df["metadata"].apply(
                lambda x, field=field: x.get(field, None)
            )

    patch_posts_df = patch_posts_df.rename(
        columns={
            "id_": "node_id",
            "evaluated_noun_chunks": "noun_chunks",
            "evaluated_keywords": "keywords",
            "cve_mentions": "cve_ids",
            "kb_mentions": "kb_ids",
        }
    )
    # Example columns in the DataFrame might differ in real code
    # Assume 'text', 'noun_chunks', 'keywords', etc. are present
    patch_posts_df["product_mentions_noun_chunks"] = None
    patch_posts_df["product_mentions_keywords"] = None
    patch_posts_df["product_mentions_subject"] = None
    patch_posts_df["product_mentions_text"] = None
    patch_posts_df["product_mentions"] = None
    patch_posts_df["windows_kbs"] = None
    patch_posts_df["edge_kbs"] = None
    # mask = patch_posts_df["product_mentions"].isna()
    # patch_posts_df.loc[mask, "product_mentions"] = patch_posts_df.loc[mask, "product_mentions"].apply(lambda x: [])
    non_conversational_idx = patch_posts_df[
        patch_posts_df["post_type"] != "Conversational"
    ].index
    # Extract product mentions only for non-conversational posts
    if len(non_conversational_idx) > 0:
        # Process noun chunks
        patch_posts_df.loc[non_conversational_idx, "product_mentions_noun_chunks"] = (
            patch_posts_df.iloc[non_conversational_idx]["noun_chunks"].apply(
                lambda x: extract_product_mentions(x, products_df, is_preprocessed=True)
            )
        )

        # Process keywords
        patch_posts_df.loc[non_conversational_idx, "product_mentions_keywords"] = (
            patch_posts_df.iloc[non_conversational_idx]["keywords"].apply(
                lambda x: extract_product_mentions(x, products_df, is_preprocessed=True)
            )
        )

        # Process subject
        patch_posts_df.loc[non_conversational_idx, "product_mentions_subject"] = (
            patch_posts_df.iloc[non_conversational_idx]["subject"].apply(
                lambda x: extract_product_mentions(x, products_df, is_preprocessed=False)
            )
        )

        # Process text
        patch_posts_df.loc[non_conversational_idx, "product_mentions_text"] = (
            patch_posts_df.iloc[non_conversational_idx]["text"].apply(
                lambda x: extract_product_mentions(x, products_df, is_preprocessed=False)
            )
        )

    # Extract build numbers
    regex_pattern = construct_regex_pattern()
    patch_posts_df["build_numbers"] = patch_posts_df["text"].apply(
        lambda x: (
            extract_build_numbers(x, regex_pattern) if pd.notna(x) else []
        )
    )

    # Extract Windows and Edge KB references
    def get_windows_kbs(row):
        kbs_from_text = (
            extract_windows_kbs(row["text"]) if pd.notna(row["text"]) else []
        )
        kbs_from_subject = (
            extract_windows_kbs(row["subject"])
            if pd.notna(row["subject"])
            else []
        )
        return list(set(kbs_from_text + kbs_from_subject))

    patch_posts_df["windows_kbs"] = patch_posts_df.apply(
        get_windows_kbs, axis=1
    )
    patch_posts_df["edge_kbs"] = patch_posts_df.apply(extract_edge_kbs, axis=1)

    def combine_kb_lists(row):
        """Combine windows and edge KB lists, handling None values."""
        windows_kbs = row["windows_kbs"] or []  # Convert None to empty list
        edge_kbs = row["edge_kbs"] or []  # Convert None to empty list
        return list(set(windows_kbs + edge_kbs))

    patch_posts_df["kb_ids"] = patch_posts_df.apply(combine_kb_lists, axis=1)

    # Combine and normalize product mentions only for non-conversational posts
    if len(non_conversational_idx) > 0:
        patch_posts_df.loc[non_conversational_idx, "product_mentions"] = (
            patch_posts_df.iloc[non_conversational_idx].apply(
                lambda row: normalize_product_mentions(
                    list(
                        set(
                            (row["product_mentions_noun_chunks"] or [])
                            + (row["product_mentions_keywords"] or [])
                            + (row["product_mentions_subject"] or [])
                            + (row["product_mentions_text"] or [])
                        )
                    ),
                    products_df,
                ),
                axis=1,
            )
        )

    def safe_convert_mentions(x):
        if isinstance(x, list):
            return convert_to_original_representation(x)
        return []  # Handle NaN and any other non-list values

    # Convert product mentions to match database format
    patch_posts_df["product_mentions"] = patch_posts_df["product_mentions"].apply(safe_convert_mentions)

    # Clean up intermediate columns
    patch_posts_df.drop(
        columns=[
            "product_mentions_noun_chunks",
            "product_mentions_keywords",
            "product_mentions_subject",
            "product_mentions_text",
        ],
        inplace=True,
    )
    # Log any records that have product mentions
    has_mentions = patch_posts_df["product_mentions"].apply(
        lambda x: bool(x) if isinstance(x, list) else False
    )
    if has_mentions.any():
        logging.info(
            f"Found {has_mentions.sum()} records with product mentions:"
        )
        # for node_id in patch_posts_df[has_mentions]["node_id"]:
        #     logging.info(f"Record {node_id} has product mentions")
    return patch_posts_df


# End Feature Engineering Patch Transformer ===================================


def extract_product_mentions(
    text: Union[str, List[str]],
    products_df: pd.DataFrame,
    threshold: int = 91,
    is_preprocessed: bool = False
) -> List[str]:
    """Extract product mentions from text with comprehensive pattern matching.

    Handles special cases like "windows10/11", version patterns like "23h2",
    and server versions while maintaining compatibility with fuzzy matching.

    Args:
        text: Input text or list of texts to search
        products_df: DataFrame containing product information
        threshold: Fuzzy matching threshold
        is_preprocessed: Whether input is pre-processed (noun chunks/keywords)

    Returns:
        List of matched product names
    """
    if pd.isna(text):
        return []

    matches: Set[str] = set()

    # Convert list input to string if needed
    search_text = ' '.join(text) if isinstance(text, list) else text
    search_text = search_text.lower()

    # Handle special Windows 10/11 pattern
    win1011_pattern = r"(?:windows|win)\s*(?:10/11|11/10)"
    if re.search(win1011_pattern, search_text, re.IGNORECASE):
        matches.update(["windows_10", "windows_11"])

    # Handle server versions
    server_patterns = {
        r"(?:windows\s+)?server\s+2025\b": "windows_server_2025",
        r"(?:windows\s+)?server\s+2022\b": "windows_server_2022",
        r"(?:windows\s+)?server\s+2019\b": "windows_server_2019",
        r"(?:windows\s+)?server\s+2016\b": "windows_server_2016",
    }

    for pattern, product in server_patterns.items():
        if re.search(pattern, search_text, re.IGNORECASE):
            matches.add(product)

    # Handle version-specific patterns
    version_patterns = {
        r"(?:windows|win)?\s*11.*?(?:version\s*)?24h2\b": "windows_11_24h2",
        r"(?:windows|win)?\s*11.*?(?:version\s*)?23h2\b": "windows_11_23h2",
        r"(?:windows|win)?\s*11.*?(?:version\s*)?22h2\b": "windows_11_22h2",
        r"(?:windows|win)?\s*10.*?(?:version\s*)?22h2\b": "windows_10_22h2",
        r"(?:windows|win)?\s*10.*?(?:version\s*)?21h2\b": "windows_10_21h2",
    }

    for pattern, product in version_patterns.items():
        if re.search(pattern, search_text, re.IGNORECASE):
            matches.add(product)

    # Handle standalone versions
    standalone_patterns = {
        r"\b24h2\b": ["windows_11_24h2"],
        r"\b23h2\b": ["windows_11_23h2"],
        r"\b22h2\b": ["windows_11_22h2", "windows_10_22h2"],
        r"\b21h2\b": ["windows_10_21h2", "windows_server_2022"]
    }

    for pattern, products in standalone_patterns.items():
        if re.search(pattern, search_text, re.IGNORECASE):
            matches.update(products)

    # Add general Windows versions if no specific version found
    if not any('windows_11' in match for match in matches):
        if re.search(r"(?:windows|win)\s*11\b", search_text, re.IGNORECASE):
            matches.add("windows_11")
    if not any('windows_10' in match for match in matches):
        if re.search(r"(?:windows|win)\s*10\b", search_text, re.IGNORECASE):
            matches.add("windows_10")

    # If no matches found and text is pre-processed, try fuzzy matching with lower threshold
    if not matches and is_preprocessed:
        # Normalize text for better matching
        search_text = re.sub(r'microsoft\s+', '', search_text)  # Remove Microsoft prefix
        search_text = re.sub(r'\s+xdr$', '', search_text)  # Remove XDR suffix
        search_text = re.sub(r'defender\s+for\s+', 'defender_', search_text)  # Normalize Defender product names

        text_series = pd.Series([search_text])
        fuzzy_matches = fuzzy_search_column(text_series, products_df, threshold=80)
        if fuzzy_matches and fuzzy_matches[0]:
            matches.update(fuzzy_matches[0])
    # For non-preprocessed text, use standard fuzzy matching as fallback
    elif not matches:
        text_series = pd.Series([search_text])
        fuzzy_matches = fuzzy_search_column(text_series, products_df, threshold)
        if fuzzy_matches and fuzzy_matches[0]:
            matches.update(fuzzy_matches[0])

    return list(matches)


def transform_symptoms(symptoms: List[Dict[str, Any]]) -> pd.DataFrame:
    # clean up document dict from mongo to align with data models
    if symptoms:
        df = pd.DataFrame(symptoms, columns=list(symptoms[0].keys()))

        if not all(
            col in df.columns
            for col in ["severity_type", "node_label", "reliability"]
        ):
            df = df.assign(
                severity_type="NST",
                node_label="Symptom",
                reliability="HIGH",
            )
        df["labels"] = df["labels"].apply(_get_first_label)

        logging.debug(f"Total Symptoms transformed: {df.shape[0]}")

        return df
    else:
        logging.warning("No Symptoms to transform.")

    return None


def transform_causes(causes: List[Dict[str, Any]]) -> pd.DataFrame:
    # clean up document dict from mongo to align with data models
    if causes:
        df = pd.DataFrame(causes, columns=list(causes[0].keys()))
        if not all(
            col in df.columns
            for col in ["severity_type", "node_label", "reliability"]
        ):
            df = df.assign(
                severity_type="NST",
                node_label="Cause",
                reliability="MEDIUM",
            )
        df["labels"] = df["labels"].apply(_get_first_label)

        logging.debug(f"Total Causes transformed: {df.shape[0]}")
        # print(f"Columns: {df.columns}")
        return df
    else:
        logging.warning("No Causes to transform.")

    return None


def transform_fixes(fixes: List[Dict[str, Any]]) -> pd.DataFrame:
    # clean up document dict from mongo to align with data models
    if fixes:
        df = pd.DataFrame(fixes, columns=list(fixes[0].keys()))

        if not all(
            col in df.columns
            for col in ["severity_type", "node_label", "reliability"]
        ):
            df = df.assign(
                severity_type="NST",
                node_label="Fix",
                reliability="MEDIUM",
            )
        df["labels"] = df["labels"].apply(_get_first_label)
        df["severity_type"] = df["severity_type"].fillna("NST")

        logging.debug(f"Total Fixes transformed: {df.shape[0]}")
        # print(f"Columns: {df.columns}")
        return df
    else:
        logging.warning("No Fixes to transform.")

    return None


def transform_tools(tools: List[Dict[str, Any]]) -> pd.DataFrame:
    # clean up document dict from mongo to align with data models
    if tools:
        df = pd.DataFrame(tools, columns=list(tools[0].keys()))
        if not all(col in df.columns for col in ["node_label", "reliability"]):
            df = df.assign(
                node_label="Tool",
                reliability="MEDIUM",
            )
        df["labels"] = df["labels"].apply(_get_first_label)

        logging.debug(f"Total Tools transformed: {df.shape[0]}")
        # print(f"Columns: {df.columns}")
        return df
    else:
        logging.warning("No Tools to transform.")

    return None


def transform_technologies(technologies: List[Dict[str, Any]]) -> pd.DataFrame:
    # clean up document dict from mongo to align with data models
    if technologies:
        df = pd.DataFrame(technologies, columns=list(technologies[0].keys()))
        if not all(col in df.columns for col in ["node_label"]):
            df = df.assign(
                node_label="Technology",
            )
        df["labels"] = df["labels"].apply(_get_first_label)

        logging.debug(f"Total Technologies transformed: {df.shape[0]}")
        # print(f"Columns: {df.columns}")
        return df
    else:
        logging.warning("No Technologies to transform.")

    return None


def combine_dicts_to_dataframe(
    dict1: Dict[str, List[Dict]], dict2: Dict[str, List[Dict]]
) -> pd.DataFrame:
    combined_data = []

    # Iterate through all keys (assuming both dicts have the same keys)
    for key in dict1.keys():
        # Extend the combined_data list with items from both dictionaries
        combined_data.extend(dict1[key])
        combined_data.extend(dict2[key])

    # Create a DataFrame from the combined list of dictionaries
    df = pd.DataFrame(combined_data)

    # Add a column to identify the original category (key)
    df["category"] = df.apply(
        lambda row: _find_category(row, dict1, dict2), axis=1
    )

    return df


def _find_category(
    row: pd.Series, dict1: Dict[str, List[Dict]], dict2: Dict[str, List[Dict]]
) -> str:
    """Find the category for a row by checking which dictionary and key it belongs to."""
    row_dict = row.to_dict()
    for key, items in dict1.items():
        if row_dict in items + dict2[key]:
            return key
    return ""


def combine_and_split_dicts(
    dict1: Dict[str, List[Dict]], dict2: Dict[str, List[Dict]]
) -> Dict[str, pd.DataFrame]:
    if not dict1 and not dict2:
        return {}

    dict1 = dict1 or {}
    dict2 = dict2 or {}
    category_dataframes = {}

    all_keys = set(dict1.keys()) | set(dict2.keys())
    for key in all_keys:
        combined_items = dict1.get(key, []) + dict2.get(key, [])
        df = pd.DataFrame(combined_items)
        category_dataframes[key] = df

    return category_dataframes


def transform_extracted_entities(
    entities_list: List[Dict], entity_type: str
) -> pd.DataFrame:
    df = pd.DataFrame(entities_list)
    # Ensure all required fields are present
    required_fields = {
        "Symptom": [
            "node_id",
            "symptom_label",
            "description",
            "source_id",
            "source_type",
            "tags",
        ],
        "Cause": [
            "node_id",
            "description",
            "source_id",
            "source_type",
            "tags",
        ],
        "Fix": ["node_id", "description", "source_id", "source_type", "tags"],
        "Tool": [
            "node_id",
            "name",
            "description",
            "source_id",
            "source_type",
            "tags",
            "source_url",
        ],
        "Technology": [
            "node_id",
            "name",
            "description",
            "source_id",
            "source_type",
            "tags",
        ],
    }
    for field in required_fields.get(entity_type, []):
        if field not in df.columns:
            df[field] = None  # Set default value if field is missing
    return df


def make_text_json_safe(text: str) -> str:
    """
    Make text safe for JSON serialization by properly escaping special characters.

    Args:
        text: Input text that may contain special characters

    Returns:
        JSON-safe string with properly escaped characters
    """
    if not isinstance(text, str):
        return text

    # Replace backslashes first to avoid double escaping
    text = text.replace('\\', '\\\\')
    # Replace quotes and other special symbols
    text = text.replace('"', '\\"')
    text = text.replace('\n', '\\n')
    text = text.replace('\r', '\\r')
    text = text.replace('\t', '\\t')
    text = text.replace('\b', '\\b')
    text = text.replace('\f', '\\f')
    return text


def _create_metadata_from_row(
    row: pd.Series,
    preserve_metadata: bool = False,
    exclude_columns: List[str] = None,
) -> dict:
    """
    Helper function to create metadata from a DataFrame row, handling NaN values and datetime objects.

    Args:
        row: DataFrame row to process
        preserve_metadata: If True, preserves existing metadata structure from source,
                         otherwise places all fields (except doc_id) in metadata
    """
    logging.debug(
        f"Creating metadata from row with keys: {row.index.tolist()}"
    )

    def process_value(value):
        """Helper to process individual values."""
        if isinstance(value, (pd.Timestamp, datetime)):
            # Handle datetime objects
            return value.isoformat()
        elif isinstance(value, (pd.Series, np.ndarray)):
            # Handle Series or arrays
            if value.size == 0:
                return []
            return value.tolist()
        elif isinstance(value, list):
            # Handle lists and nested lists
            if not value:
                return []
            return [
                (
                    process_value(sub_value)
                    if isinstance(sub_value, list)
                    else sub_value
                )
                for sub_value in value
            ]
        elif isinstance(value, dict):
            # Handle dictionaries
            return {k: process_value(v) for k, v in value.items()}
        elif isinstance(value, str):
            # Handle strings
            return make_text_json_safe(value)
        elif pd.api.types.is_scalar(value):
            # Handle scalar NaN, NAType, and primitive types
            if pd.isna(value) or isinstance(value, pd._libs.missing.NAType):
                return None
            return value
        else:
            # Convert unknown types to string
            return make_text_json_safe(str(value))

    exclude_columns = exclude_columns or ['text']
    metadata = {}
    # If preserve_metadata is True and metadata column exists, use it as base
    if preserve_metadata and 'metadata' in row:
        base_metadata = row['metadata']
        if isinstance(base_metadata, dict):
            metadata.update(base_metadata)
        elif isinstance(base_metadata, str):
            try:
                metadata.update(json.loads(base_metadata))
            except json.JSONDecodeError:
                logging.warning(
                    f"Failed to parse metadata string row:{row['node_id']}"
                )
    # Process each field in the row
    for key, value in row.items():
        if key in exclude_columns or key == 'metadata':
            continue
        try:
            processed_value = process_value(value)
            if key == 'node_id':
                # Special handling for 'node_id'
                metadata['doc_id'] = processed_value
                logging.debug(f"Set doc_id from node_id: {processed_value}")
            else:
                metadata[key] = processed_value
        except Exception as e:
            logging.error(
                f"Error processing key {key} with value"
                f" {value} ({type(value)}): {str(e)}"
            )

    logging.debug(
        f"Final metadata structure - Root keys: {list(metadata.keys())}"
    )
    logging.debug(
        f"Final metadata structure - Metadata keys: {list(metadata.keys())}"
    )

    return metadata


# =============================================================================
# Convert dataframe of KB Articles to Llama Documents
# =============================================================================


def _handle_na_text(text_value: Any) -> str:
    """Handle potential NA values in text fields.

    Args:
        text_value: Any value that might be NA

    Returns:
        str: Empty string if NA, otherwise string representation of value
    """
    if pd.isnull(text_value):
        return ""
    elif not isinstance(text_value, str):
        return str(text_value)
    return text_value


def clean_na_values_series(series: pd.Series) -> pd.Series:
    """
    Replace various forms of missing/null values in a pandas Series with Python `None`.
    Recursively handles nested data structures (lists, dicts, etc.).

    Handles all pandas NA types including:
    - pandas.NA
    - numpy.nan
    - pandas.NaT
    - None
    - String representations of NA values

    Args:
        series: Input pandas Series

    Returns:
        pandas Series with all NA values replaced with None
    """

    def clean_value(x):
        if isinstance(x, pd._libs.missing.NAType):
            return None
        if pd.api.types.is_scalar(x):
            if pd.isnull(x):
                return None
            if isinstance(x, str) and x.lower() in {
                "nan",
                "nat",
                "none",
                "null",
            }:
                return None
            return x
        if isinstance(x, (list, tuple)):
            return type(x)(clean_value(item) for item in x)
        if isinstance(x, dict):
            return {k: clean_value(v) for k, v in x.items()}
        if hasattr(x, '__iter__') and not isinstance(x, str):
            return type(x)([clean_value(item) for item in x])
        return x

    return pd.Series(
        [clean_value(x) for x in series], index=series.index, name=series.name
    )


def convert_df_to_llamadoc_kb_articles(
    df: pd.DataFrame,
) -> list[LlamaDocument]:
    """Convert a DataFrame of KB Articles to a list of LlamaDocuments."""
    llama_documents = []

    for _, row in df.iterrows():
        try:
            logging.info(f"Processing row:\n{row.to_dict()}")

            exclude_columns = [
                'text',
            ]
            # Extract metadata dynamically, excluding specified keys
            metadata_dict = _create_metadata_from_row(
                clean_na_values_series(row),
                preserve_metadata=True,
                exclude_columns=exclude_columns,
            )

            def get_clean_value(row, key, default):
                """Helper to get value from row, handling NA values"""
                value = row.get(key)
                if isinstance(value, (list, np.ndarray)):
                    return value  # Return arrays/lists as is
                if (
                    value is None
                    or isinstance(value, pd._libs.missing.NAType)
                    or (pd.api.types.is_scalar(value) and pd.isna(value))
                ):
                    return default
                return value

            # Set default values for specific fields
            defaults = {
                "node_label": get_clean_value(row, "node_label", "KBArticle"),
                "kb_id": get_clean_value(row, "kb_id", ""),
                "product_build_id": get_clean_value(
                    row, "product_build_id", ""
                ),
                "product_build_ids": get_clean_value(
                    row, "product_build_ids", []
                ),
                "cve_ids": get_clean_value(row, "cve_ids", []),
                "build_number": get_clean_value(row, "build_number", []),
                "article_url": get_clean_value(row, "article_url", ""),
                "scraped_markdown": get_clean_value(row, "scraped_markdown", ""),
                "scraped_json": get_clean_value(row, "scraped_json", ""),
                "reliability": get_clean_value(row, "reliability", ""),
                "readability": get_clean_value(row, "readability", ""),
                "excluded_embed_metadata_keys": get_clean_value(
                    row,
                    "excluded_embed_metadata_keys",
                    ["node_label", "product_build_id", "product_build_ids"],
                ),
                "excluded_llm_metadata_keys": get_clean_value(
                    row,
                    "excluded_llm_metadata_keys",
                    ["node_label", "product_build_id", "product_build_ids"],
                ),
            }

            # Update metadata with defaults
            for key, default_value in defaults.items():
                if key not in metadata_dict or metadata_dict[key] is None:
                    metadata_dict[key] = default_value
            current_time = datetime.now().isoformat()
            # update the metadata key etl_processing_status for vectorization
            metadata_dict["etl_processing_status"] = {
                "document_processed": True,
                "entities_extracted": True,
                "graph_prepared": True,
                "vector_prepared": False,
                "last_processed_at": current_time,
                "processing_version": "1.0",
            }
            # Create the LlamaDocument
            doc = LlamaDocument(
                text=_handle_na_text(row["text"]),
                doc_id=row["node_id"],
                extra_info=metadata_dict,
                excluded_embed_metadata_keys=metadata_dict[
                    "excluded_embed_metadata_keys"
                ],
                excluded_llm_metadata_keys=metadata_dict[
                    "excluded_llm_metadata_keys"
                ],
            )
            llama_documents.append(doc)

            logging.info("Created LlamaDocument", extra={"doc_id": doc.doc_id})
            logging.debug(f"Document metadata:\n{metadata_dict}")

        except Exception as e:
            logging.error(
                f"Error processing row row_data {row.to_dict()} \nerror"
                f" {str(e)}"
            )

    logging.info(
        "Completed conversion of KB Articles to LlamaDocuments "
        f"with total documents: {len(llama_documents)}"
    )
    return llama_documents


# =============================================================================
# Convert dataframe of Update Packages to Llama Documents
# =============================================================================
def convert_df_to_llamadoc_update_packages(
    df: pd.DataFrame,
) -> list[LlamaDocument]:
    """Convert a DataFrame of Update Packages to a list of LlamaDocuments."""
    llama_documents = []

    for _, row in df.iterrows():
        try:
            # Log raw input data for debugging
            logging.debug(f"Processing row:\n{row.to_dict()}")

            # Construct text content
            text_content = (
                "Update Package published on:"
                f" {row.get('published', '').strftime('%B %d, %Y')}\nBuild"
                f" Number: {row.get('build_number', '')}\nPackage Type:"
                f" {row.get('package_type', '')}\nPackage URL:"
                f" {row.get('package_url', '')}"
            )

            # Handle datetime conversion for published date
            if "published" in row and isinstance(
                row["published"], pd.Timestamp
            ):
                row["published"] = row["published"].isoformat()

            # Process downloadable packages
            downloadable_packages = row.get("downloadable_packages", [])
            if isinstance(downloadable_packages, list):
                downloadable_packages = [
                    (
                        {
                            k: v.isoformat() if isinstance(v, datetime) else v
                            for k, v in pkg.items()
                        }
                        if isinstance(pkg, dict)
                        else pkg
                    )
                    for pkg in downloadable_packages
                ]

            row = clean_na_values_series(row)
            exclude_columns = [
                'text',
                'excluded_embed_metadata_keys',
                'excluded_llm_metadata_keys',
            ]
            # Extract metadata dynamically, excluding specified keys
            metadata = _create_metadata_from_row(
                row, preserve_metadata=False, exclude_columns=exclude_columns
            )

            # Set default values for specific fields
            defaults = {
                "node_label": row.get(
                    "node_label", "UpdatePackage"
                ),  # Use node_label from row or default to "UpdatePackage"
                "build_number": row.get("build_number", ""),
                "product_build_ids": row.get("product_build_ids", []),
                "package_type": row.get("package_type", ""),
                "package_url": row.get("package_url", ""),
                "downloadable_packages": downloadable_packages,
                "reliability": row.get("reliability", ""),
                "readability": row.get("readability", "HIGH"),
                "excluded_embed_metadata_keys": [
                    "product_build_ids",
                    "node_label",
                ],
                "excluded_llm_metadata_keys": [],
            }

            # Update metadata with defaults
            for key, default_value in defaults.items():
                if key not in metadata or metadata[key] is None:
                    metadata[key] = default_value

            # Create the LlamaDocument
            doc = LlamaDocument(
                text=text_content,
                doc_id=row["node_id"],
                extra_info=metadata,
                excluded_embed_metadata_keys=metadata[
                    "excluded_embed_metadata_keys"
                ],
                excluded_llm_metadata_keys=metadata[
                    "excluded_llm_metadata_keys"
                ],
            )
            llama_documents.append(doc)

            # Log document creation details
            logging.info(f"Created KB LlamaDocument {doc.doc_id}")
            logging.debug("Document metadata", extra={"metadata": metadata})

        except Exception as e:
            # Log any errors encountered during processing
            logging.error(
                "Error processing row",
                extra={"row_data": row.to_dict(), "error": str(e)},
            )

    return llama_documents


# =============================================================================
# Convert dataframe of Symptoms to Llama Documents
# =============================================================================
def convert_df_to_llamadoc_symptoms(df: pd.DataFrame) -> list[LlamaDocument]:
    """Convert a DataFrame of Symptoms to a list of LlamaDocuments."""
    llama_documents = []

    for _, row in df.iterrows():
        try:
            # Log raw row data for debugging
            logging.debug("Processing row", extra={"row_data": row.to_dict()})
            row = clean_na_values_series(row)
            # Extract metadata, handling NaN values
            metadata = _create_metadata_from_row(row)

            # Set default values for specific fields
            defaults = {
                "node_label": row.get("node_label", "Symptom"),
                "symptom_label": row.get("symptom_label", ""),
                "severity_type": row.get("severity_type", "NST"),
                "reliability": row.get("reliability", ""),
                "tags": row.get("tags", []),
                "excluded_embed_metadata_keys": [
                    "node_label",
                    "symptom_label",
                ],
                "excluded_llm_metadata_keys": [],
            }

            # Update metadata with defaults
            for key, default_value in defaults.items():
                if key not in metadata or metadata[key] is None:
                    metadata[key] = default_value

            # Create the LlamaDocument
            doc = LlamaDocument(
                text=_handle_na_text(row["description"]),
                doc_id=row["node_id"],
                extra_info=metadata,
                excluded_embed_metadata_keys=metadata[
                    "excluded_embed_metadata_keys"
                ],
                excluded_llm_metadata_keys=metadata[
                    "excluded_llm_metadata_keys"
                ],
            )
            llama_documents.append(doc)

            # Log document creation details
            logging.info(f"Created Symptom LlamaDocument {doc.doc_id}")
            logging.debug("Document metadata", extra={"metadata": metadata})

        except Exception as e:
            # Log any errors encountered during processing
            logging.error(
                "Error processing row",
                extra={"row_data": row.to_dict(), "error": str(e)},
            )

    return llama_documents


# =============================================================================
# Convert dataframe of Causes to Llama Documents
# =============================================================================
def convert_df_to_llamadoc_causes(df: pd.DataFrame) -> list[LlamaDocument]:
    """Convert a DataFrame of Causes to a list of LlamaDocuments."""
    llama_documents = []

    for _, row in df.iterrows():
        try:
            # Log raw row data for debugging
            logging.debug("Processing row", extra={"row_data": row.to_dict()})

            # Extract metadata, handling NaN values
            metadata = _create_metadata_from_row(clean_na_values_series(row))

            # Set default values for specific fields
            defaults = {
                "node_label": row.get("node_label", "Cause"),
                "cause_label": row.get("cause_label", ""),
                "severity_type": row.get("severity_type", "NST"),
                "reliability": row.get("reliability", ""),
                "tags": row.get("tags", []),
                "excluded_embed_metadata_keys": ["node_label", "cause_label"],
                "excluded_llm_metadata_keys": [],
            }

            # Update metadata with defaults
            for key, default_value in defaults.items():
                if key not in metadata or metadata[key] is None:
                    metadata[key] = default_value

            # Create the LlamaDocument
            doc = LlamaDocument(
                text=_handle_na_text(row["description"]),
                doc_id=row["node_id"],
                extra_info=metadata,
                excluded_embed_metadata_keys=metadata[
                    "excluded_embed_metadata_keys"
                ],
                excluded_llm_metadata_keys=metadata[
                    "excluded_llm_metadata_keys"
                ],
            )
            llama_documents.append(doc)

            # Log document creation details
            logging.info(f"Created Cause LlamaDocument {doc.doc_id}")
            logging.debug("Document metadata", extra={"metadata": metadata})

        except Exception as e:
            # Log any errors encountered during processing
            logging.error(
                "Error processing row",
                extra={"row_data": row.to_dict(), "error": str(e)},
            )

    return llama_documents


# =============================================================================
# Convert dataframe of Fixes to Llama Documents
# =============================================================================
def convert_df_to_llamadoc_fixes(df: pd.DataFrame) -> list[LlamaDocument]:
    """Convert a DataFrame of Fix nodes to a list of LlamaDocuments."""
    llama_documents = []

    for _, row in df.iterrows():
        try:
            # Log raw input row data for debugging
            logging.debug("Processing row", extra={"row_data": row.to_dict()})

            # Extract metadata dynamically, excluding specified keys
            metadata = _create_metadata_from_row(clean_na_values_series(row))

            # Set default values for specific fields
            defaults = {
                "node_label": row.get(
                    "node_label", "Fix"
                ),  # Use node_label from row or default to "Fix"
                "fix_label": row.get("fix_label", ""),
                "cve_ids": row.get("cve_ids", []),
                "reliability": row.get("reliability", "HIGH"),
                "readability": row.get("readability", "HIGH"),
                "excluded_embed_metadata_keys": [
                    "node_label",
                    "fix_label",
                    "cve_ids",
                ],
                "excluded_llm_metadata_keys": [],
            }

            # Update metadata with defaults
            for key, default_value in defaults.items():
                if key not in metadata or metadata[key] is None:
                    metadata[key] = default_value

            # Create the LlamaDocument
            doc = LlamaDocument(
                text=_handle_na_text(row["description"]),
                doc_id=row["node_id"],
                extra_info=metadata,
                excluded_embed_metadata_keys=metadata[
                    "excluded_embed_metadata_keys"
                ],
                excluded_llm_metadata_keys=metadata[
                    "excluded_llm_metadata_keys"
                ],
            )
            llama_documents.append(doc)

            # Log document creation
            logging.info(f"Created Fix LlamaDocument {doc.doc_id}")
            logging.debug(f"Document metadata:\n{doc.metadata}")

        except Exception as e:
            # Log any errors encountered during processing
            logging.error(
                "Error processing row",
                extra={"row_data": row.to_dict(), "error": str(e)},
            )

    return llama_documents


# =============================================================================
# Convert dataframe of Tools to Llama Documents
# =============================================================================
def convert_df_to_llamadoc_tools(df: pd.DataFrame) -> list[LlamaDocument]:
    """Convert a DataFrame of Tool nodes to a list of LlamaDocuments."""
    llama_documents = []

    for _, row in df.iterrows():
        try:
            # Log raw input row data for debugging
            logging.debug("Processing row", extra={"row_data": row.to_dict()})

            # Extract metadata dynamically, excluding specified keys
            metadata = _create_metadata_from_row(clean_na_values_series(row))

            # Set default values for specific fields
            defaults = {
                "node_label": row.get(
                    "node_label", "Tool"
                ),  # Use node_label from row or default to "Tool"
                "tool_label": row.get("tool_label", ""),
                "name": row.get("name", ""),
                "cve_ids": row.get("cve_ids", []),
                "tool_url": row.get("tool_url", ""),
                "reliability": "HIGH",
                "readability": "HIGH",
                "excluded_embed_metadata_keys": [
                    "node_label",
                    "tool_label",
                    "node_id",
                ],
                "excluded_llm_metadata_keys": [],
            }

            # Update metadata with defaults
            for key, default_value in defaults.items():
                if key not in metadata or metadata[key] is None:
                    metadata[key] = default_value

            # Create the LlamaDocument
            doc = LlamaDocument(
                text=_handle_na_text(row["description"]),
                doc_id=row["node_id"],
                extra_info=metadata,
                excluded_embed_metadata_keys=metadata[
                    "excluded_embed_metadata_keys"
                ],
                excluded_llm_metadata_keys=metadata[
                    "excluded_llm_metadata_keys"
                ],
            )
            llama_documents.append(doc)

            # Log document creation
            logging.info(f"Created Tool LlamaDocument {doc.doc_id}")
            logging.debug(f"Document metadata:\n{metadata}")

        except Exception as e:
            # Log any errors encountered during processing
            logging.error(
                "Error processing row",
                extra={"row_data": row.to_dict(), "error": str(e)},
            )

    return llama_documents


# =============================================================================
# Convert dataframe of MSRC Posts to Llama Documents
# =============================================================================
def convert_df_to_llamadoc_msrc_posts(
    df: pd.DataFrame,
    symptom_nodes: List[Any] = None,
    cause_nodes: List[Any] = None,
    fix_nodes: List[Any] = None,
    tool_nodes: List[Any] = None,
) -> List[LlamaDocument]:
    """Convert a DataFrame of MSRC posts to a list of LlamaDocuments."""
    llama_documents = []
    # drop column `impact_type`
    # df = df.drop(columns=["impact_type"])
    for _, row in df.iterrows():
        try:
            # Log raw input row data for debugging
            logging.debug("Processing row", extra={"row_data": row.to_dict()})
            exclude_columns = ["text"]
            # Create metadata using helper function
            metadata_dict = _create_metadata_from_row(
                clean_na_values_series(row),
                preserve_metadata=True,
                exclude_columns=exclude_columns,
            )
            logging.debug(f"Generated metadata:\n{metadata_dict}")

            # Set default values for specific fields in metadata
            defaults = {
                "node_label": row.get("node_label", "MSRCPost"),
                "reliability": row.get("reliability", "HIGH"),
                "readability": row.get("readability", "HIGH"),
                "kb_ids": row.get("kb_ids", []),
                "cve_ids": row.get("cve_ids", []),
                "build_numbers": row.get("build_numbers", []),
                "product_mentions": row.get("product_mentions", []),
                "extracted_symptoms": row.get("extracted_symptoms", []),
                "extracted_causes": row.get("extracted_causes", []),
                "extracted_fixes": row.get("extracted_fixes", []),
                "extracted_tools": row.get("extracted_tools", []),
            }

            logging.debug(f"Default metadata values:\n{defaults}")

            # Update metadata with defaults
            for key, default_value in defaults.items():
                if key not in metadata_dict or metadata_dict[key] is None:
                    metadata_dict[key] = default_value
                    logging.debug(
                        "Applied default value:\n"
                        f"Key: {key}, Default Value: {default_value}"
                    )

            # Add extracted nodes while preserving nested structure
            extracted_nodes = {
                "extracted_symptoms": [
                    node.node_id
                    for node in symptom_nodes or []
                    if node.source_id == row["node_id"]
                ],
                "extracted_causes": [
                    node.node_id
                    for node in cause_nodes or []
                    if node.source_id == row["node_id"]
                ],
                "extracted_fixes": [
                    node.node_id
                    for node in fix_nodes or []
                    if node.source_id == row["node_id"]
                ],
                "extracted_tools": [
                    node.node_id
                    for node in tool_nodes or []
                    if node.source_id == row["node_id"]
                ],
            }

            # Merge extracted nodes with metadata while preserving nested structure
            metadata_dict.update(extracted_nodes)
            current_time = datetime.now().isoformat()
            # update the metadata key etl_processing_status for vectorization
            metadata_dict["etl_processing_status"] = {
                "document_processed": True,
                "nvd_extracted": True,
                "entities_extracted": True,
                "graph_prepared": True,
                "vector_prepared": False,
                "last_processed_at": current_time,
                "processing_version": "1.0",
            }
            # Create the LlamaDocument
            doc = LlamaDocument(
                text=_handle_na_text(row["text"]),
                doc_id=metadata_dict["doc_id"],
                extra_info=metadata_dict,
                excluded_embed_metadata_keys=metadata_dict[
                    "excluded_embed_metadata_keys"
                ],
                excluded_llm_metadata_keys=metadata_dict[
                    "excluded_llm_metadata_keys"
                ],
            )
            llama_documents.append(doc)

            # Log document creation
            logging.info(f"Created MSRCPost LlamaDocument {doc.doc_id}")
            logging.debug(f"Document metadata:\n{doc.metadata}")

        except Exception as e:
            # Log any errors encountered during processing
            logging.error(
                "Error processing row",
                extra={"row_data": row.to_dict(), "error": str(e)},
            )

    return llama_documents


# =============================================================================
# Convert dataframe of Patch Management Posts to Llama Documents
# =============================================================================


def convert_df_to_llamadoc_patch_posts(
    df: pd.DataFrame,
    symptom_nodes: List[Any] = None,
    cause_nodes: List[Any] = None,
    fix_nodes: List[Any] = None,
    tool_nodes: List[Any] = None,
) -> list[LlamaDocument]:
    """Convert a DataFrame of PatchManagement posts to a list of LlamaDocuments."""
    llama_documents = []
    # fmt: off
    for _, row in df.iterrows():
        try:
            # Log raw input row data for debugging
            logging.debug("Processing row", extra={"row_data": row.to_dict()})

            # Create metadata using helper function
            metadata_dict = _create_metadata_from_row(
                clean_na_values_series(row),
                preserve_metadata=True,
                exclude_columns=[],
            )
            if "cve_ids" in metadata_dict and metadata_dict["cve_ids"] == "":
                metadata_dict["cve_ids"] = []
            # fmt: off
            logging.debug(
                "Generated metadata: \n{metadata_dict}"
            )
            # fmt: on
            current_time = datetime.now().isoformat()
            # Set default values for specific fields
            defaults = {
                "node_label": row.get("node_label", "PatchManagementPost"),
                "reliability": row.get("reliability", ""),
                "readability": row.get("readability", ""),
                "conversation_link": row.get("conversation_link", ""),
                "kb_ids": row.get("kb_ids", []),
                "cve_ids": row.get("cve_ids", []),
                "build_numbers": row.get("build_numbers", []),
                "product_mentions": row.get("product_mentions", []),
                "extracted_symptoms": row.get("extracted_symptoms", []),
                "extracted_causes": row.get("extracted_causes", []),
                "extracted_fixes": row.get("extracted_fixes", []),
                "extracted_tools": row.get("extracted_tools", []),
                "excluded_embed_metadata_keys": row.get(
                    "excluded_embed_metadata_keys", []
                ),
                "excluded_llm_metadata_keys": row.get(
                    "excluded_llm_metadata_keys", []
                ),
                "etl_processing_status": row.get(
                    "etl_processing_status",
                    {
                        'document_processed': True,
                        'entities_extracted': True,
                        'graph_prepared': True,
                        'vector_prepared': True,
                        'last_processed_at': current_time,
                        'processing_version': '1.0',
                    },
                ),
            }

            # Update metadata with defaults
            for key, default_value in defaults.items():
                if key not in metadata_dict or metadata_dict[key] is None:
                    metadata_dict[key] = default_value
                    logging.debug(
                        "Applied default value",
                        extra={"key": key, "default_value": default_value},
                    )

            # Add extracted nodes
            extracted_nodes = {
                "extracted_symptoms": [
                    node.node_id
                    for node in symptom_nodes or []
                    if node.source_id == row["node_id"]
                ],
                "extracted_causes": [
                    node.node_id
                    for node in cause_nodes or []
                    if node.source_id == row["node_id"]
                ],
                "extracted_fixes": [
                    node.node_id
                    for node in fix_nodes or []
                    if node.source_id == row["node_id"]
                ],
                "extracted_tools": [
                    node.node_id
                    for node in tool_nodes or []
                    if node.source_id == row["node_id"]
                ],
            }

            # Merge extracted nodes with metadata while preserving nested structure
            metadata_dict.update(extracted_nodes)
            # update the metadata etl dict to update the vector key and the last updated key
            metadata_dict['etl_processing_status']['entities_extracted'] = True
            metadata_dict['etl_processing_status']['vector_prepared'] = True
            metadata_dict['etl_processing_status'][
                'last_processed_at'
            ] = current_time
            # Create the LlamaDocument
            doc = LlamaDocument(
                text=_handle_na_text(row["text"]),
                doc_id=row["node_id"],
                extra_info=metadata_dict,
                excluded_embed_metadata_keys=(
                    metadata_dict.get("excluded_embed_metadata_keys", [])
                    if isinstance(
                        metadata_dict.get("excluded_embed_metadata_keys"), list
                    )
                    else []
                ),
                excluded_llm_metadata_keys=(
                    metadata_dict.get("excluded_llm_metadata_keys", [])
                    if isinstance(
                        metadata_dict.get("excluded_llm_metadata_keys"), list
                    )
                    else []
                ),
            )
            llama_documents.append(doc)

            # Log document creation
            logging.info(f"Created Patch LlamaDocument doc_id: {doc.doc_id}")
            logging.debug(f"Document metadata: {doc.extra_info}")

        except Exception as e:
            # Log any errors encountered during processing
            logging.error(
                f"Error processing row_data: {row.to_dict()}\n error: {str(e)}"
            )

    # fmt: on
    return llama_documents


# End Convert dataframes to LlamaDocs ==========================


