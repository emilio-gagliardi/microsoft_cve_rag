# Purpose: Transform extracted data
# Inputs: Raw data
# Outputs: Transformed data
# Dependencies: None
import os
import logging
# from neomodel import AsyncStructuredNode
import numpy as np
import warnings
from typing import Union, List, Dict, Any
import pandas as pd
from fuzzywuzzy import fuzz, process
import re
import hashlib
import spacy
import json
from datetime import datetime
from spacy.lang.en.stop_words import STOP_WORDS
from collections import defaultdict
from application.core.models.basic_models import (
    Document,
    Vector,
    GraphNode,
    DocumentMetadata,
    VectorMetadata,
    GraphNodeMetadata,
)

# from application.services.embedding_service import EmbeddingService
from application.etl.NVDDataExtractor import ScrapingParams, NVDDataExtractor

from llama_index.core import Document as LlamaDocument
import asyncio
import marvin
from marvin.ai.text import generate_llm_response

# from microsoft_cve_rag.application.core.models import graph_db_models

marvin.settings.openai.chat.completions.model = "gpt-4o-mini"
# embedding_service = EmbeddingService.from_provider_name("fastembed")
logging.getLogger(__name__)


def normalize_mongo_kb_id(kb_id_input):
    """
    Normalize the 'kb_id' field from MongoDB documents.

    This function checks if the input is a string and converts it to a list if necessary.
    It removes any 'kb' or 'KB' prefix and ensures the 'kb_id' is in the format KB-XXXXXX or KB-XXX.XXX.XXX.XXX.
    The function returns a single item if the list has only one element, otherwise it returns the list.

    Args:
        kb_id_input (Union[str, List[str]]): The 'kb_id' field from MongoDB documents, which can be a string or a list.

    Returns:
        Union[str, List[str]]: The normalized 'kb_id' field, either as a single string or a list of strings.
    """
    if kb_id_input is None:
        return []

    if isinstance(kb_id_input, str):
        kb_id_list = [kb_id_input]
    else:
        kb_id_list = kb_id_input

    # Function to normalize a single kb_id
    def normalize_kb_id(kb_id):
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
        df["product_name"] = df["product_name"].apply(lambda x: mapping_names.get(x, x))
        df["product_architecture"] = df["product_architecture"].apply(
            lambda x: mapping.get(x, x)
        )
        df["product_architecture"] = df["product_architecture"].replace("", "NA")
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


def transform_product_builds(product_builds: List[Dict[str, Any]]) -> pd.DataFrame:
    # clean up document dict from mongo to align with data models
    if product_builds:
        df = pd.DataFrame(product_builds, columns=list(product_builds[0].keys()))

        # Apply the process_kb_id function to the 'kb_id' column
        df["kb_id"] = df["kb_id"].apply(normalize_mongo_kb_id)
        df["kb_id"] = df["kb_id"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x
        )
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
            lambda x: mapping_architectures.get(x, x)
        )
        df["product_architecture"] = df["product_architecture"].replace("", "NA")
        df["product_version"] = df["product_version"].replace("", "NV")
        df["product_name"] = df["product_name"].apply(lambda x: mapping_names.get(x, x))
        df["impact_type"] = df["impact_type"].str.lower().str.replace(" ", "_")
        df["impact_type"] = df["impact_type"].apply(lambda x: mapping_impacts.get(x, x))
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


async def async_generate_summary(text: str) -> str:
    marvin_summary_prompt = """
        Generate a highly technical summary of the following Microsoft KB Article text. This summary is intended for advanced system administrators and IT professionals specializing in modern device management with Intune MDM, Entra ID, Windows 365, and Azure.

        **Structure your response as follows:**
        1. Provide a brief, sentence-based overview that includes:
        - The primary purpose of this KB article (e.g., security update, quality improvement).
        - The specific operating systems and versions affected.
        - Any critical security issues, vulnerabilities, or bugs addressed.
        - Any prerequisites or special installation requirements.
        - Avoid adding general advice on patch management in the overview.
        - 5 sentences maximum.
        - No heading.
        Example of a good overview:'KB5040427 is a critical security update released on July 9, 2024, targeting Windows 10 Enterprise LTSC 2021, IoT Enterprise LTSC 2021, and Windows 10 version 22H2 (builds 19044.4651 and 19045.4651). The update addresses significant security vulnerabilities in RADIUS protocol (MD5 collisions) and implements enhanced BitLocker Secure Boot validation profiles. This combined SSU/LCU package requires specific prerequisites based on deployment method: KB5014032 for offline imaging or KB5005260 for WSUS deployment. The update includes SSU KB5039336 (builds 19044.4585 and 19045.4585) for improved update servicing capabilities.'

        2. Follow the overview with a section titled **Technical Breakdown**: Summarize only the specific technical content described in the KB article, with actionable details for administrators. Use subheadings that align with the content of the KB article and format commands, error codes, and configurations clearly. Do not add headings for commentary or guidelines in this prompt.

        **Vulnerabilities and Exploits**:
        Clearly state vulnerabilities and mention whether they are high impact or low impact. For example, 'This update addresses high-impact vulnerabilities including issues with Windows Installer and Remote Authentication Dial-In User Service (RADIUS) related to MD5 collision exploits. Full details are available in the July 2024 Security Updates.'

        **Guidelines for Formatting Commands, Error Codes, and Technical Content**:
        - **Commands and Scripts**: For each command, use labeled code blocks. Include only commands that are directly relevant to the KB article, avoiding placeholders. Use the following format:
        - **Powershell commands**:
            ```
            [Powershell]
            ```powershell
            # Example PowerShell command
            Set-ItemProperty -Path "HKLM:\Software\Policies\Microsoft\Windows\Installer" -Name "DisableLUAInRepair" -Value 1
            ```
        - **Registry Keys**:
            ```
            [Registry Key]
            ```registry
            `HKEY_LOCAL_MACHINE\SOFTWARE\Policies\Microsoft\Windows\Installer\DisableLUAInRepair` = `1`
            ```
        - **Intune Commands**: Ensure the command syntax is accurate and specific to Intune where applicable.
            ```
            [Intune Terminal]
            ```shell
            az intune policy set --policyName "DOCacheHost" --value "<Your MCCC Endpoint>"
            ```

        - **Error Codes**: Format error codes in a labeled code block, specifying `[Error Code]` and using backticks for clarity. For example:
            ```
            [Error Code]
            `0x80070520`
            ```

        - **Workarounds and Known Issues**:
        - Clearly detail any known issues and their workarounds. Specify whether the fix should be applied via **Autopatch, MDM, Group Policy, registry modification, or Intune command**. Be aware that microsoft has announced the deprecation of WSUS, and a preference for Autopatch or Intune is recommended where possible, but do not hallucenate workarounds or fixes outside of the KB Article content.
        - Use bullet lists. For example:
            'MCC/DHCP Option 235 Discovery Issue
            - Impact: Enterprise environments only
            - Resolution: Install KB5040525
            BitLocker Recovery Prompt
            - Trigger: Device Encryption enabled
            - Resolution: Install KB5041580
            Profile Picture Error (0x80070520)
            - Impact: Limited
            - Status: Requires support intervention'
        - Use precise instructions, including paths, commands, and exact settings, to avoid ambiguity. Example:
            ```
            If devices experience issues with Microsoft Connected Cache discovery via DHCP Option 235, set the following in Group Policy:
            [Registry Key]
            `HKEY_LOCAL_MACHINE\SOFTWARE\Policies\Microsoft\DOCacheHost` = `<MCC Endpoint>`
            ```

        3. **Installation Process and Prerequisites**:
        - Provide step-by-step installation prerequisites directly from the KB article.
        - Specify preferred update channels (e.g., Autopatch, Windows Update, WSUS) and clarify scenarios where standalone installation is necessary.
        - If uninstallation instructions are given, use labeled code blocks as with `[Powershell]` commands, ensuring that command syntax is exact.

        **Additional Guidelines**:
        - Assume the reader has a strong background in patching and Microsoft device management.
        - Clearly highlight any security risks, elevated-risk exploits, or known issues in the patch.
        - Maintain honesty and precision by adhering to the content and guidance in the KB article, while promoting modern practices where possible.
        Note. Avoid mentioning general, out-of-date advice (e.g., "Use Autopatch or Intune") unless explicitly mentioned in the KB article. Focus solely on the details directly relevant to the article's content.
        - Don't include file information in the summary. The audience can easily find that information elsewhere.
        KB Article text:
        {kb_article_text}
        """
    model_kwargs = {"max_tokens": 850, "temperature": 0.9}
    response = await generate_llm_response(
        marvin_summary_prompt.format(kb_article_text=text),
        model_kwargs=model_kwargs,
    )
    return response.response.choices[0].message.content


# Wrapper to handle async calls in apply
async def generate_summaries(texts: pd.Series) -> List[str]:
    tasks = [async_generate_summary(text) for text in texts]
    return await asyncio.gather(*tasks)


def transform_kb_articles(
    kb_articles_windows: List[Dict[str, Any]],
    kb_articles_edge: List[Dict[str, Any]]
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
        "node_label",
        "article_url",
        "summary",
    ]
    dtypes = {
            'node_id': 'str',
            'kb_id': 'str',
            'title': 'str',
            'text': 'str',
            'published': 'datetime64[ns]',
            'product_build_id': 'str',
            'product_build_ids': 'object',
            'build_number': 'object',
            'cve_ids': 'object',
            'excluded_embed_metadata_keys': 'object',
            'node_label': 'str',
            'article_url': 'str'
        }

    # Process Windows KB articles
    if kb_articles_windows:
        df_windows = pd.DataFrame(
            kb_articles_windows, columns=list(kb_articles_windows[0].keys())
        )
        # Filter out duplicates before other operations
        df_windows = df_windows.drop_duplicates(subset=["kb_id"], keep="first")

        df_windows["kb_id"] = df_windows["kb_id"].apply(normalize_mongo_kb_id)
        df_windows["kb_id"] = df_windows["kb_id"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x
        )

        df_windows = validate_and_adjust_columns(df_windows, master_columns)
        df_windows["node_label"] = "KBArticle"
        df_windows["published"] = pd.to_datetime(df_windows["published"])
        df_windows["excluded_embed_metadata_keys"] = [
            [] for _ in range(len(df_windows))
        ]
        df_windows["excluded_embed_metadata_keys"] = df_windows[
            "excluded_embed_metadata_keys"
        ].apply(
            lambda x: list(
                set(x if isinstance(x, list) else [])
                | {
                    "node_id",
                    "cve_ids",
                    "build_number",
                    "node_label",
                    "product_build_id",
                    "product_build_ids",
                }
            )
        )
        # Initialize metadata with etl_processing_status
        df_windows["metadata"] = [{
            "etl_processing_status": {
                "document_processed": True,
                "entities_extracted": False,
                "graph_prepared": False,
                "vector_prepared": False,
                "last_processed_at": None,
                "processing_version": "1.0",
            }
        } for _ in range(len(df_windows))]

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        logging.info(f"Generating summaries for {df_windows.shape[0]} Windows-based KBs")
        df_windows_has_summary = df_windows[df_windows["summary"].notna()].copy()
        df_windows_no_summary = df_windows[df_windows["summary"].isna()].copy()
        logging.info(f"Windows-based KBs with no summaries: {df_windows_no_summary.shape[0]}")
        df_windows_no_summary["summary"] = ""
        summaries = loop.run_until_complete(generate_summaries(df_windows_no_summary["text"]))
        df_windows_no_summary["summary"] = summaries
        df_windows = pd.concat([df_windows_has_summary, df_windows_no_summary])
        df_windows.sort_values(by="kb_id", ascending=True, inplace=True)

        print(f"Total Windows-based KBs transformed: {df_windows.shape[0]}")

    else:
        df_windows = pd.DataFrame(columns=master_columns)
        print("No Windows KB articles to transform.")

    # Process Edge KB articles
    if kb_articles_edge:
        df_edge = pd.DataFrame(
            kb_articles_edge, columns=list(kb_articles_edge[0].keys())
        )

        df_edge["kb_id"] = df_edge["kb_id"].apply(normalize_mongo_kb_id)
        df_edge["kb_id"] = df_edge["kb_id"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) == 1 else ""
        )

        df_edge = df_edge.drop_duplicates(subset=["kb_id"], keep="first")
        df_edge = validate_and_adjust_columns(df_edge, master_columns)
        df_edge["node_label"] = "KBArticle"
        df_edge["published"] = pd.to_datetime(df_edge["published"])
        df_edge["excluded_embed_metadata_keys"] = [[] for _ in range(len(df_edge))]
        df_edge["excluded_embed_metadata_keys"] = df_edge[
            "excluded_embed_metadata_keys"
        ].apply(
            lambda x: list(
                set(x if isinstance(x, list) else [])
                | {
                    "node_id",
                    "cve_ids",
                    "build_number",
                    "node_label",
                    "product_build_id",
                    "product_build_ids",
                }
            )
        )

        # Initialize metadata with etl_processing_status
        df_edge["metadata"] = [{
            "etl_processing_status": {
                "document_processed": True,
                "entities_extracted": False,
                "graph_prepared": False,
                "vector_prepared": False,
                "last_processed_at": None,
                "processing_version": "1.0",
            }
        } for _ in range(len(df_edge))]

        df_edge["summary"] = ""
        df_edge.sort_values(by="kb_id", ascending=True, inplace=True)
        print(f"Total Edge-based KBs transformed: {df_edge.shape[0]}")

    else:
        df_edge = pd.DataFrame(columns=master_columns)
        print("No Edge-based KB articles to transform.")

    # Combine Windows and Edge KB articles
    dfs_to_concat = []
    for df in [df_windows, df_edge]:
        if not df.empty:
            logging.info("Applying dtypes to non-empty DataFrame")
            # Apply dtypes to non-empty DataFrames
            for col, dtype in dtypes.items():
                if col in df.columns:
                    try:
                        df[col] = df[col].astype(dtype)
                    except (ValueError, TypeError):
                        print(f"Warning: Could not convert column {col} to {dtype}")
            dfs_to_concat.append(df)

    # Only concatenate if we have DataFrames to combine
    if dfs_to_concat:
        kb_articles_combined_df = pd.concat(
            dfs_to_concat,
            axis=0,
            ignore_index=True,
            copy=True
        )

        kb_articles_combined_df = kb_articles_combined_df.rename(
            columns={"id": "node_id"}
        )

        # Convert build_number to tuple for comparison (if it's a list)
        kb_articles_combined_df["build_number_tuple"] = kb_articles_combined_df[
            "build_number"
        ].apply(lambda x: tuple(x) if isinstance(x, list) else x)

        # Drop duplicates keeping first occurrence
        kb_articles_combined_df = kb_articles_combined_df.drop_duplicates(
            subset=["build_number_tuple", "kb_id", "published", "product_build_id"],
            keep="first",
        )

        # Remove the temporary tuple column
        kb_articles_combined_df = kb_articles_combined_df.drop(
            columns=["build_number_tuple"]
        )

        print(f"Total KB articles transformed: {kb_articles_combined_df.shape[0]}")
        return kb_articles_combined_df
    else:
        print("No KB articles to transform.")
        return pd.DataFrame(columns=master_columns)


def process_downloadable_packages(
    packages: Union[str, List[Dict[str, Any]], None]
) -> str:
    if packages is None or (isinstance(packages, str) and packages.strip() == ""):
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


def transform_update_packages(update_packages: List[Dict[str, Any]]) -> pd.DataFrame:
    mapping_types = {"security_hotpatch_update": "security_hotpatch"}
    # clean up document dict from mongo to align with data models
    if update_packages:
        df = pd.DataFrame(update_packages, columns=list(update_packages[0].keys()))
        df["package_type"] = df["package_type"].str.lower().str.replace(" ", "_")
        df["package_type"] = df["package_type"].apply(lambda x: mapping_types.get(x, x))
        # df["downloadable_packages"] = df["downloadable_packages"].apply(
        #     process_downloadable_packages
        # )
        df["node_label"] = "UpdatePackage"
        df["published"] = pd.to_datetime(df["published"])
        df["excluded_embed_metadata_keys"] = [[] for _ in range(len(df))]
        df["excluded_embed_metadata_keys"] = df["excluded_embed_metadata_keys"].apply(
            lambda x: list(
                set(x if isinstance(x, list) else [])
                | {
                    "product_build_ids",
                    "node_label",
                    "downloadable_packages",
                    "source",
                }
            )
        )
        df = df.rename(columns={"id": "node_id"})
        # print(df["downloadable_packages"])
        print(f"Total Update Packages transformed: {df.shape[0]}")
        # print(f"Columns: {df.columns}")
        return df
    else:
        print("No Update Packages to transform.")
        return pd.DataFrame(columns=["node_id", "package_type", "node_label", "published",
                                   "excluded_embed_metadata_keys", "downloadable_packages"])


def convert_to_list(value):
    if isinstance(value, str):
        return [value]
    elif isinstance(value, list):
        return value


def make_json_safe_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make metadata dictionary JSON-safe while preserving dictionary structure.
    Only converts to JSON string if the value itself needs to be JSON-encoded.

    Args:
        metadata: Dictionary containing metadata fields

    Returns:
        Dictionary with JSON-safe values but still in dictionary format
    """
    if not isinstance(metadata, dict):
        return metadata

    safe_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            safe_metadata[key] = make_json_safe_metadata(value)
        elif isinstance(value, (datetime, np.datetime64)):
            # Convert datetime to ISO format string
            safe_metadata[key] = value.isoformat()
        elif isinstance(value, (list, tuple)):
            # Handle lists/tuples
            safe_metadata[key] = [
                make_json_safe_metadata(item) if isinstance(item, dict) else item
                for item in value
            ]
        elif isinstance(value, (float, np.float32, np.float64)) and np.isnan(value):
            # Handle NaN values
            safe_metadata[key] = None
        elif isinstance(value, (np.int64, np.int32)):
            # Convert numpy integers to Python integers
            safe_metadata[key] = int(value)
        elif isinstance(value, (np.bool_)):
            # Convert numpy booleans to Python booleans
            safe_metadata[key] = bool(value)
        else:
            return []
    return safe_metadata


def remove_generic_text(text, threshold=80, max_match_length=500):

    initial_char_count = len(text)
    problematic_pattern = r"This metric describes the conditions beyond the attacker's control that must exist in order to exploit the vulnerability. Such conditions may require the collection of more information about the target or computational exceptions. The assessment of this metric excludes any requirements for user interaction in order to exploit the vulnerability. If a specific configuration is required for an attack to succeed, the Base metrics should be scored assuming the vulnerable component is in that configuration."
    icon_pattern = r"[^\w\s]+\s+Subscribe\s+RSS\s+PowerShell\s+[^\w\s]+\s+API"
    generic_text_patterns = [
        r"This metric reflects the context by which vulnerability exploitation is possible. The Base Score increases the more remote \(logically, and physically\) an attacker can be in order to exploit the vulnerable component.",
        problematic_pattern,
        r"This metric describes the level of privileges an attacker must possess before successfully exploiting the vulnerability.",
        r"This metric captures the requirement for a user, other than the attacker, to participate in the successful compromise the vulnerable component. This metric determines whether the vulnerability can be exploited solely at the will of the attacker, or whether a separate user \(or user-initiated process\) must participate in some manner.",
        r"Does a successful attack impact a component other than the vulnerable component\? If so, the Base Score increases and the Confidentiality, Integrity and Authentication metrics should be scored relative to the impacted component.",
        r"This metric measures the impact to the confidentiality of the information resources managed by a software component due to a successfully exploited vulnerability. Confidentiality refers to limiting information access and disclosure to only authorized users, as well as preventing access by, or disclosure to, unauthorized ones.",
        r"This metric measures the impact to integrity of a successfully exploited vulnerability. Integrity refers to the trustworthiness and veracity of information.",
        r"This metric measures the impact to the availability of the impacted component resulting from a successfully exploited vulnerability. It refers to the loss of availability of the impacted component itself, such as a networked service \(e.g., web, database, email\). Since availability refers to the accessibility of information resources, attacks that consume network bandwidth, processor cycles, or disk space all impact the availability of an impacted component.",
        r"This metric measures the likelihood of the vulnerability being attacked, and is typically based on the current state of exploit techniques, public availability of exploit code, or active, 'in-the-wild' exploitation.",
        r"The Remediation Level of a vulnerability is an important factor for prioritization. The typical vulnerability is unpatched when initially published. Workarounds or hotfixes may offer interim remediation until an official patch or upgrade is issued. Each of these respective stages adjusts the temporal score downwards, reflecting the decreasing urgency as remediation becomes final.",
        r"This metric measures the degree of confidence in the existence of the vulnerability and the credibility of the known technical details. Sometimes only the existence of vulnerabilities are publicized, but without specific details. For example, an impact may be recognized as undesirable, but the root cause may not be known. The vulnerability may later be corroborated by research which suggests where the vulnerability may lie, though the research may not be certain. Finally, a vulnerability may be confirmed through acknowledgement by the author or vendor of the affected technology. The urgency of a vulnerability is higher when a vulnerability is known to exist with certainty. This metric also suggests the level of technical knowledge available to would-be attackers.",
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

            elif pattern == icon_pattern:  # Fuzzy matching for the icon pattern
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
                        if score >= threshold and len(best_match) < max_match_length:
                            modified_text = modified_text.replace(best_match, "")
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
    if not description:
        return "NC"

    description = description.lower()

    # Define keywords with variations for each category
    category_patterns = {
        "rce": [
            "remote code execution",
            "remote execution",
            "arbitrary code execution",
            "code execution",
            "command execution"
        ],
        "privilege_elevation": [
            "elevation of privilege",
            "privilege elevation",
            "escalation of privilege",
            "privilege escalation"
        ],
        "dos": [
            "denial of service",
            "denial-of-service",
            "service denial",
            "resource exhaustion"
        ],
        "disclosure": [
            "information disclosure",
            "information leak",
            "data disclosure",
            "memory leak",
            "sensitive information"
        ],
        "tampering": [
            "tampering",
            "data manipulation",
            "unauthorized modification"
        ],
        "spoofing": [
            "spoofing",
            "impersonation",
            "authentication bypass"
        ],
        "feature_bypass": [
            "security feature bypass",
            "security bypass",
            "protection bypass"
        ],
        "availability": [
            "availability",
            "system crash",
            "system hang"
        ]
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
        "availability"
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


def _prepare_base_dataframe(msrc_posts: List[Dict[str, Any]], metadata_fields_to_move: List[str]) -> pd.DataFrame:
    """Prepare the initial dataframe from MSRC posts."""
    df = pd.DataFrame(msrc_posts, columns=list(msrc_posts[0].keys()))
    for field in metadata_fields_to_move:
        df[field] = df["metadata"].apply(lambda x: x.get(field, None))
    return df


def _process_new_records(df: pd.DataFrame, mapping_cve_category: Dict[str, str]) -> pd.DataFrame:
    """Process records that haven't been enriched with NVD data yet."""
    # Rename impact_type to cve_category only for new records
    df = df.rename(columns={"impact_type": "cve_category"})

    # Handle the mapping with None/NaN protection
    df["cve_category"] = df["cve_category"].apply(
        lambda x: mapping_cve_category.get(x, x) if pd.notna(x) else "NC"
    )
    return df


def _apply_common_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """Apply transformations common to both new and pre-processed records."""
    df["cve_category"] = df["cve_category"].fillna("NC")
    df["severity_type"] = df["severity_type"].str.lower()
    df["severity_type"] = df["severity_type"].fillna("NST")
    # df["metadata"] = df["metadata"].apply(make_json_safe_metadata)
    df["kb_ids"] = df["kb_ids"].apply(normalize_mongo_kb_id)
    df["kb_ids"] = df["kb_ids"].apply(lambda x: sorted(x, reverse=True))
    df["product_build_ids"] = df["product_build_ids"].apply(convert_to_list)
    df["node_label"] = "MSRCPost"
    df["published"] = pd.to_datetime(df["published"])

    df["excluded_embed_metadata_keys"] = df["excluded_embed_metadata_keys"].apply(
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
    df = df.rename(columns={"id_": "node_id"})
    return df


def transform_msrc_posts(msrc_posts: List[Dict[str, Any]], process_all: bool = False) -> pd.DataFrame:
    """Transform MSRC posts, handling both new and pre-processed records efficiently.

    Args:
        msrc_posts: List of MSRC post dictionaries
        process_all: If True, process all records. If False, only process new records.
    """
    logging.info(f"process New or All: {'All' if process_all else 'New'}")
    metadata_fields_to_move = [
        "revision", "title", "description", "source", "impact_type",
        "severity_type", "post_type", "post_id", "summary", "build_numbers",
        "published", "product_build_ids", "collection", "cve_category",
    ]

    if not msrc_posts:
        print("No MSRC Posts to transform.")
        return None

    # Create initial dataframe
    df = _prepare_base_dataframe(msrc_posts, metadata_fields_to_move)

    # Ensure metadata column exists and has proper structure
    if 'metadata' not in df.columns:
        df['metadata'] = [{'etl_processing_status': {}} for _ in range(len(df))]
    else:
        df['metadata'] = df['metadata'].apply(
            lambda x: {
                **(x if isinstance(x, dict) else {}),
                'etl_processing_status': x.get('etl_processing_status', {}) if isinstance(x, dict) else {}
            }
        )

    # Partition records based on whether they've been processed before
    is_processed = df["metadata"].apply(
        lambda x: x.get("etl_processing_status", {}).get("nvd_extracted", False)
    )
    preprocessed_records = df[is_processed].copy()
    new_records = df[~is_processed].copy()

    logging.info(f"Found {len(preprocessed_records)} pre-processed records and {len(new_records)} new records")

    # Determine which records to process based on process_all flag
    if process_all:
        records_to_process = df
        logging.info("Processing all records as requested")
    else:
        records_to_process = new_records
        if not preprocessed_records.empty:
            logging.info(f"Skipping {len(preprocessed_records)} pre-processed records")

    # Process records if there are any to process
    if not records_to_process.empty:
        # Initialize processing status
        current_time = datetime.now().isoformat()
        records_to_process['metadata'] = records_to_process['metadata'].apply(
            lambda x: {
                **(x if isinstance(x, dict) else {}),
                'etl_processing_status': {
                    **(x.get('etl_processing_status', {}) if isinstance(x, dict) else {}),
                    'document_processed': True,
                    'nvd_extracted': False,
                    'entities_extracted': False,
                    'graph_prepared': False,
                    'vector_prepared': False,
                    'last_processed_at': current_time,
                    'processing_version': '1.0'
                }
            }
        )

        # Set up NVD extraction
        num_cves = len(records_to_process)
        scraping_params = ScrapingParams.from_target_time(
            num_cves=num_cves, target_time_per_cve=4.0
        )
        estimated_minutes = scraping_params.estimate_total_time(num_cves) / 60
        print(f"Estimated processing time for {num_cves} records: {estimated_minutes:.1f} minutes")

        nvd_extractor = NVDDataExtractor(
            properties_to_extract=[
                "base_score", "vector_element", "impact_score", "exploitability_score",
                "attack_vector", "attack_complexity", "privileges_required",
                "user_interaction", "scope", "confidentiality", "integrity",
                "availability", "nvd_published_date", "nvd_description",
                "vector_element", "cwe_id", "cwe_name", "cwe_source", "cwe_url",
            ],
            max_records=None,
            scraping_params=scraping_params,
            headless=True,
            window_size=(1240, 1080),
            show_progress=True,
        )

        try:
            enriched_records = nvd_extractor.augment_dataframe(
                df=records_to_process,
                url_column="post_id",
                batch_size=100,
            )

            if not enriched_records.empty:
                # Update CVE categories where needed
                mask = (enriched_records['cve_category'].isin(['NC', '']))
                enriched_records.loc[mask, 'cve_category'] = \
                    enriched_records.loc[mask, 'nvd_description'].apply(extract_cve_category_from_description)

                # Log statistics about category updates
                updated_count = mask.sum()
                if updated_count > 0 and os.getenv('LOG_LEVEL', '').upper() == 'DEBUG':
                    category_stats = enriched_records.loc[mask, 'cve_category'].value_counts()
                    logging.debug(f"Updated {updated_count} CVE categories from NVD descriptions")
                    logging.debug("Category distribution for updated records:")
                    logging.debug(f"\n{category_stats}")

                # Update processing status after successful NVD extraction
                current_time = datetime.now().isoformat()
                enriched_records["metadata"] = enriched_records["metadata"].apply(
                    lambda x: {
                        **(x if isinstance(x, dict) else {}),
                        "etl_processing_status": {
                            **(x.get('etl_processing_status', {}) if isinstance(x, dict) else {}),
                            "nvd_extracted": True,
                            "last_processed_at": current_time
                        }
                    }
                )

        except Exception as e:
            logging.error(f"Error during NVD data extraction: {str(e)}")
            raise
        finally:
            nvd_extractor.cleanup()
    else:
        enriched_records = pd.DataFrame()

    # Return logic based on process_all flag
    if not process_all:
        # Return only newly processed records
        if not enriched_records.empty:
            result_df = _apply_common_transformations(enriched_records)
            result_df.sort_values(by="post_id", ascending=True, inplace=True)
            print(f"Total new MSRC Posts transformed: {result_df.shape[0]}")
            return result_df #has etl dict
        return None
    else:
        # Combine and return all records
        if not preprocessed_records.empty and not enriched_records.empty:
            combined_df = pd.concat([preprocessed_records, enriched_records], axis=0, ignore_index=True)
        elif not preprocessed_records.empty:
            combined_df = preprocessed_records
        elif not enriched_records.empty:
            combined_df = enriched_records
        else:
            logging.error("No records to process")
            return None

        # Apply common transformations and sort
        result_df = _apply_common_transformations(combined_df)
        result_df.sort_values(by="post_id", ascending=True, inplace=True)
        print(f"Total MSRC Posts transformed: {result_df.shape[0]}")
        return result_df

# End MSRC transformer ========================================

# Start Patch Transformer =====================================


nlp = spacy.load("en_core_web_lg")


def normalize_subject(subject):
    # Define the patterns to remove
    patterns_to_remove = [
        r"\[patchmanagement\]",
        r"\[External\]",
        r"\[EXTERNAL\]",
        r"🟣",  # Specifically remove this emoji
    ]

    # Remove the specified patterns including the space immediately after
    for pattern in patterns_to_remove:
        subject = re.sub(pattern + r"\s*", "", subject, flags=re.IGNORECASE)

    # Remove any remaining emojis or special symbols
    subject = re.sub(r"[^\w\s]", "", subject).strip()

    # Lowercase and tokenize the subject
    words = subject.lower().split()

    # Remove stop words
    words = [word for word in words if word not in STOP_WORDS]

    # Get the first 5 words for raw extraction
    first_five_words = "_".join(words[:5])

    # Extract key phrases using spaCy
    doc = nlp(" ".join(words))
    noun_chunks = list(doc.noun_chunks)
    key_phrases = [chunk.text.replace(" ", "_") for chunk in noun_chunks[:5]]

    # Check the similarity between the first 5 words and each key phrase
    meaningful_key_phrases = []
    for phrase in key_phrases:
        similarity = fuzz.ratio(first_five_words, phrase)
        if similarity < 70:  # Threshold can be adjusted based on needs
            meaningful_key_phrases.append(phrase)

    # Combine the first 5 words and meaningful key phrases
    combined_words = first_five_words
    if meaningful_key_phrases:
        combined_words += "_" + "_".join(meaningful_key_phrases[:2])

    # Deduplicate by keeping the first occurrence of each word
    words_list = combined_words.split("_")
    deduplicated_words = []
    seen = set()
    for word in words_list:
        if word not in seen:
            deduplicated_words.append(word)
            seen.add(word)

    # Join the deduplicated words back into a single string
    deduplicated_combined_words = "_".join(deduplicated_words)

    return deduplicated_combined_words


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
        normalized_subject = normalize_subject(row["metadata"]["subject"])
        matched = False
        for key in groups:
            if fuzz.ratio(normalized_subject, key) >= similarity_threshold:
                groups[key].append(idx)
                matched = True
                break
        if not matched:
            groups[normalized_subject].append(idx)
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
        thread_id = generate_thread_id(df.loc[sorted_group[0], "metadata"]["subject"])

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


def transform_patch_posts(patch_posts: List[Dict[str, Any]], process_all: bool = False) -> pd.DataFrame:
    """Transform patch posts, handling both new and pre-processed records efficiently.

    Args:
        patch_posts: List of patch post dictionaries
        process_all: If True, process all records. If False, only process new records.
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

    if not patch_posts:
        print("No patch posts to transform.")
        return None

    df = pd.DataFrame(patch_posts)

    # Extract metadata fields
    for field in metadata_fields_to_move:
        if field not in df.columns:
            df[field] = df["metadata"].apply(lambda x: x.get(field, None))

    # Rename columns
    df = df.rename(
        columns={
            "id_": "node_id",
            "evaluated_noun_chunks": "noun_chunks",
            "evaluated_keywords": "keywords",
            "cve_mentions": "cve_ids",
            "kb_mentions": "kb_ids",
        }
    )
    df["kb_ids"] = df["kb_ids"].apply(lambda x: sorted(x, reverse=True))
    # Add node_label
    df["node_label"] = "PatchManagementPost"

    # Partition records based on whether they've been processed before
    is_processed = df["metadata"].apply(
        lambda x: x.get("etl_processing_status", {}).get("document_processed", False)
    )
    preprocessed_records = df[is_processed].copy()
    new_records = df[~is_processed].copy()

    logging.info(f"Found {len(preprocessed_records)} pre-processed records and {len(new_records)} new records")

    # Determine which records to process based on process_all flag
    if process_all:
        records_to_process = df
        logging.info("Processing all records as requested")
    else:
        records_to_process = new_records
        if not preprocessed_records.empty:
            logging.info(f"Skipping {len(preprocessed_records)} pre-processed records")

    if not records_to_process.empty:
        # Process emails
        records_to_process = process_emails(records_to_process)

        # Update processing status in metadata
        current_time = datetime.utcnow().isoformat()
        records_to_process["metadata"] = records_to_process["metadata"].apply(
            lambda x: {
                **x,
                "etl_processing_status": {
                    "document_processed": True,
                    "entities_extracted": False,
                    "graph_prepared": False,
                    "vector_prepared": False,
                    "last_processed_at": current_time,
                    "processing_version": "1.0"  # Useful for future schema migrations
                }
            }
        )
    else:
        records_to_process = pd.DataFrame()

    # Return logic based on process_all flag
    if not process_all:
        # Return only newly processed records
        if not records_to_process.empty:
            # Make metadata JSON safe
            metadata_fields_to_move.append("build_numbers")
            records_to_process["metadata"] = records_to_process["metadata"].apply(
                lambda x: remove_metadata_fields(x, metadata_fields_to_move)
            )
            # records_to_process["metadata"] = records_to_process["metadata"].apply(make_json_safe_metadata)
            records_to_process["excluded_embed_metadata_keys"] = [[] for _ in range(len(records_to_process))]
            records_to_process["excluded_embed_metadata_keys"] = records_to_process["excluded_embed_metadata_keys"].apply(
                lambda x: list(
                    set(x if isinstance(x, list) else [])
                    | {
                        "previous_id",
                        "cve_ids",
                        "kb_ids",
                        "next_id",
                        "node_label",
                        "subject",
                        "etl_processing_status"  # Exclude processing status from embeddings
                    }
                )
            )
            # Remove duplicate columns
            records_to_process = records_to_process.loc[:, ~records_to_process.columns.duplicated()]
            records_to_process["published"] = pd.to_datetime(records_to_process["published"])
            print(f"Total new patch posts transformed: {records_to_process.shape[0]}")
            return records_to_process
        return None
    else:
        # Combine and return all records
        if not preprocessed_records.empty and not records_to_process.empty:
            combined_df = pd.concat([preprocessed_records, records_to_process], axis=0, ignore_index=True)
        elif not preprocessed_records.empty:
            combined_df = preprocessed_records
        elif not records_to_process.empty:
            combined_df = records_to_process
        else:
            logging.error("No records to process")
            return None

        # Make metadata JSON safe
        metadata_fields_to_move.append("build_numbers")
        combined_df["metadata"] = combined_df["metadata"].apply(
            lambda x: remove_metadata_fields(x, metadata_fields_to_move)
        )
        combined_df["metadata"] = combined_df["metadata"].apply(make_json_safe_metadata)
        combined_df["excluded_embed_metadata_keys"] = [[] for _ in range(len(combined_df))]
        combined_df["excluded_embed_metadata_keys"] = combined_df["excluded_embed_metadata_keys"].apply(
            lambda x: list(
                set(x if isinstance(x, list) else [])
                | {
                    "previous_id",
                    "cve_ids",
                    "kb_ids",
                    "next_id",
                    "node_label",
                    "subject",
                    "etl_processing_status"  # Exclude processing status from embeddings
                }
            )
        )
        # Remove duplicate columns
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        combined_df["published"] = pd.to_datetime(combined_df["published"])
        print(f"Total patch posts transformed: {combined_df.shape[0]}")
        return combined_df

# End Patch Transformer ========================================


def transform_symptoms(symptoms: List[Dict[str, Any]]) -> pd.DataFrame:
    # clean up document dict from mongo to align with data models
    if symptoms:
        df = pd.DataFrame(symptoms, columns=list(symptoms[0].keys()))
        if not all(
            col in df.columns for col in ["severity_type", "node_label", "reliability"]
        ):
            df = df.assign(
                severity_type="NST",
                node_label="Symptom",
                reliability="HIGH",
            )
        df["labels"] = df["labels"].apply(lambda x: x[0] if isinstance(x, list) else x)

        print(f"Total Products transformed: {df.shape[0]}")
        # print(f"Columns: {df.columns}")
        return df
    else:
        print("No Symptoms to transform.")

    return None


def transform_causes(causes: List[Dict[str, Any]]) -> pd.DataFrame:
    # clean up document dict from mongo to align with data models
    if causes:
        df = pd.DataFrame(causes, columns=list(causes[0].keys()))
        if not all(
            col in df.columns for col in ["severity_type", "node_label", "reliability"]
        ):
            df = df.assign(
                severity_type="NST",
                node_label="Cause",
                reliability="MEDIUM",
            )
        df["labels"] = df["labels"].apply(lambda x: x[0] if isinstance(x, list) else x)

        print(f"Total Causes transformed: {df.shape[0]}")
        # print(f"Columns: {df.columns}")
        return df
    else:
        print("No Causes to transform.")

    return None


def transform_fixes(fixes: List[Dict[str, Any]]) -> pd.DataFrame:
    # clean up document dict from mongo to align with data models
    if fixes:
        df = pd.DataFrame(fixes, columns=list(fixes[0].keys()))
        if not all(
            col in df.columns for col in ["severity_type", "node_label", "reliability"]
        ):
            df = df.assign(
                severity_type="NST",
                node_label="Fix",
                reliability="MEDIUM",
            )
        df["labels"] = df["labels"].apply(lambda x: x[0] if isinstance(x, list) else x)

        print(f"Total Fixes transformed: {df.shape[0]}")
        # print(f"Columns: {df.columns}")
        return df
    else:
        print("No Fixes to transform.")

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
        df["labels"] = df["labels"].apply(lambda x: x[0] if isinstance(x, list) else x)

        print(f"Total Tools transformed: {df.shape[0]}")
        # print(f"Columns: {df.columns}")
        return df
    else:
        print("No Tools to transform.")

    return None


def transform_technologies(technologies: List[Dict[str, Any]]) -> pd.DataFrame:
    # clean up document dict from mongo to align with data models
    if technologies:
        df = pd.DataFrame(technologies, columns=list(technologies[0].keys()))
        if not all(col in df.columns for col in ["node_label"]):
            df = df.assign(
                node_label="Fix",
            )
        df["labels"] = df["labels"].apply(lambda x: x[0] if isinstance(x, list) else x)

        print(f"Total Technologies transformed: {df.shape[0]}")
        # print(f"Columns: {df.columns}")
        return df
    else:
        print("No Technologies to transform.")

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
        lambda row: next(
            key for key, items in dict1.items() if row.to_dict() in items + dict2[key]
        ),
        axis=1,
    )

    return df


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
        "Cause": ["node_id", "description", "source_id", "source_type", "tags"],
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
    # Replace quotes and other special characters
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
    exclude_columns: List[str] = None
) -> dict:
    """
    Helper function to create metadata from a DataFrame row, handling NaN values and datetime objects.

    Args:
        row: DataFrame row to process
        preserve_metadata: If True, preserves existing metadata structure from source,
                         otherwise places all fields (except doc_id) in metadata
    """
    logging.debug(f"Creating metadata from row with keys: {row.index.tolist()}")

    def process_value(value):
        """Helper to process individual values."""
        if isinstance(value, (pd.Timestamp, datetime)):
            # Handle datetime objects
            return value.isoformat()
        elif isinstance(value, (pd.Series, np.ndarray)):
            # Handle Series or arrays
            return value.tolist()
        elif isinstance(value, list):
            # Handle lists and nested lists
            return [
                process_value(sub_value) if isinstance(sub_value, list) else sub_value
                for sub_value in value
            ]
        elif pd.api.types.is_scalar(value):
            # Handle scalar NaN, NAType, and primitive types
            if pd.isna(value) or isinstance(value, pd._libs.missing.NAType):
                return None
            return value
        elif isinstance(value, str):
            # Handle strings
            return make_text_json_safe(value)
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
                logging.warning(f"Failed to parse metadata string row:{row['node_id']}")
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
            logging.error(f"Error processing key {key} with value {value} ({type(value)}): {str(e)}")

    logging.debug(f"Final metadata structure - Root keys: {list(metadata.keys())}")
    logging.debug(f"Final metadata structure - Metadata keys: {list(metadata.keys())}")

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
            if isinstance(x, str) and x.lower() in {"nan", "nat", "none", "null"}:
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
        [clean_value(x) for x in series],
        index=series.index,
        name=series.name
    )


def convert_df_to_llamadoc_kb_articles(
    df: pd.DataFrame
) -> list[LlamaDocument]:
    """Convert a DataFrame of KB Articles to a list of LlamaDocuments."""
    llama_documents = []
    logging.info(
        f"Starting conversion of KB Articles to LlamaDocuments "
        f"with row count: {len(df)}"
    )
    for _, row in df.iterrows():
        try:
            logging.debug(
                f"Processing row:\n"
                f"{row.to_dict()}"
            )

            exclude_columns = [
                'text',
            ]
            # Extract metadata dynamically, excluding specified keys
            metadata_dict = _create_metadata_from_row(
                clean_na_values_series(row),
                preserve_metadata=True,
                exclude_columns=exclude_columns
            )

            def get_clean_value(row, key, default):
                """Helper to get value from row, handling NA values"""
                value = row.get(key)
                if isinstance(value, (list, np.ndarray)):
                    return value  # Return arrays/lists as is
                if value is None or isinstance(value, pd._libs.missing.NAType) or (pd.api.types.is_scalar(value) and pd.isna(value)):
                    return default
                return value

            # Set default values for specific fields
            defaults = {
                "node_label": get_clean_value(row, "node_label", "KBArticle"),
                "kb_id": get_clean_value(row, "kb_id", ""),
                "product_build_id": get_clean_value(row, "product_build_id", ""),
                "product_build_ids": get_clean_value(row, "product_build_ids", []),
                "cve_ids": get_clean_value(row, "cve_ids", []),
                "build_number": get_clean_value(row, "build_number", []),
                "article_url": get_clean_value(row, "article_url", ""),
                "reliability": get_clean_value(row, "reliability", ""),
                "readability": get_clean_value(row, "readability", ""),
                "excluded_embed_metadata_keys": get_clean_value(row, "excluded_embed_metadata_keys", [
                    "node_label",
                    "product_build_id",
                    "product_build_ids",
                ]),
                "excluded_llm_metadata_keys": get_clean_value(row, "excluded_llm_metadata_keys", [
                    "node_label",
                    "product_build_id",
                    "product_build_ids",
                ]),
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
                "processing_version": "1.0"
            }
            # Create the LlamaDocument
            doc = LlamaDocument(
                text=_handle_na_text(row["text"]),
                doc_id=row["node_id"],
                extra_info=metadata_dict,
                excluded_embed_metadata_keys=metadata_dict["excluded_embed_metadata_keys"],
                excluded_llm_metadata_keys=metadata_dict["excluded_llm_metadata_keys"],
            )
            llama_documents.append(doc)

            logging.info("Created LlamaDocument", extra={"doc_id": doc.doc_id})
            logging.debug(
                f"Document metadata:\n"
                f"{metadata_dict}"
            )

        except Exception as e:
            logging.error(f"Error processing row row_data {row.to_dict()} \nerror {str(e)}")

    logging.info(
        f"Completed conversion of DataFrame to LlamaDocuments "
        f"with total documents: {len(llama_documents)}"
    )
    return llama_documents


# =============================================================================
# Convert dataframe of Update Packages to Llama Documents
# =============================================================================
def convert_df_to_llamadoc_update_packages(
    df: pd.DataFrame
) -> list[LlamaDocument]:
    """Convert a DataFrame of Update Packages to a list of LlamaDocuments."""
    llama_documents = []

    logging.info(
        f"Starting conversion of Update Packages to LlamaDocuments "
        f"with row count: {len(df)}"
    )

    for _, row in df.iterrows():
        try:
            # Log raw input data for debugging
            logging.debug(
                f"Processing row:\n"
                f"{row.to_dict()}"
            )

            # Construct text content
            text_content = (
                f"Update Package published on: {row.get('published', '').strftime('%B %d, %Y')}\n"
                f"Build Number: {row.get('build_number', '')}\n"
                f"Package Type: {row.get('package_type', '')}\n"
                f"Package URL: {row.get('package_url', '')}"
            )

            # Handle datetime conversion for published date
            if "published" in row and isinstance(row["published"], pd.Timestamp):
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
            metadata = _create_metadata_from_row(row, preserve_metadata=False, exclude_columns=exclude_columns)

            # Set default values for specific fields
            defaults = {
                "node_label": row.get("node_label", "UpdatePackage"),  # Use node_label from row or default to "UpdatePackage"
                "build_number": row.get("build_number", ""),
                "product_build_ids": row.get("product_build_ids", []),
                "package_type": row.get("package_type", ""),
                "package_url": row.get("package_url", ""),
                "downloadable_packages": downloadable_packages,
                "reliability": row.get("reliability", ""),
                "readability": row.get("readability", "HIGH"),
                "excluded_embed_metadata_keys": ["product_build_ids", "node_label"],
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
                excluded_embed_metadata_keys=metadata["excluded_embed_metadata_keys"],
                excluded_llm_metadata_keys=metadata["excluded_llm_metadata_keys"],
            )
            llama_documents.append(doc)

            # Log document creation details
            logging.info("Created LlamaDocument", extra={"doc_id": doc.doc_id})
            logging.debug("Document metadata", extra={"metadata": metadata})

        except Exception as e:
            # Log any errors encountered during processing
            logging.error("Error processing row", extra={"row_data": row.to_dict(), "error": str(e)})

    # Log summary of the conversion process
    logging.info(
        f"Completed conversion of Update Packages to LlamaDocuments "
        f"with total documents: {len(llama_documents)}"
    )

    return llama_documents


# =============================================================================
# Convert dataframe of Symptoms to Llama Documents
# =============================================================================
def convert_df_to_llamadoc_symptoms(
    df: pd.DataFrame
) -> list[LlamaDocument]:
    """Convert a DataFrame of Symptoms to a list of LlamaDocuments."""
    llama_documents = []

    logging.info(
        f"Starting conversion of Symptoms to LlamaDocuments "
        f"with row count: {len(df)}"
    )

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
                "excluded_embed_metadata_keys": ["node_label", "symptom_label"],
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
                excluded_embed_metadata_keys=metadata["excluded_embed_metadata_keys"],
                excluded_llm_metadata_keys=metadata["excluded_llm_metadata_keys"],
            )
            llama_documents.append(doc)

            # Log document creation details
            logging.info("Created LlamaDocument", extra={"doc_id": doc.doc_id})
            logging.debug("Document metadata", extra={"metadata": metadata})

        except Exception as e:
            # Log any errors encountered during processing
            logging.error("Error processing row", extra={"row_data": row.to_dict(), "error": str(e)})

    # Log summary of the conversion process
    logging.info(
        f"Completed conversion of Symptoms to LlamaDocuments "
        f"with total documents: {len(llama_documents)}"
    )

    return llama_documents


# =============================================================================
# Convert dataframe of Causes to Llama Documents
# =============================================================================
def convert_df_to_llamadoc_causes(
    df: pd.DataFrame
) -> list[LlamaDocument]:
    """Convert a DataFrame of Causes to a list of LlamaDocuments."""
    llama_documents = []

    logging.info(
        f"Starting conversion of Causes to LlamaDocuments "
        f"with row count: {len(df)}"
    )

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
                excluded_embed_metadata_keys=metadata["excluded_embed_metadata_keys"],
                excluded_llm_metadata_keys=metadata["excluded_llm_metadata_keys"],
            )
            llama_documents.append(doc)

            # Log document creation details
            logging.info("Created LlamaDocument", extra={"doc_id": doc.doc_id})
            logging.debug("Document metadata", extra={"metadata": metadata})

        except Exception as e:
            # Log any errors encountered during processing
            logging.error("Error processing row", extra={"row_data": row.to_dict(), "error": str(e)})

    # Log summary of the conversion process
    logging.info(
        f"Completed conversion of Causes to LlamaDocuments "
        f"with total documents: {len(llama_documents)}"
    )

    return llama_documents


# =============================================================================
# Convert dataframe of Fixes to Llama Documents
# =============================================================================
def convert_df_to_llamadoc_fixes(
    df: pd.DataFrame
) -> list[LlamaDocument]:
    """Convert a DataFrame of Fix nodes to a list of LlamaDocuments."""
    llama_documents = []

    logging.info(
        f"Starting conversion of Fix nodes to LlamaDocuments "
        f"with row count: {len(df)}"
    )

    for _, row in df.iterrows():
        try:
            # Log raw input row data for debugging
            logging.debug("Processing row", extra={"row_data": row.to_dict()})

            # Extract metadata dynamically, excluding specified keys
            metadata = _create_metadata_from_row(clean_na_values_series(row))

            # Set default values for specific fields
            defaults = {
                "node_label": row.get("node_label", "Fix"),  # Use node_label from row or default to "Fix"
                "fix_label": row.get("fix_label", ""),
                "cve_ids": row.get("cve_ids", []),
                "reliability": row.get("reliability", "HIGH"),
                "readability": row.get("readability", "HIGH"),
                "excluded_embed_metadata_keys": ["node_label", "fix_label", "cve_ids"],
                "excluded_llm_metadata_keys": []
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
                excluded_embed_metadata_keys=metadata["excluded_embed_metadata_keys"],
                excluded_llm_metadata_keys=metadata["excluded_llm_metadata_keys"]
            )
            llama_documents.append(doc)

            # Log document creation details
            logging.info("Created LlamaDocument", extra={"doc_id": doc.doc_id})
            logging.debug("Document metadata", extra={"metadata": metadata})

        except Exception as e:
            # Log any errors encountered during processing
            logging.error("Error processing row", extra={"row_data": row.to_dict(), "error": str(e)})

    # Log summary of the conversion process
    logging.info(
        f"Completed conversion of Fix nodes to LlamaDocuments "
        f"with total documents: {len(llama_documents)}"
    )

    return llama_documents


# =============================================================================
# Convert dataframe of Tools to Llama Documents
# =============================================================================
def convert_df_to_llamadoc_tools(
    df: pd.DataFrame
) -> list[LlamaDocument]:
    """Convert a DataFrame of Tool nodes to a list of LlamaDocuments."""
    llama_documents = []

    logging.info(
        f"Starting conversion of Tool nodes to LlamaDocuments "
        f"with row count: {len(df)}"
    )

    for _, row in df.iterrows():
        try:
            # Log raw input row data for debugging
            logging.debug("Processing row", extra={"row_data": row.to_dict()})

            # Extract metadata dynamically, excluding specified keys
            metadata = _create_metadata_from_row(clean_na_values_series(row))

            # Set default values for specific fields
            defaults = {
                "node_label": row.get("node_label", "Tool"),  # Use node_label from row or default to "Tool"
                "tool_label": row.get("tool_label", ""),
                "name": row.get("name", ""),
                "cve_ids": row.get("cve_ids", []),
                "tool_url": row.get("tool_url", ""),
                "reliability": "HIGH",
                "readability": "HIGH",
                "excluded_embed_metadata_keys": ["node_label", "tool_label", "node_id"],
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
                excluded_embed_metadata_keys=metadata["excluded_embed_metadata_keys"],
                excluded_llm_metadata_keys=metadata["excluded_llm_metadata_keys"],
            )
            llama_documents.append(doc)

            # Log document creation details
            logging.info("Created LlamaDocument", extra={"doc_id": doc.doc_id})
            logging.debug("Document metadata", extra={"metadata": metadata})

        except Exception as e:
            # Log any errors encountered during processing
            logging.error("Error processing row", extra={"row_data": row.to_dict(), "error": str(e)})

    # Log summary of the conversion process
    logging.info(
        f"Completed conversion of Tool nodes to LlamaDocuments "
        f"with total documents: {len(llama_documents)}"
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

    logging.info(
        f"Starting conversion of MSRC posts to LlamaDocuments "
        f"with row count: {len(df)}"
    )

    for _, row in df.iterrows():
        try:
            # Log raw input row data for debugging
            logging.debug("Processing row", extra={"row_data": row.to_dict()})
            exclude_columns = ["text"]
            # Create metadata using helper function
            metadata_dict = _create_metadata_from_row(
                clean_na_values_series(row),
                preserve_metadata=True,
                exclude_columns=exclude_columns
            )
            logging.debug(
                f"Generated metadata:\n"
                f"{metadata_dict}"
            )

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

            logging.debug(
                f"Default metadata values:\n"
                f"{defaults}"
            )

            # Update metadata with defaults
            for key, default_value in defaults.items():
                if key not in metadata_dict or metadata_dict[key] is None:
                    metadata_dict[key] = default_value
                    logging.debug(
                        f"Applied default value:\n"
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
                "processing_version": "1.0"
            }
            # Create the LlamaDocument
            doc = LlamaDocument(
                text=_handle_na_text(row["text"]),
                doc_id=metadata_dict["doc_id"],
                extra_info=metadata_dict,
                excluded_embed_metadata_keys=metadata_dict["excluded_embed_metadata_keys"],
                excluded_llm_metadata_keys=metadata_dict["excluded_llm_metadata_keys"],
            )
            llama_documents.append(doc)

            # Log document creation
            logging.info("Created LlamaDocument", extra={"doc_id": doc.doc_id})
            logging.debug(
                f"Document metadata:\n"
                f"{doc.metadata}"
            )

        except Exception as e:
            # Log any errors encountered during processing
            logging.error("Error processing row", extra={"row_data": row.to_dict(), "error": str(e)})

    # Log summary of the conversion process
    logging.info(
        f"Completed conversion of MSRC posts to LlamaDocuments "
        f"with total documents: {len(llama_documents)}"
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

    logging.info(
        f"Starting conversion of PatchManagement posts to LlamaDocuments "
        f"with row count: {len(df)}"
    )

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
            logging.debug(f"Generated metadata: \n{metadata_dict}")
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
                "etl_processing_status": row.get("etl_processing_status", {
                    'document_processed': True,
                    'entities_extracted': True,
                    'graph_prepared': True,
                    'vector_prepared': True,
                    'last_processed_at': current_time,
                    'processing_version': '1.0'
                }),
            }

            # Update metadata with defaults
            for key, default_value in defaults.items():
                if key not in metadata_dict or metadata_dict[key] is None:
                    metadata_dict[key] = default_value
                    logging.debug("Applied default value", extra={"key": key, "default_value": default_value})

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
            metadata_dict['etl_processing_status']['last_processed_at'] = current_time
            # Create the LlamaDocument
            doc = LlamaDocument(
                text=_handle_na_text(row["text"]),
                doc_id=row["node_id"],
                extra_info=metadata_dict,
                excluded_embed_metadata_keys=metadata_dict["excluded_embed_metadata_keys"],
                excluded_llm_metadata_keys=metadata_dict["excluded_llm_metadata_keys"],
            )
            llama_documents.append(doc)

            # Log document creation
            logging.info(f"Created LlamaDocument doc_id: {doc.doc_id}")
            logging.debug(f"Document metadata: {doc.extra_info}")

        except Exception as e:
            # Log any errors encountered during processing
            logging.error(f"Error processing row_data: {row.to_dict()}\n error: {str(e)}")

    # Log summary of the conversion process
    logging.info(
        f"Completed conversion of PatchManagement posts to LlamaDocuments "
        f"with total documents: {len(llama_documents)}"
    )

    return llama_documents
# End Convert dataframes to LlamaDocs ==========================
