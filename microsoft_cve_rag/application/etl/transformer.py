# Purpose: Transform extracted data
# Inputs: Raw data
# Outputs: Transformed data
# Dependencies: None
import os
import logging
from typing import Union, List, Dict, Any
import pandas as pd
from fuzzywuzzy import fuzz,process
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
from llama_index.core.schema import Document as LlamaDocument
import asyncio
import marvin
from marvin.ai.text import generate_llm_response
marvin.settings.openai.chat.completions.model = 'gpt-4o-mini'
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
    kb_articles_windows: List[Dict[str, Any]], kb_articles_edge: List[Dict[str, Any]]
) -> pd.DataFrame:
    # clean up document dict from mongo to align with data models
    master_columns = [
        "id",
        "build_number",
        "kb_id",
        "published",
        "product_build_id",
        "article_url",
        "cve_ids",
        "text",
        "title",
        "embedding",
        "product_build_ids",
    ]

    def flatten_kb_id(kb_id):
        if isinstance(kb_id, list):
            return ", ".join(kb_id)
        return kb_id
    

    if kb_articles_windows:
        df_windows = pd.DataFrame(
            kb_articles_windows, columns=list(kb_articles_windows[0].keys())
        )
        # Filter out duplicates before other operations
        df_windows = df_windows.drop_duplicates(
            subset=["kb_id"],
            keep='first'
        )
        
        
        df_windows["kb_id"] = df_windows["kb_id"].apply(normalize_mongo_kb_id)
        df_windows["kb_id"] = df_windows["kb_id"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x
        )
        # df_windows["kb_ids"] = df_windows["kb_ids"].apply(lambda x: sorted(x, reverse=True))

        # df_strings = df_windows[df_windows["kb_id"].apply(lambda x: isinstance(x, str))]

        # df_windows["embedding"] = df_windows.apply(
        #     lambda row: embedding_service.generate_embeddings(row["text"]), axis=1
        # )
        
        df_windows = validate_and_adjust_columns(df_windows, master_columns)
        df_windows["node_label"] = "KBArticle"
        df_windows["published"] = pd.to_datetime(df_windows["published"])
        df_windows["excluded_embed_metadata_keys"] = [[] for _ in range(len(df_windows))]
        df_windows["excluded_embed_metadata_keys"] = df_windows["excluded_embed_metadata_keys"].apply(lambda x: list(set(x if isinstance(x, list) else []) | {"node_id", "cve_ids", "build_number", "node_label", "product_build_id", 'product_build_ids'}))
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        summaries = loop.run_until_complete(generate_summaries(df_windows['text']))
        df_windows['summary'] = summaries
        df_windows.sort_values(by="kb_id", ascending=True, inplace=True)
        # print(f"Columns: {df_windows.columns}")
        # for _, row in df_windows.iterrows():
        #     print(f"{row['kb_id']}-{row['product_build_ids']}: {row['summary']}\n")
        print(f"Total Windows-based KBs transformed: {df_windows.shape[0]}")

    else:
        df_windows = pd.DataFrame(columns=master_columns)
        print("No Windows-based KB articles to transform.")

    if kb_articles_edge:
        df_edge = pd.DataFrame(
            kb_articles_edge, columns=list(kb_articles_edge[0].keys())
        )

        df_edge["kb_id"] = df_edge["kb_id"].apply(normalize_mongo_kb_id)
        df_edge["kb_id"] = df_edge["kb_id"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) == 1 else ""
        )
        # df_edge["kb_ids"] = df_edge["kb_ids"].apply(lambda x: sorted(x, reverse=True))
        # df_edge["embedding"] = df_edge.apply(
        #     lambda row: embedding_service.generate_embeddings(row["text"]), axis=1
        # )
        # df_lists = df_edge[df_edge["kb_id"].apply(lambda x: isinstance(x, list))]
        # print("Rows with lists in 'kb_id':")
        # print(df_lists.sample(n=df_lists.shape[0]))
        df_edge = df_edge.drop_duplicates(
            subset=["kb_id"],
            keep='first'
        )
        df_edge = validate_and_adjust_columns(df_edge, master_columns)
        df_edge["node_label"] = "KBArticle"
        df_edge["published"] = pd.to_datetime(df_edge["published"])
        df_edge["excluded_embed_metadata_keys"] = [[] for _ in range(len(df_edge))]
        df_edge["excluded_embed_metadata_keys"] = df_edge["excluded_embed_metadata_keys"].apply(lambda x: list(set(x if isinstance(x, list) else []) | {"node_id", "cve_ids", "build_number", "node_label", "product_build_id", 'product_build_ids'}))
        df_edge.sort_values(by="kb_id", ascending=True, inplace=True)
        # print(f"Columns: {df_edge.columns}")
        print(f"Total Edge-based KBs transformed: {df_edge.shape[0]}")

    else:
        df_edge = pd.DataFrame(columns=master_columns)
        print("No Edge-based KB articles to transform.")

    if not df_windows.empty or not df_edge.empty:
        kb_articles_combined_df = pd.concat([df_windows, df_edge], axis=0)
        kb_articles_combined_df = kb_articles_combined_df.rename(
            columns={"id": "node_id"}
        )

        # Convert build_number to tuple for comparison (if it's a list)
        kb_articles_combined_df['build_number_tuple'] = kb_articles_combined_df['build_number'].apply(
            lambda x: tuple(x) if isinstance(x, list) else x
        )
        
        # Drop duplicates keeping first occurrence
        kb_articles_combined_df = kb_articles_combined_df.drop_duplicates(
            subset=['build_number_tuple', 'kb_id', 'published', 'product_build_id'],
            keep='first'
        )
        
        # Remove the temporary tuple column
        kb_articles_combined_df = kb_articles_combined_df.drop(columns=['build_number_tuple'])
        
        # Log the deduplication results
        logging.info(f"Removed {len(kb_articles_combined_df) - kb_articles_combined_df.shape[0]} duplicate KB articles")

        return kb_articles_combined_df
    else:
        return None

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
        df["excluded_embed_metadata_keys"] = df["excluded_embed_metadata_keys"].apply(lambda x: list(set(x if isinstance(x, list) else []) | {"product_build_ids","node_label","downloadable_packages", "source"}))
        df = df.rename(columns={"id": "node_id"})
        # print(df["downloadable_packages"])
        print(f"Total Update Packages transformed: {df.shape[0]}")
        # print(f"Columns: {df.columns}")
        return df
    else:
        print("No Update Packages to transform.")

    return None


def make_json_safe_metadata(metadata: Dict[str, Any]) -> str:
    for key, value in metadata.items():
        if isinstance(value, datetime):
            metadata[key] = value.isoformat()
    return json.dumps(metadata, default=custom_json_serializer)


def convert_to_list(value):
    if isinstance(value, str):
        return [value]
    elif isinstance(value, list):
        return value
    else:
        return []

def remove_generic_text(text, threshold=80, max_match_length=500):
    # logging.info(f"Initial character count: {len(text)}")
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
            modified_text = re.sub(pattern, '', modified_text)
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
                segments = problematic_pattern.split('. ')
                for segment in segments:
                    best_match, score = process.extractOne(segment, [modified_text], scorer=fuzz.partial_ratio)
                    
                    # Only replace if the score is high, the match length is reasonable, and it’s not too broad
                    if (score >= threshold and len(best_match) < max_match_length 
                            and len(best_match) < len(modified_text) * 0.5):
                        modified_text = modified_text.replace(best_match, '')
                        patterns_found += 1
                        # print(f"Pattern segment removed using fuzzy matching: {segment[:30]}... (Score: {score}) | modified_text len: {len(modified_text)}")
                    else:
                        # print(f"Pattern segment skipped in fuzzy matching due to length or low score: {segment[:30]}... (Score: {score})")
                        pass
            
            elif pattern == icon_pattern:  # Fuzzy matching for the icon pattern
                if re.search(icon_pattern, modified_text):
                    modified_text = re.sub(icon_pattern, '', modified_text)
                    patterns_found += 1
                    # print(f"Icon pattern found and removed with regex | modified_text len: {len(modified_text)}")
                else:
                    # Fallback to fuzzy matching if regex fails
                    # print("Icon pattern not matched by regex; attempting fuzzy matching for each segment.")
                    segments = ["Subscribe", "RSS", "PowerShell", "API"]
                    for segment in segments:
                        best_match, score = process.extractOne(segment, [modified_text], scorer=fuzz.partial_ratio)
                        
                        # Only replace if the match score is high and the length is reasonable
                        if score >= threshold and len(best_match) < max_match_length:
                            modified_text = modified_text.replace(best_match, '')
                            patterns_found += 1
                            # print(f"Icon pattern segment removed using fuzzy matching: {segment} (Score: {score}) | modified_text len: {len(modified_text)}")
                        else:
                            # print(f"Icon pattern segment skipped in fuzzy matching due to length or low score: {segment} (Score: {score})")
                            pass
                        
    final_char_count = len(modified_text)
    # logging.info(f"Final character count: {final_char_count}")
    # Return the modified text and a summary of patterns found
    return modified_text.strip(), patterns_found, len(generic_text_patterns) - patterns_found


def transform_msrc_posts(msrc_posts: List[Dict[str, Any]]) -> pd.DataFrame:
    # clean up document dict from mongo to align with data models
    metadata_fields_to_move = [
        "revision",
        "title",
        "description",
        "source",
        "impact_type",
        "severity_type",
        "post_type",
        "post_id",
        "summary",
        "build_numbers",
        "published",
        "product_build_ids",
    ]
    mapping_cve_category = {
        "Tampering": "tampering",
        "Spoofing": "spoofing",
        "Availability": "availability",
        "Elevation of Privilege": "privilege_elevation",
        "Denial of Service": "dos",
        "Information Disclosure": "disclosure",
        "Remote Code Execution": "rce",
        "Security Feature Bypass": "feature_bypass",
    }
    
    if msrc_posts:
        df = pd.DataFrame(msrc_posts, columns=list(msrc_posts[0].keys()))

        for field in metadata_fields_to_move:
            df[field] = df["metadata"].apply(lambda x: x.get(field, None))
        # df["embedding"] = df.apply(
        #     lambda row: embedding_service.generate_embeddings(row["text"]), axis=1
        # )
        df = df.rename(columns={"impact_type": "cve_category"})
        # df["cve_category"] = df["cve_category"].str.lower().str.replace(" ", "_")
        df["cve_category"] = df["cve_category"].apply(lambda x: mapping_cve_category.get(x, x))
        df["cve_category"] = df["cve_category"].fillna("NC")
        df["severity_type"] = df["severity_type"].str.lower()
        df["severity_type"] = df["severity_type"].fillna("NST")
        df["metadata"] = df["metadata"].apply(make_json_safe_metadata)
        df["kb_ids"] = df["kb_ids"].apply(normalize_mongo_kb_id)
        df["kb_ids"] = df["kb_ids"].apply(lambda x: sorted(x, reverse=True))
        df["product_build_ids"] = df["product_build_ids"].apply(convert_to_list)
        df["node_label"] = "MSRCPost"
        df["published"] = pd.to_datetime(df["published"])
        
        df["excluded_embed_metadata_keys"] = df["excluded_embed_metadata_keys"].apply(lambda x: list(set(x if isinstance(x, list) else []) | {"source", "description", "product_build_ids", "kb_ids", "build_numbers", "node_label", 'patterns_found', 'patterns_missing'}))
        df[['text', 'patterns_found', 'patterns_missing']] = df['text'].apply(lambda x: pd.Series(remove_generic_text(x)))
        
        df = df.rename(columns={"id_": "node_id"})
        df.sort_values(by="post_id", ascending=True, inplace=True)
        
        num_cves = len(df)
        scraping_params = ScrapingParams.from_target_time(
            num_cves=num_cves,
            target_time_per_cve=4.0  # (seconds)
        )
        estimated_minutes = scraping_params.estimate_total_time(num_cves) / 60
        print(f"Estimated processing time: {estimated_minutes:.1f} minutes")
        
        nvd_extractor = NVDDataExtractor(
            properties_to_extract=[
                'base_score',
                'vector_element',
                'impact_score',
                'exploitability_score',
                'attack_vector',
                'attack_complexity',
                'privileges_required',
                'user_interaction',
                'scope',
                'confidentiality',
                'integrity',
                'availability',
                'nvd_published_date',
                'nvd_description',
                'vector_element',
                'cwe_id',
                'cwe_name',
                'cwe_source',
                'cwe_url'
            ],
            max_records=None,
            scraping_params=scraping_params,
            headless=True,  # Set to False for debugging
            window_size=(1240, 1080),
            show_progress=True
        )
        
        try:
            enriched_df = nvd_extractor.augment_dataframe(
                df=df, 
                url_column='post_id',
                batch_size=100  # Adjust based on your needs
            )
            logging.debug(f"enriched_df.columns:\n{enriched_df.columns}")
            if not enriched_df.empty:
                sample_size = min(3, len(enriched_df))
                logging.debug(f"enriched_df.sample(n={sample_size}):\n{enriched_df.sample(n=sample_size)}")
            else:
                logging.debug("enriched_df is empty - no samples to display")

        except Exception as e:
            print(f"Error during NVD data extraction: {str(e)}")
            raise
            
        finally:
            nvd_extractor.cleanup()
        
        if not enriched_df.empty:
            print(f"Total MSRC Posts transformed: {enriched_df.shape[0]}")
            return enriched_df
        else:
            logging.error("NVD extraction failed - no enriched data to return")
            return None
    else:
        print("No MSRC Posts to transform.")
        return None


nlp = spacy.load("en_core_web_lg")


def normalize_subject(subject):
    # Define the patterns to remove
    patterns_to_remove = [
        r"\[patchmanagement\]",
        r"\[External\]",
        r"\[EXTERNAL\]",
        r"🟣"   # Specifically remove this emoji
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


def transform_patch_posts(patch_posts: List[Dict[str, Any]]) -> pd.DataFrame:
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
            "kb_mentions": "kb_ids"

        }
    )
    df["kb_ids"] = df["kb_ids"].apply(lambda x: sorted(x, reverse=True))
    # Add node_label
    df["node_label"] = "PatchManagementPost"

    # Process emails
    df = process_emails(df)

    # Make metadata JSON safe
    metadata_fields_to_move.append("build_numbers")
    df["metadata"] = df["metadata"].apply(
        lambda x: remove_metadata_fields(x, metadata_fields_to_move)
    )
    df["metadata"] = df["metadata"].apply(make_json_safe_metadata)
    df["excluded_embed_metadata_keys"] = [[] for _ in range(len(df))]
    df["excluded_embed_metadata_keys"] = df["excluded_embed_metadata_keys"].apply(lambda x: list(set(x if isinstance(x, list) else []) | {"previous_id", "cve_ids", "kb_ids","next_id", "node_label","subject"}))
    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    df['published'] = pd.to_datetime(df['published'])
    logging.info(f"Total Patch posts transformed: {df.shape[0]}")
    logging.debug(f"Patch df columns: {df.columns}")
    # print(df.head())

    return df


def transform_symptoms(symptoms: List[Dict[str, Any]]) -> pd.DataFrame:
    # clean up document dict from mongo to align with data models
    if symptoms:
        df = pd.DataFrame(symptoms, columns=list(symptoms[0].keys()))
        if not all(
            col in df.columns for col in ["severity_type", "node_label", "reliability"]
        ):
            df = df.assign(
                severity_type='NST',
                node_label='Symptom',
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
                severity_type='NST',
                node_label='Cause',
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
                severity_type='NST',
                node_label='Fix',
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
                node_label='Tool',
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
                node_label='Fix',
            )
        df["labels"] = df["labels"].apply(lambda x: x[0] if isinstance(x, list) else x)

        print(f"Total Technologies transformed: {df.shape[0]}")
        # print(f"Columns: {df.columns}")
        return df
    else:
        print("No Technologies to transform.")

    return None

def combine_dicts_to_dataframe(dict1: Dict[str, List[Dict]], dict2: Dict[str, List[Dict]]) -> pd.DataFrame:
    combined_data = []
    
    # Iterate through all keys (assuming both dicts have the same keys)
    for key in dict1.keys():
        # Extend the combined_data list with items from both dictionaries
        combined_data.extend(dict1[key])
        combined_data.extend(dict2[key])
    
    # Create a DataFrame from the combined list of dictionaries
    df = pd.DataFrame(combined_data)
    
    # Add a column to identify the original category (key)
    df['category'] = df.apply(lambda row: next(key for key, items in dict1.items() if row.to_dict() in items + dict2[key]), axis=1)
    
    return df

def combine_and_split_dicts(dict1: Dict[str, List[Dict]], dict2: Dict[str, List[Dict]]) -> Dict[str, pd.DataFrame]:
    category_dataframes = {}
    
    # Iterate through all keys (assuming both dicts have the same keys)
    for key in dict1.keys():
        # Combine items from both dictionaries for this category
        combined_items = dict1[key] + dict2[key]
        
        # Create a DataFrame for this category
        df = pd.DataFrame(combined_items)
        # for _, row in df.iterrows():
        #     print(f"{row['source_type']} - {row['description']}")
        # Add to our dictionary of DataFrames
        category_dataframes[key] = df
    
    return category_dataframes

def transform_extracted_entities(
    entities_list: List[Dict], entity_type: str
) -> pd.DataFrame:
    df = pd.DataFrame(entities_list)
    # Ensure all required fields are present
    required_fields = {
        'Symptom': [
            'node_id',
            'symptom_label',
            'description',
            'source_id',
            'source_type',
            'tags',
        ],
        'Cause': ['node_id','description','source_id','source_type','tags'],
        'Fix': ['node_id','description','source_id','source_type','tags'],
        'Tool': [
            'node_id',
            'name',
            'description',
            'source_id',
            'source_type',
            'tags',
            'source_url',
        ],
        'Technology': [
            'node_id',
            'name',
            'description',
            'source_id',
            'source_type',
            'tags',
        ]

    }
    for field in required_fields.get(entity_type, []):
        if field not in df.columns:
            df[field] = None  # Set default value if field is missing
    return df

# =============================================================================
# Convert dataframe of KB Articles to Llama Documents
# =============================================================================
def convert_df_to_llamadoc_kb_articles(df: pd.DataFrame, exclusion_keys: set[str] = {"text", "node_id"}) -> list[LlamaDocument]:
    llama_documents = []

    for _, row in df.iterrows():
        # Handle datetime conversion for 'published' field
        if 'published' in row and isinstance(row['published'], pd.Timestamp):
            row['published'] = row['published'].isoformat()

        # Extract metadata dynamically, excluding specified keys
        metadata = {key: row[key] for key in row.index if key not in exclusion_keys}

        # Set default values for specific fields
        defaults = {
            "source": "KBArticle",
            "kb_id": "",
            "title": "",
            "product_build_id": "",
            "product_build_ids": [],
            "cve_ids": [],
            "build_number": [],
            "article_url": "",
            "excluded_embed_metadata_keys": ["node_id", "cve_ids", "build_number", "node_label", "product_build_id", "product_build_ids"],
        }

        # Update metadata with defaults
        for key, default_value in defaults.items():
            if key not in metadata or not metadata[key]:
                metadata[key] = default_value

        # Create the LlamaDocument
        doc = LlamaDocument(
            text=row["text"],
            doc_id=row["node_id"],
            metadata=metadata,
            excluded_embed_metadata_keys=metadata["excluded_embed_metadata_keys"],
        )
        llama_documents.append(doc)

    return llama_documents

# =============================================================================
# Convert dataframe of Update Packages to Llama Documents
# =============================================================================
def convert_df_to_llamadoc_update_packages(df: pd.DataFrame, exclusion_keys: set[str] = {"text", "node_id", "downloadable_packages"}) -> list[LlamaDocument]:
    llama_documents = []

    for _, row in df.iterrows():
        # Construct text content
        text_content = (
            f"Update Package published on: {row.get('published', '').strftime('%B %d, %Y')}\n"
            f"Build Number: {row.get('build_number', '')}\n"
            f"Package Type: {row.get('package_type', '')}\n"
            f"Package URL: {row.get('package_url', '')}"
        )

        # Handle datetime conversion for published date
        if 'published' in row and isinstance(row['published'], pd.Timestamp):
            row['published'] = row['published'].isoformat()

        # Process downloadable packages
        downloadable_packages = row.get('downloadable_packages', [])
        if isinstance(downloadable_packages, list):
            downloadable_packages = [
                {k: v.isoformat() if isinstance(v, datetime) else v 
                 for k, v in pkg.items()}
                if isinstance(pkg, dict) else pkg
                for pkg in downloadable_packages
            ]

        # Extract metadata dynamically, excluding specified keys
        metadata = {key: row[key] for key in row.index if key not in exclusion_keys}

        # Set default values for specific fields
        defaults = {
            "source": "UpdatePackage",
            "build_number": "",
            "product_build_ids": [],
            "package_type": "",
            "package_url": "",
            "downloadable_packages": downloadable_packages,
            "excluded_embed_metadata_keys": ["product_build_ids", "node_label"],
        }

        # Update metadata with defaults
        for key, default_value in defaults.items():
            if key not in metadata or not metadata[key]:
                metadata[key] = default_value

        # Create the LlamaDocument
        doc = LlamaDocument(
            text=text_content,
            doc_id=row["node_id"],
            metadata=metadata,
            excluded_embed_metadata_keys=metadata["excluded_embed_metadata_keys"],
        )
        llama_documents.append(doc)

    return llama_documents

# =============================================================================
# Convert dataframe of Symptoms to Llama Documents
# =============================================================================
def convert_df_to_llamadoc_symptoms(df: pd.DataFrame, exclusion_keys: set[str] = {"text", "node_id", "description"}) -> list[LlamaDocument]:
    llama_documents = []

    for _, row in df.iterrows():
        # Extract metadata dynamically, excluding specified keys
        metadata = {key: row[key] for key in row.index if key not in exclusion_keys}

        # Set default values for specific fields
        defaults = {
            "source": "Symptom",
            "symptom_label": "",
            "severity_type": "NST",
            "reliability": "MEDIUM",
            "tags": [],
            "excluded_embed_metadata_keys": ["source", "symptom_label"],
        }

        # Update metadata with defaults
        for key, default_value in defaults.items():
            if key not in metadata or not metadata[key]:
                metadata[key] = default_value

        # Create the LlamaDocument
        doc = LlamaDocument(
            text=row["description"],
            doc_id=row["node_id"],
            metadata=metadata,
            excluded_embed_metadata_keys=metadata["excluded_embed_metadata_keys"],
        )
        llama_documents.append(doc)

    return llama_documents

# =============================================================================
# Convert dataframe of Causes to Llama Documents
# =============================================================================
def convert_df_to_llamadoc_causes(df: pd.DataFrame, exclusion_keys: set[str] = {"text", "node_id", "description"}) -> list[LlamaDocument]:
    llama_documents = []

    for _, row in df.iterrows():
        # Extract metadata dynamically, excluding specified keys
        metadata = {key: row[key] for key in row.index if key not in exclusion_keys}

        # Set default values for specific fields
        defaults = {
            "source": "Cause",
            "cause_label": "",
            "severity_type": "NST",
            "reliability": "MEDIUM",
            "tags": [],
            "excluded_embed_metadata_keys": ["source", "cause_label"],
        }

        # Update metadata with defaults
        for key, default_value in defaults.items():
            if key not in metadata or not metadata[key]:
                metadata[key] = default_value

        # Create the LlamaDocument
        doc = LlamaDocument(
            text=row["description"],
            doc_id=row["node_id"],
            metadata=metadata,
            excluded_embed_metadata_keys=metadata["excluded_embed_metadata_keys"],
        )
        llama_documents.append(doc)

    return llama_documents

# =============================================================================
# Convert dataframe of Fixes to Llama Documents
# =============================================================================
def convert_df_to_llamadoc_fixes(df: pd.DataFrame, exclusion_keys: set[str] = {"text", "node_id", "description"}) -> list[LlamaDocument]:
    llama_documents = []

    for _, row in df.iterrows():
        # Extract metadata dynamically, excluding specified keys
        metadata = {key: row[key] for key in row.index if key not in exclusion_keys}

        # Set default values for specific fields
        defaults = {
            "source": "Fix",
            "fix_label": "",
            "severity_type": "NST",
            "reliability": "MEDIUM",
            "tags": [],
            "excluded_embed_metadata_keys": ["source", "fix_label"],
        }

        # Update metadata with defaults
        for key, default_value in defaults.items():
            if key not in metadata or not metadata[key]:
                metadata[key] = default_value

        # Create the LlamaDocument
        doc = LlamaDocument(
            text=row["description"],
            doc_id=row["node_id"],
            metadata=metadata,
            excluded_embed_metadata_keys=metadata["excluded_embed_metadata_keys"],
        )
        llama_documents.append(doc)

    return llama_documents

# =============================================================================
# Convert dataframe of Tools to Llama Documents
# =============================================================================
def convert_df_to_llamadoc_tools(df: pd.DataFrame, exclusion_keys: set[str] = {"text", "node_id", "description"}) -> list[LlamaDocument]:
    llama_documents = []

    for _, row in df.iterrows():
        # Extract metadata dynamically, excluding specified keys
        metadata = {key: row[key] for key in row.index if key not in exclusion_keys}

        # Set default values for specific fields
        defaults = {
            "source": "Tool",
            "tool_label": "",
            "severity_type": "NST",
            "reliability": "MEDIUM",
            "tags": [],
            "tool_url": "",
            "excluded_embed_metadata_keys": ["source", "tool_label"],
        }

        # Update metadata with defaults
        for key, default_value in defaults.items():
            if key not in metadata or not metadata[key]:
                metadata[key] = default_value

        # Create the LlamaDocument
        doc = LlamaDocument(
            text=row["description"],
            doc_id=row["node_id"],
            metadata=metadata,
            excluded_embed_metadata_keys=metadata["excluded_embed_metadata_keys"],
        )
        llama_documents.append(doc)

    return llama_documents

# =============================================================================
# Convert dataframe of MSRC Posts to Llama Documents
# =============================================================================
def convert_df_to_llamadoc_msrc_posts(df: pd.DataFrame, symptom_nodes: list, cause_nodes: list, fix_nodes: list, tool_nodes: list, exclusion_keys: set[str] = {"text", "node_id"}) -> list[LlamaDocument]:
    llama_documents = []

    for _, row in df.iterrows():
        # Handle datetime conversion for 'published' and 'nvd_published_date' fields
        if 'published' in row and isinstance(row['published'], pd.Timestamp):
            row['published'] = row['published'].isoformat()
        if 'nvd_published_date' in row and isinstance(row['nvd_published_date'], pd.Timestamp):
            row['nvd_published_date'] = row['nvd_published_date'].isoformat()

        # Extract metadata dynamically, excluding specified keys
        metadata = {key: row[key] for key in row.index if key not in exclusion_keys}

        # Add extracted nodes and default values
        metadata.update({
            "extracted_symptoms": [node.node_id for node in symptom_nodes if node.source_id == row["node_id"]],
            "extracted_causes": [node.node_id for node in cause_nodes if node.source_id == row["node_id"]],
            "extracted_fixes": [node.node_id for node in fix_nodes if node.source_id == row["node_id"]],
            "extracted_tools": [node.node_id for node in tool_nodes if node.source_id == row["node_id"]],
            "excluded_embed_metadata_keys": row.get("excluded_embed_metadata_keys", [
                "source", "description", "product_build_ids", "kb_ids", "build_numbers"
            ]),
        })

        # Create the LlamaDocument
        doc = LlamaDocument(
            text=row["text"],
            doc_id=row["node_id"],
            metadata=metadata,
            excluded_embed_metadata_keys=metadata["excluded_embed_metadata_keys"],
        )
        llama_documents.append(doc)

    return llama_documents

# =============================================================================
# Convert dataframe of Patch Management Posts to Llama Documents
# =============================================================================
def convert_df_to_llamadoc_patch_posts(df: pd.DataFrame, exclusion_keys: set[str] = {"text", "node_id"}) -> list[LlamaDocument]:
    llama_documents = []

    for _, row in df.iterrows():
        # Convert datetime columns to ISO 8601 strings
        if 'published' in row and isinstance(row['published'], pd.Timestamp):
            row['published'] = row['published'].isoformat()
        if 'receivedDateTime' in row and isinstance(row['receivedDateTime'], pd.Timestamp):
            row['receivedDateTime'] = row['receivedDateTime'].isoformat()

        # Extract metadata dynamically, excluding specified keys
        metadata = {key: row[key] for key in row.index if key not in exclusion_keys}

        # Set default values for specific fields
        defaults = {
            "source": "PatchManagementPost",
            "reliability": "MEDIUM",
            "readability": "MEDIUM",
            "conversation_link": [],
            "kb_ids": [],
            "cve_ids": [],
            "build_numbers": [],
            "product_mentions": [],
            "excluded_embed_metadata_keys": [
                "previous_id", "cve_ids", "kb_ids", "next_id", "node_label", "subject"
            ],
        }

        # Update metadata with defaults
        for key, default_value in defaults.items():
            if key not in metadata or not metadata[key]:
                metadata[key] = default_value

        # Add extracted nodes
        metadata.update({
            "extracted_symptoms": [node.node_id for node in symptom_nodes if node.source_id == row["node_id"]],
            "extracted_causes": [node.node_id for node in cause_nodes if node.source_id == row["node_id"]],
            "extracted_fixes": [node.node_id for node in fix_nodes if node.source_id == row["node_id"]],
            "extracted_tools": [node.node_id for node in tool_nodes if node.source_id == row["node_id"]],
        })

        # Create the LlamaDocument
        doc = LlamaDocument(
            text=row["text"],
            doc_id=row["node_id"],
            metadata=metadata,
            excluded_embed_metadata_keys=metadata["excluded_embed_metadata_keys"],
        )
        llama_documents.append(doc)

    return llama_documents