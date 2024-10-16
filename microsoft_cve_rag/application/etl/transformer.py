# Purpose: Transform extracted data
# Inputs: Raw data
# Outputs: Transformed data
# Dependencies: None
from typing import Union, List, Dict, Any
import pandas as pd
from fuzzywuzzy import fuzz
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
from application.services.embedding_service import EmbeddingService


embedding_service = EmbeddingService.from_provider_name("fastembed")


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
        df["kb_ids"] = df["kb_ids"].apply(lambda x: sorted(x, reverse=True))

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
    ]

    def flatten_kb_id(kb_id):
        if isinstance(kb_id, list):
            return ", ".join(kb_id)
        return kb_id

    if kb_articles_windows:
        df_windows = pd.DataFrame(
            kb_articles_windows, columns=list(kb_articles_windows[0].keys())
        )

        df_windows["kb_id"] = df_windows["kb_id"].apply(normalize_mongo_kb_id)
        df_windows["kb_id"] = df_windows["kb_id"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x
        )
        df_windows["kb_ids"] = df_windows["kb_ids"].apply(lambda x: sorted(x, reverse=True))

        # df_strings = df_windows[df_windows["kb_id"].apply(lambda x: isinstance(x, str))]

        # df_windows["embedding"] = df_windows.apply(
        #     lambda row: embedding_service.generate_embeddings(row["text"]), axis=1
        # )
        df_windows = validate_and_adjust_columns(df_windows, master_columns)
        df_windows["node_label"] = "KBArticle"
        df_windows["published"] = pd.to_datetime(df_windows["published"])
        df_windows.sort_values(by="kb_id", ascending=True, inplace=True)
        # print(f"Columns: {df_windows.columns}")
        # for _, row in df_windows.iterrows():
        #     print(row)
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
        df_edge["kb_ids"] = df_edge["kb_ids"].apply(lambda x: sorted(x, reverse=True))
        # df_edge["embedding"] = df_edge.apply(
        #     lambda row: embedding_service.generate_embeddings(row["text"]), axis=1
        # )
        # df_lists = df_edge[df_edge["kb_id"].apply(lambda x: isinstance(x, list))]
        # print("Rows with lists in 'kb_id':")
        # print(df_lists.sample(n=df_lists.shape[0]))
        df_edge = validate_and_adjust_columns(df_edge, master_columns)
        df_edge["node_label"] = "KBArticle"
        df_edge["published"] = pd.to_datetime(df_edge["published"])
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
    mapping_impacts = {
        "information_disclosure": "disclosure",
        "elevation_of_privilege": "privilege_elevation",
        "nit": "NIT",
    }
    mapping_severity = {"nit": "NIT"}
    if msrc_posts:
        df = pd.DataFrame(msrc_posts, columns=list(msrc_posts[0].keys()))

        for field in metadata_fields_to_move:
            df[field] = df["metadata"].apply(lambda x: x.get(field, None))
        # df["embedding"] = df.apply(
        #     lambda row: embedding_service.generate_embeddings(row["text"]), axis=1
        # )
        df["impact_type"] = df["impact_type"].str.lower().str.replace(" ", "_")
        df["impact_type"] = df["impact_type"].apply(lambda x: mapping_impacts.get(x, x))
        df["impact_type"] = df["impact_type"].fillna("NIT")
        df["severity_type"] = df["severity_type"].str.lower()
        df["severity_type"] = df["severity_type"].apply(
            lambda x: mapping_severity.get(x, x)
        )
        df["severity_type"] = df["severity_type"].fillna("NST")
        df["metadata"] = df["metadata"].apply(make_json_safe_metadata)
        df["kb_ids"] = df["kb_ids"].apply(normalize_mongo_kb_id)
        df["kb_ids"] = df["kb_ids"].apply(lambda x: sorted(x, reverse=True))
        df["product_build_ids"] = df["product_build_ids"].apply(convert_to_list)
        df["node_label"] = "MSRCPost"
        df["published"] = pd.to_datetime(df["published"])
        df = df.rename(columns={"id_": "node_id"})
        df.sort_values(by="post_id", ascending=True, inplace=True)
        # ids_check = []
        # for _, row in df.iterrows():
        #     print(f"{row['post_id']} - {row['product_build_ids']}")
        #         ids_check.append(row["node_id"])
        # quoted_list = ", ".join(f'"{item}"' for item in ids_check)
        # print(quoted_list)
        print(f"Total MSRC Posts transformed: {df.shape[0]}")
        # print(f"Columns: {df.columns}")
        return df
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

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    df['published'] = pd.to_datetime(df['published'])
    print(f"Total Patch posts transformed: {df.shape[0]}")
    # print(f"Columns: {df.columns}")
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
