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

    if isinstance(kb_id_input, str):
        kb_id_list = [kb_id_input]
    else:
        kb_id_list = kb_id_input

    # Function to normalize a single kb_id
    def normalize_kb_id(kb_id):
        # Remove any 'kb' prefix
        kb_id = kb_id.replace("kb", "").replace("KB", "")
        # Ensure the kb_id is in the format KB-XXXXXX or KB-XXX.XXX.XXX.XXX
        return f"KB-{kb_id}"

    # Extract substrings and replace the total strings
    processed_list = [normalize_kb_id(s.split("_")[0]) for s in kb_id_list]

    # Remove duplicates by converting the list to a set and back to a list
    processed_list = list(set(processed_list))

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
        df = df.rename(
            columns={
                "cve_id": "cve_ids",
                "kb_id": "kb_ids",
                "build_number": "build_numbers",
            }
        )
        print(f"Total Products transformed: {df.shape[0]}")
        print(f"Columns: {df.columns}")
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
        print(f"Columns: {df.columns}")
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

    if kb_articles_windows:
        df_windows = pd.DataFrame(
            kb_articles_windows, columns=list(kb_articles_windows[0].keys())
        )

        df_windows["kb_id"] = df_windows["kb_id"].apply(normalize_mongo_kb_id)
        df_windows["kb_id"] = df_windows["kb_id"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x
        )
        # df_windows["embedding"] = df_windows.apply(
        #     lambda row: embedding_service.generate_embeddings(row["text"]), axis=1
        # )
        df_windows = validate_and_adjust_columns(df_windows, master_columns)
        df_windows["node_label"] = "KBArticle"
        df_windows.sort_values(by="kb_id", ascending=True, inplace=True)
        print(f"Columns: {df_windows.columns}")
        # for _, row in df_windows.iterrows():
        #     print(row)
        print(f"Total Windows-based KBs transformed: {df_windows.shape[0]}")

    else:
        df_windows = pd.DataFrame(columns=list(kb_articles_windows[0].keys()))
        print("No Windows-based KB articles to transform.")

    if kb_articles_edge:
        df_edge = pd.DataFrame(
            kb_articles_edge, columns=list(kb_articles_edge[0].keys())
        )

        df_edge["kb_id"] = df_edge["kb_id"].apply(normalize_mongo_kb_id)
        df_edge["kb_id"] = df_edge["kb_id"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x
        )
        # df_edge["embedding"] = df_edge.apply(
        #     lambda row: embedding_service.generate_embeddings(row["text"]), axis=1
        # )
        df_edge = validate_and_adjust_columns(df_edge, master_columns)
        df_edge["node_label"] = "KBArticle"
        df_edge.sort_values(by="kb_id", ascending=True, inplace=True)
        # print(f"Columns: {df_edge.columns}")
        print(f"Total Edge-based KBs transformed: {df_edge.shape[0]}")

    else:
        df_edge = pd.DataFrame(columns=list(kb_articles_edge[0].keys()))
        print("No Edge-based KB articles to transform.")

    if not df_windows.empty or not df_edge.empty:
        kb_articles_combined_df = pd.concat([df_windows, df_edge], axis=0)
        kb_articles_combined_df = kb_articles_combined_df.rename(
            columns={"id": "node_id"}
        )

        return kb_articles_combined_df
    else:
        return None


def custom_json_serializer(obj):
    """Custom JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


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
        df = df.rename(columns={"id": "node_id"})
        # print(df["downloadable_packages"])
        print(f"Total Update Packages transformed: {df.shape[0]}")
        print(f"Columns: {df.columns}")
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
    }
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
        df["severity_type"] = df["severity_type"].fillna("NST")
        df["metadata"] = df["metadata"].apply(make_json_safe_metadata)
        df["kb_ids"] = df["kb_ids"].apply(normalize_mongo_kb_id)
        df["product_build_ids"] = df["product_build_ids"].apply(convert_to_list)
        df["node_label"] = "MSRCPost"
        df = df.rename(columns={"id_": "node_id"})
        df.sort_values(by="post_id", ascending=True, inplace=True)
        # ids_check = []
        # for _, row in df.iterrows():
        #     print(f"{row['post_id']} - {row['product_build_ids']}")
        #         ids_check.append(row["node_id"])
        # quoted_list = ", ".join(f'"{item}"' for item in ids_check)
        # print(quoted_list)
        print(f"Total MSRC Posts transformed: {df.shape[0]}")
        print(f"Columns: {df.columns}")
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
                df.at[idx, "previous_id"] = df.loc[sorted_group[i - 1], "node_id"]

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
        thread_df["sortDate"] = thread_df["metadata"].apply(
            lambda x: x["receivedDateTime"]
        )
        thread_df = thread_df.sort_values(by="sortDate")

        for _, email in thread_df.iterrows():
            print(f"\nEmail ID: {email['id_']}")
            print(f"Subject: {email['metadata']['subject']}")
            print(f"Received: {email['metadata']['receivedDateTime']}")
            print(f"From: {email['metadata'].get('from', 'N/A')}")
            print(f"Previous Email: {email['previous_id'] or 'None'}")
            print(f"Next Email: {email['next_id'] or 'None'}")
            print(f"\nContent Preview: {str(email['text'])[:100]}...")
            print(f"{'-'*50}")


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
    # clean up document dict from mongo to align with data models
    if patch_posts:

        df = pd.DataFrame(patch_posts, columns=list(patch_posts[0].keys()))
        for field in metadata_fields_to_move:
            df[field] = df["metadata"].apply(lambda x: x.get(field, None))
        # print(f"datatypes: {df.dtypes}")
        # df["embedding"] = df.apply(
        #     lambda row: embedding_service.generate_embeddings(row["text"]), axis=1
        # )
        df['kb_ids'] = pd.Series([[] for _ in range(len(df))])
        df["node_label"] = "PatchManagementPost"
        df = df.rename(
            columns={
                "id_": "node_id",
                "evaluated_noun_chunks": "noun_chunks",
                "evaluated_keywords": "keywords",
                "cve_mentions": "cve_ids",
                "kb_mentions": "kb_ids"

            }
        )

        grouped_patch_posts_df = process_emails(df)
        df["metadata"] = df["metadata"].apply(make_json_safe_metadata)
        print(f"Total Patch posts transformed: {df.shape[0]}")
        print(f"Columns: {df.columns}")
        # print_threads(grouped_patch_posts_df)
        return grouped_patch_posts_df
    else:
        print("No patch posts to transform.")

    return None


def transform(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    transformed_data = []
    print("begin transform process")
    # for record in data:
    #     # Transform to Document
    #     document = Document(
    #         embedding=embedding_service.generate_embedding(record.get("text", "")),
    #         metadata=DocumentMetadata(**record.get("metadata", {})),
    #         text=record.get("text", ""),
    #         # Add other fields as needed
    #     )
    #     transformed_data.append({"document": document})

    #     # Transform to Vector
    #     vector = Vector(
    #         embedding=embedding_service.generate_embedding(record.get("text", "")),
    #         metadata=VectorMetadata(**record.get("metadata", {})),
    #         text=record.get("text", ""),
    #         # Add other fields as needed
    #     )
    #     transformed_data.append({"vector": vector})

    #     # Transform to GraphNode
    #     graph_node = GraphNode(
    #         embedding=embedding_service.generate_embedding(record.get("text", "")),
    #         metadata=GraphNodeMetadata(**record.get("metadata", {})),
    #         text=record.get("text", ""),
    #         # Add other fields as needed
    #     )
    #     transformed_data.append({"graph_node": graph_node})

    return transformed_data
