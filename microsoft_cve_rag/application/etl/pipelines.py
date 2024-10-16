from application.etl import extractor
from application.etl import transformer
from application.etl import loader
from application.etl import neo4j_migrator
from application.services.graph_db_service import (
    build_relationships,
    ProductService,
    ProductBuildService,
    MSRCPostService,
    KBArticleService,
    UpdatePackageService,
    PatchManagementPostService,
    SymptomService,
    CauseService,
    FixService,
    ToolService,
    TechnologyService,
)
from application.services.document_service import DocumentService
from application.core.models.basic_models import Document
from application.services.llama_index_service import (
    extract_entities_relationships,
)

from typing import Dict, Any
from datetime import datetime
import time
import asyncio
import re
import json
from datetime import datetime, timedelta
import pandas as pd
from fuzzywuzzy import fuzz

async def incremental_ingestion_pipeline(
    db_name: str, collection_name: str, query: Dict[str, Any]
):
    # A daily incremental ingestion pipeline that extracts data from a MongoDB database and loads it into a vector database and the graph database.
    # A shortcut to call full_ingestion_pipeline() with the start_date and end_date parameters set to the last 24 hours.
    response = {
        "message": "incremental ingestion pipeline complete.",
        "status": "success",
        "code": 200,
    }
    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()
    print(f"start_date: {start_date}, end_date: {end_date}")
    response = await full_ingestion_pipeline(start_date, end_date)
    print(f"response: {response}")
    return response


async def full_ingestion_pipeline(start_date: datetime, end_date: datetime = None):
    response = {
        "message": "full ingestion pipeline complete.",
        "status": "in progress",
        "code": 200,
    }
    print("Begin data extraction ===============================================")
    start_time = time.time()

    # product_docs = await asyncio.to_thread(extractor.extract_products, None)
    # product_build_docs = await asyncio.to_thread(
    #     extractor.extract_product_builds, start_date, end_date, None
    # )
    # product_build_docs_by_build = sorted(product_build_docs, key=sort_by_build_number)
    # kb_article_docs_windows, kb_article_docs_edge = await asyncio.to_thread(
    #     extractor.extract_kb_articles, start_date, end_date, None
    # )
    # update_package_docs = extractor.extract_update_packages(start_date, end_date, None)
    msrc_docs = extractor.extract_msrc_posts(start_date, end_date, 5)
    patch_docs = extractor.extract_patch_posts(start_date, end_date, 5)

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Time taken to extract all data: {int(minutes)} min : {int(seconds)} sec")

    print("Begin data transformation ==========================================")
    start_time = time.time()

    # products_df = await asyncio.to_thread(transformer.transform_products, product_docs)
    # product_builds_df = await asyncio.to_thread(
    #     transformer.transform_product_builds, product_build_docs_by_build
    # )
    # kb_articles_combined_df = await asyncio.to_thread(
    #     transformer.transform_kb_articles, kb_article_docs_windows, kb_article_docs_edge
    # )
    # update_packages_df = await asyncio.to_thread(
    #     transformer.transform_update_packages, update_package_docs
    # )
    msrc_posts_df = await asyncio.to_thread(transformer.transform_msrc_posts, msrc_docs)
    
    patch_posts_df = await asyncio.to_thread(
        transformer.transform_patch_posts, patch_docs
    )
    
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Time taken to transform all data: {int(minutes)} min : {int(seconds)} sec")

    print("Begin Llama Index Processing ====================================")
    start_time = time.time()

    # EXTRACT SYMPTOMS, CAUSES, FIXES, AND TOOLS
    # FROM MSRCPosts and PatchManagementPosts. TODO: include KB Articles kg extraction
    # With Llama Index.
    # print(f"{msrc_posts_df.shape} : {msrc_posts_df.columns}")
    msrc_llm_extracted_data = await extract_entities_relationships(msrc_posts_df, "MSRCPost"
    )
    msrc_total = 0
    for key, value in msrc_llm_extracted_data.items():
        if not isinstance(value, list):
            print(f"Invalid value for {key}: {value}")
        else:
            count = len(value)
            msrc_total += count
    print(f"{msrc_total} non-empty msrc items returned by extract_entities_relationships")
    
    patch_llm_extracted_data = await extract_entities_relationships(
        patch_posts_df, "PatchManagementPost"
    )
    patch_total = 0
    for key, value in patch_llm_extracted_data.items():
        if not isinstance(value, list):
            print(f"Invalid value for {key}: {value}")
        else:
            count = len(value)
            patch_total += count
    print(f"{patch_total} non-empty patch items returned by extract_entities_relationships")
    combined_df = transformer.combine_and_split_dicts(msrc_llm_extracted_data, patch_llm_extracted_data)
    all_symptoms_df = combined_df['symptoms']
    print(f"{all_symptoms_df.shape} : {all_symptoms_df.sample(n=min(all_symptoms_df.shape[0], 6))}")
    all_causes_df = combined_df['causes']
    print(f"{all_causes_df.shape} : {all_causes_df.sample(n=min(all_causes_df.shape[0], 6))}")
    all_fixes_df = combined_df['fixes']
    print(f"{all_fixes_df.shape} : {all_fixes_df.sample(n=min(all_fixes_df.shape[0], 6))}")
    all_tools_df = combined_df['tools']
    print(f"{all_tools_df.shape} : {all_tools_df.sample(n=min(all_tools_df.shape[0], 6))}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(
        f"Time taken to extract Symptoms, Causes,Fixes, and Tools: {int(minutes)} min : {int(seconds)} sec"
    )

    print("Begin data loading ==============================================")
    start_time = time.time()

    # product_load_response = await loader.load_products_graph_db(products_df)
    # product_nodes = product_load_response["nodes"]
    # product_node_ids = product_load_response["insert_ids"]
    # print(product_node_ids)

    # product_build_load_response = await loader.load_product_builds_graph_db(
    #     product_builds_df
    # )
    # product_build_nodes = product_build_load_response["nodes"]
    # product_build_node_ids = product_build_load_response["insert_ids"]

    # # KBArticle nodes <class 'application.core.models.graph_db_models.KBArticle'>
    # kb_article_load_response = await loader.load_kbs_graph_db(kb_articles_combined_df)
    # kb_nodes = kb_article_load_response["nodes"]
    # kb_node_ids = kb_article_load_response["insert_ids"]

    # UpdatePackage nodes <class 'application.core.models.graph_db_models.UpdatePackage'>
    # update_packages_load_response = await loader.load_update_packages_graph_db(
    #     update_packages_df
    # )
    # update_package_nodes = update_packages_load_response["nodes"]
    # update_package_node_ids = update_packages_load_response["insert_ids"]

    # MSRCPost nodes <class 'application.core.models.graph_db_models.MSRCPost'>
    msrc_posts_load_response = await loader.load_msrc_posts_graph_db(msrc_posts_df)
    msrc_post_nodes = msrc_posts_load_response["nodes"]
    # msrc_post_node_ids = msrc_posts_load_response["insert_ids"]

    # PatchManagementPost nodes <class 'application.core.models.graph_db_models.PatchManagementPost'>
    patch_posts_load_response = await loader.load_patch_posts_graph_db(patch_posts_df)
    patch_post_nodes = patch_posts_load_response["nodes"]
    # patch_post_node_ids = patch_posts_load_response["insert_ids"]
    symptom_load_response = await loader.load_symptoms_graph_db(all_symptoms_df)
    symptom_nodes = symptom_load_response["nodes"]
    # for node in symptom_nodes:
    #     print(node, end="\n\n")
    cause_load_response = await loader.load_causes_graph_db(all_causes_df)
    cause_nodes = cause_load_response["nodes"]
    fix_load_response = await loader.load_fixes_graph_db(all_fixes_df)
    fix_nodes = fix_load_response["nodes"]
    tool_load_response = await loader.load_tools_graph_db(all_tools_df)
    tool_nodes = tool_load_response["nodes"]
    # technology_load_response = await loader.load_technologies_graph_db(
    #     all_technologies_df
    # )
    # technology_nodes = technology_load_response["nodes"]

    nodes_dict = {
        # "Product": product_nodes,
        # "ProductBuild": product_build_nodes,
        # "UpdatePackage": update_package_nodes,
        # "KBArticle": kb_nodes,
        "MSRCPost": msrc_post_nodes,
        "PatchManagementPost": patch_post_nodes,
        "Symptom": symptom_nodes,
        "Cause": cause_nodes,
        "Fix": fix_nodes,
        "Tool": tool_nodes,
        # "Technology": technology_nodes,
    }

    await build_relationships(nodes_dict)

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Time taken to load all data: {int(minutes)} min : {int(seconds)} sec")
    print("Full Ingestion complete ============================================")
    response = {
        "message": "full ingestion pipeline complete.",
        "status": "Complete",
        "code": 200,
    }
    return response


def prepare_products(df):
    df["product_name"] = df["product_name"].str.replace("_", " ")
    df["product_full"] = df.apply(
        lambda row: " ".join(
            [
                part
                for part in [
                    row["product_name"],
                    row["product_version"],
                    row["product_architecture"],
                ]
                if part not in ["NV", "NA"]
            ]
        ),
        axis=1,
    )
    df["product_name_version"] = df.apply(
        lambda row: " ".join(
            [
                part
                for part in [row["product_name"], row["product_version"]]
                if part not in ["NV"]
            ]
        ),
        axis=1,
    )

    # Add additional search terms for "edge"
    edge_terms = ["microsoft edge", "chromium", "chromiumbased", "chromium based"]
    df.loc[df["product_name"] == "edge", "product_full"] = df.loc[
        df["product_name"] == "edge", "product_full"
    ].apply(lambda x: x + " " + " ".join(edge_terms))
    df.loc[df["product_name"] == "edge", "product_name_version"] = df.loc[
        df["product_name"] == "edge", "product_name_version"
    ].apply(lambda x: x + " " + " ".join(edge_terms))

    return df


def fuzzy_search(text, search_terms, threshold=80):
    if pd.isna(text):
        return []
    matches = set()
    for term in search_terms:
        if fuzz.partial_ratio(text.lower(), term.lower()) >= threshold:
            matches.add(term)
    return list(matches)


def fuzzy_search_column(column, products_df, threshold=80):
    product_mentions = []
    for text in column:
        if isinstance(text, list):
            text = " ".join(text)
        matches = fuzzy_search(text, products_df["product_full"], threshold)
        if not matches:
            matches = fuzzy_search(text, products_df["product_name_version"], threshold)
        if not matches:
            matches = fuzzy_search(text, products_df["product_name"], threshold)
        product_mentions.append(matches)
    return product_mentions


def retain_most_specific(mentions, products_df):
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
        if not any(mention in other for other in specific_mentions if other != mention):
            final_mentions.add(mention)

    return list(final_mentions)


def convert_to_original_representation(mentions):
    return [mention.replace(" ", "_") for mention in mentions]


def construct_regex_pattern():
    max_digits_per_group = (4, 4, 5, 5)
    pattern = (
        r"\b"
        + r"\.".join(
            [r"\d{1," + str(max_digits) + r"}" for max_digits in max_digits_per_group]
        )
        + r"\b"
    )
    return pattern


# Function to extract build numbers from text
def extract_build_numbers(text, pattern):
    matches = re.findall(pattern, text)
    build_numbers = [[int(part) for part in match.split(".")] for match in matches]
    return build_numbers


def extract_windows_kbs(text):
    pattern = r"(?i)KB[-\s]?\d{6,7}"
    matches = re.findall(pattern, text)
    # Convert matches to uppercase and ensure the format is "KB-123456"
    matches = [match.upper().replace(" ", "").replace("-", "") for match in matches]
    matches = [f"KB-{match[2:]}" for match in matches]
    return list(set(matches))


def extract_edge_kbs(row):
    edge_kbs = []
    if "edge" in row["product_mentions"]:
        for build_number in row["build_numbers"]:
            build_str = ".".join(map(str, build_number))
            edge_kbs.append(f"KB-{build_str}")
    return list(set(edge_kbs))


async def update_email_records(emails_df, document_service):
    for index, row in emails_df.iterrows():
        document_id = row["node_id"]
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
                print(f"doc: {document_id} updated")
            else:
                print(f"{document_id} not updated")
        except Exception as e:
            print(
                f"Exception {e}. Data input: doc id: {document_id}\n{row['product_mentions']}\n{row['build_numbers']}\n{row['kb_mentions']}"
            )


async def patch_feature_engineering_pipeline(
    start_date: datetime, end_date: datetime = None
):

    response = {
        "message": "patch feature engineering pipeline complete.",
        "status": "in progress",
        "code": 200,
    }
    print("Begin data extraction ===============================================")
    start_time = time.time()
    product_docs = await asyncio.to_thread(extractor.extract_products, None)
    patch_docs = await asyncio.to_thread(
        extractor.extract_patch_posts, start_date, end_date, None
    )
    products_df = await asyncio.to_thread(transformer.transform_products, product_docs)
    patch_posts_df = await asyncio.to_thread(
        transformer.transform_patch_posts, patch_docs
    )

    products_df = prepare_products(products_df)
    # for _, row in products_df.iterrows():
    #     print(row)

    patch_posts_df["windows_kbs"] = None
    patch_posts_df["edge_kbs"] = None
    patch_posts_df["product_mentions_noun_chunks"] = fuzzy_search_column(
        patch_posts_df["noun_chunks"], products_df
    )
    patch_posts_df["product_mentions_keywords"] = fuzzy_search_column(
        patch_posts_df["keywords"], products_df
    )
    patch_posts_df["product_mentions"] = patch_posts_df.apply(
        lambda row: list(
            set(row["product_mentions_noun_chunks"] + row["product_mentions_keywords"])
        ),
        axis=1,
    )
    patch_posts_df["product_mentions"] = patch_posts_df["product_mentions"].apply(
        lambda mentions: retain_most_specific(mentions, products_df)
    )
    patch_posts_df["product_mentions"] = patch_posts_df["product_mentions"].apply(
        convert_to_original_representation
    )
    regex_pattern = construct_regex_pattern()
    patch_posts_df["build_numbers"] = patch_posts_df["text"].apply(
        lambda x: extract_build_numbers(x, regex_pattern) if pd.notna(x) else []
    )

    patch_posts_df["windows_kbs"] = patch_posts_df["text"].apply(
        lambda x: extract_windows_kbs(x) if pd.notna(x) else []
    )
    patch_posts_df["edge_kbs"] = patch_posts_df.apply(extract_edge_kbs, axis=1)

    patch_posts_df["kb_mentions"] = patch_posts_df.apply(
        lambda row: list(set(row["windows_kbs"] + row["edge_kbs"])), axis=1
    )

    patch_posts_df.drop(
        columns=[
            "windows_kbs",
            "edge_kbs",
            "product_mentions_noun_chunks",
            "product_mentions_keywords",
        ],
        inplace=True,
    )
    # for _, row in patch_posts_df.iterrows():
    #     print(f"{row['node_id']}-{row['build_numbers']}-{row['kb_mentions']}")
    document_service = DocumentService(collection_name="docstore")
    await update_email_records(patch_posts_df, document_service)

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(
        f"Time taken to process patch emails: {int(minutes)} min : {int(seconds)} sec"
    )
    print(
        "Patch feature engineering complete ============================================"
    )
    response = {
        "message": "Patch feature engineering complete.",
        "status": "Complete",
        "code": 200,
    }
    return response


# END PATCH FEATURE ENGINEERING PIPELINE

# BEGIN NEO4J V1 MIGRATION PIPELINE


def process_json_file(file_path):
    with open(file_path, "r", encoding="utf-8-sig") as file:
        content = file.read()

    fixed_content = fix_escape_chars(content)

    try:
        data = json.loads(fixed_content)
        print("Successfully parsed the file after fixing escape characters.")
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


def normalize_and_rename(df, column_name):
    # Normalize the nested dictionaries into columns
    normalized_df = pd.json_normalize(df[column_name])

    # Rename the columns to remove the nesting prefixes
    normalized_df.columns = [col.split(".")[-1] for col in normalized_df.columns]

    # Drop the original column and concatenate the normalized DataFrame
    df = df.drop(columns=[column_name])
    df = pd.concat(
        [df.reset_index(drop=True), normalized_df.reset_index(drop=True)], axis=1
    )

    return df


# migrate_neo4j_v1_pipeline
async def migrate_neo4j_v1_pipeline(start_date: datetime, end_date: datetime = None):

    response = {
        "message": "neo4j v1 migration pipeline complete.",
        "status": "in progress",
        "code": 200,
    }
    print("Begin neo4j v1 migration ===============================================")
    start_time = time.time()

    v2_products = await neo4j_migrator.load_v2_products()
    v2_msrc_posts = await neo4j_migrator.load_v2_msrc_posts()
    v2_patch_posts = await neo4j_migrator.load_v2_patch_management_posts()

    # Load V1 nodes
    v1_nodes_file_path = r"C:\\Users\\emili\\PycharmProjects\\microsoft_cve_rag\\microsoft_cve_rag\\application\\data\\v1_all_nodes_narrow.json"

    # The JSON file contains invalid escape characters, so we need to fix them before loading the data into a pandas dataframe
    data, error = process_json_file(v1_nodes_file_path)

    if data:
        temp_df = pd.DataFrame(data)
    else:
        print(f"error: {error}")
        raise ValueError("no data to process")
    # Extract the first label from 'Labels' (since it's a list)
    temp_df["Labels"] = temp_df["Labels"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
    )
    # Create subsets of the DataFrame based on 'Labels'
    node_types = temp_df["Labels"].unique()

    # create subsets of the temp_df dataframe for each node type
    df_v1_nodes_subsets = {
        node_type: temp_df[temp_df["Labels"] == node_type] for node_type in node_types
    }

    # process properties dictionary on each subset dictionary
    for node_type, df in df_v1_nodes_subsets.items():
        df_v1_nodes_subsets[node_type] = normalize_and_rename(df, "Properties")

    # Initialize mappings and processed IDs
    global_node_mapping = {}
    global_v1_to_v2_mapping = {}
    global_processed_v1_ids = []
    global_nodes_dict = {}
    # Process each node type separately
    for node_type, df_subset in df_v1_nodes_subsets.items():
        # Load corresponding V2 nodes
        if node_type == "AffectedProduct":
            # v2_nodes is a list of dicts, not neomodel asyncstructurednodes
            v2_nodes = v2_products
            prompt_node_type = "AffectedProduct"

        elif node_type == "MSRCSecurityUpdate":
            # v2_nodes is a list of dicts, not neomodel asyncstructurednodes
            v2_nodes = v2_msrc_posts
            prompt_node_type = "MSRCSecurityUpdate"

        elif node_type == "PatchManagement":
            # v2_nodes is a list of dicts, not neomodel asyncstructurednodes
            v2_nodes = v2_patch_posts
            prompt_node_type = "PatchManagement"

        elif node_type == "Symptom":
            v2_nodes = []
            prompt_node_type = node_type

        elif node_type == "Cause":
            v2_nodes = []
            prompt_node_type = node_type

        elif node_type == "Fix":
            v2_nodes = []
            prompt_node_type = node_type

        elif node_type == "Tool":
            v2_nodes = []
            prompt_node_type = node_type

        elif node_type == "Technology":
            v2_nodes = []
            prompt_node_type = node_type

        else:
            v2_nodes = []
            prompt_node_type = node_type

        # Transform nodes for this type
        nodes_to_create, node_mapping, v1_to_v2_mapping, processed_v1_ids = (
            await neo4j_migrator.transform_nodes(df_subset, v2_nodes, prompt_node_type)
        )

        # Merge the mappings and processed IDs
        global_node_mapping.update(node_mapping)
        global_v1_to_v2_mapping.update(v1_to_v2_mapping)
        global_processed_v1_ids.extend(processed_v1_ids)

        # transform dicts into dfs
        if nodes_to_create:
            match node_type:
                case "AffectedProduct":
                    print("Case AffectedProduct")
                    global_nodes_dict[node_type] = None
                    print("load AffectedProduct with bulk_create()")
                case "MSRCSecurityUpdate":
                    print("Case MSRCSecurityUpdate")
                    global_nodes_dict[node_type] = None
                    print("load MSRCSecurityUpdate with bulk_create()")
                case "PatchManagement":
                    print("Case PatchManagement")
                    global_nodes_dict[node_type] = None
                    print("load PatchManagement with bulk_create()")
                case "Symptom":

                    symptoms_narrow_df = await asyncio.to_thread(
                        transformer.transform_symptoms, nodes_to_create
                    )
                    symptoms_df = await asyncio.to_thread(
                        normalize_and_rename, symptoms_narrow_df, "properties"
                    )
                    symptoms_df.drop(
                        columns=["version", "architecture", "build_number"],
                        inplace=True,
                    )

                    symptom_load_response = await loader.load_symptoms_graph_db(
                        symptoms_df
                    )
                    # actual Neomodel AsyncStructured Nodes
                    symptom_nodes = symptom_load_response["nodes"]
                    global_nodes_dict[node_type] = symptom_nodes
                case "Cause":
                    print("Case Cause")
                    causes_narrow_df = await asyncio.to_thread(
                        transformer.transform_causes, nodes_to_create
                    )
                    causes_df = await asyncio.to_thread(
                        normalize_and_rename, causes_narrow_df, "properties"
                    )
                    causes_df.drop(
                        columns=["version", "architecture", "build_number"],
                        inplace=True,
                    )

                    causes_load_response = await loader.load_causes_graph_db(causes_df)
                    # actual Neomodel AsyncStructured Nodes
                    cause_nodes = causes_load_response["nodes"]
                    global_nodes_dict[node_type] = cause_nodes
                case "Fix":
                    print("Case Fix")
                    fixes_narrow_df = await asyncio.to_thread(
                        transformer.transform_fixes, nodes_to_create
                    )
                    fixes_df = await asyncio.to_thread(
                        normalize_and_rename, fixes_narrow_df, "properties"
                    )
                    fixes_df.drop(
                        columns=["version", "architecture", "build_number"],
                        inplace=True,
                    )

                    fixes_load_response = await loader.load_fixes_graph_db(fixes_df)
                    # actual Neomodel AsyncStructured Nodes
                    fix_nodes = fixes_load_response["nodes"]
                    global_nodes_dict[node_type] = fix_nodes
                case "Tool":
                    print("Case Tool")

                    tools_narrow_df = await asyncio.to_thread(
                        transformer.transform_tools, nodes_to_create
                    )
                    tools_df = await asyncio.to_thread(
                        normalize_and_rename, tools_narrow_df, "properties"
                    )
                    tools_df.drop(
                        columns=["version", "architecture", "build_number"],
                        inplace=True,
                    )
                    tools_load_response = await loader.load_tools_graph_db(tools_df)
                    # actual Neomodel AsyncStructured Nodes
                    tool_nodes = tools_load_response["nodes"]
                    global_nodes_dict[node_type] = tool_nodes
                case "Technology":
                    print("Case Technology")

                    technologies_narrow_df = await asyncio.to_thread(
                        transformer.transform_technologies, nodes_to_create
                    )
                    technologies_df = await asyncio.to_thread(
                        normalize_and_rename, technologies_narrow_df, "properties"
                    )
                    technologies_df.drop(
                        columns=["version", "architecture", "build_number"],
                        inplace=True,
                    )

                    technologies_load_response = (
                        await loader.load_technologies_graph_db(technologies_df)
                    )
                    # actual Neomodel AsyncStructured Nodes
                    technology_nodes = technologies_load_response["nodes"]
                    global_nodes_dict[node_type] = technology_nodes
                case _:
                    print("Default case")

    # BEGIN RELATIONSHIP PROCESSING
    v1_relationships_file_path = r"C:\\Users\\emili\\PycharmProjects\\microsoft_cve_rag\\microsoft_cve_rag\\application\\data\\v1_relationships_properties.json"
    missing_nodes = neo4j_migrator.identify_missing_nodes(
        v1_relationships_file_path, global_processed_v1_ids
    )

    # Transform relationships
    relationships = await neo4j_migrator.transform_relationships(
        v1_relationships_file_path,
        global_node_mapping,
        global_v1_to_v2_mapping,
        missing_nodes,
    )
    # Optionally, create relationships in Neo4j here
    for rel in relationships[:10]:
        print(f"relationship:\n{rel}")

    # await neo4j_migrator.build_migration_relationships(global_nodes_dict, relationships)

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Time taken to neo4j v1 migration: {int(minutes)} min : {int(seconds)} sec")
    print(
        "neo4j v1 migration pipeline complete ==========================================="
    )
    response = {
        "message": "neo4j v1 migration pipeline complete.",
        "status": "Complete",
        "code": 200,
    }
    return response


# Custom sorting function
def sort_by_build_number(item):
    return tuple(item["build_number"])
