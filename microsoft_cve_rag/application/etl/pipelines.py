import os
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
from application.services.vector_db_service import VectorDBService
from application.services.embedding_service import (
    EmbeddingService,
    LlamaIndexEmbeddingAdapter,
    )
from application.core.models.basic_models import Document, DocumentMetadata
from application.services.llama_index_service import (
    extract_entities_relationships,
    LlamaIndexVectorService
)
from application.etl.MongoPipelineLoader import MongoPipelineLoader
from application.app_utils import get_app_config
from llama_index.core import Document as LlamaDocument

from typing import Dict, Any
import time
import asyncio
import re
import json
from datetime import datetime, timedelta, timezone
import pandas as pd
from fuzzywuzzy import fuzz
import logging

settings = get_app_config()

logging.getLogger(__name__)

def validate_pipeline(pipeline):
    for stage in pipeline:
        for key, value in stage.items():
            if not isinstance(key, str):
                raise ValueError(f"Invalid key type in pipeline stage: {type(key)}")
    return pipeline

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
    logging.info(f"start_date: {start_date}, end_date: {end_date}")
    response = await full_ingestion_pipeline(start_date, end_date)
    logging.info(f"response: {response}")
    return response


async def full_ingestion_pipeline(start_date: datetime, end_date: datetime = None):
    response = {
        "message": "full ingestion pipeline complete.",
        "status": "in progress",
        "code": 200,
    }
    logging.info("Begin mongo feature engineering =====================================\n")
    start_time = time.time()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    etl_dir = os.path.join(base_dir, "etl")
    # pipeline_loader = MongoPipelineLoader(base_directory=etl_dir)
    # # print(f"Base directory: {pipeline_loader.get_base_directory()}")
    # # create a dictionary of pipeline configurations
    # pipelines_arguments_config = {
    #     "fe_product_build_ids.yaml": {
    #         "start_date": start_date.isoformat(),
    #         "end_date": end_date.isoformat() if end_date else datetime.now(timezone.utc).isoformat(),
    #         "exclude_collections": ['archive_stable_channel_notes', 'beta_channel_notes', 'mobile_stable_channel_notes', 'windows_update'],
    #     },
    #     "fe_msrc_kb_ids.yaml": {
    #         "start_date": start_date.isoformat(),
    #         "end_date": end_date.isoformat() if end_date else datetime.now().isoformat()
    #     },
    #     "fe_kb_article_cve_ids.yaml": {
    #         "start_date": start_date.isoformat(),
    #         "end_date": end_date.isoformat() if end_date else datetime.now().isoformat()
    #     },
    #     "fe_kb_article_product_build_ids.yaml": {
    #         "start_date": start_date.isoformat(),
    #         "end_date": end_date.isoformat() if end_date else datetime.now().isoformat()
    #     }
    # }

    # mongo_db_config = {
    #     "fe_product_build_ids.yaml": {
    #         "mongo_collection": "docstore",
    #         "mongo_db": "report_docstore"
    #     },
    #     "fe_msrc_kb_ids.yaml": {
    #         "mongo_collection": "docstore",
    #         "mongo_db": "report_docstore"
    #     },
    #     "fe_kb_article_cve_ids.yaml": {
    #         "mongo_collection": "microsoft_kb_articles",
    #         "mongo_db": "report_docstore"
    #     },
    #     "fe_kb_article_product_build_ids.yaml": {
    #         "mongo_collection": "microsoft_kb_articles",
    #         "mongo_db": "report_docstore"
    #     },
    # }
    
    # for yaml_file, arguments in pipelines_arguments_config.items():
    #     try:
    #         # Load and render the pipeline using MongoPipelineLoader
    #         resolved_pipeline = pipeline_loader.get_pipeline(yaml_file, arguments)
            
    #         validated_pipeline = validate_pipeline(resolved_pipeline)
             
    #         # Get mongo db and collection details
    #         db_details = mongo_db_config.get(yaml_file, {})
    #         mongo_collection = db_details.get("mongo_collection")
    #         mongo_db = db_details.get("mongo_db")
            
            
    #         document_service = DocumentService(mongo_db, mongo_collection)
    #         docs = document_service.aggregate_documents(validated_pipeline)
            
    #         if docs:
    #             logging.info(f"Aggregation pipeline returned: {len(docs)}\n")
    #             for doc in docs:
    #                 logging.info(f"doc id: {doc['_id']}")
    #     except ValueError as e:
    #         logging.error(f"Pipeline validation failed for {yaml_file}: {e}")
    #     except Exception as e:
    #         logging.error(f"Failed to load pipeline {yaml_file}: {e}")
        
    # response = await patch_feature_engineering_pipeline(start_date, end_date)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    logging.info(f"Time taken for feature engineering: {int(minutes)} min : {int(seconds)} sec")
    logging.info("End mongo feature engineering =======================================\n")
    
    logging.info("Begin data extraction ===============================================\n")
    start_time = time.time()

    # product_docs = await asyncio.to_thread(extractor.extract_products, None)
    # product_build_docs = await asyncio.to_thread(
    #     extractor.extract_product_builds, start_date, end_date, None
    # )
    # product_build_docs_by_build = sorted(product_build_docs, key=sort_by_build_number)
    # kb_article_docs_windows, kb_article_docs_edge = await asyncio.to_thread(
    #     extractor.extract_kb_articles, start_date, end_date, 2
    # )
    # update_package_docs = extractor.extract_update_packages(start_date, end_date, None)
    msrc_docs = extractor.extract_msrc_posts(start_date, end_date, 2)
    patch_docs = extractor.extract_patch_posts(start_date, end_date, 2)
    for doc in msrc_docs:
        print(f"{doc['metadata']['post_id']}")
    for doc in patch_docs:
        print(f"{doc['metadata']['id']}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    logging.info(f"Time taken to extract all data: {int(minutes)} min : {int(seconds)} sec")
    logging.info("Done with data extraction ===============================================\n")
    
    logging.info("Begin data transformation ==========================================")
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
    print(f"after transform msrc_posts_df cols:\n{msrc_posts_df.columns}")
    patch_posts_df = await asyncio.to_thread(
        transformer.transform_patch_posts, patch_docs
    )
    print(f"after transform patch_posts_df cols:\n{patch_posts_df.columns}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    logging.info(f"Time taken to transform all data: {int(minutes)} min : {int(seconds)} sec")
    logging.info("Done with data transformation ==========================================\n")
    # return response

    logging.info("Begin Graph extraction ------------------------------\n")
    start_time = time.time()

    # EXTRACT SYMPTOMS, CAUSES, FIXES, AND TOOLS
    # FROM MSRCPosts and PatchManagementPosts.
    # Custom entity extraction
    # if isinstance(msrc_posts_df, pd.DataFrame) and not msrc_posts_df.empty:
    #     print(f"{msrc_posts_df.shape} : {msrc_posts_df.columns}")
    
    # msrc_llm_extracted_data = await extract_entities_relationships(msrc_posts_df, "MSRCPost"
    # )
    # msrc_total = 0
    # for key, value in msrc_llm_extracted_data.items():
    #     if not isinstance(value, list):
    #         print(f"Invalid value for {key}: {value}")
    #     else:
    #         count = len(value)
    #         msrc_total += count
    # print(f"{msrc_total} non-empty msrc items returned by extract_entities_relationships")
    
    # patch_llm_extracted_data = await extract_entities_relationships(
    #     patch_posts_df, "PatchManagementPost"
    # )
    # patch_total = 0
    # for key, value in patch_llm_extracted_data.items():
    #     if not isinstance(value, list):
    #         print(f"Invalid value for {key}: {value}")
    #     else:
    #         count = len(value)
    #         patch_total += count
    # print(f"{patch_total} non-empty patch items returned by extract_entities_relationships")
    # combined_df = transformer.combine_and_split_dicts(msrc_llm_extracted_data, patch_llm_extracted_data)
    # all_symptoms_df = combined_df['symptoms']
    # print(f"{all_symptoms_df.shape} : {all_symptoms_df.sample(n=min(all_symptoms_df.shape[0], 6))}")
    # all_causes_df = combined_df['causes']
    # print(f"{all_causes_df.shape} : {all_causes_df.sample(n=min(all_causes_df.shape[0], 6))}")
    # all_fixes_df = combined_df['fixes']
    # print(f"{all_fixes_df.shape} : {all_fixes_df.sample(n=min(all_fixes_df.shape[0], 6))}")
    # all_tools_df = combined_df['tools']
    # print(f"{all_tools_df.shape} : {all_tools_df.sample(n=min(all_tools_df.shape[0], 6))}")
    
    logging.info("Done with Graph extraction -----------------------------\n")
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    logging.info(
        f"Time taken to extract entities: {int(minutes)} min : {int(seconds)} sec"
    )
    
    logging.info("Begin data loading ==========================================\n")
    start_time = time.time()
    logging.info("Begin graph database loading ------------------------------\n")
    
    # product_load_response = await loader.load_products_graph_db(products_df)
    # product_nodes = product_load_response["nodes"]
    # # product_node_ids = product_load_response["insert_ids"]
    # # print(product_node_ids)

    # product_build_load_response = await loader.load_product_builds_graph_db(
    #     product_builds_df
    # )
    # product_build_nodes = product_build_load_response["nodes"]
    # # product_build_node_ids = product_build_load_response["insert_ids"]

    # kb_article_load_response = await loader.load_kbs_graph_db(kb_articles_combined_df)
    # kb_nodes = kb_article_load_response["nodes"]
    # # kb_node_ids = kb_article_load_response["insert_ids"]
    
    # update_packages_load_response = await loader.load_update_packages_graph_db(
    #     update_packages_df
    # )
    # update_package_nodes = update_packages_load_response["nodes"]
    # # update_package_node_ids = update_packages_load_response["insert_ids"]

    # msrc_posts_load_response = await loader.load_msrc_posts_graph_db(msrc_posts_df)
    # msrc_post_nodes = msrc_posts_load_response["nodes"]
    # # msrc_post_node_ids = msrc_posts_load_response["insert_ids"]
    
    # patch_posts_load_response = await loader.load_patch_posts_graph_db(patch_posts_df)
    # patch_post_nodes = patch_posts_load_response["nodes"]
    # # patch_post_node_ids = patch_posts_load_response["insert_ids"]
    # symptom_load_response = await loader.load_symptoms_graph_db(all_symptoms_df)
    # symptom_nodes = symptom_load_response["nodes"]
    # cause_load_response = await loader.load_causes_graph_db(all_causes_df)
    # cause_nodes = cause_load_response["nodes"]
    # fix_load_response = await loader.load_fixes_graph_db(all_fixes_df)
    # fix_nodes = fix_load_response["nodes"]
    # tool_load_response = await loader.load_tools_graph_db(all_tools_df)
    # tool_nodes = tool_load_response["nodes"]
    # technology_load_response = await loader.load_technologies_graph_db(
    #     all_technologies_df
    # )
    # technology_nodes = technology_load_response["nodes"]

    # nodes_dict = {
    #     "Product": product_nodes,
    #     "ProductBuild": product_build_nodes,
    #     "UpdatePackage": update_package_nodes,
        # "KBArticle": kb_nodes,
        # "MSRCPost": msrc_post_nodes,
    #     "PatchManagementPost": patch_post_nodes,
    #     "Symptom": symptom_nodes,
    #     "Cause": cause_nodes,
    #     "Fix": fix_nodes,
    #     "Tool": tool_nodes,
    # }

    # await build_relationships(nodes_dict)
    
    logging.info("Done with graph database loading ------------------------------\n")
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    logging.info(
        f"Time taken to upsert graph entities: {int(minutes)} min : {int(seconds)} sec"
    )
    logging.info("Begin vector database loading ------------------------------\n")
    start_time = time.time()
    try:
        # Initialize VectorDBService with your configuration
        vector_db_service = VectorDBService(
            embedding_config=settings["EMBEDDING_CONFIG"],
            vectordb_config=settings["VECTORDB_CONFIG"],
        )
        try:
            llama_vector_service = await LlamaIndexVectorService.initialize(
                vector_db_service=vector_db_service,
                persist_dir=settings["VECTORDB_CONFIG"]['persist_dir']
            )
            logging.info("initialized llama vector service")
            
            llama_documents = []

            # new caller functions start
            # if isinstance(kb_articles_combined_df, pd.DataFrame) and not kb_articles_combined_df.empty:
            #     llama_documents += convert_df_to_llamadoc_kb_articles(kb_articles_combined_df)

            # if isinstance(update_packages_df, pd.DataFrame) and not update_packages_df.empty:
            #     llama_documents += convert_df_to_llamadoc_update_packages(update_packages_df)

            # if isinstance(all_symptoms_df, pd.DataFrame) and not all_symptoms_df.empty:
            #     llama_documents += convert_df_to_llamadoc_symptoms(all_symptoms_df)

            # if isinstance(all_causes_df, pd.DataFrame) and not all_causes_df.empty:
            #     llama_documents += convert_df_to_llamadoc_causes(all_causes_df)

            # if isinstance(all_fixes_df, pd.DataFrame) and not all_fixes_df.empty:
            #     llama_documents += convert_df_to_llamadoc_fixes(all_fixes_df)

            # if isinstance(all_tools_df, pd.DataFrame) and not all_tools_df.empty:
            #     llama_documents += convert_df_to_llamadoc_tools(all_tools_df)

            # if isinstance(msrc_posts_df, pd.DataFrame) and not msrc_posts_df.empty:
            #     llama_documents += convert_df_to_llamadoc_msrc_posts(msrc_posts_df, symptom_nodes, cause_nodes, fix_nodes, tool_nodes)

            # if isinstance(patch_posts_df, pd.DataFrame) and not patch_posts_df.empty:
            #     llama_documents += convert_df_to_llamadoc_patch_posts(patch_posts_df, symptom_nodes, cause_nodes, fix_nodes, tool_nodes)

            # new caller functions end
            if llama_documents:
                # Upsert documents to vector store
                await llama_vector_service.upsert_documents(llama_documents)
                logging.info("Documents upserted to vector store")
            else:
                print("No documents to upsert")

        except Exception as e:
            logging.info(f"Error in llama workflow in pipeline: {e}")
            raise

        finally:
            # Ensure cleanup happens even if there's an error
            if 'llama_vector_service' in locals():
                await llama_vector_service.aclose()
            await vector_db_service.aclose()

    except Exception as e:
        logging.info(f"Error in full ingestion pipeline: {e}")
        raise


    logging.info("Done with vector database loading ------------------------------\n")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    logging.info(f"Time taken to upsert vector data: {int(minutes)} min : {int(seconds)} sec")
    logging.info("Begin document updating ------------------------------\n")
    start_time = time.time()
    
    msrc_exclude_columns = ['node_id', 'metadata','excluded_embed_metadata_keys','excluded_llm_metadata_keys','text','node_label','patterns_found','patterns_missing','revision', 'published','title','description']
    patch_exclude_columns = ['node_id', 'metadata','excluded_embed_metadata_keys','excluded_llm_metadata_keys','noun_chunks', 'keywords','text','node_label',]
    
    print(f"before document upsert - msrc_df.columns:\n{msrc_posts_df.columns}")
    # Update MSRC Posts
    if isinstance(msrc_posts_df, pd.DataFrame) and not msrc_posts_df.empty:
        docstore_service = DocumentService(db_name="report_docstore", collection_name="docstore")
        for _, row in msrc_posts_df.iterrows():
            logging.info(f"\nProcessing document {row['node_id']} ================")
            logging.info(f"Original row data:\n{row.to_dict()}")
            logging.info(f"Columns being excluded: {msrc_exclude_columns}")
            
            metadata = DocumentMetadata(
                id=row['node_id'],
                **{col: row[col] for col in row.index 
                if col not in msrc_exclude_columns}  # Remove hasattr check
            )
            logging.info(f"Created DocumentMetadata:\n{metadata.model_dump(exclude_none=False)}")
            
            doc = Document(
                id_=row['node_id'],
                metadata=metadata,
                text=row['text']
            )
            logging.info(f"Created Document instance:\n{doc.model_dump(exclude_none=False)}")
            
            update_result = docstore_service.update_document(row['node_id'], doc, False)
            logging.info(f"Update result for document {row['node_id']}: {update_result}")
            logging.info("=" * 50)
        logging.info(f"Updated {len(msrc_posts_df)} MSRC posts")
    print()
    print(f"before document upsert - patch_posts_df.columns:\n{patch_posts_df.columns}")
    # Update Patch Management Posts
    if isinstance(patch_posts_df, pd.DataFrame) and not patch_posts_df.empty:
        docstore_service = DocumentService(db_name="report_docstore", collection_name="docstore")
        
        for _, row in patch_posts_df.iterrows():
            logging.info(f"\nProcessing document {row['node_id']} ================")
            logging.info(f"Original row data:\n{row.to_dict()}")
            logging.info(f"Columns being excluded: {patch_exclude_columns}")
            
            # Convert row data, handling receivedDateTime specially
            metadata_dict = {
                col: (row[col].isoformat() if col == 'receivedDateTime' else row[col])
                for col in row.index 
                if col not in patch_exclude_columns
            }
            
            metadata = DocumentMetadata(
                id=row['node_id'],
                **metadata_dict
            )
            logging.info(f"Created DocumentMetadata:\n{metadata.model_dump(exclude_none=False)}")
            
            doc = Document(
                id_=row['node_id'],
                metadata=metadata,
                text=row['text']
            )
            logging.info(f"Created Document instance:\n{doc.model_dump(exclude_none=False)}")
            
            update_result = docstore_service.update_document(row['node_id'], doc, False)
            logging.info(f"Update result for document {row['node_id']}: {update_result}")
            logging.info("=" * 50)
        logging.info(f"Updated {len(patch_posts_df)} Patch Management posts")

    # Update KB Articles
    # if isinstance(kb_articles_combined_df, pd.DataFrame) and not kb_articles_combined_df.empty:
    #     kb_service = DocumentService(db_name="report_docstore", collection_name="microsoft_kb_articles")
    #     for _, row in kb_articles_combined_df.iterrows():
    #         doc = Document(
    #             id_=row['node_id'],
    #             text=row['text'],
    #             title=row['title']
    #         )
    #         kb_service.update_document(row['node_id'], doc)
    #     logging.info(f"Updated {len(kb_articles_combined_df)} KB Articles")
        
    logging.info("End document updating ------------------------------\n")
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    logging.info(f"Time taken to upsert mongo data: {int(minutes)} min : {int(seconds)} sec")
    
    logging.info("Full Ingestion complete =====================================\n")
    
    response = {
        "message": "full ingestion pipeline complete.",
        "status": "Complete",
        "code": 200,
    }
    return response
# END FULL INGESTION PIPELINE ===============================================

# BEGIN PATCH FEATURE ENGINEERING HERLPERS ==================================

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


def validate_pipeline(pipeline):
    for stage in pipeline:
        for key, value in stage.items():
            if not isinstance(key, str):
                raise ValueError(f"Invalid key type in pipeline stage: {type(key)}")
    return pipeline

