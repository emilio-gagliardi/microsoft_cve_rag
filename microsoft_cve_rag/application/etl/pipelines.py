import os
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any
import asyncio
import re
import json
import pandas as pd
from pymongo.cursor import CursorType
from application.etl import extractor
from application.etl import transformer
from application.etl import loader
from application.etl import neo4j_migrator
from application.services.graph_db_service import (
    build_relationships_in_batches,
)
from application.services.document_service import DocumentService
from application.services.vector_db_service import VectorDBService
from application.core.models.basic_models import Document, DocumentMetadata
from application.services.llama_index_service import (
    extract_entities_relationships,
    LlamaIndexVectorService,
    CustomDocumentTracker,
    track_dataframe_documents,
)
from application.etl.MongoPipelineLoader import MongoPipelineLoader
from application.app_utils import get_app_config
from tqdm import tqdm
import numpy as np

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

import pprint


async def execute_single_pipeline(
    pipeline_loader: MongoPipelineLoader,
    yaml_file: str,
    mongo_db_config: Dict[str, Dict[str, Any]],
    arguments: Dict[str, Any]
):
    try:
        logging.info(f"Loading pipeline from: {yaml_file}")
        # Load and validate pipeline
        pipeline = pipeline_loader.get_pipeline(yaml_file, arguments)
        validate_pipeline(pipeline)
        # pprint.pprint(pipeline)
        # Get mongo db and collection details
        db_details = mongo_db_config.get(yaml_file, {})
        mongo_collection = db_details.get("mongo_collection")
        mongo_db = db_details.get("mongo_db")

        if not mongo_db or not mongo_collection:
            raise ValueError(f"Missing MongoDB configuration for {yaml_file}")

        # Execute pipeline
        document_service = DocumentService(mongo_db, mongo_collection)
        result = document_service.aggregate_documents(pipeline)

        if result:
            logging.info(f"Pipeline {yaml_file} processed {len(result)} documents")
            for doc in result:
                logging.debug(f"Processed document: {doc.get('_id', 'No ID')}")
        return result

    except Exception as e:
        logging.error(f"Failed to execute pipeline {yaml_file}: {str(e)}")
        raise


async def run_feature_engineering(
    pipeline_loader: MongoPipelineLoader,
    pipelines_arguments_config: Dict[str, Dict[str, Any]],
    mongo_db_config: Dict[str, Dict[str, Any]]
):
    """Run all feature engineering pipelines concurrently."""

    try:
        # Create tasks for each pipeline
        tasks = []
        for yaml_file, arguments in pipelines_arguments_config.items():
            tasks.append(execute_single_pipeline(pipeline_loader, yaml_file, mongo_db_config, arguments))

        # Execute all pipelines concurrently and wait for completion
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for any exceptions
        for result, (yaml_file, _) in zip(results, pipelines_arguments_config.items()):
            if isinstance(result, Exception):
                logging.error(f"Pipeline {yaml_file} failed: {str(result)}")
                raise result

        return results

    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}")
        raise


async def full_ingestion_pipeline(start_date: datetime, end_date: datetime = None):
    response = {
        "message": "full ingestion pipeline complete.",
        "status": "in progress",
        "code": 200,
    }

    # Feature Engineering Stage
    logging.info("Begin mongo feature engineering =====================================\n")
    start_time = time.time()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    etl_dir = os.path.join(base_dir, "etl")
    pipeline_loader = MongoPipelineLoader(base_directory=etl_dir)

    # Create missing KB Articles.
    # requires differrent processing than other pipelines because
    # we need to insert documents into a collection that doesn't have a unique index constraint.
    yaml_file = "create_missing_kb_articles_no_merge.yaml"
    kb_articles_arguments_config = {
        yaml_file: {
           "start_date": start_date.isoformat(),
           "end_date": end_date.isoformat() if end_date else datetime.now().isoformat()
        }
    }
    arguments = kb_articles_arguments_config[yaml_file]
    kb_articles_mongo_db_config = {
        yaml_file: {
            "mongo_collection": "docstore",
            "mongo_db": "report_docstore"
        }
    }
    new_kbs_cursor = await execute_single_pipeline(
        pipeline_loader,
        yaml_file,
        kb_articles_mongo_db_config,
        arguments
    )
    new_kb_docs = list(new_kbs_cursor)
    if new_kb_docs:
        mongo_collection = "microsoft_kb_articles"
        mongo_db = "report_docstore"
        document_service = DocumentService(mongo_db, mongo_collection)
        # Convert docs to include id_ field and pass all fields through
        document_instances = [
            Document(
                id_=str(doc.get('_id')),
                metadata=DocumentMetadata(id=str(doc.get('_id'))),
                **{k: v for k, v in doc.items() if k not in ['_id']}
            )
            for doc in new_kb_docs
        ]
        results = document_service.create_documents(document_instances)
        print(f"Inserted {len(document_instances)} new KB articles into {mongo_collection}.")

    # create a dictionary of pipeline configurations
    pipelines_arguments_config = {
        "fe_product_build_ids.yaml": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat() if end_date else datetime.now(timezone.utc).isoformat(),
            "exclude_collections": ['archive_stable_channel_notes', 'beta_channel_notes', 'mobile_stable_channel_notes', 'windows_update'],
        },
        "fe_msrc_kb_ids.yaml": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat() if end_date else datetime.now().isoformat()
        },
        "fe_kb_article_cve_ids.yaml": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat() if end_date else datetime.now().isoformat()
        },
        "fe_kb_article_product_build_ids.yaml": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat() if end_date else datetime.now().isoformat()
        }
    }

    mongo_db_config = {
        "fe_product_build_ids.yaml": {
            "mongo_collection": "docstore",
            "mongo_db": "report_docstore"
        },
        "fe_msrc_kb_ids.yaml": {
            "mongo_collection": "docstore",
            "mongo_db": "report_docstore"
        },
        "fe_kb_article_cve_ids.yaml": {
            "mongo_collection": "microsoft_kb_articles",
            "mongo_db": "report_docstore"
        },
        "fe_kb_article_product_build_ids.yaml": {
            "mongo_collection": "microsoft_kb_articles",
            "mongo_db": "report_docstore"
        },
    }

    try:
        # Run feature engineering stage completely before moving to extraction
        async with asyncio.Lock():
            # Run main feature engineering
            await run_feature_engineering(pipeline_loader, pipelines_arguments_config, mongo_db_config)

            # Run patch feature engineering after main feature engineering
            patch_response = await patch_feature_engineering_pipeline(start_date, end_date)
            if patch_response["code"] not in [200, 204]:  # Accept both success and no content
                logging.error(f"Patch feature engineering failed: {patch_response['message']}")
                return patch_response
            elif patch_response["code"] == 204:
                logging.info(f"Patch feature engineering: {patch_response['message']}")

            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes, seconds = divmod(elapsed_time, 60)
            logging.info(f"Time taken for feature engineering: {int(minutes)} min : {int(seconds)} sec")
            logging.info("End mongo feature engineering =======================================\n")

            # Only start extraction after feature engineering is completely done
            logging.info("Begin Full ETL data extraction ======================================")
            start_time = time.time()

            try:
                extracted_docs = await extract_all_documents(start_date, end_date)

                end_time = time.time()
                elapsed_time = end_time - start_time
                minutes, seconds = divmod(elapsed_time, 60)
                logging.info(f"Time taken to extract all data: {int(minutes)} min : {int(seconds)} sec")
                logging.info("Done with data extraction ===============================================\n")
            except Exception as e:
                logging.error(f"Extraction stage failed: {str(e)}")
                response["status"] = "failed"
                response["message"] = f"Extraction failed: {str(e)}"
                response["code"] = 500
                return response

    except Exception as e:
        logging.error(f"Feature engineering stage failed: {str(e)}")
        response["status"] = "failed"
        response["message"] = f"Feature engineering failed: {str(e)}"
        response["code"] = 500
        return response

    # Begin transformation stage with extracted documents
    logging.info("Begin Full ETL data transformation =====================================")
    start_time = time.time()

    products_df = None
    if extracted_docs.get("products"):
        products_df = await asyncio.to_thread(
            transformer.transform_products,
            extracted_docs["products"]
        )
    product_builds_df = None
    if extracted_docs.get("product_builds"):
        product_builds_df = await asyncio.to_thread(
            transformer.transform_product_builds,
            extracted_docs["product_builds"]
        )
        if product_builds_df is not None and not product_builds_df.empty:
            logging.info(f"num product build df rows: {product_builds_df.shape[0]}")
    kb_articles_combined_df = None
    if extracted_docs.get("kb_articles") and len(extracted_docs["kb_articles"]) >= 2:
        kb_articles_combined_df = await asyncio.to_thread(
            transformer.transform_kb_articles,
            extracted_docs["kb_articles"][0],
            extracted_docs["kb_articles"][1]
        )
        if kb_articles_combined_df is not None and not kb_articles_combined_df.empty:
            logging.info(f"num kb articles df rows: {kb_articles_combined_df.shape[0]}")
    update_packages_df = None
    if extracted_docs.get("update_packages"):
        update_packages_df = await asyncio.to_thread(
            transformer.transform_update_packages,
            extracted_docs["update_packages"]
        )
    msrc_posts_df = None
    if extracted_docs.get("msrc_posts"):
        for doc in extracted_docs['msrc_posts']:
            logging.info(f"msrc id: {doc['metadata']['id']}")
        msrc_posts_df = await asyncio.to_thread(
            transformer.transform_msrc_posts,
            extracted_docs["msrc_posts"]
        )
        if msrc_posts_df is not None and not msrc_posts_df.empty:
            logging.info(f"num msrc_posts_df rows: {msrc_posts_df.shape[0]}")
    patch_posts_df = None
    if extracted_docs.get("patch_posts"):
        patch_posts_df = await asyncio.to_thread(
            transformer.transform_patch_posts_v2,
            extracted_docs["patch_posts"]
        )

        if patch_posts_df is not None and not patch_posts_df.empty:
            logging.info(f"num patch_posts_df rows: {patch_posts_df.shape[0]}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    logging.info(f"Time taken to transform all data: {int(minutes)} min : {int(seconds)} sec")
    logging.info("Done with data transformation ==========================================\n")

    logging.info("Begin Graph extraction ------------------------------\n")
    start_time = time.time()

    # EXTRACT SYMPTOMS, CAUSES, FIXES, AND TOOLS
    # FROM MSRCPosts and PatchManagementPosts.
    # Custom entity extraction
    msrc_llm_extracted_data = None
    if isinstance(msrc_posts_df, pd.DataFrame) and not msrc_posts_df.empty:
        logging.debug(f"{msrc_posts_df.shape} : {msrc_posts_df.columns}")

    msrc_llm_extracted_data, msrc_failed_extractions, msrc_empty_extractions = await extract_entities_relationships(
        msrc_posts_df,
        "MSRCPost"
    )

    msrc_total = 0
    for key, value in msrc_llm_extracted_data.items():
        if not isinstance(value, list):
            logging.error(f"Invalid value for {key}: {value}")
        else:
            count = len(value)
            msrc_total += count
    logging.info(f"{msrc_total} non-empty msrc items returned by extract_entities_relationships")

    patch_llm_extracted_data = None
    if isinstance(patch_posts_df, pd.DataFrame) and not patch_posts_df.empty:
        logging.debug(f"{patch_posts_df.shape} : {patch_posts_df.columns}")

    patch_llm_extracted_data, patch_failed_extractions, patch_empty_extractions = await extract_entities_relationships(
        patch_posts_df,
        "PatchManagementPost"
    )

    patch_total = 0
    for key, value in patch_llm_extracted_data.items():
        if not isinstance(value, list):
            logging.error(f"Invalid value for {key}: {value}")
        else:
            count = len(value)
            patch_total += count
    logging.info(f"{patch_total} non-empty patch items returned by extract_entities_relationships")

    combined_df = None
    all_symptoms_df = None
    all_causes_df = None
    all_fixes_df = None
    all_tools_df = None

    if (msrc_llm_extracted_data is not None or patch_llm_extracted_data is not None):
        combined_df = transformer.combine_and_split_dicts(msrc_llm_extracted_data, patch_llm_extracted_data)

        if combined_df and 'symptoms' in combined_df:
            all_symptoms_df = combined_df['symptoms']
            if not all_symptoms_df.empty:
                logging.debug(f"{all_symptoms_df.shape} : {all_symptoms_df.sample(n=min(all_symptoms_df.shape[0], 6))}")

        if combined_df and 'causes' in combined_df:
            all_causes_df = combined_df['causes']
            if not all_causes_df.empty:
                logging.debug(f"{all_causes_df.shape} : {all_causes_df.sample(n=min(all_causes_df.shape[0], 6))}")

        if combined_df and 'fixes' in combined_df:
            all_fixes_df = combined_df['fixes']
            if not all_fixes_df.empty:
                logging.debug(f"{all_fixes_df.shape} : {all_fixes_df.sample(n=min(all_fixes_df.shape[0], 6))}")

        if combined_df and 'tools' in combined_df:
            all_tools_df = combined_df['tools']
            if not all_tools_df.empty:
                logging.debug(f"{all_tools_df.shape} : {all_tools_df.sample(n=min(all_tools_df.shape[0], 6))}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    logging.info(
        f"Time taken to extract entities: {int(minutes)} min : {int(seconds)} sec")
    logging.info("Done with Graph extraction -----------------------------\n")
    logging.info("Begin data loading ==========================================\n")

    logging.info("Begin graph database loading ------------------------------\n")
    start_time = time.time()

    # Initialize node lists
    product_nodes = []
    product_build_nodes = []
    kb_nodes = []
    update_package_nodes = []
    msrc_post_nodes = []
    patch_post_nodes = []
    symptom_nodes = []
    cause_nodes = []
    fix_nodes = []
    tool_nodes = []

    node_list_mapping = {
        "Product": product_nodes,
        "ProductBuild": product_build_nodes,
        "KBArticle": kb_nodes,
        "UpdatePackage": update_package_nodes,
        "MSRCPost": msrc_post_nodes,
        "PatchManagementPost": patch_post_nodes,
        "Symptom": symptom_nodes,
        "Cause": cause_nodes,
        "Fix": fix_nodes,
        "Tool": tool_nodes
    }

    # Initialize error tracking
    error_dict = {
        "Product": [],
        "ProductBuild": [],
        "KBArticle": [],
        "UpdatePackage": [],
        "MSRCPost": [],
        "PatchManagementPost": [],
        "Symptom": [],
        "Cause": [],
        "Fix": [],
        "Tool": []
    }

    # Load all nodes first
    node_loading_tasks = [
        ("Product", loader.load_products_graph_db(products_df)),
        ("ProductBuild", loader.load_product_builds_graph_db(product_builds_df)),
        ("KBArticle", loader.load_kbs_graph_db(kb_articles_combined_df)),
        ("UpdatePackage", loader.load_update_packages_graph_db(update_packages_df)),
        ("MSRCPost", loader.load_msrc_posts_graph_db(msrc_posts_df)),
        ("PatchManagementPost", loader.load_patch_posts_graph_db(patch_posts_df)),
        ("Symptom", loader.load_symptoms_graph_db(all_symptoms_df)),
        ("Cause", loader.load_causes_graph_db(all_causes_df)),
        ("Fix", loader.load_fixes_graph_db(all_fixes_df)),
        ("Tool", loader.load_tools_graph_db(all_tools_df))
    ]

    # Load nodes with progress tracking
    nodes_dict = {}
    with tqdm(total=len(node_loading_tasks), desc="Loading nodes") as pbar:
        for node_type, task in node_loading_tasks:
            try:
                task_response = await task
                # Handle successful inserts - more defensive check
                nodes = task_response.get("nodes", [])
                if nodes:  # This will handle both None and empty list cases
                    nodes_dict[node_type] = nodes
                    node_list_mapping[node_type].extend(nodes)
                    logging.info(f"Loaded {len(nodes)} {node_type} nodes")

                # Handle errors if present
                if task_response.get("errors"):
                    error_dict[node_type].extend(task_response["errors"])
                    logging.warning(f"Encountered {len(task_response['errors'])} errors while loading {node_type} nodes")

                # Log partial success
                if task_response["code"] == 207:
                    logging.info(f"Partial success for {node_type}: {task_response['message']}")

            except Exception as e:
                logging.error(f"Error loading {node_type} nodes: {str(e)}")
                raise
            pbar.update(1)

    # Log error summary
    llm_extraction_types = ["Symptom", "Cause", "Fix", "Tool"]
    for node_type, errors in error_dict.items():
        if errors:
            logging.error(f"\n{node_type} Loading Errors:")
            id_field = "source_id" if node_type in llm_extraction_types else "node_id"
            for error in errors:
                logging.error(f"  - {id_field}: {error[id_field]}, Error: {error['message']}")

    # Calculate stats with proper handling of missing nodes
    stats = {}
    for node_type in node_list_mapping.keys():
        successful = len(nodes_dict.get(node_type, []))  # Default to empty list if missing
        failed = len(error_dict.get(node_type, []))      # Default to empty list if missing

        # Count empty extractions for this node type from both MSRC and Patch sources
        empty_count = 0
        # Only add lengths if the lists actually exist in the defaultdict
        if node_type in msrc_empty_extractions and msrc_empty_extractions[node_type]:
            empty_count += len(msrc_empty_extractions[node_type])
        if node_type in patch_empty_extractions and patch_empty_extractions[node_type]:
            empty_count += len(patch_empty_extractions[node_type])

        stats[node_type] = {
            "successful": successful,
            "failed": failed,
            "empty_extractions": empty_count,
            "total_attempted": successful + failed + empty_count
        }

    # Build relationships with batching and checkpointing
    checkpoint_file = "C:/Users/emili/PycharmProjects/microsoft_cve_rag/microsoft_cve_rag/application/data/graph_db/relationship_checkpoint.json"
    batch_size = 1000  # Adjust based on available memory

    try:
        await build_relationships_in_batches(
            nodes_dict,
            batch_size=batch_size,
            checkpoint_file=checkpoint_file
        )
    except Exception as e:
        logging.error(f"Error building relationships: {str(e)}")
        logging.info("You can resume the process later using the checkpoint file")
        raise
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"\nGraph database loading completed in {duration:.2f} seconds")

    logging.info("Begin vector database loading ------------------------------\n")
    start_time = time.time()
    try:
        # Initialize VectorDBService with your configuration
        vector_db_service = VectorDBService(
            embedding_config=settings["EMBEDDING_CONFIG"],
            vectordb_config=settings["VECTORDB_CONFIG"],
        )
        try:
            # Set insert-optimized mode for bulk loading
            await vector_db_service.update_collection_params(optimize_for="insert")

            llama_vector_service = await LlamaIndexVectorService.initialize(
                vector_db_service=vector_db_service,
                persist_dir=settings["VECTORDB_CONFIG"]['persist_dir']
            )
            logging.debug("initialized llama vector service")

            # Create indexes for frequently queried fields
            payload_indexes = [
                ("post_type", {"type": "keyword"}),
                ("collection", {"type": "keyword"}),
                ("published", {"type": "keyword"}),
                ("source_type", {"type": "keyword"}),
                ("severity_type", {"type": "keyword"}),
                ("entity_type", {"type": "keyword"})
            ]

            for field_name, schema in payload_indexes:
                try:
                    await vector_db_service.create_payload_index(field_name, schema)
                except Exception as e:
                    logging.warning(f"Failed to create index for {field_name}: {e}")

            llama_documents = []
            dataframe_conversions = []
            # Define root-level keys for each document type
            kb_articles_root_keys = ['node_id', 'kb_id', 'published', 'title', 'text', 'build_number', 'product_build_id', 'article_url', 'excluded_embed_metadata_keys', 'summary']
            update_packages_root_keys = ['node_id', 'kb_id', 'product_build_id', 'product_build_ids',  'package_type', 'package_url', 'downloadable_packages', 'excluded_embed_metadata_keys']
            symptoms_root_keys = ['node_id', 'symptom_label', 'description', 'source_id', 'source_type', 'tags', 'source_id']
            causes_root_keys = ['node_id', 'cause_label', 'description', 'source_id', 'source_type', 'tags', 'source_id']
            fixes_root_keys = ['node_id', 'fix_label', 'description', 'source_id', 'source_type', 'tags', 'source_id']
            tools_root_keys = ['node_id', 'tool_label', 'description', 'source_id', 'source_type', 'tags', 'tool_url']
            msrc_posts_root_keys = ['node_id', 'excluded_embed_metadata_keys', 'excluded_llm_metadata_keys', 'text', 'embedding']
            patch_posts_root_keys = ['node_id', 'excluded_embed_metadata_keys', 'excluded_llm_metadata_keys', 'text', 'embedding']

            if isinstance(kb_articles_combined_df, pd.DataFrame) and not kb_articles_combined_df.empty:
                dataframe_conversions.append(('kb_articles', kb_articles_combined_df))

            if isinstance(update_packages_df, pd.DataFrame) and not update_packages_df.empty:
                dataframe_conversions.append(('update_packages', update_packages_df))

            if isinstance(all_symptoms_df, pd.DataFrame) and not all_symptoms_df.empty:
                dataframe_conversions.append(('symptoms', all_symptoms_df))

            if isinstance(all_causes_df, pd.DataFrame) and not all_causes_df.empty:
                dataframe_conversions.append(('causes', all_causes_df))

            if isinstance(all_fixes_df, pd.DataFrame) and not all_fixes_df.empty:
                dataframe_conversions.append(('fixes', all_fixes_df))

            if isinstance(all_tools_df, pd.DataFrame) and not all_tools_df.empty:
                dataframe_conversions.append(('tools', all_tools_df))

            if isinstance(msrc_posts_df, pd.DataFrame) and not msrc_posts_df.empty:
                dataframe_conversions.append(('msrc_posts', msrc_posts_df))

            if isinstance(patch_posts_df, pd.DataFrame) and not patch_posts_df.empty:
                dataframe_conversions.append(('patch_posts', patch_posts_df))

            async def convert_dataframe(name: str, df: pd.DataFrame):
                logging.info(f"converting {name} df to LLamaDoc")
                if name == 'kb_articles':
                    return transformer.convert_df_to_llamadoc_kb_articles(df)
                elif name == 'update_packages':
                    return transformer.convert_df_to_llamadoc_update_packages(df)
                elif name == 'symptoms':
                    return transformer.convert_df_to_llamadoc_symptoms(df)
                elif name == 'causes':
                    return transformer.convert_df_to_llamadoc_causes(df)
                elif name == 'fixes':
                    return transformer.convert_df_to_llamadoc_fixes(df)
                elif name == 'tools':
                    return transformer.convert_df_to_llamadoc_tools(df)
                elif name == 'msrc_posts':
                    return transformer.convert_df_to_llamadoc_msrc_posts(df, symptom_nodes, cause_nodes, fix_nodes, tool_nodes)
                elif name == 'patch_posts':
                    return transformer.convert_df_to_llamadoc_patch_posts(df, symptom_nodes, cause_nodes, fix_nodes, tool_nodes)

            conversion_tasks = [
                asyncio.create_task(convert_dataframe(name, df))
                for name, df in dataframe_conversions
            ]

            converted_docs = await asyncio.gather(*conversion_tasks)

            for docs in converted_docs:
                if docs:
                    llama_documents.extend(docs)

            if llama_documents:
                doc_tracker = CustomDocumentTracker(
                    persist_path=os.path.join(settings["VECTORDB_CONFIG"]['persist_dir'], 'doc_tracker.json')
                )

                # Configure batch size based on document count
                batch_size = min(max(1000, len(llama_documents) // 10), 5000)
                logging.info(f"Using batch size of {batch_size} for {len(llama_documents)} documents")

                try:
                    # UPSERT with optimized batch processing and proper await
                    # nodes_created = await llama_vector_service.upsert_documents(
                    #     documents=llama_documents,
                    #     verify_upsert=True,
                    #     wait=False,  # Don't wait for indexing
                    #     show_progress=True,
                    #     batch_size=batch_size
                    # )
                    nodes_created = None
                    if nodes_created is not None:
                        logging.info(f" - Created {nodes_created} nodes from {len(llama_documents)} documents")

                        # Track documents after successful upsert
                        tracking_tasks = []
                        for name, df in dataframe_conversions:
                            tracking_task = None
                            if name == 'kb_articles':
                                tracking_task = track_dataframe_documents(df, doc_tracker, kb_articles_root_keys)
                            elif name == 'update_packages':
                                tracking_task = track_dataframe_documents(df, doc_tracker, update_packages_root_keys)
                            elif name == 'symptoms':
                                tracking_task = track_dataframe_documents(df, doc_tracker, symptoms_root_keys)
                            elif name == 'causes':
                                tracking_task = track_dataframe_documents(df, doc_tracker, causes_root_keys)
                            elif name == 'fixes':
                                tracking_task = track_dataframe_documents(df, doc_tracker, fixes_root_keys)
                            elif name == 'tools':
                                tracking_task = track_dataframe_documents(df, doc_tracker, tools_root_keys)
                            elif name == 'msrc_posts':
                                tracking_task = track_dataframe_documents(df, doc_tracker, msrc_posts_root_keys)
                            elif name == 'patch_posts':
                                tracking_task = track_dataframe_documents(df, doc_tracker, patch_posts_root_keys)

                            if tracking_task:
                                tracking_tasks.append(tracking_task)

                        if tracking_tasks:
                            # Wait for all tracking tasks to complete concurrently
                            await asyncio.gather(*tracking_tasks)
                            logging.info(f" - Successfully tracked {len(tracking_tasks)} document groups")

                        # Save tracked documents
                        doc_tracker._save_catalog()
                        logging.info(" - Saved document tracking catalog")
                    else:
                        logging.error("Failed to create nodes - upsert returned None")
                        # raise ValueError("Upsert operation failed")

                except Exception as e:
                    logging.error(f"Error during document upsert or tracking: {str(e)}")
                    raise
            else:
                logging.info("No documents to process")
        except Exception as e:
            logging.error(f"Error in llama workflow in pipeline: {e}")
            raise

        finally:
            # Switch back to search-optimized mode before closing
            try:
                await vector_db_service.update_collection_params(optimize_for="search")
            except Exception as e:
                logging.warning(f"Failed to switch to search mode: {e}")

            # Ensure cleanup happens even if there's an error
            if 'llama_vector_service' in locals():
                await llama_vector_service.aclose()
            await vector_db_service.aclose()

    except Exception as e:
        logging.error(f"Error in full ingestion pipeline: {e}")
        raise

    logging.info("Done with vector database loading ------------------------------\n")

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    logging.info(f"Time taken to upsert vector data: {int(minutes)} min : {int(seconds)} sec")

    logging.info("Begin document updating ------------------------------\n")
    start_time = time.time()

    msrc_exclude_columns = [
        'node_id',
        'metadata',
        'excluded_embed_metadata_keys',
        'excluded_llm_metadata_keys',
        'text',
        'node_label',
        'patterns_found',
        'patterns_missing',
        'revision',
        'published',
        'title',
        'description',
        'embedding',
        'post_type',
        'collection',
        'kb_ids',
        'source',
        'build_numbers',
        'impact_type',
        'severity_type',
        'product_build_ids',
        'summary',
        'cve_fixes',
        'cve_mentions',
        ]

    patch_exclude_columns = [
        'node_id',
        'metadata',
        'receivedDateTime',
        'published',
        'subject',
        'topic',
        'build_numbers',
        'kb_ids',
        'excluded_embed_metadata_keys',
        'excluded_llm_metadata_keys',
        'noun_chunks',
        'keywords',
        'text',
        'node_label',
        'embedding',
        'post_type',
        'collection',
        'conversation_link',
        ]
    kb_exclude_columns = [
        'node_id',
        'excluded_embed_metadata_keys',
        'text',
        'node_label',
        'title',
        'product_build_id',
        'product_build_ids',
        'build_number',
        'kb_id',
        'published',
        'article_url',
        'embedding'
        ]

    NULL_VALUE_REPLACEMENTS = {
        pd.NA: None,
        "None": None,
        "none": None,
        "NULL": None,
        "null": None,
        pd.NaT: None,
        np.nan: None
    }

    def convert_and_replace_nulls(df: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical columns to object type and replace null values."""
        # Get categorical columns
        categorical_columns = df.select_dtypes(include=['category']).columns

        # Convert categorical columns to object type
        if not categorical_columns.empty:
            df = df.copy()
            for col in categorical_columns:
                df[col] = df[col].astype('object')

        # Replace null values
        return df.replace(NULL_VALUE_REPLACEMENTS)

    logging.info("STARTING MSRC DOCUMENT UPDATES")
    # print(f"before document upsert - msrc_df.columns:\n{msrc_posts_df.columns}")
    # Update MSRC Posts
    if isinstance(msrc_posts_df, pd.DataFrame) and not msrc_posts_df.empty:

        msrc_posts_df = convert_and_replace_nulls(msrc_posts_df)
        docstore_service = DocumentService(db_name="report_docstore", collection_name="docstore")
        for _, row in msrc_posts_df.iterrows():
            try:
                logging.debug(f"Original row data:\n{row.to_dict()}")
                logging.debug(f"Columns being excluded: {msrc_exclude_columns}")

                exclude_fields = set(msrc_exclude_columns)
                # Convert row data into metadata dictionary with explicit Series handling
                metadata_dict = {}
                for col in row.index:
                    if col not in exclude_fields:
                        value = row[col]
                        if isinstance(value, pd.Series):
                            # If we get a Series, log it and take the first non-null value
                            logging.warning(f"Found Series instead of scalar for column {col}: {value}")
                            non_null_values = value.dropna()
                            value = non_null_values.iloc[0] if not non_null_values.empty else None

                        if isinstance(value, (pd.Timestamp, np.datetime64)):
                            value = pd.Timestamp(value).to_pydatetime()
                        elif value is pd.NA:
                            value = None

                        metadata_dict[col] = value

                # Update ETL status
                metadata_dict['etl_processing_status'] = {
                    "document_processed": True,
                    "nvd_extracted": True,
                    "entities_extracted": True,
                    "graph_prepared": True,
                    "vector_prepared": True,
                    "last_processed_at": datetime.now().isoformat(),
                    "processing_version": "1.0"
                }

                # Create DocumentMetadata - it will handle all the nested dictionary logic now
                metadata = DocumentMetadata(
                    id=row['node_id'],
                    **metadata_dict
                )
                logging.debug(f"Created DocumentMetadata:\n{metadata.model_dump(exclude_none=False)}")

                doc = Document(
                    id_=row['node_id'],
                    metadata=metadata,
                    text=row['text'],
                    embedding=row['embedding']
                    )
                logging.debug(f"Created Document instance:\n{doc.model_dump(exclude_none=False)}")

                update_result = docstore_service.update_document(
                    row['node_id'],
                    doc,
                    False
                )
                logging.info(f"Update result for document {row['node_id']}: {update_result}")
            except Exception as e:
                logging.error(f"Error processing document {row['node_id']}: {str(e)}")
                logging.error(f"Row data that caused error:\n{row.to_dict()}")
                continue

        logging.info(f"Updated {len(msrc_posts_df)} MSRC posts")
    logging.info("STARTING PATCH DOCUMENT UPDATES")

    # Update Patch Management Posts
    if isinstance(patch_posts_df, pd.DataFrame) and not patch_posts_df.empty:
        patch_posts_df = convert_and_replace_nulls(patch_posts_df)
        docstore_service = DocumentService(db_name="report_docstore", collection_name="docstore")

        for _, row in patch_posts_df.iterrows():
            logging.debug(f"Original row data:\n{row.to_dict()}")
            logging.debug(f"Columns being excluded: {patch_exclude_columns}")

            # Extract original metadata
            # original_metadata = {}
            # if 'metadata' in row:
            #     source_metadata = row['metadata']
            #     if isinstance(source_metadata, str):
            #         try:
            #             original_metadata = json.loads(source_metadata)
            #         except json.JSONDecodeError:
            #             original_metadata = {}

            #     elif isinstance(source_metadata, dict):
            #         original_metadata = source_metadata

            # Get new/modified fields only
            new_metadata = {}
            exclude_fields = set(patch_exclude_columns)

            for col in row.index:
                col_name = col if isinstance(col, str) else str(col)
                value = row[col]
                # Check if the field itself exists (is not NA)
                if col_name not in exclude_fields and value is not pd.NA:
                    if col_name == "receivedDateTime" and isinstance(value, pd.Timestamp):
                        value = value.isoformat()
                    elif col_name == "published" and isinstance(value, pd.Timestamp):
                        value = value.to_pydatetime()
                    new_metadata[col_name] = value

            # Add ETL status
            new_metadata['etl_processing_status'] = {
                    "document_processed": True,
                    "entities_extracted": True,
                    "graph_prepared": True,
                    "vector_prepared": True,
                    "last_processed_at": datetime.now().isoformat(),
                    "processing_version": "1.0"
                }

            # Combine metadata and create instances
            # final_metadata = {**original_metadata, **new_metadata}
            final_metadata = {**new_metadata}
            metadata = DocumentMetadata(
                id=row['node_id'],
                **final_metadata
            )

            # Create document with optional fields
            doc = Document(
                id_=row['node_id'],
                metadata=metadata,
                text=row['text'] if 'text' in row and not pd.isna(row['text']) else "",
                embedding=row['embedding'] if 'embedding' in row and not pd.isna(row['embedding']) else None
            )
            logging.debug(f"Created Document instance:\n{doc.model_dump(exclude_none=False)}")

            update_result = docstore_service.update_document(
                row['node_id'],
                doc,
                False
            )
            logging.info(f"Update result for document {row['node_id']}: {update_result}")
        logging.info(f"Updated {len(patch_posts_df)} Patch Management posts")

    logging.info("STARTING KB ARTICLE DOCUMENT UPDATES")

    if isinstance(kb_articles_combined_df, pd.DataFrame) and not kb_articles_combined_df.empty:
        kb_articles_combined_df = convert_and_replace_nulls(kb_articles_combined_df)
        kb_service = DocumentService(db_name="report_docstore", collection_name="microsoft_kb_articles")
        for _, row in kb_articles_combined_df.iterrows():
            # Convert row data into update dictionary, excluding specified columns
            update_data = {
                col: (
                    row[col].to_pydatetime() if isinstance(row[col], pd.Timestamp)
                    else None if row[col] is pd.NA
                    else row[col]
                )
                for col in row.index
                if col not in kb_exclude_columns
            }
            # Add text and title fields to update_data explicitly
            if 'text' in row.index:
                update_data['text'] = row['text']
            if 'title' in row.index:
                update_data['title'] = row['title']
            if 'scraped_markdown' in row.index:
                update_data['scraped_markdown'] = row['scraped_markdown']
            if 'scraped_json' in row.index:
                update_data['scraped_json'] = row['scraped_json']
            # update the etl_processing_status dictionary with latest status values
            if 'metadata' in row:
                current_metadata = row['metadata']
                if isinstance(current_metadata, str):
                    try:
                        current_metadata = json.loads(current_metadata)
                    except json.JSONDecodeError:
                        current_metadata = {}

                if not isinstance(current_metadata, dict):
                    current_metadata = {}

                # Get or create etl_processing_status
                etl_status = current_metadata.get('etl_processing_status', {})
                if not isinstance(etl_status, dict):
                    etl_status = {}

                # Update ETL status
                etl_status.update({
                    'vector_prepared': True,
                    'entities_extracted': True,
                    'graph_prepared': True,
                    'last_processed_at': datetime.now().isoformat()
                })

                # Update metadata in update_data
                current_metadata['etl_processing_status'] = etl_status
                update_data['metadata'] = current_metadata
            # Create filter to match documents
            filter_query = {
                "id": row['node_id'],
                "product_build_id": row['product_build_id']
            }

            # matched_documents_count = kb_service.collection.count_documents(filter_query)
            # logging.info(f"Matched documents count for filter {filter_query}: {matched_documents_count}")
            # Update all matching documents
            update_result = kb_service.update_documents(filter_query, update_data)
            logging.info(f"Update result for document {row['node_id']}: {update_result} documents updated")

            # updated_document = kb_service.collection.find_one(filter_query)
            # logging.info(f"Updated document for node_id {row['node_id']}:\n{updated_document}")
            # if update_result == 0:
            #     print(f"filter_query:\n{filter_query}")
            #     print(f"update_data:\n{update_data}")
            #     logging.warning(f"No documents were updated for node_id {row['node_id']}. Check filter and update_data.")

        logging.info(f"Updated {len(kb_articles_combined_df)} KB Articles")

    logging.info("End document updating ------------------------------\n")
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    logging.info(f"Time taken to upsert mongo data: {int(minutes)} min : {int(seconds)} sec")

    if msrc_failed_extractions:
        logging.warning(f"Failed to extract {len(msrc_failed_extractions)} entities from MSRC posts")
        for entity in msrc_failed_extractions:
            logging.warning(f"Failed to extract entity: {entity}")
    if patch_failed_extractions:
        logging.warning(f"Failed to extract {len(patch_failed_extractions)} entities from Patch posts")
        for entity in patch_failed_extractions:
            logging.warning(f"Failed to extract entity: {entity}")

    logging.info("Full Ingestion complete =====================================\n")

    def format_extraction_issue(source_id: str, node_type: str, issue_type: str, message: str = None) -> str:
        base_msg = f"- source: {source_id} - {node_type} - {issue_type}"
        return f"{base_msg} - {message}" if message else base_msg

    # Process MSRC Posts
    msrc_issues = []
    # Add failed extractions
    for entity in msrc_failed_extractions:
        msrc_issues.append(format_extraction_issue(
            getattr(entity, 'source_id', 'unknown'),
            getattr(entity, 'node_label', 'unknown'),
            'Failed Extraction',
            getattr(entity, 'error', '')
        ))
    # Add empty extractions
    for node_type, items in msrc_empty_extractions.items():
        if items:  # Only process if there are actual items
            for item in items:
                msrc_issues.append(format_extraction_issue(
                    item.get('source_id', 'unknown'),
                    node_type,
                    'Empty Extraction'
                ))
    # Add loading errors
    for node_type, errors in error_dict.items():
        for error in errors:
            if error.get('source_type') == 'MSRCPost':
                msrc_issues.append(format_extraction_issue(
                    error.get('source_id', 'unknown'),
                    node_type,
                    'Failed Loading',
                    error.get('message', '')
                ))

    # Process Patch Posts
    patch_issues = []
    # Add failed extractions
    for entity in patch_failed_extractions:
        patch_issues.append(format_extraction_issue(
            getattr(entity, 'source_id', 'unknown'),
            getattr(entity, 'node_label', 'unknown'),
            'Failed Extraction',
            getattr(entity, 'error', '')
        ))
    # Add empty extractions
    for node_type, items in patch_empty_extractions.items():
        if items:  # Only process if there are actual items
            for item in items:
                patch_issues.append(format_extraction_issue(
                    item.get('source_id', 'unknown'),
                    node_type,
                    'Empty Extraction'
                ))
    # Add loading errors
    for node_type, errors in error_dict.items():
        for error in errors:
            if error.get('source_type') == 'PatchManagementPost':
                patch_issues.append(format_extraction_issue(
                    error.get('source_id', 'unknown'),
                    node_type,
                    'Failed Loading',
                    error.get('message', '')
                ))

    # Check if there are any issues to report
    has_issues = bool(msrc_issues or patch_issues or
                      any(items for items in msrc_empty_extractions.values()) or
                      any(items for items in patch_empty_extractions.values()))

    print("\nExtraction and Loading Results:")
    print("=" * 80)

    if not has_issues:
        print("No failures or empty extractions in batch")
        logging.info("Empty extraction state:")
        logging.info(f"MSRC empty extractions: {dict(msrc_empty_extractions)}")
        logging.info(f"Patch empty extractions: {dict(patch_empty_extractions)}")
        logging.info(f"Stats: {stats}")
    else:
        # Display results
        if msrc_issues:
            print("\nMSRC Posts Issues:")
            print("\n".join(sorted(msrc_issues)))

        if patch_issues:
            print("\nPatch Management Posts Issues:")
            print("\n".join(sorted(patch_issues)))

        # Display summary stats
        print("\nSummary Statistics:")
        print("-" * 40)
        for node_type, stat in stats.items():
            if stat["failed"] > 0 or stat["empty_extractions"] > 0:
                print(f"{node_type}:")
                if stat["failed"] > 0:
                    print(f"  Failed: {stat['failed']}/{stat['total_attempted']}")
                if stat["empty_extractions"] > 0:
                    print(f"  Empty: {stat['empty_extractions']}/{stat['total_attempted']}")

    print("=" * 80)

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


# def fuzzy_search(text, search_terms, threshold=80):
#     if pd.isna(text):
#         return []
#     matches = set()
#     for term in search_terms:
#         if fuzz.partial_ratio(text.lower(), term.lower()) >= threshold:
#             matches.add(term)
#     return list(matches)


# def fuzzy_search_column(column, products_df, threshold=80):
#     product_mentions = []
#     for text in column:
#         if isinstance(text, list):
#             text = " ".join(text)
#         matches = fuzzy_search(text, products_df["product_full"], threshold)
#         if not matches:
#             matches = fuzzy_search(text, products_df["product_name_version"], threshold)
#         if not matches:
#             matches = fuzzy_search(text, products_df["product_name"], threshold)
#         product_mentions.append(matches)
#     return product_mentions


# def retain_most_specific(mentions, products_df):
#     specific_mentions = set()
#     for mention in mentions:
#         parts = mention.split()
#         if len(parts) == 3:
#             product_name, product_version, product_architecture = parts
#             if product_version != "NV" and product_architecture != "NA":
#                 specific_mentions.add(mention)
#         elif len(parts) == 2:
#             product_name, product_version = parts
#             if product_version != "NV":
#                 specific_mentions.add(mention)
#         else:
#             specific_mentions.add(mention)

#     # Remove less specific mentions
#     final_mentions = set()
#     for mention in specific_mentions:
#         if not any(mention in other for other in specific_mentions if other != mention):
#             final_mentions.add(mention)

#     return list(final_mentions)


# def convert_to_original_representation(mentions):
#     return [mention.replace(" ", "_") for mention in mentions]


# def construct_regex_pattern():
#     max_digits_per_group = (4, 4, 5, 5)
#     pattern = (
#         r"\b"
#         + r"\.".join(
#             [r"\d{1," + str(max_digits) + r"}" for max_digits in max_digits_per_group]
#         )
#         + r"\b"
#     )
#     return pattern


# # Function to extract build numbers from text
# def extract_build_numbers(text, pattern):
#     matches = re.findall(pattern, text)
#     build_numbers = [[int(part) for part in match.split(".")] for match in matches]
#     return build_numbers


# def extract_windows_kbs(text):
#     pattern = r"(?i)KB[-\s]?\d{6,7}"
#     matches = re.findall(pattern, text)
#     # Convert matches to uppercase and ensure the format is "KB-123456"
#     matches = [match.upper().replace(" ", "").replace("-", "") for match in matches]
#     matches = [f"KB-{match[2:]}" for match in matches]
#     return list(set(matches))


# def extract_edge_kbs(row):
#     edge_kbs = []
#     if "edge" in row["product_mentions"]:
#         for build_number in row["build_numbers"]:
#             build_str = ".".join(map(str, build_number))
#             edge_kbs.append(f"KB-{build_str}")
#     return list(set(edge_kbs))


async def update_email_records(emails_df, document_service):
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
                kb_mentions=row["kb_ids"],
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
                f"Exception {e}. Data input: doc id: {document_id}\n{row['product_mentions']}\n{row['build_numbers']}\n{row['kb_mentions']}"
            )

    logging.info(f"Processed {total_processed} documents, updated {total_updated} documents")
    if not_updated_docs:
        logging.debug(f"Documents not updated: {', '.join(map(str, not_updated_docs))}")


async def patch_feature_engineering_pipeline(
    start_date: datetime, end_date: datetime = None
):

    response = {
        "message": "patch feature engineering pipeline inomplete.",
        "status": "in progress",
        "code": 500,
    }
    logging.info(
        "Patch Feature Engineering data extraction ==========================="
    )
    start_time = time.time()
    # Ensure both dates are timezone-aware
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date and end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    product_docs = await asyncio.to_thread(
        extractor.extract_products,
        None
    )

    patch_docs = await asyncio.to_thread(
        extractor.patch_fe_extractor,
        start_date,
        end_date,
        max_records=None,
    )
    if not patch_docs:
        response = {
            "message": "No Patch documents found for the specified time period.",
            "status": "NoRecords",
            "code": 204  # Using 204 to indicate success but no content
        }
        return response
    products_df = await asyncio.to_thread(transformer.transform_products, product_docs)
    products_df = prepare_products(products_df)
    patch_posts_df = await asyncio.to_thread(
        transformer.patch_fe_transformer,
        patch_docs,
        products_df
    )

    document_service = DocumentService(collection_name="docstore")
    await update_email_records(patch_posts_df, document_service)

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    logging.info(
        f"Time taken to feature engineer Patch Posts: {int(minutes)} min : {int(seconds)} sec"
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


# def validate_pipeline(pipeline):
#     for stage in pipeline:
#         for key, value in stage.items():
#             if not isinstance(key, str):
#                 raise ValueError(f"Invalid key type in pipeline stage: {type(key)}")
#     return pipeline


async def extract_all_documents(start_date, end_date):
    """Extract all document types concurrently."""
    async def extract_with_logging(extraction_func, doc_type, *args):
        try:
            result = await asyncio.to_thread(extraction_func, *args)
            if isinstance(result, tuple):
                # Handle kb_articles case which returns two values
                logging.info(f"Extracted {len(result[0])} Windows-based {doc_type}")
                logging.info(f"Extracted {len(result[1])} Edge-based {doc_type}")
                return result
            else:
                logging.info(f"Extracted {len(result)} {doc_type}")
                return result
        except Exception as e:
            logging.error(f"Failed to extract {doc_type}: {str(e)}")
            raise

    try:
        # Create extraction tasks
        extraction_tasks = {
            "products": extract_with_logging(
                extractor.extract_products, "Products", None
            ),
            "product_builds": extract_with_logging(
                extractor.extract_product_builds, "Product Builds", start_date, end_date, None
            ),
            "kb_articles": extract_with_logging(
                extractor.extract_kb_articles, "KB Articles", start_date, end_date, None
            ),
            "update_packages": extract_with_logging(
                extractor.extract_update_packages, "Update Packages", start_date, end_date, None
            ),
            "msrc_posts": extract_with_logging(
                extractor.extract_msrc_posts, "MSRC Posts", start_date, end_date, None
            ),
            "patch_posts": extract_with_logging(
                extractor.extract_patch_posts, "Patch Posts", start_date, end_date, None
            ),
        }

        # Execute all extraction tasks concurrently
        results = await asyncio.gather(*extraction_tasks.values(), return_exceptions=True)

        # Process results and check for exceptions
        extracted_docs = {}
        for (key, result) in zip(extraction_tasks.keys(), results):
            if isinstance(result, Exception):
                logging.error(f"Extraction failed for {key}: {str(result)}")
                raise result
            extracted_docs[key] = result

        # Sort product builds by build number
        if "product_builds" in extracted_docs:
            extracted_docs["product_builds"] = sorted(
                extracted_docs["product_builds"],
                key=sort_by_build_number
            )

        # Sort MSRC posts by post_id
        if "msrc_posts" in extracted_docs:
            extracted_docs["msrc_posts"] = sorted(
                extracted_docs["msrc_posts"],
                key=lambda x: x.get("id_", "")
            )

        # Sort KB articles by kb_id
        if "kb_articles" in extracted_docs:
            windows_kb, edge_kb = extracted_docs["kb_articles"]
            # Filter out edge KB articles with None kb_id before sorting
            filtered_edge_kb = [kb for kb in edge_kb if kb.get("kb_id") is not None]
            extracted_docs["kb_articles"] = (
                sorted(windows_kb, key=lambda x: x.get("kb_id", "")),
                sorted(filtered_edge_kb, key=lambda x: x.get("kb_id", ""))
            )

        return extracted_docs

    except Exception as e:
        logging.error(f"Document extraction failed: {str(e)}")
        raise
