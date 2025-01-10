# Qdrant Upsert Performance Optimization Blueprint

This document summarizes recommended strategies, configurations, and code samples for optimizing **upsert performance** in Qdrant.

## Current State of Vector Upserting in the Application

Your application, built on **FastAPI**, exposes endpoints (via `curl` or other HTTP clients) to run a data processing pipeline. Within this pipeline, you have multiple “service” modules—one for each database (including the vector database)—that provide CRUD-like operations on those data stores.

### FastAPI Endpoint and ETL Pipeline
- **FastAPI** is the main entry point: you can trigger the ETL pipeline by sending HTTP requests to the exposed endpoint(s).
- Inside the pipeline (as shown in your `pipelines.py`), you collect data from several DataFrames, convert them into `llama_documents`, and then upsert them into Qdrant.

### Vector Service Module
- The **`VectorDBService`** (in `vector_db_service.py`) currently handles Qdrant interactions, including creating collections, generating embeddings (via FastEmbed or Ollama), and upserting data.
- The service implements asynchronous CRUD operations (`create_vector`, `get_vector`, `update_vector`), but **does not** yet leverage:
  - **Batching** (it upserts one `PointStruct` at a time)
  - **Configuration optimizations** (e.g., `HnswConfigDiff`, `OptimizersConfigDiff`)
- There's also an `update_vector` method that sets `wait=True`. You mentioned this was to ensure “other inserts didn't conflict.” However, `wait=True` **does not** handle concurrency conflicts in Qdrant; it simply waits for the indexing operation to complete before returning.
  - Qdrant resolves conflicts based on point IDs (if two inserts with the same ID occur, the last upsert overwrites the point).
  - If your IDs differ, Qdrant will store them as different points, whether you use `wait=True` or not.

### Misunderstanding `wait=True`
- `wait=True` in Qdrant tells the server to **finish** applying the upsert before returning control.
  - **Pro**: You know the data has been fully indexed and is ready for querying immediately after the call.
  - **Con**: Slower pipeline throughput if you're inserting large batches, as each upsert call waits for indexing.
- `wait=False` means the method returns once Qdrant **acknowledges** receiving the data, but may still be building indexes in the background.
  - This **does not** create or solve concurrency conflicts. It simply improves insertion speed by not blocking on the operation to finish.

### Summary of the Current State
- **Works End-to-End**: You can upsert documents from different DataFrames, and see them appear in the Qdrant UI.
- **Lacks Batching**: Each document is upserted individually, causing repeated overhead (network, index updates).
- **No Custom HNSW/Optimizer Config**: The service uses Qdrant defaults (e.g., `ef_construct=100`), which can slow inserts if you have many points.
- **Potential Overuse of `wait=True`**: The pipeline may be waiting unnecessarily between each insert, reducing throughput without providing concurrency protection.

**Takeaway**: You can maintain the same structure (CRUD via a service class) but introduce **batch upserts** and **optimized collection configs** to significantly speed up your insertion process. Also, consider using `wait=False` for big inserts unless you need immediate query availability for each document.


## Key Strategies

1. **Batch Your Upserts**
   - Reduce repeated overhead by inserting points in chunks rather than one-by-one.
   - Qdrant supports batching via:
     - `async upsert(...)` with a list of points
     - `upload_points(...)` for auto-chunking
     - `batch_update_points(...)` if you need mixed insert/delete/payload updates in one call

2. **Configure Partial Payload Indexing**
   - Only index the payload fields you actually need to filter on.
   - Use `create_payload_index(...)` for each field to be indexed. Fields *not* indexed remain as unindexed payload (still stored, not searchable).

3. **Tune HNSW Index Parameters**
   - `ef_construct` (lower = faster insert, higher = better recall)
   - `m` (links per node in the HNSW graph; lower = faster build, higher = better recall)
   - `max_indexing_threads`, `full_scan_threshold`, etc.
   - Adjust via `HnswConfigDiff` in `update_collection(...)`.

4. **Adjust Optimizers**
   - `indexing_threshold` (delay index building until you have a certain number of vectors)
   - `memmap_threshold`, `default_segment_number`, etc.
   - Adjust via `OptimizersConfigDiff` in `update_collection(...)`.

5. **Allocate More Docker Resources**
   - Increase CPU & memory in Docker Desktop or `docker-compose` so Qdrant has more capacity to build indexes quickly.

6. **Optional**: Toggle “Loose” vs. “Tight” Settings
   - Before heavy inserts, **lower** `ef_construct` and **raise** thresholds to minimize overhead.
   - After inserts complete, **raise** `ef_construct` to improve query recall.

---

## Implementation Tasks

### 1. Collection Configuration Optimization [Status: Completed]
- [x] 1.1 Enhance ensure_collection_exists_async method
  - Add HNSW config with optimized parameters
  - Add Optimizer config with higher indexing thresholds
  - Add payload schema definition
  - Make parameters configurable via vectordb_config
- [x] 1.2 Add configuration validation
  - Validate HNSW parameters
  - Validate Optimizer parameters
  - Ensure backward compatibility

### 2. Batch Processing Enhancement [Status: Completed]
- [x] 2.1 Enhance upsert_points method
  - Implement configurable batch sizes
  - Add parallel processing with asyncio.gather
  - Add retry logic with exponential backoff
  - Add progress tracking
  - Make wait parameter configurable
- [x] 2.2 Add batch processing validation
  - Validate batch sizes
  - Add error handling per batch
  - Implement batch failure recovery

### 3. Collection Parameter Management [Status: Completed]
- [x] 3.1 Add update_collection_params method
  - Implement mode switching (insert vs search)
  - Add HNSW parameter updates
  - Add Optimizer parameter updates
- [x] 3.2 Add create_payload_index method
  - Implement payload index creation
  - Add index validation
  - Add error handling
    Candidates for payload indexes (spans all document fields):
    - `build_numbers` (list of lists of ints)
    - `kb_ids` (list of strings)
    - `collection` (string)
    - `cwe_id` (string)
    - `cwe_name` (string)
    - `nvd_description` (string)
    - `post_id` (string)
    - `post_type` (string)
    - `products` (list of strings)
    - `published` (string)
    - `title` (string)
    - `entity_type` (string)
    - `source_type` (string)
    - `symptom_label` (string)
    - `cause_label` (string)
    - `fix_label` (string)
    - `tool_label` (string)
    - `tool_url` (string)
    - `tags` (string)
    - `severity_type` (string)
### 4. Testing and Validation [Status: Pending]
- [ ] 4.1 Unit Tests
  - Test collection creation with different configs
  - Test batch processing with various sizes
  - Test parameter updates
- [ ] 4.2 Integration Tests
  - Test end-to-end insert workflow
  - Test mode switching
  - Test error scenarios

### 5. Documentation [Status: Completed]
- [x] 5.1 Code Documentation
  - Add detailed docstrings
  - Add usage examples
  - Document configuration options
- [x] 5.2 Performance Documentation
  - Document optimization strategies
  - Add performance benchmarks
  - Add configuration guidelines

## Implementation Timeline
- Start Date: 2025-01-07
- Target Completion: 2025-01-14

## Task Dependencies
1. Collection Configuration must be completed before Batch Processing
2. Both must be completed before Parameter Management
3. Testing can begin after each major component
4. Documentation should be updated alongside implementation

## Progress Tracking
- [x] Phase 1: Collection Configuration (100%)
- [x] Phase 2: Batch Processing (100%)
- [x] Phase 3: Parameter Management (100%)
- [ ] Phase 4: Testing (Pending)
- [x] Phase 5: Documentation (100%)

## Implementation Example: ETL Pipeline

### Overview
This example demonstrates how to use the optimized VectorDBService in a real-world ETL pipeline for ingesting various document types (KB articles, update packages, symptoms, causes, fixes, tools, MSRC posts, and patch posts).

### Code Example
```python
# Initialize VectorDBService with configuration
vector_db_service = VectorDBService(
    embedding_config=settings["EMBEDDING_CONFIG"],
    vectordb_config=settings["VECTORDB_CONFIG"],
)

try:
    # 1. Set insert-optimized mode for bulk loading
    await vector_db_service.update_collection_params(optimize_for="insert")
    
    # 2. Initialize LlamaIndex service
    llama_vector_service = await LlamaIndexVectorService.initialize(
        vector_db_service=vector_db_service,
        persist_dir=settings["VECTORDB_CONFIG"]['persist_dir']
    )

    # 3. Create indexes for frequently queried fields
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

    # 4. Process documents in parallel
    conversion_tasks = [
        asyncio.create_task(convert_dataframe(name, df))
        for name, df in dataframe_conversions
    ]
    converted_docs = await asyncio.gather(*conversion_tasks)

    # 5. Optimize batch size and upsert
    if llama_documents:
        # Configure batch size based on document count
        batch_size = min(max(1000, len(llama_documents) // 10), 5000)
        
        # UPSERT with optimized batch processing
        await llama_vector_service.upsert_documents(
            llama_documents,
            verify_upsert=True,
            batch_size=batch_size,
            wait=False,  # Don't wait for indexing
            show_progress=True
        )

finally:
    # 6. Switch back to search-optimized mode before closing
    try:
        await vector_db_service.update_collection_params(optimize_for="search")
    except Exception as e:
        logging.warning(f"Failed to switch to search mode: {e}")
    
    # 7. Cleanup
    if 'llama_vector_service' in locals():
        await llama_vector_service.aclose()
    await vector_db_service.aclose()
```

### Key Features

1. **Mode Switching**
   - Starts in insert-optimized mode for bulk loading
   - Switches back to search-optimized mode when done
   - Handles mode switching errors gracefully

2. **Index Management**
   - Creates indexes for commonly queried fields
   - Uses keyword type for exact matching
   - Implements error handling for index creation

3. **Batch Processing**
   - Dynamic batch size calculation based on document count
   - Minimum batch size: 1000 documents
   - Maximum batch size: 5000 documents
   - Progress tracking enabled

4. **Parallel Processing**
   - Converts documents in parallel using asyncio
   - Handles multiple document types efficiently
   - Maintains document tracking

5. **Error Handling**
   - Graceful handling of index creation failures
   - Mode switching error recovery
   - Proper resource cleanup

### Performance Considerations

1. **Batch Size Selection**
   - Uses 10% of total documents as base
   - Capped between 1000 and 5000
   - Adjusts automatically based on document count

2. **Index Creation**
   - Creates indexes before bulk insert
   - Focuses on frequently queried fields
   - Non-blocking index creation

3. **Resource Management**
   - Proper cleanup in finally block
   - Async context management
   - Error recovery strategies

### Document Types Supported
- KB Articles
- Update Packages
- Symptoms
- Causes
- Fixes
- Tools
- MSRC Posts
- Patch Posts

Each document type maintains its own schema and tracking metadata while sharing the optimized vector storage infrastructure.

## Example 1: Batched Upsert (Using `async upsert`)

```python
import uuid
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

async def batch_upsert_points(async_client: QdrantClient, collection_name: str, documents):
    points_batch = []

    for doc in documents:
        embedding = await generate_embeddings_async(doc.text)  # your embedding logic
        point_id = str(uuid.uuid4())
        points_batch.append(
            PointStruct(
                id=point_id,
                vector=embedding[0],
                payload={
                    "metadata": doc.metadata,  # or doc.metadata.model_dump() if needed
                    "text": doc.text,
                }
            )
        )

    # Upsert all points in one request
    result = await async_client.upsert(
        collection_name=collection_name,
        points=points_batch,
        wait=False  # faster, won't wait for indexing to finish
    )
    print(f"Batched upsert result: {result.status}")

## Example 2: Partial Payload Indexing
### Only index the fields you plan to filter on:

```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import PayloadSchemaType

async def create_indexes(async_client: QdrantClient, collection_name: str):
    # Index the "tags" field for text-based search
    await async_client.create_payload_index(
        collection_name=collection_name,
        field_name="tags",
        field_schema=PayloadSchemaType.TEXT,
        wait=True
    )

    # Index the "category" field for exact matches
    await async_client.create_payload_index(
        collection_name=collection_name,
        field_name="category",
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True
    )

    # Any other fields remain in payload but are not indexed
    print("Payload indexes created for 'tags' and 'category'.")

## Example 3: Tuning HNSW and Optimizers via update_collection()

```python

from qdrant_client import QdrantClient
from qdrant_client.http.models import HnswConfigDiff, OptimizersConfigDiff

async def tune_collection_params(async_client: QdrantClient, collection_name: str):
    # Define HNSW settings
    my_hnsw_config = HnswConfigDiff(
        m=16,                    # Links per node
        ef_construct=32,         # Lower for faster insert; can raise after big insert
        full_scan_threshold=10000,
        max_indexing_threads=2,  # More threads => faster index build => higher CPU usage
        on_disk=False            # Set True if index is large & you want to reduce RAM usage
    )

    # Define Optimizer settings
    my_optimizers_config = OptimizersConfigDiff(
        indexing_threshold=50000,     # Build index after 50k vectors
        memmap_threshold=50000,       # Use memmapped storage after 50k vectors
        default_segment_number=2      # Fewer segments => merges happen less frequently
    )

    # Apply new configs
    result = await async_client.update_collection(
        collection_name=collection_name,
        hnsw_config=my_hnsw_config,
        optimizers_config=my_optimizers_config
    )

    if result:
        print(f"Collection '{collection_name}' updated successfully!")
    else:
        print(f"Failed to update collection '{collection_name}'")

## Recommended Workflow
1. Before Bulk Insert
- Set lower ef_construct, higher indexing_threshold, and a smaller number of segments to minimize overhead.
- Insert data in batches.
2. After Bulk Insert
- Optionally re-run update_collection() with higher ef_construct if you need better recall.
- Lower indexing_threshold or set it to None so normal incremental indexing resumes for smaller subsequent updates.

## Example 4: Increase Docker Resources
If using Docker Desktop:

Docker Desktop > Settings > Resources: Increase CPUs and Memory allocated to Docker.
For example, if your machine has 8 cores and 16 GB RAM, you could allocate 4 cores & 8 GB to Docker.
If using docker-compose:
```yaml
version: '3.7'
services:
  qdrant:
    image: qdrant/qdrant:v1.2.0
    container_name: qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_data:/qdrant/storage
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: '8G'

## FAQ
1. Can I Call update_collection() Multiple Times per Day?
Yes. Qdrant doesn't impose a limit on reconfiguring collection settings. Commonly, you'd do it before and after large batch inserts. Just note that each reconfiguration can trigger partial index rebuilds, so you don't want to toggle settings too frequently if you can avoid it.

2. What's the Recommended Range for ef_construct?
- Typical practical range: 8–200.
- Lower (~8–32) = faster inserts, less accurate recall.
- Higher (~100–200) = slower inserts, better recall.
- Qdrant's default is often around 100, which is a middle ground.
- Do I Need to Re-Create the Collection for These Settings?
- No. You can apply them to an existing collection with update_collection(...). No need to drop/re-create as long as you keep the same data.

3. Are Batch Sizes Configurable in All Methods?
- upload_points(): batch_size defaults to 64 (you can override).
- upsert() with a list doesn't automatically chunk; you handle the chunking if needed.
- batch_update_points() is all-or-nothing, so if you want chunking there, you'd also handle it manually.

## Usage Guide

### 1. Collection Configuration
```python
# Initialize with optimized settings for inserts
vector_service = VectorDBService(
    embedding_config={
        "embedding_provider": "fastembed",
        "fastembed_model_name": "BAAI/bge-small-en-v1.5"
    },
    vectordb_config={
        "tier1_collection": "my_collection",
        "distance_metric": "cosine",
        "hnsw_config": {
            "m": 16,
            "ef_construct": 32,
            "max_indexing_threads": 4
        },
        "optimizer_config": {
            "indexing_threshold": 50000,
            "default_segment_number": 2
        }
    }
)
```

### 2. Batch Processing
```python
# Process points in optimized batches
result = await vector_service.upsert_points(
    points=points_list,
    batch_size=1000,  # Will be automatically optimized
    wait=False,       # Don't wait for indexing
    show_progress=True
)

print(f"Inserted {result['successful_points']} points")
print(f"Performance: {result['points_per_second']} points/second")
```

### 3. Dynamic Optimization
```python
# Switch to insert-optimized mode for bulk operations
await vector_service.update_collection_params(optimize_for="insert")

# Perform bulk inserts...

# Switch back to search-optimized mode
await vector_service.update_collection_params(optimize_for="search")
```

### 4. Payload Index Management
```python
# Create indexes for frequently filtered fields
await vector_service.create_payload_index(
    field_name="post_type",
    field_schema={"type": "keyword", "index": True}
)

await vector_service.create_payload_index(
    field_name="published",
    field_schema={"type": "keyword", "index": True}
)

# List all active indexes
indexes = await vector_service.list_payload_indexes()
```

## Performance Optimization Guidelines

### 1. Insert Performance
- Use `wait=False` for faster inserts
- Set larger batch sizes (1000-5000) for bulk operations
- Use insert-optimized mode with:
  - Lower `ef_construct` (16-32)
  - Higher `indexing_threshold` (50000+)
  - Fewer segments (2-4)

### 2. Search Performance
- Switch to search-optimized mode after bulk inserts
- Use appropriate payload indexes for filtered fields
- Optimize HNSW parameters:
  - Higher `ef_construct` (100+)
  - Lower `indexing_threshold` (20000)
  - More segments (4-8)

### 3. Memory Management
- Monitor batch sizes to prevent OOM errors
- Use `on_disk=True` for very large collections
- Adjust `memmap_threshold` based on available RAM

### 4. Error Handling
- Use retry logic for network issues
- Monitor failed batches in upsert results
- Implement appropriate error recovery strategies
