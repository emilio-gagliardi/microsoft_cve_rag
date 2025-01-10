# Qdrant Upsert Performance Optimization Blueprint

This document summarizes recommended strategies, configurations, and code samples for optimizing **upsert performance** in Qdrant, based on our discussions.

---

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
```

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

```

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


```

## FAQ
1. Can I Call update_collection() Multiple Times per Day?
Yes. Qdrant doesn’t impose a limit on reconfiguring collection settings. Commonly, you’d do it before and after large batch inserts. Just note that each reconfiguration can trigger partial index rebuilds, so you don’t want to toggle settings too frequently if you can avoid it.

2. What’s the Recommended Range for ef_construct?
- Typical practical range: 8–200.
- Lower (~8–32) = faster inserts, less accurate recall.
- Higher (~100–200) = slower inserts, better recall.
- Qdrant’s default is often around 100, which is a middle ground.
- Do I Need to Re-Create the Collection for These Settings?
- No. You can apply them to an existing collection with update_collection(...). No need to drop/re-create as long as you keep the same data.

3. Are Batch Sizes Configurable in All Methods?
- upload_points(): batch_size defaults to 64 (you can override).
- upsert() with a list doesn’t automatically chunk; you handle the chunking if needed.
- batch_update_points() is all-or-nothing, so if you want chunking there, you’d also handle it manually.
