# Refactor Patch Management Posts Transformation Function for Historical Thread Merging

## Objective
1. Modify the Patch Management transformation function to include historical thread lookup.
2. Create a helper function to query and retrieve existing threads by matching metadata.
3. Refactor the thread creation logic to prevent duplication when a matching thread exists and reconnect messages within the same thread. The batching process creates arbitrary start and end dates for each ETL ingestion. This prematurely cuts the relationship between messages in the same thread.
4. Ensure backward compatibility with existing thread generation logic. The logic orders the documents chronologically and then computes the thread_id, next_id, and previous_id for each document.
5. When the helper function returns existing documents that match on metadata, the refactored transformation function should walk through all the matching docs within a thread and ensure all the next_id and previous_id values are correctly set.

## Context
- **Files Affected**:
  - `transformer.py`: Contains the existing patch management transformation function `transform_patch_posts` that needs to be refactored.
  - `transformer.py`: Will house the new helper function for querying historical threads.

## Low-Level Tasks (Ordered from Start to Finish)
1. **Create a new helper function in `transformer.py`**:
   - Name: `find_existing_thread(metadata: Dict) -> Optional[str]`
   - Logic:
     - Accept metadata as input (e.g., subject, timestamp).
     - Search the historical thread data for a match based on fuzzy similarity.
     - Return the thread ID if a match is found, else return `None`.
     - Add logging to indicate matches and non-matches.

2. **Refactor the transformation function in `transformer.py`**:
   - Update the logic to:
     - Use `find_existing_thread()` before creating a new thread.
     - If a thread is found, associate the current data with the existing thread ID.
     - If no thread is found, proceed with creating a new thread.

5. **Optimize performance**:
   - Profile the helper function for large historical datasets.
   - Implement caching or indexing if necessary for efficiency.
   - Assume that conversations quickly stale so don't cache more than 30 days.

6. **Update documentation**:
   - Document the new behavior of the transformation function in `project_notes.txt`.
   - Add comments and docstrings in the codebase explaining the rationale for changes.

