# Dask Distributed DataFrames

## When to Use Dask

As a general rule of thumb:

* **100,000 rows:** Start *considering* Dask. At this point, you might not *need* it, but it's worth evaluating if your Pandas code is becoming slow or memory-intensive.
* **1,000,000 rows:** Seriously *consider* Dask. If you're consistently working with datasets of this size, Dask can likely provide significant performance improvements.
* **10,000,000 rows or more:** Dask is likely *necessary*. Pandas will struggle to handle datasets of this size efficiently.

## Example: Deduplication with Dask

Here's how you might adapt your deduplication code to use Dask:

```python
import dask.dataframe as dd
import pandas as pd

# Assuming your data is in a CSV file
ddf = dd.read_csv('your_data.csv')

# Scoring function (works on a Dask Series)
def calculate_score(row):
    score = (
        (row['summary'].notna() & (row['summary'].str.strip() != '')).astype(int) * 8 +
        (row['cve_ids'].apply(lambda x: isinstance(x, list) and len(x) > 0)).astype(int) * 4 +
        (row['title'].notna() & (row['title'].str.strip() != '')).astype(int) * 2 +
        (row['scraped_markdown'].notna() & (row['scraped_markdown'].str.strip() != '')).astype(int) +
        (row['text'].notna() & (row['text'].str.strip() != '')).astype(int)
    )
    return score

# Apply the scoring function
ddf['score'] = ddf.apply(calculate_score, axis=1, meta=('score', 'int'))

# Sort and deduplicate
ddf = ddf.sort_values(['kb_id', 'score'], ascending=[True, False])
ddf = ddf.groupby('kb_id').first().reset_index()

# Compute the result (convert to Pandas DataFrame)
deduplicated_df = ddf.compute()


## Scaling Dask with Multiple GPU Servers

### To scale Dask to use multiple GPU servers:

1. Set up your servers: Install the OS, GPU drivers, and necessary software (Python, Dask, cuDF).
2. Start the Dask scheduler: On one of your servers, start the Dask scheduler.
3. Start the Dask workers: On each of your servers, start a Dask worker process, telling it to connect to the scheduler.
4. Connect your client: In your Python code, create a Dask client and connect it to the scheduler.
Submit your tasks: Create Dask DataFrames and submit your computations. Dask will distribute the tasks to the workers and use the GPUs if available.

### Example using SSHCluster

```python
from dask.distributed import Client, SSHCluster

# Define the cluster
cluster = SSHCluster(
    hosts=["worker1", "worker2", "worker3", "worker4"],  # Replace with your worker hostnames
    connect_options={"username": "your_username"},  # Replace with your username
    worker_options={"nthreads": 4, "memory_limit": "8GB"},  # Adjust as needed
    scheduler_options={"port": 0, "dashboard_address": ":8787"},
)

# Connect to the cluster
client = Client(cluster)

# Now you can use dask as before
import dask.dataframe as dd
df = dd.read_csv("s3://bucket/data/*.csv")
result = df.groupby("id").value.mean().compute()
print(result)

# Close the client when done
client.close()
```
