The ETL pipeline is not finding all the KB articles when creating `microsoft_kb_articles`. The collection `docstore` has records that match as KB articles and we need an aggregation pipeline or python workflow to find these records and create them in the collection `microsoft_kb_articles`.

The following mongo query can find the records in `docstore` that match as KB articles:
```mongodb
{
  "metadata.title": { "$regex": "kb\\d{6,7}", "$options": "i" },
  "metadata.published": {
    "$gte": ISODate("2024-06-15T00:00:00Z"),
    "$lte": ISODate("2024-06-30T23:59:59Z")
  }
}
```

We then need to create a route in the `etl_routes.py` to handle this workflow. All routes need pydantic models V2 with swagger docs. The route should also return a pydantic model with standardized response structure as other routes, i.e. `response, status, code` keys. All mongo credentials are loaded from environment variables.

Draft for the code below:
```python
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field

app = FastAPI(
    title="KB Article API",
    description="API for creating new KB articles in the MongoDB collection.",
    version="1.0.0"
)

# Pydantic models with swagger docs (using Pydantic v2 style)
class KBCreate(BaseModel):
    id: str = Field(..., description="Unique record identifier")
    build_number: Dict[str, Any] = Field(..., description="Build number details")
    kb_id: str = Field(..., description="Knowledge base id")
    published: Dict[str, Any] = Field(..., description="Published details")
    article_url: str = Field(..., description="URL of the article")
    cve_id: Dict[str, Any] = Field(..., description="CVE id details")
    cve_ids: Dict[str, Any] = Field(..., description="List of CVE ids")
    product_build_ids: Dict[str, Any] = Field(..., description="List of product build ids")
    metadata: Dict[str, Any] = Field(..., description="Metadata information")
    summary: str = Field(..., description="Summary of the record")
    text: str = Field(..., description="Full text content")
    title: str = Field(..., description="Title of the record")
    scraped_markdown: str = Field(..., description="Scraped markdown content")
    scraped_json: Dict[str, Any] = Field(..., description="Scraped JSON content")
    # This field will be forced to null since the value is unknown.
    product_build_id: Optional[str] = Field(default=None, description="Foreign key to product build, set to null if unknown")

# Pydantic response model
class KBCreateResponse(BaseModel):
    response: str = Field(..., description="Response message")
    status: str = Field(..., description="Status of the operation")
    code: int = Field(..., description="HTTP status code")

# Setup MongoDB connection using environment variable placeholders
MONGO_URI = os.getenv("MONGO_URI", "mongodb://<username>:<password>@<host>:<port>")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "your_db_name")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "your_collection_name")

client = AsyncIOMotorClient(MONGO_URI)
db = client[MONGO_DB_NAME]
collection = db[MONGO_COLLECTION_NAME]

@app.post("/create-kb-article", response_model=KBCreateResponse, summary="Create a new KB article", tags=["KB Articles"])
async def create_kb_article(record: KBCreate) -> KBCreateResponse:
    """
    Create a new KB article in the MongoDB collection.

    All required fields are provided in the request body.
    The 'product_build_id' is set to null because the actual UUID is not available.
    """
    # Convert the Pydantic model to a dict
    record_data = record.dict()
    # Enforce product_build_id to be null (None in Python)
    record_data["product_build_id"] = None

    result = await collection.insert_one(record_data)
    if not result.inserted_id:
        raise HTTPException(status_code=500, detail="Record creation failed")
    return KBCreateResponse(response="Record created successfully", status="success", code=200)
```

Create a test file in the `tests` directory that makes a request to the FastAPI endpoint with sample data.
