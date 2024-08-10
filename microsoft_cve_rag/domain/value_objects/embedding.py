# embedding.py
# Purpose: This script defines the Embedding value object.
# Explanation: A value object is an object that contains certain values and is used to pass data around.
# In this case, an Embedding has a list of numbers representing text in a machine-readable format.
# Regular Class vs. Pydantic: We use a Pydantic model here because it provides strong data validation and serialization capabilities.
# Relationships: This script may be used by services and repositories that handle embedding data.
# Example Usage:
# from microsoft_cve_rag.domain.value_objects.embedding import Embedding

from pydantic import BaseModel
from typing import List


class Embedding(BaseModel):
    values: List[float]
