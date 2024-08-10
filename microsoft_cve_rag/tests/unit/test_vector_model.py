# tests/unit/test_vector_model.py

import pytest
from pydantic import ValidationError
from application.core.models import Vector, VectorMetadata
from application.config import DEFAULT_EMBEDDING_CONFIG


def test_vector_model():
    metadata = VectorMetadata(cve_fixes="fix1", cve_mentions="mention1", tags="tag1")
    vector = Vector(
        embedding=[0.1] * DEFAULT_EMBEDDING_CONFIG.embedding_length,
        metadata=metadata,
        text="example text",
    )

    assert vector.id_ is not None
    assert vector.embedding is not None
    assert vector.metadata == metadata
    assert vector.text == "example text"
    assert vector.created_at is not None
    assert vector.updated_at is not None


def test_vector_model_validation():
    with pytest.raises(ValidationError):
        Vector(
            embedding=[0.1] * (DEFAULT_EMBEDDING_CONFIG.embedding_length - 1)
        )  # Invalid embedding length
