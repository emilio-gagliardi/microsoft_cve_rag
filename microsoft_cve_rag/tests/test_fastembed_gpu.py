"""Test FastEmbed GPU compatibility and initialization."""
import sys
from typing import Optional, Tuple
import torch  # noqa: F401 Required for ONNX Runtime
import onnxruntime as ort
# import fastembed
ort.preload_dlls()

from qdrant_client.qdrant_fastembed import TextEmbedding  # noqa: E402
providers = ["CUDAExecutionProvider"]


def test_fastembed_gpu() -> Tuple[bool, Optional[str]]:
    """Test if FastEmbed can initialize and use GPU.

    Returns:
        Tuple[bool, Optional[str]]: Success status and error message if any
    """
    try:
        # Initialize with Snowflake Small model
        embedder = TextEmbedding(
            model_name="BAAI/bge-small-en",
            providers=providers
        )

        # Try a simple embedding to verify functionality
        test_text = ["Testing GPU compatibility with FastEmbed"]
        embeddings = embedder.embed(test_text)

        # Convert to list to trigger actual computation
        _ = list(embeddings)

        return True, None

    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    success, error = test_fastembed_gpu()

    if success:
        print("\033[92mSuccess! FastEmbed initialized and used GPU without error.\033[0m")
        sys.exit(0)
    else:
        print(f"\033[91mFastEmbed GPU test failed with error:\n{error}\033[0m")
        sys.exit(1)
