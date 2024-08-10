class EmbeddingConfig:
    def __init__(self, model_name: str, embedding_length: int, model_card: str):
        self.model_name = model_name
        self.embedding_length = embedding_length
        self.model_card = model_card


# Default embedding configuration
DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig(
    model_name="snowflake-arctic-embed-l",
    embedding_length=1024,
    model_card="https://huggingface.co/Snowflake/snowflake-arctic-embed-l",
)
