from typing import List, Union, Optional
from transformers import PreTrainedTokenizerBase
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
from llama_index.core import Document


class DocumentChunker:
    """
    A class for splitting documents into manageable chunks using Llama Index's SentenceSplitter,
    SentenceWindowNodeParser, or Spacy for comparison.

    Attributes:
    -----------
    splitter : SentenceSplitter or SentenceWindowNodeParser
        An instance of Llama Index's splitter class, used for splitting documents.

    tokenizer : PreTrainedTokenizerBase
        A tokenizer from the transformers library used for handling tokenization.

    Methods:
    --------
    split_only(text: str, method: str, num_sentences: int) -> list:
        Splits the given document text into nodes or chunks based on the selected method.

    tokenize_only(text: str) -> list:
        Tokenizes the text using the provided tokenizer.

    split_sentences_then_tokenize(text: str) -> list:
        Splits the document into sentences and then tokenizes them, ensuring token limits are respected.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
        window_size: int = 3,
    ):
        """
        Initialize the DocumentChunker with a tokenizer and splitting configuration.

        Parameters:
        -----------
        tokenizer : PreTrainedTokenizerBase
            The tokenizer to be used for encoding the text.

        chunk_size : int, optional
            The maximum size of a chunk in tokens. Defaults to 1024.

        chunk_overlap : int, optional
            The number of tokens overlapping between chunks for context preservation. Defaults to 20.

        window_size : int, optional
            The number of sentences on either side to include in the window for context preservation.
            Defaults to 3.
        """
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.window_size = window_size

        # Initializing Llama Index splitters
        self.llama_splitter = SentenceSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        self.window_splitter = SentenceWindowNodeParser.from_defaults(
            window_size=self.window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_sentence",
        )

    def split_only(
        self, text: str, method: str = "llama", num_sentences: int = 3
    ) -> List[Union[Document, str]]:
        """
        Split the text into chunks using either LlamaIndex's SentenceSplitter, SentenceWindowNodeParser, or Spacy.

        Args:
            text (str): The input text to be split into chunks or nodes.
            method (str): The method to use for splitting. Either "llama", "window", or "spacy".
            num_sentences (int): Number of sentences per chunk when using Spacy.

        Returns:
            List[Union[Document, str]]: A list of Llama Index Document nodes or text chunks.
        """
        if method == "llama":
            print("Using SentenceSplitter")
            return self.llama_splitter.get_nodes_from_documents(
                [Document(text=text)], show_progress=False
            )
        elif method == "window":
            print("Using SentenceWindowNodeParser")
            return self.window_splitter.get_nodes_from_documents(
                [Document(text=text)], show_progress=False
            )
        elif method == "spacy":
            print("Using Spacy for sentence splitting")
            return self._split_with_spacy(text, num_sentences)
        else:
            raise ValueError("Invalid method. Use 'llama', 'window', or 'spacy'.")

    def _split_with_spacy(self, text: str, num_sentences: int) -> List[str]:
        """
        Split the text into chunks using Spacy, grouping sentences.

        Args:
            text (str): The input text to be split into chunks.
            num_sentences (int): Number of sentences to include in each chunk.

        Returns:
            List[str]: A list of text chunks.
        """
        try:
            import spacy

            nlp = spacy.load("en_core_web_lg")
        except ImportError:
            raise ImportError(
                "Spacy and the 'en_core_web_sm' model are required for Spacy-based splitting. "
                "Install them using: pip install spacy && python -m spacy download en_core_web_sm"
            )

        doc = nlp(text)
        sentences = list(doc.sents)
        chunks = []

        # Group sentences into chunks based on num_sentences parameter
        for i in range(0, len(sentences), num_sentences):
            chunk = " ".join([sent.text for sent in sentences[i : i + num_sentences]])
            chunks.append(chunk)

        return chunks

    def tokenize_only(self, text: str) -> List[int]:
        """
        Tokenize the text without splitting into chunks.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            List[int]: A list of token IDs.
        """
        return self.tokenizer.encode(text)

    def split_sentences_then_tokenize(self, text: str) -> List[str]:
        """Split into sentences first, then tokenize."""
        nodes = self.split_only(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for node in nodes:
            if isinstance(node, Document):
                sentence = node.get_text()  # Get the text from the node
            else:
                sentence = node
            sentence_tokens = self.tokenize_only(sentence)
            if current_length + len(sentence_tokens) > self.chunk_size:
                chunks.append(self.tokenizer.decode(current_chunk))
                current_chunk = sentence_tokens
                current_length = len(sentence_tokens)
            else:
                current_chunk.extend(sentence_tokens)
                current_length += len(sentence_tokens)

        if current_chunk:
            chunks.append(self.tokenizer.decode(current_chunk))

        return chunks

    def tokenize_then_split(self, text: str) -> List[str]:
        """Tokenize first, then try to split at sentence boundaries."""
        encoded = self.tokenize_only(text)
        chunks = []
        start = 0
        while start < len(encoded):
            end = start + self.chunk_size
            chunk_ids = encoded[start:end]

            # Try to find a sentence boundary
            sentence_end = self.find_sentence_boundary(chunk_ids)
            if sentence_end > 0:
                end = start + sentence_end

            chunk_text = self.tokenizer.decode(encoded[start:end])
            chunks.append(chunk_text)
            start = end - self.chunk_overlap

        return chunks

    def find_sentence_boundary(self, token_ids: List[int]) -> int:
        """Find the last sentence boundary in a list of token IDs."""
        sentence_end_tokens = [
            self.tokenizer.encode(".")[
                0
            ],  # Assuming tokenizer.encode returns a list, take the first element
            self.tokenizer.encode("!")[0],
            self.tokenizer.encode("?")[0],
        ]
        for i in reversed(range(len(token_ids))):
            if token_ids[i] in sentence_end_tokens:
                return i + 1
        return -1  # No sentence boundary found

if __name__ == "__main__":
    pass
    # chunker = DocumentChunker(tokenizer, chunk_size=1024, chunk_overlap=200)
# llama_chunks = chunker.split_only(text, method="window")
# print(f"num llama chunks: {len(llama_chunks)}")
# for idx, chunk in enumerate(llama_chunks):

#     window_context = chunk.metadata.get("window")
#     original_sentence = chunk.metadata.get("original_sentence")
#     print(f"chunk text {idx} - {chunk.get_text()}")
#     print("Original Sentence:", original_sentence)
#     print("Window Context:", window_context)
#     print()
# spacy_chunks = chunker.split_only(text, method="spacy", num_sentences=3)
# print(f"\n\nnum spacy chunks: {len(spacy_chunks)}")
# for idx, chunk in enumerate(spacy_chunks):
#     print(f"sentence {idx} - {chunk}")
# tokens = chunker.tokenize_only(text)
# for token in tokens:
#     print(token)