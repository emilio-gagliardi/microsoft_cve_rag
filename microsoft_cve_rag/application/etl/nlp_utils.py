import spacy
from pydantic import BaseModel
import marvin
import asyncio
from enum import Enum
from application.app_utils import get_openai_api_key

# Load spaCy model
nlp = spacy.load("en_core_web_lg")
marvin.settings.openai.api_key = get_openai_api_key()


def count_syllables(word):
    word = word.lower()
    syllable_count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        syllable_count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            syllable_count += 1
    if word.endswith("e"):
        syllable_count -= 1
    if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
        syllable_count += 1
    if syllable_count == 0:
        syllable_count = 1
    return syllable_count


def calculate_flesch_kincaid_grade(text):
    """
    The Flesch-Kincaid readability score is a metric used to assess the readability of a text. It provides an indication of the level of education required to understand the text. The score is calculated based on the average sentence length and the average number of syllables per word.
    """
    doc = nlp(text)

    total_words = len([token for token in doc if not token.is_punct])
    total_sentences = len(list(doc.sents))
    total_syllables = sum(
        count_syllables(token.text) for token in doc if not token.is_punct
    )

    if total_sentences == 0 or total_words == 0:
        return 0

    flesch_kincaid_grade = (
        (0.39 * (total_words / total_sentences))
        + (11.8 * (total_syllables / total_words))
        - 15.59
    )
    return flesch_kincaid_grade


class ReliabilityType(Enum):
    LOW = "The text has a low reliability"
    MEDIUM = "The text has a medium reliability"
    HIGH = "The text has high reliability"


async def classify_reliability(text: str) -> ReliabilityType:
    """
    Classifies the reliability of the given text.
    """
    instructions = """
    Evaluate the reliability of the text based on the following criteria:
    - Presence of verifiable facts or sources
    - Consistency of information
    - Objectivity of the language used
    - Absence of exaggerated claims or sensationalism
    - Presence of sufficient text to fully explain the problem, solution, cause or tool
    """

    result = await marvin.classify_async(
        text,
        labels=ReliabilityType,
        instructions=instructions,
        model_kwargs={"temperature": 0.3},
    )

    return result


# Example usage
# email_text = "I have observed symptoms in my network and found a solution that resolved the problem."
# flesch_kincaid_score = calculate_flesch_kincaid_grade(email_text)
# print(f"Flesch-Kincaid Grade Level: {flesch_kincaid_score}")

# Example usage
# async def main():
#     text_to_classify = "According to a recent study published in Nature, global temperatures have risen by 1.1°C since pre-industrial times."
#     reliability = await classify_reliability(text_to_classify)
#     print(f"The text reliability is classified as: {reliability.name}")
