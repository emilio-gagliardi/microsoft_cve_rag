import spacy
import re

# Load spaCy model
nlp = spacy.load("en_core_web_lg")


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


# Example usage
email_text = "I have observed symptoms in my network and found a solution that resolved the problem."
flesch_kincaid_score = calculate_flesch_kincaid_grade(email_text)
print(f"Flesch-Kincaid Grade Level: {flesch_kincaid_score}")
