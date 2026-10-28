import re

import emoji
import nltk


# function to remove urls and emojis, etc

def cleanup(text):
    """Replace unwanted content with spaces without shifting positions."""
    if not isinstance(text, str):
        return text

    # remove links
    text = re.sub(
        r"http\S+|www\S+",
        lambda match: " " * len(match.group()),
        text,
        flags=re.IGNORECASE,
    )

    characters = list(text)

    # remove emojis
    # NB: later on could try keeping them to see diff
    for match in emoji.emoji_list(text):
        start = match["match_start"]
        end = match["match_end"]
        characters[start:end] = " " * (end - start)

    text = "".join(characters)

    # Replace any remaining emoji variation selectors with spaces.
    return text.replace("\ufe0e", " ").replace("\ufe0f", " ")


# function to tokenize while retaining position
def tokenize_with_positions(text, stop_words, sentence_tokenizer, word_tokenizer):
    """Tag complete sentences and retain word positions for BERT"""
    if not isinstance(text, str):
        return []

    word_records = []

    # tokenizer gives us the start and end position of each word in the sentence
    for sentence_start, sentence_end in sentence_tokenizer.span_tokenize(text):
        sentence = text[sentence_start:sentence_end]

        words = word_tokenizer.tokenize(sentence)
        positions = list(word_tokenizer.span_tokenize(sentence))

        # Tag the complete sentence BEFORE lowercasing or removing stopwords.
        # Punctuation remains available as grammatical context.
        tagged_words = nltk.pos_tag(words)

        for (word, pos), (start, end) in zip(tagged_words, positions):
            # NB: we need to convert to lower case here because
            # BERT expects lowercase words
            # also, lowercasing later can change positions so be careful
            word = word.lower()

            # remove/ignore stop words
            if word in stop_words:
                continue

            # add to our list of words all the info
            # word, start and end positions of the word
            # Update: now we all add the POS tag
            word_records.append({
                "word": word,
                "pos": pos,
                "start": sentence_start + start,
                "end": sentence_start + end,
            })

    return word_records


def load_nlp_tools():
    """Load the baseline resources; downloads happen only during an explicit run."""
    from nltk.corpus import stopwords
    from nltk.tokenize import NLTKWordTokenizer, PunktTokenizer

    for resource_path, package in (
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt_tab/english", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ):
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(package, quiet=True, raise_on_error=True)
            except Exception as error:
                raise RuntimeError(f"Missing NLTK resource {package}. Install it with python -m nltk.downloader {package}.") from error
    return set(stopwords.words("english")), PunktTokenizer("english"), NLTKWordTokenizer()
