
import re
import nltk
import unidecode


def _remove_newline(text):
    """ Removes new line symbols

    Args:
        text (str): original text

    Returns:
        (str): text without new line symbols
    """

    regex_pattern = "[\r|\n|\r\n]"
    return re.sub(regex_pattern, " ", text)


def _replace_dates(text):
    """ Replaces dates, months and days of the week for keywords

    Args:
        text (str): original text

    Returns:
        (str): preprocessed text
    """

    date_pattern = "(\d+)(/|.)(\d+)(/|.)(\d+)"
    new_text = re.sub(date_pattern, " data ", text)

    month_pattern = "janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro"
    new_text = re.sub(month_pattern, " mes ", new_text)

    day_pattern = "segunda|terça|quarta|quinta|sexta|sabado|sábado|domingo"
    new_text = re.sub(day_pattern, " diasemana ", new_text)
    return new_text


def _replace_numbers(text):
    """ Replaces numbers with the keyword 'numero'

    Args:
        text (str): original text

    Returns:
        (str): preprocessed text
    """

    return re.sub("[0-9]+", " numero ", text)


def _replace_negation_words(text):
    """ Replaces negation words with the keyword 'negação'

    Args:
        text (str): original text

    Returns:
        (str): preprocessed text
    """

    return re.sub("não|ñ|nao", " negação ", text)


def _remove_additional_whitespaces(text):
    """ Removes additional whitespaces

    Args:
        text (str): original text

    Returns:
        (str): preprocessed text
    """

    new_text = re.sub("\s+", " ", text)
    new_text = new_text.strip()
    return new_text


def _remove_stopwords_punctuation(text, stopwords):
    """ Removes stopwords and punctuation

    Args:
        text (str): original text
        stopwords (list): list of stopwords

    Returns:
        (str): preprocessed text
    """

    tokens = nltk.tokenize.word_tokenize(text)
    words = [t for t in tokens if t.isalpha() and t not in stopwords]
    return " ".join(words)


def _remove_accent_marks(text):
    """ Removes accent marks

    Args:
        text (str): original text

    Returns:
        (str): preprocessed text
    """

    return unidecode.unidecode(text)


def _text_stemmer(text, stemmer):
    """ Reduces each word of the text to its stem/root

    Args:
        text (str): original text
        stemmer (class): class of the stemmer

    Returns:
        (str): preprocessed text
    """

    return " ".join([stemmer.stem(word) for word in text.split()])


def text_preprocessing(text, stopwords, stemmer):
    """ Run the text preprocessing pipeline

    Args:
        text (str): original text
        stopwords (list): list of stopwords
        stemmer (class): class of the stemmer

    Returns:
        (str): preprocessed text
    """

    new_text = text.lower()
    new_text = _remove_newline(new_text)
    new_text = _replace_dates(new_text)
    new_text = _replace_numbers(new_text)
    new_text = _replace_negation_words(new_text)
    new_text = _remove_additional_whitespaces(new_text)
    new_text = _remove_stopwords_punctuation(new_text, stopwords)
    new_text = _remove_accent_marks(new_text)
    new_text = _text_stemmer(new_text, stemmer)
    return new_text
