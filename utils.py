
import re
import nltk
import unidecode
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


######################
# TEXT PREPROCESSING #
######################
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


####################
# MODEL EVALUATION #
####################
def _create_cm_boxlabels(cm, percentage=True):
    """ Create the confusion matrix boxlabels (count and percentage)

    Args:
        cm (object): confusion matrix

    Returns:
        (list): list of boxlabels
    """

    blanks = ['' for i in range(cm.size)]

    if len(cm)==2:
        group_labels = ["TN\n", "FP\n", "FN\n", "TP\n"]
    else:
        group_labels = blanks

    group_counts = ["{0:0.0f}\n".format(value) for value in cm.flatten()]
    if percentage:
        group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,
                                                                group_counts,
                                                                group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cm.shape[0], cm.shape[1])
    return box_labels


def _create_summary_text(cm):
    """ Create model statistics summary

    Args:
        cm (object): confusion matrix

    Returns:
        (str): model statistics summary
    """

    accuracy  = np.trace(cm) / float(np.sum(cm))

    if len(cm)==2:
        #Metrics for Binary Confusion Matrices
        precision = cm[1,1] / sum(cm[:,1])
        recall    = cm[1,1] / sum(cm[1,:])
        f1_score  = 2*precision*recall / (precision + recall)
        stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                     accuracy,precision,recall,f1_score)
    else:
        stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    return stats_text


def evaluate_model(model,
                   X_train,
                   y_train,
                   X_valid,
                   y_valid,
                   categories='auto',
                   percentages=True,
                   figsize=None,
                   cmap='Blues',
                   title=None):
    """ Show confusion matrix and model statistics summary

    Args:
        model (object): machine learning model
        X_train (list): training data
        y_train (list): training labels
        X_valid (list): validation data
        y_valid (list): validation labels
        categories (str, optional): categories that will be displayed on the x and y axis. Defaults to 'auto'.
        percentages (bool, optional): if True show percentages. Defaults to True.
        figsize (tuple, optional): figure size. Defaults to None.
        cmap (str, optional): colormap. Defaults to 'Blues'.
        title (str, optional): plot's title. Defaults to None.
    """

    # Model training
    model_name = re.sub(r'\([^)]*\)', '', str(model))
    model.fit(X_train, y_train)

    # Creating plot data (confusion matrix, boxlabels and summary text)
    plot_data = {'train': {}, 'valid': {}}
    plot_data['train']['cm'] = confusion_matrix(y_train, model.predict(X_train))
    plot_data['valid']['cm'] = confusion_matrix(y_valid, model.predict(X_valid))
    for split in ["train", "valid"]:
        plot_data[split]['boxlabels'] = _create_cm_boxlabels(plot_data[split]['cm'], percentages)
        plot_data[split]['summary'] = _create_summary_text(plot_data[split]['cm'])

    # Plot Confusion Matrix and Summary Text
    if figsize==None:
        figsize = plt.rcParams.get('figure.figsize')
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    i = 0
    for split in ["train", "valid"]:
        sns.heatmap(plot_data[split]['cm'], annot=plot_data[split]['boxlabels'], fmt="", 
                    cmap=cmap, cbar=False, xticklabels=categories, yticklabels=categories, ax=ax[i])
        ax[i].set_ylabel('True label')
        ax[i].set_xlabel('Predicted label' + plot_data[split]['summary'])
        ax[i].set_title(f"{model_name} ({split})")
        i += 1
    fig.subplots_adjust(wspace=0.8)
