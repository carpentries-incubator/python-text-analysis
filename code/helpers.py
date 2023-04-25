import pathlib
import parse
import pandas
import logging
import matplotlib.pyplot as plt
from collections import defaultdict

import nltk # preprocess_text function
import re # preprocess_text function
# prep lemmatizer
nltk.download('wordnet') # preprocess_text function
from nltk.stem import WordNetLemmatizer # preprocess_text function
# prep stopwords
from nltk.corpus import stopwords # preprocess_text function
nltk.download('stopwords') # preprocess_text function

# TODO make a requirements.txt too that can be used with pip

def create_file_list(directory, filter_str='*'):
    """
    Example:
        corpus_file_list = create_file_list("python-text-analysis/data", "*.txt")
    """
    files = pathlib.Path(directory).glob(filter_str)
    files_to_analyze = list(map(str, files))
    return files_to_analyze

def parse_authors_titles(data_dir, corpus_file_list):
    import parse
    authors = []
    titles = []
    for filename in corpus_file_list:
        bookdata = parse.search(data_dir+"{author}-{title}.txt", filename)
        if bookdata is not None:
            authors.append(bookdata["author"])
            titles.append(bookdata["title"])
        else:
            print(f"Problem processing {filename}")
    return authors, titles

def parse_into_dataframe(pattern, items, col_name="Item"):
    """
    Example:
        data = parse_into_dataframe(corpus_file_list, "python-text-analysis/data/{Author}-{Title}.txt", col_name="File")
    """
    results = []
    p = parse.compile(pattern)
    for item in items:
        result = p.search(item)
        if result is not None:
            result.named[col_name] = item
            results.append(result.named)

    return pandas.DataFrame.from_dict(results).sort_values('Author')


def lemmatize_files(tokenizer, corpus_file_list):
    """
    Example:
        data["Lemma_File"] = lemmatize_files(tokenizer, corpus_file_list)
    """
    logging.warning("This function is computationally intensive. It may take several minutes to finish running.")
    N = len(corpus_file_list)
    lemma_filename_list = []
    for i, filename in enumerate(corpus_file_list):
        progress_log = f"{i+1} out of {N}: Lemmatizing {filename}"
        print(progress_log)
        logging.info(progress_log)
        lemma_filename = filename + ".lemmas"
        lemma_filename_list.append(lemma_filename)
        open(lemma_filename, "w", encoding="utf-8").writelines(
            token.lemma_.lower() + "\n"
            for token in tokenizer(open(filename, "r", encoding="utf-8").read())
        )
    return lemma_filename_list


def preprocess_text(text: str, remove_stopwords: bool, verbose: bool) -> list:
    """Function that cleans the input text by going to:
    - remove numbers
    - remove special characters
    - remove stopwords
    - convert to lowercase
    - remove excessive white spaces
    Arguments:
        text (str): text to clean
        remove_stopwords (bool): whether to remove stopwords
    Returns:
        str: cleaned text
    """
    # remove numbers and special characters
    text = re.sub("[^A-Za-z]+", " ", text)

    # 1. create tokens with help from nltk
    tokens = nltk.word_tokenize(text)
    if verbose:
        print('Tokens', tokens)
    # 2. lowercase and strip out any extra whitepaces with .strip()
    tokens = [w.lower().strip() for w in tokens]
    if verbose:
        print('Lowercase', tokens)

    # 3. convert tokens to lemmatized versions
    wnl = WordNetLemmatizer()  # fast lemmatizer
    tokens = [wnl.lemmatize(w) for w in tokens]
    if verbose:
        print('Lemmas', tokens)

    # remove stopwords
    if remove_stopwords:
        # 2. check if it's a stopword
        tokens = [w for w in tokens if
                  not w.lower() in stopwords.words("english")]
        if verbose:
            print('StopRemoved', tokens)

    # return a list of cleaned tokens
    return tokens


def var_explained_plot(model):
    """
    Example:
        var_explained_plot(model)
    """
    yvals = model.explained_variance_ratio_ * 100
    xvals = range(len(yvals))
    plt.plot(xvals, yvals)
    plt.xlabel("Topic Number")
    plt.ylabel("Percent Explained")
    plt.title("Dropoff of Variance Explained")
    plt.show()

def lsa_plot(data, model, x="X", y="Y", xlabel="Topic X", ylabel="Topic Y", title="My LSA Plot", groupby=None, colors={}):
    """
    Example:
        colors = {
            "austen": "red",
            "chesterton": "blue",
            "dickens": "green",
            "dumas": "orange",
            "melville": "cyan",
            "shakespeare": "magenta"
        }

        lsa_plot(data, model, groupby="Author", colors=colors)
    """
    xR2 = round(model.explained_variance_ratio_[1] * 100, 2)
    yR2 = round(model.explained_variance_ratio_[2] * 100, 2)
    if groupby is not None:
        colormap = defaultdict(lambda: "black", colors)
        data["Color"] = data[groupby].map(colormap)
        for group, items in data.groupby(by=groupby):
            items.plot(
                x, y,
                label=group,
                c="Color",
                kind="scatter",
                ax=plt.gca(),
                figsize=[5, 5],
                xlim=[-1, 1],
                ylim=[-1, 1],
                title=title,
                xlabel=f"{xlabel} ({xR2}%)",
                ylabel=f"{ylabel} ({yR2}%)"
            )
    else:
        data.plot(
                x, y,
                kind="scatter",
                ax=plt.gca(),
                figsize=[5, 5],
                xlim=[-1, 1],
                ylim=[-1, 1],
                title=title,
                xlabel=f"{xlabel} ({xR2}%)",
                ylabel=f"{ylabel} ({yR2}%)"
            )

    plt.show()

def showTopics(vectorizer, model, topic_number=1, n=10):
    """
    Example:
        showTopics(vectorizer, model, topic_number=1, n=5)
    """
    terms = vectorizer.get_feature_names_out()
    weights = model.components_[topic_number]
    df = pandas.DataFrame({"Term": terms, "Weight": weights})
    tops = df.sort_values(by=["Weight"], ascending=False)[0:n]
    bottoms = df.sort_values(by=["Weight"], ascending=False)[-n:]
    print(pandas.concat([tops, bottoms]))
