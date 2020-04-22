from sklearn.datasets import fetch_20newsgroups
import logging
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def splitData(data):
    return data.data, data.target


def get_splited_words(sent):
    """
    Only keep characters in sent and split into words.
    """
    letters = re.sub("[^a-zA-Z]", " ", sent)
    words = letters.lower().split(" ")
    words = [w for w in words if w != ""]
    return words


def remove_stopwords(lst):
    """
    Remove stopwords and those words' length <= 3 from a lst of words
    """
    stops = stopwords.words("english")
    stemmer = SnowballStemmer("english")

    res = [stemmer.stem(w) for w in lst]
    res = [w for w in res if w not in stops and len(w) > 3]

    return res


def remove_noise(data):
    """
    remove all unnecessary characters from all data
    """
    result = []
    for sentence in data:
        words = get_splited_words(sentence)
        res = remove_stopwords(words)
        result.append(res)
    return result


def get_freq(data):
    """
    data: A list of list of words
    return: A counter obj which contains frequence of each single word.
    """
    res = Counter()
    for lst in data:
        res.update(Counter(lst))
    print "There are {} words.".format(len(res.items()))
    return res


def get_freq_words(freq, lower):
    """
    freq: A counter obj contains frequence of each word.
    return: words whose frequence lower than threshold.
    """
    res = []
    for k, v in freq.items():
        if v <= lower :
            res.append(k)
    print "There are {} words frequence lower than {}".format(len(res), lower)
    return res


def remove_by_freq(data, removes):
    res = []
    for lst in data:
        r = [w for w in lst if w not in removes]
        res.append(r)
    return res


def preprocess(data, lower):
    data = remove_noise(data)

    frequence = get_freq(data)
    removes = set(get_freq_words(frequence, lower))
    data = remove_by_freq(data, removes)

    print "There are {} words in preprocessed data".format(len(get_freq(data).items()))

    res = []
    for lst in data:
        res.append(" ".join(lst))
    return res


if __name__ == "__main__":
    # manually select 6 categories from dataset.
    categories = ["comp.os.ms-windows.misc", "misc.forsale", "rec.sport.baseball", "sci.electronics",
                  "talk.politics.mideast", "alt.atheism"]

    # remove some maybe useless parts from each text file.
    removes = ("headers", "footers", "quotes")

    train_set = fetch_20newsgroups(subset='train', categories=categories, shuffle=True,
                                   random_state=422, remove=removes)

    test_set = fetch_20newsgroups(subset='test', categories=categories, shuffle=True,
                                  random_state=422, remove=removes)

    train_X, train_y = splitData(train_set)
    test_X, test_y = splitData(test_set)

    print "train set have " + str(len(train_X))
    print "test set have " + str(len(test_X))

    lower = 5

    train_X_processed = preprocess(train_X, lower)

    test_X_processed = preprocess(test_X, lower)