import pickle
from nltk.corpus import stopwords
import string
import pandas as pd
from scipy import spatial
import utils


def build_index():
    # считывает сырые данные и строит индекс
    input_file = open('index.pkl', 'rb')
    data = pd.read_csv('abcnews-date-text.csv')
    word_2_vec = utils.load_vectors('/Users/karim/FastText/crawl-300d-2M.vec', 1000000)
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    index = pickle.load(input_file)
    input_file.close()
    return index, data, word_2_vec, stop_words


def score(query, document, word_2_vec):
    # возвращает какой-то скор для пары запрос-документ
    # больше -- релевантнее
    query_vector = utils.average_sentence_vector(query, word_2_vec)
    document_vector = utils.average_sentence_vector(document, word_2_vec)
    return 1 - spatial.distance.cosine(query_vector, document_vector)


def retrieve(query, inverted_index, data, stop_words):
    # возвращает начальный список релевантных документов
    # (желательно, не бесконечный)
    query_words = utils.my_lemmatizer(query, stop_words)
    lists = []

    for word in query_words:
        if word in inverted_index:
            lists.append(inverted_index[word])
    answers = []
    for index in utils.lists_intersections(lists):
        answers.append(data.iloc[index, 1])
    return answers
