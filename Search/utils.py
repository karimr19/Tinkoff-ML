import io
from itertools import islice
import numpy as np
from nltk.corpus import wordnet
from nltk import pos_tag, WordNetLemmatizer
from nltk.tokenize import word_tokenize


def load_vectors(fname, limit):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in islice(fin, limit):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data


def get_wordnet_pos(treebank_tag):
    my_switch = {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV,
    }
    for key, item in my_switch.items():
        if treebank_tag.startswith(key):
            return item
    return wordnet.NOUN


def my_lemmatizer(sent, stop_words):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(sent)
    tokenized_sent = [w.lower() for w in word_tokens if not w.lower() in stop_words]
    pos_tagged = [(word, get_wordnet_pos(tag))
                  for word, tag in pos_tag(tokenized_sent)]
    return [lemmatizer.lemmatize(word, tag)
            for word, tag in pos_tagged]


def lists_intersections(lists):
    if not lists:
        return []
    rows_cnt = len(lists)
    pointers = [0 for i in range(rows_cnt)]
    common_elements_cnt = 0
    common_elements = list()
    for i in range(0, len(lists[0])):
        pivot = lists[0][i]
        checked_lists_cnt = 1
        for j in range(1, rows_cnt):
            while pointers[j] < len(lists[j]) and pivot > lists[j][pointers[j]]:
                pointers[j] += 1
            if pointers[j] == len(lists[j]):
                if lists[j][pointers[j] - 1] != pivot:
                    break
            elif lists[j][pointers[j]] != pivot:
                break
            checked_lists_cnt += 1
        if checked_lists_cnt == rows_cnt:
            common_elements_cnt += 1
            common_elements.append(pivot)
        # Здесь нужно подобрать индекс
        if common_elements_cnt == 10:
            break
    return common_elements


def average_sentence_vector(sentence, word_2_vec):
    sum_vector = np.zeros((300,), dtype='float32')
    words = word_tokenize(sentence)
    vectors_cnt = 0
    for word in words:
        if word in word_2_vec:
            sum_vector = np.add(sum_vector, word_2_vec[word])
            vectors_cnt += 1
    if vectors_cnt > 0:
        return np.divide(sum_vector, len(words))
    return sum_vector
