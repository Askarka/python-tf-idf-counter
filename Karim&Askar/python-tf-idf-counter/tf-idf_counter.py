import math
import os
import re

import nltk
import pymorphy2

from bs4 import BeautifulSoup
from nltk.corpus import stopwords

stopWords = stopwords.words("russian")

folder = 'data'


def lineWithoutStopWords(line):
    return [word for word in line if word not in stopWords]


morph = pymorphy2.MorphAnalyzer()


def normalForm(line):
    return [morph.parse(word)[0].normal_form for word in line if len(word) > 2]


if __name__ == '__main__':
    lemmas_tf = []
    tokens_tf = []

    all_docs_lemmas = []
    all_docs_tokens = []

    lemmas_idf = []
    tokens_idf = []

    text_without_tags = []

    progress_counter = 0

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.html'):
                text = open(folder + '/' + file, 'r')
                text = [line.lower() for line in text]
                # print(text)
                for line in text:
                    # text_without_tags.append(re.sub(r"<[^>]+>", "", line, flags=re.S))
                    soup = BeautifulSoup(line, features="html.parser")
                    text_line = soup.getText()
                    text_line.rstrip()
                    text_line_cleared = text_line.replace('\n', '')
                    if (text_line_cleared != '\n') and (text_line_cleared != '') :
                        text_without_tags.append(text_line_cleared)
                # print(text_without_tags)
                text_without_tags = [re.sub(r'[^\w\s]', ' ', line, flags=re.UNICODE) for line in text_without_tags]
                tokenized_text = [nltk.word_tokenize(line) for line in text_without_tags]
                tokenized_text_without_stop_words = [lineWithoutStopWords(line) for line in tokenized_text]
                tokenized_text_without_stop_words_flatten = [item for sublist in tokenized_text_without_stop_words for
                                                             item in sublist]

                all_docs_tokens.append(tokenized_text_without_stop_words_flatten)

                lemmatized_text = [normalForm(line) for line in tokenized_text_without_stop_words]
                lemmatized_text_flatten = [item for sublist in lemmatized_text for item in sublist]

                all_docs_lemmas.append(lemmatized_text_flatten)

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.html'):
                text = open(folder + '/' + file, 'r')
                text = [line.lower() for line in text]
                text = [re.sub(r'[^\w\s]', ' ', line, flags=re.UNICODE) for line in text]
                tokenized_text = [nltk.word_tokenize(line) for line in text]
                tokenized_text_without_stop_words = [lineWithoutStopWords(line) for line in tokenized_text]
                tokenized_text_without_stop_words_flatten = [item for sublist in tokenized_text_without_stop_words for
                                                             item in sublist]
                doc_tokens = list(set(tokenized_text_without_stop_words_flatten))
                doc_tokens.sort()

                for token in doc_tokens:
                    tokens_tf.append(tokenized_text_without_stop_words_flatten.count(token) / len(
                        tokenized_text_without_stop_words_flatten))
                    n_qi = 0
                    for doc in all_docs_tokens:
                        if doc.__contains__(token):
                            n_qi += 1
                    if (n_qi != 0):
                        tokens_idf.append(math.log(len(all_docs_tokens)/n_qi))
                    else:
                        tokens_idf.append(0.0)

                tokens_file = open('tokens_tf-idf/'+'tf-idf_' + file.split('_')[1].replace('html', 'txt'), "w+")
                for i in range(len(doc_tokens)):
                    tokens_file.write(doc_tokens[i] + ' ' + str(tokens_tf[i]) + ' ' + str(tokens_idf[i]) + '\n')
                tokens_file.close()

                lemmatized_text = [normalForm(line) for line in tokenized_text_without_stop_words]
                lemmatized_text_flatten = [item for sublist in lemmatized_text for item in sublist]

                doc_lemmas = list(set(lemmatized_text_flatten))
                doc_lemmas.sort()

                for lemma in doc_lemmas:
                    lemmas_tf.append(lemmatized_text_flatten.count(lemma) / len(lemmatized_text_flatten))
                    n_qi = 0
                    for doc in all_docs_lemmas:
                        if doc.__contains__(lemma):
                            n_qi += 1
                    if (n_qi != 0):
                        lemmas_idf.append(math.log(len(all_docs_lemmas)/n_qi))
                    else:
                        lemmas_idf.append(0.0)

                lemmas_file = open('lemmas_tf-idf/'+'tf-idf_' + file.split('_')[1].replace('html', 'txt'), "w+")
                # print(len(doc_lemmas))
                for i in range(len(doc_lemmas)):
                    # print(i)
                    lemmas_file.write(doc_lemmas[i] + ' ' + str(lemmas_tf[i]) + ' ' + str(lemmas_idf[i]) + '\n')
                # print()
                lemmas_file.close()


                lemmas_tf = []
                tokens_tf = []

                lemmas_idf = []
                tokens_idf = []

                progress_counter += 1
                print(str(progress_counter) + ' /115 or ' + str((progress_counter/155*100)) + '%')
