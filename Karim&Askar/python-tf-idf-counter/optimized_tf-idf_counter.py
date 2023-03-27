import math
import os
import pymorphy2

from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

FOLDER = 'data'
FILES_AMOUNT = 115


# nltk.download('stopwords')

def get_text():
    all_texts = []
    for i in range(FILES_AMOUNT):
        file = f'site_{i + 1}.html'
        soup = BeautifulSoup(open(os.path.join('data', file), 'r'), 'html.parser')
        all_texts.append(soup.get_text().replace('\n', ' ').replace('\t', ' ').replace('/', ' ').replace('\'', ' '))
    print('- text received')
    return all_texts


def tokenize(texts):
    tokens = []
    nltk_stopwords = stopwords.words('english')
    punctuation = ['!', '(', ')', '-', '[', ']', '{', '}',
                   ';', '?', '@', '#', '$', '%', ':', '\'', '"', '\\',
                   ',', '.', '/', '^', '&', ';', '*', '_', '``', '\'\'', '--', '—', '|', '–', '·', '..']
    for i in range(FILES_AMOUNT):
        texts[i] = word_tokenize(texts[i])
        text_without_stopwords = []
        for word in texts[i]:
            if word not in nltk_stopwords and word not in punctuation:
                text_without_stopwords.append(word)
        tokens.append(text_without_stopwords)
    print('- tokens allocated')
    return tokens


def lemmatize(tokens):
    lemmas = []
    morph = pymorphy2.MorphAnalyzer()
    for i in range(FILES_AMOUNT):
        lemmas.append([])
        for token in tokens[i]:
            lemmas[i].append(morph.parse(token)[0].normal_form)
    print('- lemmatization done')
    return lemmas


def leave_unique_elements_and_sort(list_of_list_of_elements):
    list_with_lists_of_unique_elements = []
    for i in range(FILES_AMOUNT):
        to_sort = list(set(list_of_list_of_elements[i]))
        to_sort.sort()
        list_with_lists_of_unique_elements.append(to_sort)
    return list_with_lists_of_unique_elements


def count_tfs(list_of_list_of_elements, unique_list_of_list_of_elements):
    tfs = []
    for i in range(len(unique_list_of_list_of_elements)):
        tfs.append([])
        for j in range(len(unique_list_of_list_of_elements[i])):
            tfs[i].append(list_of_list_of_elements[i].count(unique_list_of_list_of_elements[i][j]) / len(
                list_of_list_of_elements[i]))
    return tfs


def count_idfs(unique_list_of_list_of_elements):
    idfs = []
    for i in range(len(unique_list_of_list_of_elements)):
        idfs.append([])
        for j in range(len(unique_list_of_list_of_elements[i])):
            n_qi = 0
            for k in range(len(unique_list_of_list_of_elements)):
                if unique_list_of_list_of_elements[i][j] in unique_list_of_list_of_elements[k]:
                    n_qi += 1
            idfs[i].append(math.log(FILES_AMOUNT / n_qi))
    return idfs


def write_files(type_of_word, unique_list_of_list_of_elements, tfs, idfs):
    for i in range(FILES_AMOUNT):
        file = open(f'new_tf-idf_files/{type_of_word}s_tf-idf/tf-idf_{i+1}.txt', 'w+')
        for j in range(len(unique_list_of_list_of_elements[i])):
            file.write(f'{unique_list_of_list_of_elements[i][j]} {tfs[i][j]} {idfs[i][j]}\n')
        file.close()
    print(f'files ready: {type_of_word}')


if __name__ == '__main__':
    texts = get_text()
    # print(texts)
    tokens = tokenize(texts)
    # print(tokens)
    lemmas = lemmatize(tokens)
    # print(lemmas)
    sorted_unique_tokens = leave_unique_elements_and_sort(tokens)
    # print(sorted_unique_tokens)
    sorted_unique_lemmas = leave_unique_elements_and_sort(lemmas)
    # print(sorted_unique_lemmas)
    tokens_tfs = count_tfs(tokens, sorted_unique_tokens)
    # print(tokens_tfs)
    lemmas_tfs = count_tfs(lemmas, sorted_unique_lemmas)
    # print(lemmas_tfs)
    tokens_idfs = count_idfs(sorted_unique_tokens)
    # print(tokens_idfs)
    lemmas_idfs = count_idfs(sorted_unique_lemmas)
    # print(lemmas_idfs)
    write_files('token', sorted_unique_tokens, tokens_tfs, tokens_idfs)
    write_files('lemma', sorted_unique_lemmas, lemmas_tfs, lemmas_idfs)
