#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 20:28:12 2020

@author: mao15
"""
import os
import html2text
import re
import gensim
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import pickle
import unicodedata

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def extract(html, filename):
    text = html2text.html2text(html)
    text = text.lower()
    x1 = re.search("(?m)^.{0,7}item.+[\n]{0,3}.*[\n]{0,3}business", text)
    x2 = re.finditer(
        r"(?m)^.{0,7}(item|i tem).+[\n]{0,3}.*[\n]{0,3}(unresolved staff comments|u nresolved staff comments)", text)
    end_pos = 0
    for m in x2:
        if m.end() > end_pos:
            end_pos = m.end()
    # print(x1)
    # print(end_pos)
    if x1 and x2:
        # print(x1.span()[0],end_pos)
        if end_pos - x1.span()[1] > 2000:
            return(text[x1.span()[0]:end_pos])
        else:
            print(text[x1.span()[0]:end_pos])
            print("%s content too short!\n" % filename)
    else:
        print("%s content can not located!\n" % filename)


# 该函数用于将business和riskfactor session从10k里抽取出来
def get_raw_dict(path):
    counter = 0
    os.chdir(path)  # 到达相应工作路径

    raw_dict = dict()  # 创建一个字典型的数据结构，存放数据，每一个key代表财报文件名，key对应的value是文件内容

    for filename in os.listdir(os.getcwd()):
        if filename[-4:] == 'html':
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    htmlhandle = f.read()
            except Exception as e:
                print(e)
                continue

            if filename[-6] == 'k':
                raw_dict[filename[0:7]] = extract(
                    html=htmlhandle, filename=filename)
                counter += 1
                print(filename, " finished! ----- ", counter)
    return raw_dict


def normtxt(txt):
    return unicodedata.normalize("NFKD", txt)


def clean(doc):
    doc = normtxt(txt=doc)
    doc = re.sub('[^a-zA-Z\s]', '', doc)
    stop_free = ' '.join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join([ch for ch in stop_free if ch not in exclude])
    normalized = ' '.join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


def html2bow(path):
    dict_a = get_raw_dict(path)
    # p_dict = preprocess_raw(dict_a)
    # ensure the doc would not be too short, longer than 1000
    for x in list(dict_a):
        try:
            if len(dict_a[x]) < 1000:
                # print(x)
                dict_a.pop(x)
        except Exception as e:
            dict_a.pop(x)
            print(e)

    doc_complete = []
    lable = []
    for d in dict_a:
        doc_complete.append(dict_a[d])
        lable.append(int(d[:2]))
    doc_clean = [clean(doc).split() for doc in doc_complete]
    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    # save lable and dictionary using pickle
    # wb 写入 二进制
    with open('../dictionary.pkl', 'wb') as f:
        pickle.dump(dictionary, f)
        pickle.dump(lable, f)
        pickle.dump(doc_term_matrix, f)
    return doc_term_matrix


def html2tfidf(path):
    bow_corpus = html2bow(path)
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    return corpus_tfidf


if __name__ == '__main__':
    # print('hello')
    # get_raw_dict('C:\\Users\\think\\Dropbox\\FordhamClass\\textual analysis\\test_docs')
    result, industry = html2bow(
        '/home/clarence/Dropbox/FordhamClass/textual analysis/test_docs')
