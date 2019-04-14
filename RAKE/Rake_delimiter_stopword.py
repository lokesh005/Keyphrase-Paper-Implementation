"""
Implementation of paper = EmbedRank: Unsupervised Key-phrase Extraction using Sentence Embeddings
Link: https://arxiv.org/pdf/1801.04470.pdf
"""
import os
import random
import string
import time

import numpy as np
from nltk import SnowballStemmer
from nltk import ngrams
from nltk import word_tokenize
from nltk.chunk.regexp import *
from nltk.corpus import stopwords

stemmer = SnowballStemmer('english')

stop_words = set(stopwords.words('english'))

LATIN_1_CHARS = (  # To convert unicodes into understandable punctuations
    ('\xe2\x80\x99', "'"),
    ('\xef\xac\x81', ''),
    ('- ', ''),
    (' - ', ''),
    ('\xc3\xa9', 'e'),
    ('\xe2\x80\x90', '-'),
    ('\xe2\x80\x91', '-'),
    ('\xe2\x80\x92', '-'),
    ('\xe2\x80\x93', '-'),
    ('\xe2\x80\x94', '-'),
    ('\xe2\x80\x94', '-'),
    ('\xe2\x80\x98', "'"),
    ('\xe2\x80\x9b', "'"),
    ('\xe2\x80\x9c', '"'),
    ('\xe2\x80\x9c', '"'),
    ('\xe2\x80\x9d', '"'),
    ('\xe2\x80\x9e', '"'),
    ('\xe2\x80\x9f', '"'),
    ('\xe2\x80\xa6', '...'),
    ('\xe2\x80\xb2', "'"),
    ('\xe2\x80\xb3', "'"),
    ('\xe2\x80\xb4', "'"),
    ('\xe2\x80\xb5', "'"),
    ('\xe2\x80\xb6', "'"),
    ('\xe2\x80\xb7', "'"),
    ('\xe2\x81\xba', "+"),
    ('\xe2\x81\xbb', "-"),
    ('\xe2\x81\xbc', "="),
    ('\xe2\x81\xbd', "("),
    ('\xe2\x81\xbe', ")"),
    ('\xef\x82\xb7', ''),
    ('\xef\x82\xa7', '')
)

grammar = r"""
  NP:
    {<JJ.*>*<NN||NNS>+}          # Chunk everything
  """


def get_keywords(txt):
    """
    :param txt: Paragraph
    :return: Candidate keyphrases
    """
    # print "step1"
    txt = txt.decode('utf-8').encode('ascii', 'ignore').strip()

    for _hex, _char in LATIN_1_CHARS:
        txt = txt.replace(_hex, _char)

    txt = " ".join(txt.split())  # For removing whitespaces
    txt = txt.lower()

    phr = []
    c = []
    for word in word_tokenize(txt):
        if word in stop_words or word in list(string.punctuation):
            phr.append(' '.join(c))
            c = []
        else:
            c.append(word)

    keyword = []
    for p in phr:
        if p == '':
            continue
        if len(word_tokenize(p)) == 1:
            keyword.append(p)
        else:
            for i in range(1, len(word_tokenize(p))):
                grams = ngrams(p.decode('utf_8').lower().split(), i)
                for gram in grams:
                    keyword.append(' '.join(gram))

    return list(set(keyword))


def main():

    begin = time.time()

    os.chdir("./data")

    files = os.listdir('.')

    precision = []
    recall = []
    f1_score = []

    for le in range(1, 10):
        init_txt = [f for f in files if f.endswith('.txt')]
        init_txt = random.sample(init_txt, 30)

        text_name = []

        for i in init_txt:
            text_name.append(re.sub('.txt', '', i))  # Removing '.txt'

        init_key = [f for f in files if f.endswith('.key')]
        key_name = []
        for i in init_key:
            key_name.append(re.sub('.key', '', i))  # Removing '.key'

        pos_inp_in_out = []
        txt = []
        for i in text_name:
            if i in key_name:
                pos_inp_in_out.append(key_name.index(i))
                txt.append(i + ".txt")

        key_file = []
        for i in pos_inp_in_out:
            key_file.append(init_key[i])

        pre = []
        rec = []
        f1_ = []

        for i in range(len(txt)):
                # print("i= " + str(i))
                # print("Txt="+str(txt[i]))

            text = open(txt[i], 'r')
            text = (text.read()).decode('utf-8').encode('ascii', 'ignore').strip()

            for _hex, _char in LATIN_1_CHARS:
                text = text.replace(_hex, _char)

            cand_keyword = get_keywords(text)

            y_pred = []
            for item in cand_keyword:
                y_pred.append(str(item))

            # if len(y_pred) > n:
            #    y_pred = y_pred[0:n]

            # print("Key=" + key_file[i])
            phrases = open(key_file[i], mode='r')
            phr = phrases.read()
            y_true = phr.split('\n')

            # if len(y_true) > n:
            #    y_true = y_true[0:n]

            tp = 0
            fp = 0
            fn = 0

            for k in y_pred:
                if k in y_true:
                    tp += 1
                elif k not in y_true:
                    fp += 1

            for k in y_true:
                if k not in y_pred:
                    fn += 1

            p = tp * 1.0 / (tp + fp)
            print("p=" + str(p))

            r = tp * 1.0 / (tp + fn)
            print("r=" + str(r))

            f1 = 2 * p * r * 1.0 / (p + r + 0.1)

            pre.append(p)
            rec.append(r)
            f1_.append(f1)

        precision.append(np.mean(pre))
        recall.append(np.mean(rec))
        f1_score.append(np.mean(f1_))

    print
    print("######################################################3")
    # print("n=" + str(n))
    print("Precision = " + str(np.mean(precision)))
    print("Recall = " + str(np.mean(recall)))
    print("F1-score = " + str(np.mean(f1_score)))

    print("--- %s seconds ---" % (time.time() - begin))


if __name__ == '__main__':
    main()
