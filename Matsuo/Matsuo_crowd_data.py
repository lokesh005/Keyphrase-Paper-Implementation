import collections
import math
import os
import random
import re
import string
import time
from itertools import islice

import numpy as np
import pandas as pd
from nltk import SnowballStemmer
from nltk import word_tokenize, sent_tokenize
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


def take(n, iterable):
    return list(islice(iterable, n))


def step12(text):
    rem = list(string.punctuation)
    rem.remove('.')
    rem.remove('?')
    rem.remove('!')

    orig_text = text
    orig_text = orig_text.translate(None, ''.join(rem))
    stemmed_text = []
    for word in word_tokenize(orig_text):
        if word not in stop_words:
            stemmed_text.append(stemmer.stem(word).lower())

    orig_text = ' '.join(stemmed_text)

    text = text.translate(None, ''.join(rem))
    n = len(word_tokenize(orig_text))
    print "Length of text = " + str(n)
    stemmed_text = []
    for word in word_tokenize(text):
        if word not in stop_words and len(word) > 3 and not word.isdigit():
            stemmed_text.append(stemmer.stem(word).lower())

    d = dict((x, stemmed_text.count(x)) for x in set(stemmed_text))

    cand_keywords = collections.defaultdict(lambda: 0)
    for key, val in d.iteritems():
        if val > 2:
            cand_keywords[key] = val

    total_cand = len(cand_keywords)
    if total_cand < 2:
        c = 0
        for key, val in d.iteritems():
            cand_keywords[key] = val
            c += 1
            if c > 2:
                break
        cand_keywords = sorted(cand_keywords.items(), key=lambda x: x[1], reverse=True)
    else:
        cand_keywords = sorted(cand_keywords.items(), key=lambda x: x[1], reverse=True)[:int(round(0.3 * total_cand))]
    print "Total Candidates = " + str(total_cand)
    return n, orig_text, cand_keywords


def h(x):
    if x == 0:
        return 0
    return -x*math.log10(x)


def step3(n, orig_text, cand_keywords):
    print "Candidates = "+str(cand_keywords)
    key = []
    for k, v in cand_keywords:
        key.append(k)

    s1 = orig_text.split('.')  # sent_tokenize(orig_text)
    sentence = []
    word = []
    for s in s1:
        if len(s) > 10:
            sentence.append(s)
            for w in word_tokenize(s):
                if len(w) > 3 and w not in stop_words and not w.isdigit():
                    word.append(w)

    q_word = list(set(word))
    matrix = np.zeros((len(q_word), len(cand_keywords)))

    for ind_cand in range(len(cand_keywords)):
        for ind_term in range(len(q_word)):
            count = 0
            for s in sentence:
                if cand_keywords[ind_cand][0] in s and q_word[ind_term] in s:
                    count += 1
            matrix[ind_term, ind_cand] = count

    row_name = q_word[:]
    col_name = cand_keywords[:]
    col_name_copy = col_name[:]
    matrix1 = matrix[:, :]

    cluster = []
    clust = []
    flag = 1
    variable = False

    while len(cluster) != 0 or flag == 1:
        if col_name == col_name_copy and flag == 0:
            break
        col_name = col_name_copy[:]
        matrix = matrix1

        for i in range(len(col_name)-1):
            for j in range(i+1, (len(col_name))):
                if col_name[i][0] in clust:
                    variable = False
                    break
                elif col_name[j][0] in clust:
                    continue

                if col_name[i][1] == 0 or col_name[j][1] == 0:
                    break
                sum1 = math.log10(2)

                for w in row_name:
                    a = matrix[row_name.index(w)][i]
                    b = matrix[row_name.index(w)][j]
                    sum1 += (h((a*1.0/col_name[i][1])+(b*1.0/col_name[j][1])) -
                             h(a*1.0/col_name[i][1]) - h(b*1.0/col_name[j][1]))*0.5

                count = 0

                x = col_name[i][0].split('_')
                y = col_name[j][0].split('_')
                x.extend(y)
                for s in sentence:
                    c1 = 0
                    for xx in x:
                        if xx in s:
                            c1 += 1

                    if len(x) == c1:
                        count += 1

                num = n*count
                if num == 0:
                    m = 0
                else:
                    m = math.log10(num*1.0/(col_name[i][1]*col_name[j][1]))

                if sum1 > 0.95*math.log10(2) or m > math.log10(2):
                    cluster.append(str(col_name[i][0])+"_"+str(col_name[j][0]))
                    clust.append(col_name[i][0])
                    clust.append(col_name[j][0])

                    matrix1 = np.delete(matrix, (i, j), 1)
                    matrix1 = np.append(matrix1, [[0]] * matrix.shape[0], 1)

                    for ind_term in range(len(row_name)):
                        count = 0
                        x = col_name[i][0].split('_')
                        y = col_name[j][0].split('_')
                        x.extend(y)
                        for s in sentence:
                            c1 = 0
                            for xx in x:
                                if xx in s:
                                    c1 += 1
                            if len(x) == c1 and row_name[ind_term] in s:
                                count += 1

                        matrix1[ind_term, (matrix1.shape[1]-1)] = count

                    count = 0
                    x = col_name[i][0].split('_')
                    y = col_name[j][0].split('_')
                    x.extend(y)
                    for s in sentence:
                        c1 = 0
                        for xx in x:
                            if xx in s:
                                c1 += 1

                        if len(x) == c1:
                            count += 1

                    if count != 0:
                        col_name_copy.remove(col_name[i])
                        col_name_copy.remove(col_name[j])
                        col_name_copy.append((str(col_name[i][0]) + "_" + str(col_name[j][0]), count))

                    if count == 0:
                        matrix1 = matrix[:]

                    variable = True
                if variable:
                    variable = False
                    break

        flag = 0
    cluster = col_name_copy[:]
    pc = collections.defaultdict(lambda: 0)
    lis = list(matrix1.sum(axis=0))

    for c in range(len(cluster)):
        pc[cluster[c]] = lis[c]*1.0/n

    return pc, cluster, matrix1, row_name


def step4(orig_text, pc, cluster, matrix1, row_name):
    print "Clusters = " + str(cluster)
    words = collections.defaultdict(lambda: 0)
    for w in row_name:
        for s in sent_tokenize(orig_text):
            if w in s:
                words[w] += len(word_tokenize(s))

    x = collections.defaultdict(lambda: 0)
    nam = []
    freq = []
    x2 = []
    for i in range(len(row_name)):
        lis = []
        for c in range(len(cluster)):
            num = (matrix1[i, c] - pc[cluster[c]]*words[row_name[i]])**2
            den = pc[cluster[c]]*words[row_name[i]]
            val = num*1.0/den
            lis.append(val)

        nam.append(row_name[i])
        freq.append(words[row_name[i]])
        x2.append(sum(lis) - max(lis))
        x[row_name[i]] = sum(lis) - max(lis)

    dd = pd.DataFrame({'X_Sq': x2, 'Name': nam, 'Freq': freq})
    dd = dd.sort_values('X_Sq', ascending=False)
    dd.to_csv("Compare2.csv", index=False)
    return sorted(x.items(), key=lambda x1: x1[1], reverse=True)


def main():
    begin = time.time()
    print "#"*20 + "MAIN" + "#"*20
    os.chdir('..')
    os.chdir('./crowd_data')
    files = os.listdir('.')

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

    precision = []
    recall = []
    f1_score = []
    for i in range(len(txt)):
        print "File = " + str(txt[i])
        text = open(txt[i], 'r')
        text = (text.read()).decode('utf-8').encode('ascii', 'ignore').strip()

        for _hex, _char in LATIN_1_CHARS:
            text = text.replace(_hex, _char)

        n, orig_text, cand_keywords = step12(text)
        pc, cluster, matrix1, row_name = step3(n, orig_text, cand_keywords)
        item = step4(orig_text, pc, cluster, matrix1, row_name)
        key = []
        val = []
        for k, v in item:
            key.append(k)
            val.append(v)

        phrases = open(key_file[i], mode='r')
        phr = phrases.read()
        y_true = phr.split('\n')

        if len(y_true) > len(key):
            y_true = y_true[0:len(key)]
        else:
            key = key[0:len(y_true)]

        tp = 0
        fp = 0
        fn = 0

        for k in key:
            if k in y_true:
                tp += 1
            elif k not in y_true:
                fp += 1

        for k in y_true:
            if k not in key:
                fn += 1

        p = tp * 1.0 / (tp + fp)
        print("p= " + str(p))

        r = tp * 1.0 / (tp + fn)
        print("r= " + str(r))

        f1 = 2 * p * r * 1.0 / (p + r + 0.1)

        precision.append(p)
        recall.append(r)
        f1_score.append(f1)
        print "\n\n"

    print("######################################################3")
    print "LEn" + str(len(txt))
    print("Precision = " + str(np.mean(precision)))
    print("Recall = " + str(np.mean(recall)))
    print("F1-score = " + str(np.mean(f1_score)))

    print "Time " + str(time.time() - begin)


if __name__ == '__main__':
    main()
