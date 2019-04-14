"""
Implementation of Matsuo paper
"""

import collections
import math
import string

import numpy as np
import pandas as pd
from nltk import SnowballStemmer
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

STEMMER = SnowballStemmer('english')

STOP_WORDS = set(stopwords.words('english'))

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


def step12(text):
    """
    :param text:
    :return: length, orig_text, cand_keywords
    """
    rem = list(string.punctuation)
    rem.remove('.')
    rem.remove('?')
    rem.remove('!')

    orig_text = text
    orig_text = orig_text.translate(None, ''.join(rem))
    stemmed_text = []
    for word in word_tokenize(orig_text):
        if word not in STOP_WORDS:
            stemmed_text.append(STEMMER.stem(word).lower())

    orig_text = ' '.join(stemmed_text)

    text = text.translate(None, ''.join(rem))
    length = len(word_tokenize(orig_text))

    stemmed_text = []
    for word in word_tokenize(text):
        if word not in STOP_WORDS and len(word) > 3 and not word.isdigit():
            stemmed_text.append(STEMMER.stem(word).lower())

    dictionary = dict((x, stemmed_text.count(x)) for x in set(stemmed_text))

    cand_keywords = collections.defaultdict(lambda: 0)
    for key, val in dictionary.iteritems():
        if val > 2:
            cand_keywords[key] = val

    total_cand = len(cand_keywords)
    if total_cand < 2:
        cnt = 0
        for key, val in dictionary.iteritems():
            cand_keywords[key] = val
            cnt += 1
            if cnt > 2:
                break
        cand_keywords = sorted(cand_keywords.items(), key=lambda x: x[1], reverse=True)
    else:
        cand_keywords = sorted(cand_keywords.items(),
                               key=lambda x: x[1], reverse=True)[:int(round(0.3 * total_cand))]

    return length, orig_text, cand_keywords


def h(x):
    """
    :param x: integer
    :return: -x*math.log10(x)
    """
    if x == 0:
        return 0
    return -x*math.log10(x)


def step3(length, orig_text, cand_keywords):

    key = []
    for keys, _ in cand_keywords:
        key.append(keys)

    tokens = orig_text.split('.')  # sent_tokenize(orig_text)
    sentences = []
    word = []
    for token in tokens:
        if len(token) > 10:
            sentences.append(token)
            for w in word_tokenize(token):
                if len(w) > 3 and w not in STOP_WORDS and not w.isdigit():
                    word.append(w)

    q_word = list(set(word))

    matrix = np.zeros((len(q_word), len(cand_keywords)))

    for ind_cand in range(len(cand_keywords)):
        for ind_term in range(len(q_word)):
            count = 0
            for sent in sentences:
                if cand_keywords[ind_cand][0] in sent and q_word[ind_term] in sent:
                    count += 1
            matrix[ind_term, ind_cand] = count
            # print "matrix["+str(ind_term) + "," +str(ind_cand)+"]="+str(count)

    row_name = q_word[:]
    col_name = cand_keywords[:]
    col_name_copy = col_name[:]
    matrix1 = matrix[:, :]
    cluster = []
    clust = []
    flag = 1
    variable = False

    while cluster or flag == 1:
        if col_name == col_name_copy and flag == 0:
            break
        col_name = col_name_copy[:]
        matrix = matrix1

        # print "Len" + str(len(col_name))
        for i in range(len(col_name)-1):
            for j in range(i+1, (len(col_name))):
                # print "i="+str(i)+" j="+str(j)
                if col_name[i][0] in clust:
                    # print "BRRR"
                    # print clust
                    # print col_name[i][0]
                    variable = False
                    break
                elif col_name[j][0] in clust:
                    # print "CCCC"
                    # print clust
                    # print col_name[i][0]
                    continue

                if col_name[i][1] == 0 or col_name[j][1] == 0:
                    break
                sum1 = math.log10(2)

                for word in row_name:
                    # if w is not
                    val1 = matrix[row_name.index(word)][i]
                    # print "SSS" + str(matrix.shape)
                    val2 = matrix[row_name.index(word)][j]
                    # print a, b
                    # print cand_keywords[i][0]
                    # print w
                    # print a
                    # print b
                    # print h((a*1.0/cand_keywords[i][1])+(b*1.0/cand_keywords[j][1]))
                    sum1 += (h((val1*1.0/col_name[i][1])+(val2*1.0/col_name[j][1])) -
                             h(val1*1.0/col_name[i][1]) - h(val2*1.0/col_name[j][1]))*0.5

                count = 0

                col1 = col_name[i][0].split('_')
                col2 = col_name[j][0].split('_')
                col1.extend(col2)
                for sent in sentences:
                    cnt1 = 0
                    for word in col1:
                        if word in sent:
                            cnt1 += 1

                    if len(col1) == cnt1:
                        count += 1

                print col_name_copy
                print count
                num = length * count
                print num
                print col_name[i][1]*col_name[j][1]
                if num == 0:
                    val = 0
                else:
                    val = math.log10(num*1.0/(col_name[i][1]*col_name[j][1]))
                print "m=" + str(val)
                print "sum1=" + str(sum1)
                if sum1 > 0.95*math.log10(2) or val > math.log10(2):
                    print "TIN"

                    cluster.append(str(col_name[i][0])+"_"+str(col_name[j][0]))
                    clust.append(col_name[i][0])
                    clust.append(col_name[j][0])

                    matrix1 = np.delete(matrix, (i, j), 1)
                    matrix1 = np.append(matrix1, [[0]] * matrix.shape[0], 1)

                    print col_name_copy
                    print "SISST" + str(col_name_copy)
                    print col_name[i], col_name[j]
                    # col_name_copy.remove(col_name[i])
                    print "j="+str(j)+"i="+str(i)
                    print col_name_copy
                    print col_name
                    print col_name[j]
                    # col_name_copy.remove(col_name[j])
                    # print str(col_name[i][0]) + "_" + str(col_name[j][0])
                    # print len(s)

                    for ind_term in range(len(row_name)):
                        count = 0
                        col1 = col_name[i][0].split('_')
                        col2 = col_name[j][0].split('_')
                        col1.extend(col2)
                        for sent in sentences:
                            cnt1 = 0
                            for word in col1:
                                if word in sent:
                                    cnt1 += 1
                            if len(col1) == cnt1 and row_name[ind_term] in sent:
                                count += 1

                        matrix1[ind_term, (matrix1.shape[1]-1)] = count

                    count = 0
                    col1 = col_name[i][0].split('_')
                    col2 = col_name[j][0].split('_')
                    col1.extend(col2)
                    for sent in sentences:
                        cnt1 = 0
                        for word in col1:
                            if word in sent:
                                cnt1 += 1

                        if len(col1) == cnt1:
                            count += 1

                    if count != 0:
                        col_name_copy.remove(col_name[i])
                        col_name_copy.remove(col_name[j])
                        col_name_copy.append(
                            (str(col_name[i][0]) + "_" + str(col_name[j][0]), count)
                        )

                    if count == 0:
                        matrix1 = matrix[:]

                    # print col_name_copy
                    variable = True

                if variable:
                    variable = False
                    break

        flag = 0
        # print matrix

    cluster = col_name_copy[:]
    pc = collections.defaultdict(lambda: 0)
    lis = list(matrix1.sum(axis=0))
    print lis
    for clust in range(len(cluster)):
        pc[cluster[clust]] = lis[clust] * 1.0 / length

    print "Finale" + str(row_name)
    print pc
    return pc, cluster, matrix1, row_name


def step4(orig_text, pc, cluster, matrix1, row_name):
    print "Cluster"+str(cluster)
    words = collections.defaultdict(lambda: 0)
    for word in row_name:
        for sent in sent_tokenize(orig_text):
            if word in sent:
                words[word] += len(word_tokenize(sent))

    print words
    x = collections.defaultdict(lambda: 0)
    nam = []
    freq = []
    x2 = []
    for i in range(len(row_name)):
        lis = []
        for c in range(len(cluster)):
            num = (matrix1[i, c] - pc[cluster[c]]*words[row_name[i]])**2
            print "num"+str(num)
            den = pc[cluster[c]]*words[row_name[i]]
            print "den"+str(den)
            val = num*1.0/den
            print "val"+str(val)
            lis.append(val)
        print sum(lis)
        print max(lis)
        nam.append(row_name[i])
        freq.append(words[row_name[i]])
        x2.append(sum(lis) - max(lis))
        x[row_name[i]] = sum(lis) - max(lis)

    data = pd.DataFrame({'X_Sq': x2, 'Name': nam, 'Freq': freq})
    data = data.sort_values('X_Sq', ascending=False)
    data.to_csv("Compare2.csv", index=False)
    return sorted(x.items(), key=lambda x1: x1[1], reverse=True)


def main():
    """
    :return: Keywords_baseline.csv
    """
    print "#"*20 + "MAIN" + "#"*20

    text = open("a.txt", 'r')
    text = (text.read()).decode('utf-8').encode('ascii', 'ignore').strip()

    for _hex, _char in LATIN_1_CHARS:
        text = text.replace(_hex, _char)

    length, orig_text, cand_keywords = step12(text)
    pc, cluster, matrix1, row_name = step3(length, orig_text, cand_keywords)
    item = step4(orig_text, pc, cluster, matrix1, row_name)
    key = []
    val = []
    for k, v in item:
        key.append(k)
        val.append(v)

    data = pd.DataFrame(data={'Key': key, 'Val': val})
    data.to_csv("Keywords_baseline.csv", index=False)
    print item


if __name__ == '__main__':
    main()
