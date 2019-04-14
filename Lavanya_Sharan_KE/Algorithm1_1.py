"""

Implementation of paper = EmbedRank: Unsupervised Keyphrase Extraction using Sentence Embeddings
Link: https://arxiv.org/pdf/1801.04470.pdf

"""

import collections
import itertools
import string
import time

import numpy as np
import pandas as pd
import simple_gensim
from nltk import word_tokenize, pos_tag
from nltk.chunk import *
from nltk.chunk.regexp import *
from nltk.corpus import stopwords
from scipy import spatial

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


# Step1
def step1(txt):
    print("step1")
    cp = RegexpParser(grammar)
    tokens = word_tokenize(txt)
    chunked = ne_chunk(pos_tag(tokens))
    sentence = chunked.leaves()
    keyword = []
    for tre in cp.parse(sentence):
        if type(tre) == Tree:
            keyword.append(" ".join([token for token, pos in tre.leaves()]))

    cand_keyword = []
    for k in keyword:
        if len(k) > 3 and k.replace('-', '') not in cand_keyword:
            cand_keyword.append(k.replace('-', ''))

    return cand_keyword


# Step2
def step2(txt, cand_keyword, model):
    print("step2")
    start_alpha = 0.01
    infer_epoch = 1000
    tokens = word_tokenize(txt)

    doc_vec = model.infer_vector(tokens, alpha=start_alpha, steps=infer_epoch)

    key_vec = collections.defaultdict(lambda: 0)
    cos_val = collections.defaultdict(lambda: 0)

    for adn in cand_keyword:
        phr = word_tokenize(adn)
        key_vec[adn] = model.infer_vector(phr, alpha=start_alpha, steps=infer_epoch)
        cos_val[adn] = 1 - spatial.distance.cosine(key_vec[adn], doc_vec)

    return key_vec, doc_vec, cos_val


def mmr(cand_key, key_vec, cos_val, lambd):
    print("mmr")
    max_cos_key_doc = cos_val[max(cos_val, key=cos_val.get)]
    ncos = collections.defaultdict(lambda: 0)

    for adn in cand_key:
        ncos[adn] = cos_val[adn] * 1.0 / max_cos_key_doc

    # For sim between key and doc
    norm_cos = collections.defaultdict(lambda: 0)

    for adn in cand_key:
        norm_cos[adn] = 0.5 + (ncos[adn] - np.mean(ncos.values())) * 1.0 / np.std(ncos.values())

    pairs = itertools.combinations(cand_key, 2)
    lis = []
    for i in pairs:
        # if (i[0], i[1]) not in lis and (i[1], i[0]) not in lis:
        lis.append((i[0], i[1]))
        lis.append((i[1], i[0]))

    key_cos_val = collections.defaultdict(lambda: 0)

    for i in lis:
        key_cos_val[str(i[0]) + '_' + str(i[1])] = 1 - spatial.distance.cosine(key_vec[i[0]], key_vec[i[1]])

    nkey_cos_val = collections.defaultdict(lambda: 0)

    arr = np.array(lis)
    mean_key_val = collections.defaultdict(lambda: 0)
    std_key_val = collections.defaultdict(lambda: 0)

    for i in lis:
        ind = np.where(arr == i[0])
        kk = []
        for u in ind[0]:
            kk.append(lis[u])

        for_max = []
        for j in kk:
            for_max.append(key_cos_val[str(j[0]) + "_" + str(j[1])])

        m = max(for_max)
        mean_key_val[str(i[0])] = np.mean(for_max)

        std_key_val[str(i[0])] = np.std(for_max)
        if std_key_val[str(i[0])] == 0:
            dev = np.std(for_max)
            print(i[0] + "::" + str(for_max))
            print(i[0] + "::" + str(dev))
            std_key_val[str(i[0])] = dev
        nkey_cos_val[str(i[0]) + '_' + str(i[1])] = key_cos_val[str(i[0]) + '_' + str(i[1])] * 1.0 / m

    norm_keys_cos = collections.defaultdict(lambda: 0)

    for i in lis:
        norm_keys_cos[str(i[0]) + "_" + str(i[1])] = 0.5 + (nkey_cos_val[str(i[0]) + "_" + str(i[1])] -
                                                            mean_key_val[str(i[0])]) * 1.0 / std_key_val[str(i[0])]

    mmr_vals = collections.defaultdict(lambda: 0)
    kk = []
    for adn in cand_key:
        ind = np.where(arr == adn)
        for u in ind[0]:
            kk.append(lis[u])

        for_max = []
        for j in kk:
            for_max.append(norm_keys_cos[str(j[0]) + "_" + str(j[1])])

        m = max(for_max)

        mmr_vals[adn] = lambd * norm_cos[adn] - (1 - lambd) * m

    return sorted(mmr_vals.items(), key=lambda x: x[1], reverse=True)


def main():
    start = time.time()
    model = simple_gensim.models.Doc2Vec.load("doc2vec.bin")

    para = [line.rstrip('\n') for line in open('Doc.txt', 'r')]

    text = ""
    c = 0
    count = 0
    df = pd.DataFrame()
    d = collections.defaultdict(lambda: 0)
    para_start = time.time()
    for line in para:
        text = text + " " + line
        keyphrases = []
        if line == '' and c == 0:
            text = text.decode('utf-8').encode('ascii', 'ignore').strip()

            for _hex, _char in LATIN_1_CHARS:
                text = text.replace(_hex, _char)

            text = " ".join(text.split())
            print(text)
            table = string.maketrans("", "")

            cand_keyword = step1((text.lower()).translate(table, string.punctuation))
            key_vec, doc_vec, cos_val = step2((text.lower()).translate(table, string.punctuation),
                                              cand_keyword, model)
            mmr_vals = mmr(cand_keyword, key_vec, cos_val, lambd=0.5)

            for lin in mmr_vals:
                keyphrases.append(lin[0])
                if d[lin[0]] == 0:
                    d[lin[0]] = 1
                else:
                    d[lin[0]] += 1
                # print line[0]
                # f.write(str(lin[0])+"\t"+str(lin[1]))
                # f.write('\n')
            # f.write("Time =" + str(time.time() - para_start))
            # f.write('\n\n')
            df = df.append({'Text': text, 'Keyword': keyphrases}, ignore_index=True)
            count += 1
            # print "Para" + str(count)
            c = 1
            print("Para" + str(count) + ": Time =" + str(time.time() - para_start))
        elif c == 1:
            c = 0
            text = ""
            para_start = time.time()

    df.to_csv("TrainText_with_keys.csv", index=False)

    # f = open('Keyphrases_finale.txt', 'w')
    data = pd.DataFrame(columns=['Key', 'Count'])
    # for key, val in d.iteritems():
    # data = data.append({'Key': d.keys(), 'Count': d.values()}, ignore_index=True)
    data['Key'] = d.keys()
    data['Count'] = d.values()
    data.to_csv("Keyphrase_Counter.csv", index=False)

    # for phra in keyphrases:
    #    f.write(phra)
    #    f.write('\n')
    # f.close()

    print("Time =" + str(time.time() - start))


if __name__ == '__main__':
    main()
