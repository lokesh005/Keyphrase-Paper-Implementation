import pke
import os
import re
import random
import numpy as np
import time


def main():
    begin = time.time()
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

        # initialize keyphrase extraction model, here TopicRank
        extractor = pke.TopicRank(input_file=txt[i])

        # load the content of the document, here document is expected to be in raw
        # format (i.e. a simple text file) and preprocessing is carried out using nltk
        extractor.read_document(format='raw')

        # keyphrase candidate selection, in the case of TopicRank: sequences of nouns
        # and adjectives
        extractor.candidate_selection()

        # candidate weighting, in the case of TopicRank: using a random walk algorithm
        extractor. candidate_weighting()

        # N-best selection, keyphrases contains the 10 highest scored candidates as
        # (keyphrase, score) tuples
        keyphrases = extractor.get_n_best(n=10)

        key = []
        val = []
        for k, v in keyphrases:
            key.append(k)
            val.append(v)

        phrases = open(key_file[i], mode='r')
        phr = phrases.read()
        y_true = phr.split('\n')

        if len(y_true) > len(key):
            y_true = y_true[0:len(key)]
        else:
            key = key[0:len(y_true)]

        da = []
        for y in y_true:
            da.append(y.lower())

        y_true = da
        print(key)
        print(y_true)

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
