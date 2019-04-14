import os
import time

import simple_gensim
import pandas as pd
from nltk import SnowballStemmer
from nltk import word_tokenize
from rake_nltk import Rake
from scipy import spatial

stemmer = SnowballStemmer('english')

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


def main():
    begin = time.time()
    # model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
    # nlp = spacy.load('en_core_web_lg')
    os.chdir('..')
    model = simple_gensim.models.Doc2Vec.load('doc2vec.bin')
    print "Model Loaded"

    orig_data = pd.read_csv("Asset_with_files_name.csv")

    golden_data = pd.read_csv("Asset_manager_golden.csv")

    golden_keywords = []
    for ind, row in golden_data.iterrows():
        text = row['Text']
        text = text.decode('utf-8').encode('ascii', 'ignore').strip()

        for _hex, _char in LATIN_1_CHARS:
            text = text.replace(_hex, _char)

        text = " ".join(text.split())

        r = Rake()
        r.extract_keywords_from_text(text.decode('utf-8').encode('ascii', 'ignore').strip())
        keywords = r.get_ranked_phrases()
        for key in keywords:
            # sp = key.split()
            # stemm = []
            # for s in sp:
            #    stemm.append(stemmer.stem(s).lower())
            golden_keywords.append(key)

    golden_keywords = set(golden_keywords)

    gol_vec = []
    for gol_val in golden_keywords:
        a = word_tokenize(gol_val)
        a = model.infer_vector(a)
        gol_vec.append(a)

    df = pd.DataFrame()

    for ind, row in orig_data.iterrows():
        # print "ind" + str(ind)
        if ind == 337:
            break

        para_start = time.time()
        text = row['Text']
        text = text.decode('utf-8').encode('ascii', 'ignore').strip()

        for _hex, _char in LATIN_1_CHARS:
            text = text.replace(_hex, _char)

        text = " ".join(text.split())

        r = Rake()
        r.extract_keywords_from_text(text.decode('utf-8').encode('ascii', 'ignore').strip())
        y_pred = r.get_ranked_phrases()

        pred = []
        for v in y_pred:
            # sp = v.split()  # v[0].split()
            # stemm = []
            # for s in sp:
            #    stemm.append(stemmer.stem(s).lower())
            pred.append(v)

        sum1 = 0
        pred_vec = []
        for pred_val in pred:
            p = word_tokenize(pred_val)
            p = model.infer_vector(p)
            pred_vec.append(p)

        for i in range(len(gol_vec)):
            cos = []
            for p in range(len(pred_vec)):
                cos.append(1 - spatial.distance.cosine(gol_vec[i], pred_vec[p]))
            sum1 += max(cos)

        df = df.append({'File': row['File'], 'Wted_Freq_cos': sum1, 'Text': text},
                       ignore_index=True)

        print "Para" + str(ind) + ": Time =" + str(time.time() - para_start)

    dd = pd.DataFrame()
    for _file in df['File'].unique():
        d = df.loc[df['File'] == _file, :]
        d = d.sort_values('Wted_Freq_cos', ascending=False)
        dd = dd.append(d)

    mi = min(list(dd['Wted_Freq_cos']))
    ma = max(list(dd['Wted_Freq_cos']))
    norm_freq = []
    for freq in list(dd['Wted_Freq_cos']):
        norm_freq.append(((freq - mi) * (5 - 0) * 1.0 / (ma - mi)) + 0)
    dd['Wted_Freq_cos'] = norm_freq

    dd.to_csv("Rake_cos.csv", index=False)
    print "Total time = " + str(time.time() - begin)


if __name__ == '__main__':
    main()
