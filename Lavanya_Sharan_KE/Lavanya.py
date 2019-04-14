import os
from keywordextraction import *
import Algorithm1_1 as a1
import pandas as pd
from nltk import SnowballStemmer
import time
stemmer = SnowballStemmer('english')


def main():
    begin = time.time()
    os.chdir("..")
    orig_data = pd.read_csv("Asset_with_files_name.csv")
    print "HHHHHHH"
    print os.listdir('.')
    print orig_data.head()
    data = pd.read_csv("Keyphrase_Counter.csv")
    keywords = list(data['Key'])

    df = pd.DataFrame()
    os.chdir('./Lavanya_Sharan_KE')

    # load keyword classifier
    preload = 0
    classifier_type = 'logistic'

    keyword_classifier = get_keywordclassifier(preload, classifier_type)['model']

    for ind, row in orig_data.iterrows():
        para_start = time.time()
        text = row['Text']
        text = text.decode('utf-8').encode('ascii', 'ignore').strip()

        for _hex, _char in a1.LATIN_1_CHARS:
            text = text.replace(_hex, _char)

        text = " ".join(text.split())

        # extract top k keywords
        top_k = 550
        y_pred = extract_keywords(text, keyword_classifier, top_k, preload)

        # #####################################################################333
        # Part 1:: Stemming
        golden_keywords = []
        for key in keywords:
            sp = key.split()
            stemm = []
            for s in sp:
                stemm.append(stemmer.stem(s).lower())
            golden_keywords.append(' '.join(stemm))

        count = 0
        para_key = []
        freq_sum = 0
        for v in y_pred:
            sp = v.split()  # v[0].split()
            stemm = []
            for s in sp:
                stemm.append(stemmer.stem(s).lower())

            if ' '.join(stemm) in golden_keywords:
                freq_sum += int(data.loc[golden_keywords.index(' '.join(stemm)), 'Count'])
                para_key.append(v)
                count += 1

        df = df.append({'Text': text, 'File_name': row['File'], 'Frequency': freq_sum,
                        'Common_Keyphrase': para_key, 'Golden_key_len': len(golden_keywords),
                        'Para_key_len': len(y_pred)}, ignore_index=True)

        print "Para" + str(ind) + ": Time =" + str(time.time() - para_start)

    dd = pd.DataFrame()
    for _file in df['File_name'].unique():
        d = df.loc[df['File_name'] == _file, :]
        d = d.sort_values('Frequency', ascending=False)
        dd = dd.append(d)

    mi = min(list(dd['Frequency']))
    ma = max(list(dd['Frequency']))
    norm_freq = []
    for freq in list(dd['Frequency']):
        norm_freq.append((freq - mi)*1.0/ma)
    dd['Frequency'] = norm_freq

    dd.to_csv("Lavanya_Result_Test_file_finale.csv", index=False)
    print "Total time = " + str(time.time() - begin)


if __name__ == '__main__':
    main()
