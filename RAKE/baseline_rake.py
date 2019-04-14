import os
import time

import pandas as pd
from nltk import SnowballStemmer
from rake_nltk import Rake

STEMMER = SnowballStemmer('english')

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
    """
    :return: "Rake_Result_Test_file_finale.csv"
    """
    begin = time.time()
    os.chdir('..')
    orig_data = pd.read_csv("Asset_with_files_name.csv")

    data = pd.read_csv("Keyphrase_Counter.csv")
    keywords = list(data['Key'])

    data = pd.DataFrame()

    for ind, row in orig_data.iterrows():
        para_start = time.time()
        text = row['Text']
        text = text.decode('utf-8').encode('ascii', 'ignore').strip()

        for _hex, _char in LATIN_1_CHARS:
            text = text.replace(_hex, _char)

        text = " ".join(text.split())

        rake = Rake()
        rake.extract_keywords_from_text(text.decode('utf-8').encode('ascii', 'ignore').strip())
        y_pred = rake.get_ranked_phrases()

        # #####################################################################333
        # Part 1:: Stemming
        golden_keywords = []
        for key in keywords:
            sp = key.split()
            stemm = []
            for s in sp:
                stemm.append(STEMMER.stem(s).lower())
            golden_keywords.append(' '.join(stemm))

        count = 0
        para_key = []
        freq_sum = 0
        for v in y_pred:
            sp = v.split()  # v[0].split()
            stemm = []
            for s in sp:
                stemm.append(STEMMER.stem(s).lower())

            if ' '.join(stemm) in golden_keywords:
                freq_sum += int(data.loc[golden_keywords.index(' '.join(stemm)), 'Count'])
                para_key.append(v)
                count += 1

        data = data.append({'Text': text, 'File_name': row['File'], 'Frequency': freq_sum,
                        'Common_Keyphrase': para_key, 'Golden_key_len': len(golden_keywords),
                        'Para_key_len': len(y_pred)}, ignore_index=True)

        print "Para" + str(ind) + ": Time =" + str(time.time() - para_start)

    data_frame = pd.DataFrame()
    for _file in data['File_name'].unique():
        data_selected = data.loc[data['File_name'] == _file, :]
        data_selected = data_selected.sort_values('Frequency', ascending=False)
        data_frame = data_frame.append(data_selected)

    minimum = min(list(data_frame['Frequency']))
    maximum = max(list(data_frame['Frequency']))
    norm_freq = []
    for freq in list(data_frame['Frequency']):
        norm_freq.append(((freq - minimum) * (5 - 0) * 1.0 / (maximum - minimum)) + 0)
    data_frame['Frequency'] = norm_freq
    os.chdir('./RAKE')
    data_frame.to_csv("Rake_Result_Test_file_finale.csv", index=False)
    print "Total time = " + str(time.time() - begin)


if __name__ == '__main__':
    main()
