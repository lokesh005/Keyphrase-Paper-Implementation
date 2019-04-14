In this we are trying to find the keyword which can represent the whole text. We have used three algorithms: 

1. **Using Rake-nltk package:** In this we have used rake package of the python.(See Rake.py file)
2. **Lavanya KE:** In this we have used Logistic Regression model. For each observation of training set all keywords associated with the article are termed as positive examples and rest are termed as negative examples.
 Features that we have used are:<br/>
    a) Frequency of words: TF, TF-IDF score, Wiki Freq. <br/> 
    b) Structure of word: Term length, capitalisation. <br/>
    c) Type of word: Named Entity, Noun Phrases, trigram. <br/>
    d) Relationship between word and input text: First occurance in text, Distance between occurance in text. <br/>
For each keyword classifier examines 10 associated features and assigns a probability p of the candidate being a keyword. If p>0.75 then candidate is selected as a keyword. (See Lavanya_Sharan_KE directory).
 
3. **Embeded:** This is an unsupervised model in which we are using Doc2vec to assign the vectors to both the document and keyphrase.(See Embeded.py file)<br/>
  a)  Phrases that consist of any number of adjectives and one or more nouns are considered to be  the candidate keyphrases.<br/>
  b)  Now we use doc2vec to find the cosine similarity between document and candidate keyphrase.<br/>
  c) Now we rank  keyphrases using Maximal Marginal Relevance (MMR) to avoid redundancy problem.<br/>

                `MMR := arg max [λ · cossim (Ci , doc) − (1 − λ) max cossim(Ci , Cj )]`

When λ = 1 MMR computes a standard, relevance ranked list, while when λ = 0 it computes a maximal diversity ranking of the documents in R. So we will keep λ = 0.5, so that relevance and diversity parts of the equation have equal importance.
             Please refer paper for further information(EmbedRank: Unsupervised Keyphrase Extraction using Sentence Embeddings) 


We have used precision, recall and f1-score as a measure to compare between algorithms. The code that we have used is: 
```
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
  re = tp*1.0/(tp+fn)
  f1 = 2*p*re*1.0/(p + re)
```
We finally took the average of all the precisions, recalls and 
f1-scores from each algorithm

Accurcy and Computation time for the above algorithm taking Crowd500 dataset are:

Method | Precision | Recall  | F1-score | Time | Dataset |
-------|------------|--------|----------|------|---------|
Embeded | 0.2779 | 0.3421 | 0.2906 | 578.2 | (Crowd)|
Rake | 0.1896 | 0.2087 | 0.1493 | 0.41 | (Crowd)|
Lavanya Sharan KE |  0.7555  | 0.1697|  0.2418|  52| (Crowd)|
Embeded gensim ngram with cosine | 0.267 |.03672 | 0.2444 | 2.08 | (Crowd) |
Stopword and punc as delimiter and finding ngrams | 0.122 | 0.5931| 0.1709| 0.42 | (Crowd) |


### Descrpition of files:</br>
*Rake.py*: Implementation of RAKE algorithm</br>
*Embeded.py*: Containes main function of Embeded algorithm</br>
*Algorithm1_1.py*: Implementation of Embeded function</br>

*Rake_cos.py*: Find the keywords using rake-nltk package in both golden text and test set and then further used cosine similarity to get the score.

*Rake_delimiter_stopword.py*: Used stopwords and punctuations as a delimiter to split the text in both golden text and test set.

**Lavanya_Sharan_KE folder**</br>
*Lavanya.py*: Implementation of the algorithm</br>
*keywordextraction.py*: Used to find the keywords by training the model</br>
*features.py*: Used to create features from the text</br>

### References:
1. *EmbedRank*: Unsupervised Keyphrase Extraction using Sentence Embeddings (Link: https://arxiv.org/pdf/1801.04470.pdf).
2. *RAKE-NLTK*: Link: https://pypi.python.org/pypi/rake-nltk
3. *Lavanya’s KE*: https://people.csail.mit.edu/lavanya/keywordfinder


