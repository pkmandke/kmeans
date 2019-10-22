# Data description


# ./tobacco

## TDIDF
1. **TFIDF_1.sav:** - text_depositions uncleaned \#7995
  * Basic porter stemmer with english stopwords and normalized features.
  * Tokenized using nltk.word_tokenize.
2. **TFIDF_2.sav** - test depositions cleaned \#4553

<!-- 2. **TFIDF_2.sav:** Ignoring tokens that are smaller than 4 characters in length. -->

## Kmeans

1. Kmeans_1_1.sav - 10 clusters
  * Kmeans clustering with TFIDF values extracted from the original unclean 7995 docs.
  * TFIDF_1.sav and tokens_1.sav used.
2. Kmeans_2_1.sav - 10 clusters
  * Kmeans clustering with TFIDF values extracted from the cleaned(encoding-wise?) 4553 docs in ceph/shared/tobacco/text_*_clean/*
  * TFIDF_2.sav and tokens_2.sav used


## Tokens

Unique Tokens in every file stored as a dict in a class. Only unique tokens in each doc are stored.

1. **tokens_1.sav**
    * Uncleaned 7995 docs. 

2. **tokens_2.sav**
  * Cleaned 4553 docs. 