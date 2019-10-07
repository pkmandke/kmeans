'''
Title: Evaluating K-means

CS5604@VT

Team: TML

Author: Prathamesh Mandke

Date created: 10/03/2019

'''

BS_PATH = '/mnt/ceph/tml/clustering/kmeans/obj/'
DATA_PATH = '/mnt/ceph/shared/tobacco/text_files_depositions_clean/'

import joblib
from kmeans import *
import numpy as np
import nltk
import os

class SavLoader:

    def __init__(self, tf_path=None, km_path=None, tk_path=None):

        if tf_path:
            self.tfidf = joblib.load(tf_path)
        if km_path:
            self.km = joblib.load(km_path)
        if tk_path:
            self.tk_path = joblib.load(tk_path)

    def eval_cluster(self, n_clus=1, tokenize=False):

        self.k = self.km.km

        doc_idx = np.where(self.k.labels_ == n_clus)
        if tokenize:
            with open(self.tfidf.doc_list[doc_idx[0][0]], 'rb') as f:
                com_tk = set(p_tokenize(f.read().decode('utf-8', 'ignore')))
        else:
            com_tk = set(self.tk_path.doc_tokens[self.tfidf.doc_list[doc_idx[0][0]].split('/')[-1]])

        for _ in doc_idx[1:]:
            if tokenize:
                with open(self.tfidf.doc_list[doc_idx[0][_]], 'rb') as f:
                    tk1 = p_tokenize(f.read().decode('utf-8', 'ignore'))
                    com_tk = list(set(tk1) & set(com_tk))
            else:
                com_tk = list(set(self.tk_path.doc_tokens[self.tfidf.doc_list[doc_idx[0][0]].split('/')[-1]]) & com_tk)

        print("Total common terms in cluster {} with {} documents are {}".format(n_clus, len(doc_idx[0]), len(com_tk)))

        return com_tk

class SavTokenize:

    def __init__(self, path=None, savPath=None):
        self.path = path
        self.doc_tokens = dict()
        self.savPath = savPath
        nltk.download('punkt')

    def tokenize(self):
        for _ in os.listdir(self.path):
            print("Tokenizing {}".format(_), end='\r')
            self.doc_tokens[_] = list(set(p_tokenize(open(self.path + '/' + _, 'rb').read().decode('utf-8', 'ignore'))))
        print("Complete")

    def save(self):
        joblib.dump(self, self.savPath)


def main():
    sav = SavLoader(tf_path=BS_PATH + 'TFIDF_1.sav', km_path=BS_PATH + 'kmeans_1_1.sav', tk_path=BS_PATH + 'tokens_1.sav')
    print("Starting eval")
    for _ in range(10):
        com_terms = sav.eval_cluster(n_clus=_)
    print("Eval done")
    #print(com_terms)
def main_():
    sav = SavTokenize(path=DATA_PATH, savPath=BS_PATH + 'tokens_2.sav')
    
    sav.tokenize()
    
    sav.save()
    
main_()

if __name__ == '__main__':
#    sav = SavTokenize(DATA_PATH, BS_PATH + 'tokens_1.sav')

#    sav.tokenize()

#    sav.save()
    pass
    #main()
