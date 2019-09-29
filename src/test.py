# For testing Kmeans 

'''

Title: K-Means clustering of documents

CS5604@VT

Team: TML

Author: Prathamesh Mandke

Date created: 09/28/2019

'''
# conda activate preproc

from kmeans import *

if __name__ == '__main__':
    
    '''print("Computing TFIDF vectors:")
    t1 = time.monotonic()
    tfidf = TFIDF()
    tfidf.fit_transform()
    print("Done in {}s.".format(timedelta(seconds=time.monotonic() - t1)))
    tfidf.save(SAVE_PATH + 'TFIDF_1.sav')
   '''
    
    tfidf = joblib.load(SAVE_PATH + 'TFIDF_1.sav')
    km = Kmeans(n_clusters=20, doc_list=tfidf.doc_list.copy())
    data = tfidf.tfidf_matrix
    del tfidf
    print("Computing Kmeans clusters:")
    t1 = time.monotonic()
    km.fit(data)
    print("Done in {}s".format(timedelta(seconds=time.monotonic() - t1)))
    km.save(SAVE_PATH + 'Kmeans_2.sav')