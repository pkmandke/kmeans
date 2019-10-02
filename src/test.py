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
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--new_tf', type=bool, default=True)
    parser.add_argument('--n_idf', type=str, default='1')
    parser.add_argument('--n_clus', default=10, type=int)
    parser.add_argument('--n_km', default='1', type=str)

    args = parser.parse_args()
    if args.new_tf:
        print("Computing TFIDF vectors:")
        t1 = time.monotonic()
        tfidf = TFIDF()
        tfidf.fit_transform()
        print("Done in {}s.".format(timedelta(seconds=time.monotonic() - t1)))
        joblib.dump(tfidf, SAVE_PATH + 'TFIDF_' + args.n_idf + '.sav')
        #tfidf.save(SAVE_PATH + 'TFIDF_' + args.n_idf + '.sav')
    else:
        tfidf = joblib.load(SAVE_PATH + 'TFIDF_' + args.n_idf + '.sav')

    sys.exit(0)

    km = Kmeans(n_clusters=args.n_clus, doc_list=tfidf.doc_list.copy())
    data = tfidf.tfidf_matrix.copy()
    del tfidf
    print("Computing Kmeans clusters:")
    t1 = time.monotonic()
    km.fit(data)
    print("Done in {}s".format(timedelta(seconds=time.monotonic() - t1)))
    joblib.dump(km, SAVE_PATH + 'kmeans_' + args.n_idf.sav + '_' + args.n_km + '.sav')
