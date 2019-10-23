# For testing Kmeans

'''

Title: Testing Kmeans algorithm.

CS5604@VT

Team: TML

Author: Prathamesh Mandke

Date created: 09/28/2019

'''

# conda activate preproc - deprecated

# conda activate kmeans

from kmeans import *
import argparse

RD_PATH = '/mnt/ceph/tml/clustering/kmeans/obj/tobacco/'

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--new_tf', type=bool, default=False)
    parser.add_argument('--n_idf', type=str, default='2')
    parser.add_argument('--n_clus', default=10, type=int) # N/A for CH score compute
    parser.add_argument('--n_km', default='1', type=str) # N/A for CH score compute
    parser.add_argument('--ch_idx', default=True, type=bool)

    args = parser.parse_args()
    print("Start")
    t_st = time.monotonic()

    if args.new_tf:
        print("Computing TFIDF vectors:")
        t1 = time.monotonic()
        tfidf = TFIDF()
        tfidf.fit_transform()
        print("Done in {}s.".format(timedelta(seconds=time.monotonic() - t1)))
        joblib.dump(tfidf, SAVE_PATH + 'TFIDF_' + args.n_idf + '.sav')
        #tfidf.save(SAVE_PATH + 'TFIDF_' + args.n_idf + '.sav')
    else:
        tfidf = joblib.load(RD_PATH + 'TFIDF_' + args.n_idf + '.sav')

    ch_index = dict()

    data = tfidf.tfidf_matrix.copy()
    if args.ch_idx:
        for cluster in [5, 10, 12, 15, 20, 25]:
            km = Kmeans(n_clusters=cluster, doc_list=tfidf.doc_list.copy())
            print("Running K-Means with {}clusters".format(cluster))
            t1 = time.monotonic()
            km.fit(data)
            print("Done in {}s".format(timedelta(seconds=time.monotonic() - t1)))
            joblib.dump(km, SAVE_PATH + 'kmeans_' + args.n_idf + '_' + str(cluster) + '.sav')

            ch_index[cluster] = km.compute_chi_index(data)

            print("CH score is {}".format(ch_index[cluster]))

        joblib.dump(ch_index, SAVE_PATH + 'ch_score_dict.sav')
    else:
        km = Kmeans(n_clusters=args.n_clus, doc_list=tfidf.doc_list.copy())
        data = tfidf.tfidf_matrix.copy()
        del tfidf
        print("Computing Kmeans clusters:")
        t1 = time.monotonic()
        km.fit(data)
        print("Done in {}s".format(timedelta(seconds=time.monotonic() - t1)))
        joblib.dump(km, SAVE_PATH + 'kmeans_' + args.n_idf + '_' + args.n_km + '.sav')


    print("Done. Total time taken {}s".format(timedelta(seconds=time.monotonic() - t_st)))
    
if __name__ == '__main__':
    main()
