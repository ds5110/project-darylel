import read_files 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import sys

def main():
    df = read_files.main()
    df = df[df.county != "unk"]
    df = df.groupby('county')['text'].apply(' '.join).reset_index()

    # parse command line argument
    tfVectorizer = CountVectorizer()
    tfidfVectorizer = TfidfVectorizer()

    # sparse_matrix & feature_names defined here, and used below
    tf_sparse_matrix = tfVectorizer.fit_transform(df['text'])
    tfidf_sparse_matrix = tfidfVectorizer.fit_transform(df['text'])
    tf_feature_names = tfVectorizer.get_feature_names_out() 
    tfidf_feature_names = tfidfVectorizer.get_feature_names_out() 

    # Get highest tf words in the first book
    for i in range(0,8):
        tf = tf_sparse_matrix[i,:].toarray()[0]
        tfidf = tfidf_sparse_matrix[i,:].toarray()[0]


        # Sort tfidf from large to small (default sort is increasing)
        tf_sorted_indices = np.argsort(-tf) # these are the indices of the sort
        sorted_tf = [tf[j] for j in tf_sorted_indices] # this is the sorted array
        tf_sorted_features = [tf_feature_names[j] for j in tf_sorted_indices] # features sorted by tfidf

        tfidf_sorted_indices = np.argsort(-tfidf) # these are the indices of the sort
        sorted_tfidf = [tfidf[j] for j in tfidf_sorted_indices] # this is the sorted array
        tfidf_sorted_features = [tfidf_feature_names[j] for j in tfidf_sorted_indices] # features sorted by tfidf

        fig, ax = plt.subplots(1,2,figsize=(24,5))
        
        ax[0].bar(tf_sorted_features[:10], sorted_tf[:10], 
                width=1, alpha=.5, edgecolor='black')
        ax[0].set_title('TF of ' + df['county'][i] + " county")
        ax[0].set_xticklabels(tf_sorted_features[:10],rotation=45)

        ax[1].bar(tfidf_sorted_features[:10], sorted_tfidf[:10], 
                width=1, alpha=.5, edgecolor='black')
        ax[1].set_title('TF-IDF of ' + df['county'][i] + " county")
        ax[1].set_xticklabels(tfidf_sorted_features[:10],rotation=45)

    plt.show()


if __name__ == '__main__':
    main()
