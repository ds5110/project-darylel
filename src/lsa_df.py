import read_files 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from PIL import Image
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd

def main():
    df = read_files.main()
    df = df[df.county != "unk"]
    df = df.reset_index(drop=True)
    # df = df.groupby('county')['text'].apply(' '.join).reset_index()
    X = df['text']
    vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True)
    Xc = vectorizer.fit_transform(X) # .toarray()

    # Performed a truncated SVD
    rank = 8
    tsvd = TruncatedSVD(n_components=rank)
    Xt = tsvd.fit_transform(Xc)

    # get color map
    cm = np.zeros(df['text'].shape)
    unique_county = pd.DataFrame(data = df.county.unique())
    for i in range(len(df['county'])):
        cm[i] = unique_county.index[unique_county[0] == df.at[i,'county']].tolist()[0]

    # Show a scatter plot of all documents
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].scatter(Xt[:, 0], Xt[:, 1], c=cm, s=50, edgecolor="k", cmap="Set1")
    ax[0].set_xlabel('t0')
    ax[0].set_ylabel('t1')
    ax[0].grid()

    ax[1].scatter(Xt[:, 0], Xt[:, 1], c=cm, s=50, edgecolor="k", cmap="Set1")
    ax[1].set_xlabel('t0')
    ax[1].set_ylabel('t1')
    ax[1].grid()
    ax[1].set_xlim(0,1)

    plt.show()

if(__name__ == '__main__'):
    main()