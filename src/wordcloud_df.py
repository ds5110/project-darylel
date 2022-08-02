import read_files 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from PIL import Image

def main():
    df = read_files.main()
    df = df[df.county != "unk"]
    df = df.groupby('county')['text'].apply(' '.join).reset_index()

    # parse command line argument
    tfVectorizer = CountVectorizer()
    tfidfVectorizer = TfidfVectorizer(sublinear_tf=True)

    # sparse_matrix & feature_names defined here, and used below
    tf_sparse_matrix = tfVectorizer.fit_transform(df['text'])
    tfidf_sparse_matrix = tfidfVectorizer.fit_transform(df['text'])
    tf_feature_names = tfVectorizer.get_feature_names_out() 
    tfidf_feature_names = tfidfVectorizer.get_feature_names_out() 

    ### map mask
    androscogin_ori = np.array(Image.open("../figs/androscogin_county.png"))
    androscogin_mask = androscogin_ori.copy()
    androscogin_mask[androscogin_mask.sum(axis=2) == 0] = 255


    # Get highest tf words in the first book
    for i in range(0,1):
        tf = tf_sparse_matrix[i,:].toarray()[0]
        tfidf = tfidf_sparse_matrix[i,:].toarray()[0]

        tf_dict = dict(zip(tf_feature_names, tf))
        tfidf_dict = dict(zip(tfidf_feature_names, tfidf))

        tf_wordcloud = WordCloud(max_words=100, mask=androscogin_mask, contour_width=3, contour_color='firebrick', background_color="white").generate_from_frequencies(tf_dict)
        tfidf_wordcloud = WordCloud(max_words=100, mask=androscogin_mask, contour_width=3, contour_color='firebrick', background_color="white").generate_from_frequencies(tfidf_dict)

        fig, ax = plt.subplots(1,2,figsize=(24,5))
        ax[0].imshow(tf_wordcloud, interpolation='bilinear')
        ax[0].set_title('TF of ' + df['county'][i] + " county")
        ax[0].axis("off")

        ax[1].imshow(tfidf_wordcloud, interpolation='bilinear')
        ax[1].set_title('TF-IDF of ' + df['county'][i] + " county")
        ax[1].axis("off")
    plt.show()



    
    


if __name__ == '__main__':
    main()
