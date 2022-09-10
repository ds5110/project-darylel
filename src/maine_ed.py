'''
Gungyeom (James) Kim & Daryle Lamoureux
DS5110 Summer 2022
Final Project

This project consumes text documents from interviews conducted by the Maine Ed Forum.
It then analyzes and visualizes the data from those files.
'''
import os
import sys
import re
from datetime import date
from dotenv import load_dotenv
import pandas as pd
import requests
from requests.structures import CaseInsensitiveDict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from PIL import Image
load_dotenv()
# Un-comment the bellow nltk modules if you have not previously loaded them
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('omw-1.4')

def add_stopwords():
    '''
    Extend stopwords from .txt file
    Return:
        stops: list of stopwords
    '''
    # Initialize list to hold custom stopword
    stops = list()

    # Read stopwords from txt file
    with open('custom_stopwords.txt', 'r', encoding='utf-8') as f:
        data = f.read()

    # Create a list of custom stopwords
    stops = data.split('\n')

    return stops

def make_call(content, params=None):
    '''
    Reusable function to make API calls

    Parameters:
        content: The name of the information to be retried, i.e. snippets, conversations, etc.
        params: Dictionary of parameters for the API call
    Returns:
        resp_json: JSON with the response data
    '''
    # Get API authentication token from .env
    # User will have to have a token issued by LVN/Maine Ed Forum and save it in a .env file locally
    auth_token = os.environ.get("API_TOKEN")

    # URL for API call
    # This currently uses version v1. If there is version update in the future, update the link here
    url = 'https://api.lvn.org/v1/' + content

    # Define header for API call
    headers = CaseInsensitiveDict()
    headers["Accept"] = "application/json"
    headers["Authorization"] = "Bearer " + auth_token

    if params is not None:
        resp = requests.get(url, headers=headers, params=params)
    else:
        resp = requests.get(url, headers=headers)

    if str(resp.status_code).startswith('5'):
        print('ERROR: 500 Server Error - Try again later')
        exit(0)

    # GET call as JSON
    resp_json = resp.json()

    return resp_json

def get_convo_ids():
    '''
    Get all the conversation IDs

    Returns:
        all_ids: List of all conversation IDs
    '''
    # Initialize the empty list
    all_ids = list()

    # Parameters for the API call
    # Pagination will need to be adjusted if conversations becomes > 100
    params = {
        "collection_ids": "150",
        "page_size": "100"
    }

    # Make the API call
    resp = make_call('conversations', params)

    # Parse the response and add to the list
    for s in resp:
        all_ids.append(s.get('id'))

    return all_ids

def get_conversations(ids):
    '''
    Get all conversations included in the IDs list

    Parameters:
        ids: List of conversation IDs
    Returns:
        conversations: Pandas dataframe with all the conversations
    '''
    # Initialize conversations list
    conversations = list()

    for conversation_id in ids:
        # Make the API call for the conversation with the given conversation ID
        resp = make_call('conversations/' + str(conversation_id), {})

        if resp is not None:
            for s in resp.get('snippets'):
                # Initialize conversation list
                conversation = []

                # Recreate each conversation and add to the conversations list
                conversation.append(s['speaker_name'])
                conversation.append(s['is_facilitator'])

                # Consolidate all the words
                temp = list()
                # Create a list of tags per conversation
                tags = list()
                for word in s['words']:
                    temp.append(word[0])
                sentence = ' '.join(temp)
                sentence = preprocess(sentence)
                conversation.append(sentence)

                # Assign the appropriate group to each conversations
                session = resp.get('title')
                if 'educator' in session.lower():
                    conversation.append('educator')
                elif 'teacher' in session.lower():
                    conversation.append('educator')
                elif 'youth' in session.lower():
                    conversation.append('youth')
                elif 'community' in session.lower():
                    conversation.append('community')
                elif 'industry' in session.lower():
                    conversation.append('industry')
                else:
                    conversation.append('unknown')
                # Make a list of all the associated tags
                highlights = resp.get('highlights')
                if highlights is not None:
                    for highlight in highlights:
                        for tag in highlight['tags']:
                            tags.append(tag)
                conversation.append(tags)
                conversation.append(session)
                conversation.append(conversation_id)
                # Normalize all 'Unknown' to 'Maine' and
                # Augusta -> kennebec and Bridgton, -> cumberland
                loc = resp.get('location').get('name')
                if loc != 'Maine' and loc != 'Unknown':
                    loc = loc.split(' ')
                    if loc[0] == 'Augusta,':
                        conversation.append('kennebec')
                    elif loc[0] == 'Bridgton,':
                        conversation.append('cumberland')
                    elif loc[0] == 'Presque':
                        conversation.append('aroostook')
                    else:
                        conversation.append(loc[0].lower())
                else:
                    conversation.append('maine')
                conversation.append(resp.get('participant_count'))
                conversations.append(conversation)

    return conversations

def preprocess(line):
    '''
    Create and preprocess a word list from file content

    Inputs:
        line: Sentence (string) to be preprocessed
    Returns:
        word_list: List of preprocessed words from the sentence
    '''
    # Add nltk.corpus stopwords and extend the stopwords using add_stopwords()
    new_stops = add_stopwords()
    stops = stopwords.words('english')
    stops.extend(new_stops)
    stops = set(stops)

    # Initialize WordNet Lemmatizer to lemmatize the word corpus
    wnl = WordNetLemmatizer()

    # Parse the document and create a wordlist
    word_list = []
    # Remove the notes in []
    line = re.sub(r'\[.*?\]', '', line)
    # Remove punctuation
    line = re.sub(r'[^\w\s]', ' ', line)

    # Add each word/token to the word list
    line = line.split()
    for l in line:
        if len(l) > 0 and (l.isnumeric() is False) and l.lower() not in stops:
            word_list.append(wnl.lemmatize(l).lower())

    return word_list

def create_dataframe(data, columns):
    '''
    Create a Pandas dataframe from a list of data

    Inputs:
        data: List of data elements to be converted
        columns: List of column headers
    Returns:
        df: Pandas dataframe with corresponding column headers
    '''
    df = pd.DataFrame(data, columns=columns)

    return df

def create_bar(tf, tf_feature_names, tfidf, tfidf_feature_names, category):
    '''
    Creates a bar graph of the most common (tf) and least common (tfidf) words
    Inputs:
        tf: total count vector
        tf_feature_names: names of the features in the count vector
        tfidf: tfidf weighted vector
        tfidf_features_names: names of the features in the tfidf vector
        category: the category to be plotted
    '''
    # Sort tfidf from large to small (default sort is increasing)
    # These are the indices of the sort
    tf_sorted_indices = np.argsort(-tf)
    # This is the sorted array
    sorted_tf = [tf[j] for j in tf_sorted_indices]
    # Features sorted by tfidf
    tf_sorted_features = [tf_feature_names[j] for j in tf_sorted_indices]

    # These are the indices of the sort
    tfidf_sorted_indices = np.argsort(-tfidf)
    # This is the sorted array
    sorted_tfidf = [tfidf[j] for j in tfidf_sorted_indices]
    # Features sorted by tfidf
    tfidf_sorted_features = [tfidf_feature_names[j] for j in tfidf_sorted_indices]

    plt.figure()
    plt.bar(tf_sorted_features[:25], sorted_tf[:25], width=1, alpha=.5, edgecolor='black')
    plt.title('Most Used Words in ' + category.title())
    plt.xticks(tf_sorted_features[:25], rotation=65)
    plt.tight_layout()
    if category.isalnum() is False:
        category = 'tag-' + category[1:]
    plt.savefig('../figs/bar/' + category + '-most-bar.png')
    plt.show()

    plt.figure()
    plt.bar(tfidf_sorted_features[:25], sorted_tfidf[:25], width=1, alpha=.5, edgecolor='black')
    plt.title('Least Used Words in ' + category.title())
    plt.xticks(tfidf_sorted_features[:25], rotation=65)
    plt.tight_layout()
    if category.isalnum() is False:
        category = 'tag-' + category[1:]
    plt.savefig('../figs/bar/' + category + '-least-bar.png')
    plt.show()

    # The commented out code below creates a side by side image of TF and TF-IDF
    '''
    fig, ax = plt.subplots(1,2,figsize=(24,5))

    # Currently displays 25 words. Change [:25] to the desired value for >25 or <25
    ax[0].bar(tf_sorted_features[:25], sorted_tf[:25],
            width=1, alpha=.5, edgecolor='black')
    ax[0].set_title('Most Used (TF) Words in ' + category.title())
    ax[0].set_xticks(tf_sorted_features[:25])
    ax[0].set_xticklabels(tf_sorted_features[:25],rotation=65)

    # Currently displays 25 words. Change [:25] to the desired value for >25 or <25
    ax[1].bar(tfidf_sorted_features[:25], sorted_tfidf[:25],
            width=1, alpha=.5, edgecolor='black')
    ax[1].set_title('Least Used (TF-IDF) Words in ' + category.title())
    ax[1].set_xticks(tfidf_sorted_features[:25])
    ax[1].set_xticklabels(tfidf_sorted_features[:25],rotation=65)

    # Save figure as an image
    if category[0].isalnum() is False:
        category = 'tag-' + category[1:]
    plt.savefig('../figs/bar/' + category + '-bar-' + str(date.today()) + '.png')

    # Display figure on the screen
    plt.show()
    '''

def create_cloud(tf_dict, tfidf_dict, title, mapImage):
    '''
    Generate the cloud visualizations

    Inputs:
        tf_dict: dictionary of total word counts
        tfidf_dict: dictionary of tf-idf word weights
        title: string representing the title of the visualization
        mapImage: the name of the image to be used for visualization
    '''
    # Create map mask
    img = np.array(Image.open('../maps/' + mapImage + '.png'))
    img_mask = img.copy()
    img_mask[img_mask.sum(axis=2) == 0] = 255

    # Create cloud
    tf_cloud = WordCloud(max_words=100, mask=img_mask, contour_width=3, \
        contour_color='firebrick', background_color='white').generate_from_frequencies(tf_dict)
    tfidf_cloud = WordCloud(max_words=100, mask=img_mask, contour_width=3, \
        contour_color='firebrick', background_color="white").generate_from_frequencies(tfidf_dict)

    # Plot the visualization
    plt.figure()
    plt.imshow(tf_cloud, interpolation='bilinear')
    plt.title('Most Used Words in ' + title.title())
    plt.axis('off')
    if title.isalnum() is False:
        title = 'tag-' + title[1:]
    plt.tight_layout()
    plt.savefig('../figs/cloud/' + title + '-most-cloud.png')
    plt.show()

    plt.figure()
    plt.imshow(tfidf_cloud, interpolation='bilinear')
    plt.title('Least Used Words in ' + title.title())
    plt.axis('off')
    if title.isalnum() is False:
        title = 'tag-' + title[1:]
    plt.tight_layout()
    plt.savefig('../figs/cloud/' + title + '-least-cloud.png')
    plt.show()

    # The code commented out below creates a side by side image of TF and TF-IDF
    '''
    fig, ax = plt.subplots(1,2,figsize=(24,5))
    ax[0].imshow(tf_cloud, interpolation='bilinear')
    ax[0].set_title('Most Used (TF) Words in ' + title.title())
    ax[0].axis("off")

    ax[1].imshow(tfidf_cloud, interpolation='bilinear')
    ax[1].set_title('Least Used (TF-IDF) Words in ' + title.title())
    ax[1].axis("off")

    # Save figure as an image
    # Save figure as an image
    if title[0].isalnum() is False:
        title = 'tag-' + title[1:]
    plt.savefig('../figs/cloud/' + title + '-cloud-' + str(date.today()) + '.png')

    # Display figure on the screen
    plt.show()
    '''

def create_plot(df, category, style):
    '''
    Plot and show the wordcloud from a Pandas dataframe

    Inputs:
        df: Pandas dataframe
        type: bar, cloud
        category: String with the name of the category (i.e., "maine, counties, groups, etc.)
    Return: message indicating success or an error
    '''
    # Initialize message
    message = ''

    # Intialize text vectorizers
    tf_vectorizer = CountVectorizer()
    tf_idf_vectorizer = TfidfVectorizer(sublinear_tf=True)

    # Create a copy of the dataframe
    df = df.copy()

    # Removing facilitator
    df = df[df.facilitator != True]

    # Rebuild sentences from the lemmatized words
    df['sentence'] = df['sentence'].apply(lambda x: ' '.join(x))

    # Get names of all the counties that are in the dataset
    data_counties = df['county'].drop_duplicates().to_list()
    counties = sorted(data_counties)

    # Get names of all the groups
    data_groups = df['group'].drop_duplicates().to_list()
    groups = sorted(data_groups)

    # Get all the tags
    all_tags = df['tags'].tolist()
    tags = sorted(list(set([x for tag in all_tags for x in tag])))

    # Plot state data
    if category == 'maine':
        df['county'].replace(counties, 'maine')
        df = df.groupby('county').agg({'sentence': lambda x: ' '.join(x)}).reset_index()
        for county in data_counties:
            if county == 'maine':
                # Count vectorizer
                tf_sparse = tf_vectorizer.fit_transform(df['sentence'])
                tf_features = tf_vectorizer.get_feature_names_out()
                tf = tf_sparse[counties.index(county),:].toarray()[0]
                tf_dict = dict(zip(tf_features,tf))

                # Tf-idf Vectorizer
                tfidf_sparse = tf_idf_vectorizer.fit_transform(df['sentence'])
                tfidf_features = tf_idf_vectorizer.get_feature_names_out()
                tfidf = tfidf_sparse[counties.index(county),:].toarray()[0]
                tfidf_dict = dict(zip(tfidf_features,tfidf))

                # Generate cloud or bar visualization
                if style == 'cloud':
                    # Create cloud plot
                    create_cloud(tf_dict, tfidf_dict, county, county)
                elif style == 'bar':
                    #Create bar graph visualization
                    create_bar(tf, tf_features, tfidf, tfidf_features, county)
                else:
                    return 'Error: Available choices only: <cloud | bar>'

                message = 'success'

    # Plot the counties data
    if category == 'counties':
        df = df.groupby('county').agg({'sentence': lambda x: ' '.join(x)}).reset_index()

        # Create a visualization for each county
        for county in counties:
            if county == 'maine':
                pass
            else:
                # Count vectorizer
                tf_sparse = tf_vectorizer.fit_transform(df['sentence'])
                tf_features = tf_vectorizer.get_feature_names_out()
                tf = tf_sparse[counties.index(county),:].toarray()[0]
                tf_dict = dict(zip(tf_features,tf))

                # Tf-idf Vectorizer
                tfidf_sparse = tf_idf_vectorizer.fit_transform(df['sentence'])
                tfidf_features = tf_idf_vectorizer.get_feature_names_out()
                tfidf = tfidf_sparse[counties.index(county),:].toarray()[0]
                tfidf_dict = dict(zip(tfidf_features,tfidf))

                # Generate cloud or bar visualization
                if style == 'cloud':
                    # Create cloud visualization
                    create_cloud(tf_dict, tfidf_dict, county, county)
                elif style == 'bar':
                    #Create bar graph visualization
                    create_bar(tf, tf_features, tfidf, tfidf_features, county)
                else:
                    return 'Error: Available choices only: <cloud | bar>'

            message = 'success'

    # Plot the groups data
    if category == 'groups':
        df = df.groupby('group').agg({'sentence': lambda x: ' '.join(x)}).reset_index()

        # Create a visualization for each county
        for group in groups:
            # Count vectorizer
            tf_sparse = tf_vectorizer.fit_transform(df['sentence'])
            tf_features = tf_vectorizer.get_feature_names_out()
            tf = tf_sparse[groups.index(group),:].toarray()[0]
            tf_dict = dict(zip(tf_features,tf))

            # Tf-idf Vectorizer
            tfidf_sparse = tf_idf_vectorizer.fit_transform(df['sentence'])
            tfidf_features = tf_idf_vectorizer.get_feature_names_out()
            tfidf = tfidf_sparse[groups.index(group),:].toarray()[0]
            tfidf_dict = dict(zip(tfidf_features,tfidf))

            # Generate cloud or bar visualization
            if style == 'cloud':
                # Create cloud visualization
                create_cloud(tf_dict, tfidf_dict, group, 'maine')
            elif style == 'bar':
                #Create bar graph visualization
                create_bar(tf, tf_features, tfidf, tfidf_features, group)
            else:
                return 'Error: Available choices only: <cloud | bar>'

        message = 'success'

    # Plot the tags data
    if category == 'tags':
        # Create a visualization for each county
        for tag in tags:
            # Create a dataframe with all the words associated with a tag
            new_df = df[pd.DataFrame(df.tags.tolist()).isin([tag]).any(1).values]
            new_df = new_df.groupby('facilitator').agg({'sentence': lambda x: ' '.join(x)}).reset_index()

            # TF Vectorizer
            tf_sparse = tf_vectorizer.fit_transform(new_df['sentence'])
            tf_features = tf_vectorizer.get_feature_names_out()
            tf = tf_sparse[0,:].toarray()[0]
            tf_dict = dict(zip(tf_features,tf))

            # Tf-idf Vectorizer
            tfidf_sparse = tf_idf_vectorizer.fit_transform(new_df['sentence'])
            tfidf_features = tf_idf_vectorizer.get_feature_names_out()
            tfidf = tfidf_sparse[0,:].toarray()[0]
            tfidf_dict = dict(zip(tfidf_features,tfidf))

            # Generate cloud or bar visualization
            if style == 'cloud':
                # Create cloud visualization
                create_cloud(tf_dict, tfidf_dict, tag, 'maine')
            elif style == 'bar':
                #Create bar graph visualization
                create_bar(tf, tf_features, tfidf, tfidf_features, tag)
            else:
                return 'Error: Available choices only: <cloud | bar>'

        message = 'success'

    return message

def main(sys_argv):
    '''
    Main code to get conversations from Maine Ed Forum using the API
    '''
    # Get all conversation IDs
    conversation_ids = get_convo_ids()

    # Get all conversations included in the conversation_ids list
    all_conversations = get_conversations(conversation_ids)

    # Create a dataframe from the conversations list
    columns = ['speaker','facilitator', 'sentence', 'group', 'tags', 'title', 'conversation_id', 'county','participant_count']
    df = create_dataframe(all_conversations, columns)

    # Enable CLI commands to get data
    if len(sys_argv) == 1:
        dfgb = df.groupby(['county'])
        df1 = dfgb.agg({'title': pd.Series.nunique})
        df2 = dfgb.agg(lambda x: x.drop_duplicates('title', keep='first').participant_count.sum())
        df = pd.concat([df1, df2['participant_count']],1)

        print(df)
        # testing
        #print("""\n############################################
            # THE NUMBER OF CONVERSATIONS HAPPENED AND #
            # THE NUMBER OF PARTICIPANT INVOLVED TOTAL #
            # WITHIN EACH COUNTY                       #
            ############################################""")

        result = 'success'
    elif len(sys_argv) <= 2:
        # Print the help message
        print("""
        Generates bar chart or word clouds of important words within categories using data by Ed Maine Forums from LVN API\n
        $python3 maine_ed.py help
        : print help\n
        $python3 maine_ed.py
        : print out the number of conversations happened and the number of participant involved total within each county\n
        $python3 maine_ed.py <maine | counties | groups | tags > <bar | cloud>
        : maine: plot <bar | cloud> of most frequent word from every conversation
        : counties: plot <bar | cloud> for every county in Maine where a conversation took place
        : groups: plot <bar | cloud> for every group that took part in the conversations
        : tags: plot <bar | cloud> for every tag that has been added to responses in the conversations
        """)

        result = 'success'
    elif len(sys_argv) >= 2:
        # Processing dataframe according to sys_argv[1]
        result = create_plot(df, sys_argv[1], sys_argv[2])
    else:
        # Print error message if none of the above apply
        print('''\nERROR: Use the command:
        $python3 maine_ed.py <maine | counties | groups | tags> <bar | cloud>''')

    # Print this message if unable to create a visualization
    if result != 'success':
        print('Unable to process the request. Please try again.')

if __name__ == "__main__":
    main(sys.argv)
