'''
Gungyeom (James) Kim & Daryle Lamoureux 
DS5110 Summer 2022
Final Project

This project consumes text documents from interviews conducted by the Maine Ed Forum.
It then analyzes and visualizes the data from those files.
'''
import os
import sys
from dotenv import load_dotenv
import re
import requests
from requests.structures import CaseInsensitiveDict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
import pandas as pd
load_dotenv()
import read_files 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from PIL import Image

def make_call(content, params={}):
    '''
    Reusable function to make API calls

    Parameters:
        content: The name of the information to be retried, i.e. snippets, conversations, etc.
        params: Dictionary of parameters for the API call
    Returns:
        resp_json: JSON with the response data
    '''
    # Get API authentication token from .env
    auth_token = os.environ.get("API_TOKEN")

    # URL for API call
    url = 'https://api.lvn.org/v1/' + content

    # Define header for API call
    headers = CaseInsensitiveDict()
    headers["Accept"] = "application/json"
    headers["Authorization"] = "Bearer " + auth_token

    if params != {}:
        resp = requests.get(url, headers=headers, params=params)
    else:
        resp = requests.get(url, headers=headers)

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
                conversation.append(resp.get('location').get('name'))
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
    # Create set of English stopwords from nltk.corpus
    new_stops = ["according", "across", "actual", "actually", "additionally", "afar", "ago", "ah",
        "already", "almost", "along", "also", "although", "always", "among", "apparently", "apart",
        "around", "b", "bit", "bring", "brings", "come", "comes", "coming", "completely", "cough",
        "could", "easily", "especially", "even", "every", "everything", "forth", "get", "gets",
        "getting", "go", "goes", "going", "gosh", "got", "gotten", "happen", "happens", "h", "hi",
        "ii", "let", "lets", "like", "likes", "little", "look", "looks", "lot", "lots", "make",
        "makes", "making", "many", "may", "maybe", "might", "much", "oh", "okay", "oops", "p",
        "pas", "perhaps", "pop", "pops", "put", "puts", "pretty", "putting", "quite", "really",
        "sec", "said", "say", "saying", "says", "see", "seen", "sees", "seem", "seems", "somebody",
        "something", "sort", "sorts", "specifically", "still", "strongly", "stuff", "sure", "take",
        "takes", "thing", "things", "today", "told", "totally", "u", "uh", "umf", "unless", "upon",
        "using", "vo", "way", "well", "went", "whew", "whoa", "would", "wow", "x", "yeah", "yep",
        "yes", "yet", "z"]
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
        if len(l) > 0 and (l.isnumeric() == False) and l.lower() not in stops:
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

def create_wordcloud(df, category):
    '''
    Plot and show the wordcloud from a Pandas dataframe

    Inputs:
        df: Pandas dataframe
        category: "state", "androscogin", "aroostook", "cumberland", "franklin", 
                "hancock", "kennebec", "knox", "lincoln", "penobscot", "piscataquis", 
                "sagadahoc", "somerset", "waldo", "washington", "york"
    Returns:
        null
    '''
    # parse command line argument
    tfVectorizer = CountVectorizer()
    tfidfVectorizer = TfidfVectorizer(sublinear_tf=True)

    # sparse_matrix & feature_names defined here, and used below
    
    tf_sparse_matrix = tfVectorizer.fit_transform(df['sentence'])
    tfidf_sparse_matrix = tfidfVectorizer.fit_transform(df['sentence'])
    tf_feature_names = tfVectorizer.get_feature_names_out() 
    tfidf_feature_names = tfidfVectorizer.get_feature_names_out() 

    ### map mask
    img = np.array(Image.open("../figs/" + category + ".png"))
    img_mask = img.copy()
    img_mask[img_mask.sum(axis=2) == 0] = 255

    counties = {"aroostook": 1, "kennebec": 2, "cumberland":3, "knox":4, "lincoln":5, "somerset":8, "washington":10, "york": 11}
    
    tf = tf_sparse_matrix[counties[category],:].toarray()[0]
    tfidf = tfidf_sparse_matrix[counties[category],:].toarray()[0]

    tf_dict = dict(zip(tf_feature_names, tf))
    tfidf_dict = dict(zip(tfidf_feature_names, tfidf))

    tf_wordcloud = WordCloud(max_words=100, mask=img_mask, contour_width=3, contour_color='firebrick', background_color="white").generate_from_frequencies(tf_dict)
    tfidf_wordcloud = WordCloud(max_words=100, mask=img_mask, contour_width=3, contour_color='firebrick', background_color="white").generate_from_frequencies(tfidf_dict)

    fig, ax = plt.subplots(1,2,figsize=(24,5))
    ax[0].imshow(tf_wordcloud, interpolation='bilinear')
    ax[0].set_title('TF of ' + category)
    ax[0].axis("off")

    ax[1].imshow(tfidf_wordcloud, interpolation='bilinear')
    ax[1].set_title('TF-IDF of ' + category)
    ax[1].axis("off")

    plt.show()


def main(sys_argv):
    '''
    Main code to get conversations from Maine Ed Forum using the API
    '''
    # Enable CLI commands to get data
    if len(sys_argv) >= 2:

        # Get all conversation IDs
        conversation_ids = get_convo_ids()

        # Get all conversations included in the conversation_ids list
        all_conversations = get_conversations(conversation_ids)

        # Create a dataframe from the conversations list
        columns = ['speaker','facilitator', 'sentence', 'group', 'tags', 'title', 'conversation_id', 'county','participant_count']
        df = create_dataframe(all_conversations, columns)

        # Removing facilitator
        df = df[df.facilitator != True]

        # Coverting type of each cell in df['sentence'] from list to str
        # To apply, fit function later
        df['sentence'] = df['sentence'].apply(lambda x: ' '.join(x))

        # replacing subordianted county
        df = df.replace(['Augusta, Kennebec County, Maine', 'Presque Isle, Aroostook County, Maine'],['Kennebec County, Maine','Aroostook County, Maine'])

        # Processing dataframe according to sys_argv[1]
        if sys_argv[1] == 'counties':
            df = df.groupby('county').agg({'sentence': lambda x: ' '.join(x)}).reset_index()
        elif sys_argv[1] == 'groups':
            df = df.groupby('group').agg({'sentence': lambda x: ' '.join(x)}).reset_index()
        elif sys_argv[1] == 'state':
            pass
        elif sys_argv[1] == 'tags':
            pass
        else:
            print('\nERROR: Use the command -- python3 maine_ed.py <counties | groups | state>')
        
        


        # testing
        print("processed df")
        print(df)
        print(df.shape)

        # Action according to sys_argv[2]
        if sys_argv[3] == "bar":
            pass
        elif sys_argv[3] == "cloud":
            create_wordcloud(df, sys_argv[2]) # only work with countines at this moment have to implement for state
        else:
            pass

    else:
        print('\nERROR: Use the command -- python3 maine_ed.py <counties | groups | state>')

if __name__ == "__main__":
    main(sys.argv)
