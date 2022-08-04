'''
Gungyeom (James) Kim & Daryle Lamoureux 
DS5110 Summer 2022
Final Project

This project consumes text documents from interviews conducted by the Maine Ed Forum.
It then analyzes and visualizes the data from those files.
'''
import os
from dotenv import load_dotenv
import json
import re
import time
import requests
from requests.structures import CaseInsensitiveDict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
import pandas as pd
load_dotenv()

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
    #ids = [1722]

    for x in ids:
        # Make the API call for the conversation with the given conversation ID
        resp = make_call('conversations/' + str(x), {})

        if resp is not None:
            for s in resp.get('snippets'):
                # Initialize conversation list
                conversation = []

                # Recreate each conversation and add to the conversations list
                conversation.append(s['speaker_name'])
                conversation.append(s['is_facilitator'])
                # Consolidate all the words
                temp = list()
                for word in s['words']:
                    temp.append(word[0])
                sentence = ' '.join(temp)
                sentence = preprocess(sentence)
                conversation.append(sentence)
                conversation.append(resp.get('title'))
                conversation.append(x)
                conversation.append(resp.get('location').get('name'))
                conversations.append(conversation)

        # Wait before making the next call to prevent rate limits in the API
        time.sleep(10)

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
    #ps = PorterStemmer()

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
            #word_list.append(ps.stem(l))
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

def main():
    '''
    Main code to get conversations from Maine Ed Forum using the API
    '''
    # Get all conversation IDs
    conversation_ids = get_convo_ids()

    # Get all conversations included in the conversation_ids list
    all_conversations = get_conversations(conversation_ids)

    # Create a dataframe from the conversations list
    columns = ['speaker','facilitator', 'sentence', 'title', 'conversation_id', 'county']
    df = create_dataframe(all_conversations, columns)

    print(df.head())
    print(df.shape)

if __name__ == "__main__":
    main()
