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
nltk.download('stopwords')
nltk.download('omw-1.4')
import pandas as pd
load_dotenv()
import read_files 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from PIL import Image

### for choropleth map
# import geopandas as gpd
# import adjustText as aT


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
                sentence = preprocess(sentence, s['speaker_name'])
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

def preprocess(line, extra_stopwords):
    '''
    Create and preprocess a word list from file content

    Inputs:
        line: Sentence (string) to be preprocessed
        extra_stopwords: extra stopwords provided, e.g. speaker's name
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
    
    # Adding speaker's name
    if(len(extra_stopwords)>0):
        stops.append(extra_stopwords)
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

def create_plot(df, style, category):
    '''
    Plot and show the wordcloud from a Pandas dataframe

    Inputs:
        df: Pandas dataframe
        type: bar, cloud
        category: "maine", "counties", "androscogin", "aroostook", "cumberland", "franklin", 
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

    # current available category
    countiesDict = {"aroostook": 0, "cumberland":1, "kennebec": 2,  "knox":3, "lincoln":4, "somerset":6, "washington":8, "york": 9}
    if(category=='maine'):
        print(tf_feature_names)
        tf = tf_sparse_matrix[0,:].toarray()[0]
        tfidf = tfidf_sparse_matrix[0,:].toarray()[0]
    elif(category=='counties'):
        for county in countiesDict:
            create_plot(df, style, county)
    elif(category in countiesDict):
        tf = tf_sparse_matrix[countiesDict[category],:].toarray()[0]
        tfidf = tfidf_sparse_matrix[countiesDict[category],:].toarray()[0]

    if(category!='counties'):
        if(style == 'cloud'):
            ### map mask
            img = np.array(Image.open("../figs/" + category + ".png"))
            img_mask = img.copy()
            img_mask[img_mask.sum(axis=2) == 0] = 255

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
        elif(style=='bar'):
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
            ax[0].set_title('TF of ' + category)
            ax[0].set_xticklabels(tf_sorted_features[:10],rotation=45)

            ax[1].bar(tfidf_sorted_features[:10], sorted_tfidf[:10], 
                    width=1, alpha=.5, edgecolor='black')
            ax[1].set_title('TF-IDF of ' + category)
            ax[1].set_xticklabels(tfidf_sorted_features[:10],rotation=45)
        plt.show()

### Require geopandas installed
'''
def create_choro(df):
'''
'''
    Plot choropleth map categorized by counties associated with given Pandas data frame
    Input:
        df: Pandas data frame, e.g. data frame with features, county, participant number and number of conversation
    Return:
        None
'''
''' 
    # Within each county how many conversations have happened there and how many people total
    fp = "../ref/Maine_County_Boundary_Polygons_Dissolved_Feature.zip"
    map_df = gpd.read_file(fp)

    merged = map_df.set_index('COUNTY').join(df.set_index('county'))
    merged['title'] = merged['title'].fillna(0)

    # Here we find the center points, copy our original df to a new df, and then set 
    # the geometry column to the newly created center points column 
    # (because a GeoPandas df can only have one geometry column)
    merged["center"] = merged["geometry"].centroid
    merged_points = merged.copy()
    merged_points.set_geometry("center", inplace = True)

    # set a variable that will call whatever column we want to visualise on the map
    variable = 'title'
    # set the range for the choropleth
    vmin, vmax = 0, 10
    # create figure and axes for Matplotlib
    fig, ax = plt.subplots(1, figsize=(5, 12))
    # create map
    merged.plot(column=variable, cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8', label='')

    # Label county on map
    texts = []
    for x, y, label in zip(merged_points.geometry.x, merged_points.geometry.y, merged_points.index):
        texts.append(plt.text(x-len(label)/25, y, label, fontsize = 8))
    aT.adjust_text(texts, force_points=0.3, force_text=0.8, expand_points=(1,1), expand_text=(1,1))

    # remove the axis
    ax.axis('off')
    # add a title
    ax.set_title('Number of Conversation by county', fontdict={'fontsize': '25', 'fontweight' : '3'})
    # create an annotation for the data source
    # ax.annotate('Ed Maine Forum',xy=(0.1, .08),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=12, color='#555555')
    # Create colorbar as a legend
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # empty array for the data range
    sm._A = []
    # add the colorbar to the figure
    cbar = fig.colorbar(sm)
    plt.show()
'''

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

    # Removing facilitator
    df = df[df.facilitator != True]

    # Coverting type of each cell in df['sentence'] from list to str
    # To apply, fit function later
    df['sentence'] = df['sentence'].apply(lambda x: ' '.join(x))

    # replacing subordianted county
    df = df.replace(['Augusta, Kennebec County, Maine', 'Presque Isle, Aroostook County, Maine'],['Kennebec County, Maine','Aroostook County, Maine'])

    
    # Enable CLI commands to get data
    if (len(sys_argv) == 1):
        # df = df.groupby('county').agg({'title': pd.Series.nunique}).reset_index()
        dfgb = df.groupby(['county'])
        df1 = dfgb.agg({'title': pd.Series.nunique})
        df2 = dfgb.agg(lambda x: x.drop_duplicates('title', keep='first').participant_count.sum())
        df = pd.concat([df1, df2['participant_count']],1)
        
        # testing
        print("""\n############################################
# THE NUMBER OF CONVERSATIONS HAPPENED AND #
# THE NUMBER OF PARTICIPANT INVOLVED TOTAL #
# WITHIN EACH COUNTY                       #
############################################""")
        print(df)
    
    elif(len(sys_argv)>=2):
        countiesList = ["androscogin", "aroostook", "cumberland", "franklin", 
                "hancock", "kennebec", "knox", "lincoln", "penobscot", "piscataquis", 
                "sagadahoc", "somerset", "waldo", "washington", "york"]
        groupsList = []

        # Processing dataframe according to sys_argv[1]
        if (sys_argv[1] == 'help'):
            print("""
            You can get bar chart or wordcloud of important word within categories using data by Ed Maine Forums from LVN API
            $python3 maine_ed.py
            : print help\n
            $python3 maine_ed.py
            : print out the number of conversations happened and the number of participant involved total within each county\n
            $python3 maine_ed.py <counties | groups | tags | maine |  name of county | name of group> <bar | cloud>
            : counties: plot <bar | cloud> for every county in Maine if it exist in the dataframe
            : groups: not implemented yet, (plot <bar | cloud> for every group)
            : tags: not implemented yet
            : maine: plot <bar | cloud> of most frequent word from every conversation
            : name of county: plot <bar | cloud> for specified county if it exist in the dataframe
            : name of group: not implemented yet, (plot <bar | cloud> for specified group)
            """)
            return
        elif sys_argv[1] == 'counties' or sys_argv[1] in countiesList:
            df = df.groupby('county').agg({'sentence': lambda x: ' '.join(x)}).reset_index()
        elif sys_argv[1] == 'groups'or sys_argv[1] in groupsList:
            df = df.groupby('group').agg({'sentence': lambda x: ' '.join(x)}).reset_index()
        elif sys_argv[1] == 'maine':
            df = pd.DataFrame({'sentence': [' '.join(df['sentence'].tolist())]})
        elif sys_argv[1] == 'tags':
            pass
        else:
            print('''\nERROR: FLAG NOT FOUND
            Use the command 
            $python3 maine_ed.py <counties | groups | tags | maine |  name of county | name of group> <bar | cloud>
            ''')
        
        # testing
        print("processed df")
        print(df)
        print(df.shape)

        # Action according to sys_argv[2]
        if ((len(sys_argv) > 2) and (sys_argv[2] in ['bar','cloud'])):
            create_plot(df, sys_argv[2], sys_argv[1])
        else:
            print('''\nERROR: MISSING FLAG
            Need to provide <bar | cloud>. Use the command
            $python3 maine_ed.py <counties | groups | tags | maine |  name of county | name of group> <bar | cloud>''')    
    else:
        print('''\nERROR: TOO MANY FLAG
        Use the command
        $python3 maine_ed.py <counties | groups | tags | maine |  name of county | name of group> <bar | cloud>''')

if __name__ == "__main__":
    main(sys.argv)
