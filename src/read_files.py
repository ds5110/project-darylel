'''
Gungyeom (James) Kim & Daryle Lamoureux 
DS5110 Summer 2022
Final Project

This project consumes text documents from interviews conducted by the Maine Ed Forum.
It then analyzes and visualizes the data from those files.
'''

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('wordnet')
import re
import time
import matplotlib.pyplot as plt
import pandas as pd

def read_file(doc):
    '''
    Read a file

    Arugments:
        doc: The name of the document, including the suffix (i.e. .txt) to be read
    Returns:
        lines: A list of each line read from the document
    '''
    # Read file
    with open('../data/Maine_Ed_2050/' + doc, encoding="utf-8") as f:
        lines = f.readlines()

    return lines

def enhance_data(lines, participants, doc):
    '''
    Add the participant's county and the name of the file to the dataset

    Arguments:
        lines: List of lines read from the file
        participants: Dictionary with participant's name as key and their county as the value
        doc: the name of the file being read
    Returns:
        x: List of lists with each sublist containing [participant name, question/answer, participant county, file name]
    '''
    x = []
    # Parse the lines list and county data to each sublist
    for value in lines:
        if value[0] in participants.keys():
            x.append(value + [participants[value[0]]] + [doc])

    return x

def create_preprocess(lines):
    '''
    Create and preprocess a word list from file content
    '''
    # Create set of English stopwords from nltk.corpus
    stops = set(stopwords.words('english'))

    # Initialize WordNet Lemmatizer to lemmatize the word corpus
    wnl = WordNetLemmatizer()

    # Parse the document and create a wordlist
    lines_cleaned = []
    for line in lines:
        word_list = []
        # Remove the notes in []
        line[1] = re.sub(r'\[.*?\]', '', line[1])
        # Remove punctuation
        line[1] = re.sub(r'[^\w\s]', ' ', line[1])

        # Add each word/token to the word list
        line1 = line[1].split()
        for l in line1:
            if len(l) > 0 and l.lower() not in stops:
                #word_list.append(ps.stem(l))
                word_list.append(wnl.lemmatize(l).lower())

        line[1] = word_list
        lines_cleaned.append(line)

    return lines_cleaned

def unique_list(words):
    '''
    Take a list of words/tokens and return a list of unique words

    Arguments:
        words: List of words
    Returns:
        unique_list = list of a set of unique words
    '''
    # Create a the list of unique words in the corpus
    unique_list = list(set(words))

    return unique_list

def freq(unique, words):
    '''
    Create a frequency mapping {word: frequency}

    Arguments:
        unique: List of unique words in the document(s)
        words: List of all words in the document(s)
    Results: 
        frequency: Dictionary with the unique word as the key and the frequency count as the value
    '''
    # Create word frequency mapping
    frequency = {}
    for u in unique:
        count = words.count(u)
        if u not in frequency.keys():
            frequency[u] = count

    return frequency

def reformat_text(lines):
    '''
    Remove the formatting from the file to keep answers and questions 'whole'

    Arguments:
        lines: List of lists with the lines read from the file
    Returns:
        x_formatted: Lists of lists where the sublist is made of participant name and question/answer
    '''
    x = []
    x_formatted = []
    # If a line starts with a space/spaces, append it to the previous line
    for l in lines:
        if l[0] != ' ':
            x.append(re.sub(' {2,}', ' ', l))
        else:
            x[-1] = x[-1].strip() + re.sub(' {2,}', ' ', l)

    # Split question/answer from speaker and add to the list
    for line in x:
        if line != '\n':
            x_formatted.append(line.split(': '))

    return x_formatted

def sub(txts, title):
    total_lines = []
    for text in txts:
        lines = read_file(text)
        total_lines = total_lines + reformat_text(lines)

    word_list = create_preprocess(total_lines)

    vocabulary = unique_list(word_list)

    frequency = freq(vocabulary, word_list)

    df = pd.DataFrame.from_dict({'word': frequency.keys(), 'frequency': frequency.values()})
    # print(df.head())

    # print(df.sort_values('frequency', ascending=False).head(50))

    # t = time.time()
    # # Write word frequencies to file
    # df.sort_values('frequency', ascending=False).to_csv('file_' + title + '.csv', sep=',')
    


def main():
    '''
    Main function
    '''
    # Lists of cohorts/sessions with corresponding documents
    educate = ['Educate_Maine_2021_Symposium.txt']
    farmers = ['Maine_Farmers_ 012722.txt', 'Maine_Farmers_020122.txt', 'Maine_Farmers_020922.txt']
    kennebec = ['Kennebec_County_Educators.txt', 'Kennebec_County_Community_020122.txt']
    presque_isle = ['University_of_Maine_Presque_Isle_Pre-Service_Teachers_2.txt', 'University_of_Maine_Presque_Isle_Pre-Service_Teachers_1.txt']
    community = ['Community_Caring_Collaborative_1.txt', 'Community_Caring_Collaborative_2.txt']
    all = ['Educate_Maine_2021_Symposium.txt', 'Maine_Farmers_ 012722.txt', 'Maine_Farmers_020122.txt', 'Maine_Farmers_020922.txt', 'Kennebec_County_Educators.txt', 'Kennebec_County_Community_020122.txt', 'University_of_Maine_Presque_Isle_Pre-Service_Teachers_2.txt', 'University_of_Maine_Presque_Isle_Pre-Service_Teachers_1.txt', 'Community_Caring_Collaborative_1.txt', 'Community_Caring_Collaborative_2.txt']

    # List of all Maine counties
    counties = ['Androscoggin', 'Aroostook', 'Cumberland', 'Franklin', 'Hancock', 'Kennebec', 'Knox', 'Lincoln', 'Oxford', 'Penobscot', 'Piscataquis', 'Sagadahoc', 'Somerset', 'Waldo', 'Washington', 'York']
    # Dictionary of participants and their counties
    participants = {'Abby': 'Washington', 'Corey': 'Washington', 'Julie': 'Washington', 'Jane': 'Washington', 'Anna': 'Washington',
        'Mandy': 'Washington', 'Dante': 'Washington', 'Charlie': 'Washington', 'Jason': 'unk', 'Jennifer': 'unk', 'Nancy': 'unk', 
        'Heather': 'unk', 'Rob': 'unk', 'Kelsey': 'unk', 'Patty': 'unk', 'Kim': 'unk', 'Shelly': 'unk', 'Katie': 'unk', 'Doris': 'unk',
        'Jackie': 'Kennebec', 'Faye': 'Kennebec', 'Mark': 'Kennebec', 'Amanda': 'Kennebec', 'Lindsay': 'Kennebec', 'Tanya': 'Kennebec',
        'Emmanuel': 'Kennebec', 'Emily': 'Aroostook', 'Donna': 'York', 'Seren': 'Androscoggin', 'Steve': 'Androscoggin',
        'Nick King': 'Cumberland', 'Nick': 'Cumberland', 'Rhiannon Hampso...': 'Knox', 'Rhiannon': 'Knox', 'Seth Kroeck': 'Cumberland',
        'Seth': 'Cumberland', 'Christian Brayd...': 'Cumberland', 'Christian': 'Cumberland', 'Jean': 'York', 'Frank': 'York', 'Lexi': 'unk',
        'Logan': 'unk', 'Elena': 'unk', 'Taylor': 'unk', 'Reece': 'unk', 'Caitlyn': 'Hancock',
        'Keana': 'Aroostook', 'Madison': 'Aroostook', 'Twyla': 'Washington'}

    # Parse all document (change 'all' to one of the cohorts above or perform this on a single document)
    results = []
    for doc in all:
        lines = read_file(doc)
        lines = reformat_text(lines)
        lines = enhance_data(lines, participants, doc)
        lines = create_preprocess(lines)
        results += lines

    # Make the dataframe
    d = {'name':[line[0] for line in results], 
        'text':[str(line[1]) for line in results], 
        'county':[line[2] for line in results], 
        'doc':[line[3] for line in results]}
    df = pd.DataFrame(data = d)

    # Get all words in results
    words = []
    for line in results:
        # print(line)
        words += line[1]

    # Create a list of unique terms in words
    vocab = sorted(unique_list(words))

    # Print length of words and length of vocab
    print(len(words))
    print(len(vocab))

    # Print frequency dictionary
    #print(freq(vocab, words))
    return df

if __name__ == '__main__':
    main()
