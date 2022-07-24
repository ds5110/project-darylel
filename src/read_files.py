from operator import index
import re
import time
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt

def read_file(doc):
    '''
    Read a file
    '''
    # Read file
    with open('./Maine_Ed_2050/' + doc, encoding="utf-8") as f:
        lines = f.readlines()

    return lines

def create_preprocess(lines):
    '''
    Create and preprocess a word list from file content
    '''
    # List of interviewer and participant names
    participants = ['Abby', 'Alana', 'Amanda', 'Anna', 'Brayd', 'Caitlyn', 'Charlie', 'Chase', 'Christian', 'Corey', \
        'Dante', 'Donna', 'Doris', 'Elena', 'Emmanuel', 'Emily', 'Faye', 'Frank', 'Hampso', 'Heather', 'Jackie', 'Jane', \
        'Jason', 'Jean', 'Jennifer', 'Julie', 'Katie', 'Keana', 'Kelsey', 'Kim', 'King', 'Kroeck', 'Lexi', 'Lindsay', \
        'Logan', 'Madison', 'Mandy', 'Mark', 'Nancy', 'Nick', 'Patty', 'Reece', 'Rhiannon', 'Rob', 'Sarah', 'Seren', 'Seth', \
        'Shelly', 'Steve', 'Tanya', 'Taylor', 'Twyla']

    # Create set of English stopwords from nltk.corpus
    stops = set(stopwords.words('english'))

    # Initialize Porter Stemmer to stem the word corpus
    ps = PorterStemmer()

    # Parse the document and create a wordlist
    word_list = []
    for line in lines:
        # Remove the notes in []
        line = re.sub(r'\[.*?\]', '', line)
        # Remove punctuation
        line = re.sub(r'[^\w\s]', ' ', line)

        # Add each word/token to the word list
        line = line.split()
        for l in line:
            if len(l) > 0 and l.lower() not in stops and l not in participants:
                word_list.append(ps.stem(l))

    return word_list

def unique_list(words):
    '''
    Take a list of words/tokens and return a list of unique words
    '''
    # Create a the list of unique words in the corpus
    unique_list = list(set(words))

    return unique_list

def freq(unique, words):
    '''
    Create a frequency mapping {word: frequency}
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
    '''
    x = []
    # If a line starts with a space/spaces, append it to the previous line
    for i in range(len(lines)):
        if lines[i][0] != ' ':
            x.append(re.sub(' {2,}', ' ', lines[i]))
        else:
            x[-1] = x[-1].strip() + re.sub(' {2,}', ' ', lines[i])

    return x

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
    educate = ['./data/Maine_Ed_2050/Educate_Maine_2021_Symposium.txt']
    farmers = ['Maine_Farmers_ 012722.txt', 'Maine_Farmers_020122.txt', 'Maine_Farmers_020922.txt']
    kennebec = ['Kennebec_County_Educators.txt', 'Kennebec_County_Community_020122.txt']
    presque_isle = ['University_of_Maine_Presque_Isle_Pre-Service_Teachers_2.txt', 'University_of_Maine_Presque_Isle_Pre-Service_Teachers_1.txt']
    community = ['Community_Caring_Collaborative_1.txt', 'Community_Caring_Collaborative_2.txt']

    txtss = [educate, farmers, kennebec, presque_isle, community]
    lines = read_file(educate[0])
    lines = reformat_text(lines)
    print(lines)

    #print(lines)

    
    #for txts, title in zip(txtss, ["educate", "farmers", "kennebec", "presque_isle", "community"]):
        #sub(txts, title)

if __name__ == '__main__':
    main()
