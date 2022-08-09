# DS5110 Final Project: Education ACTION Forum of Maine

* Stakeholder: Jennifer Chace (Executive Director)
* Contributors: James Kim and Daryle Lamoureux (The Roux Institute at Northeastern University)

## Overview
The Education ACTION Forum of Maine (Maine Ed Forum) has been conducting interviews across the state among youth, educators, community members and representatives from the state's industries. The goal of these conversations is to find insight and trends by groups, by county, etc. to understand better the concerns of the community in regards to education and education-related topics.

## Methodology
This project analyzes the conversations that have taken place to date by accessing these conversations using the Maine Ed Forum's API. The analysis performed data analysis to identify key terms and trends within each conversation, whether by county, group or otherwise. This was primarly done using traditional word count as well as TF/IDF to see if there are differences in the most common words as well as the last common words encountered in these conversations.

## Results
### EDA: one or more figures sufficient to assess project feasibility

![fig1](/figs/Figure_1.png)

#### Description

Simple bar chart of frequency of 5 most occurring words in each text file.


## Technical Implementation
Data Source: API provided by Maine Ed Forum to access data in real-time

Code base: Python using the following libraries: os, dotenv, re, requests, nltk, pandas, matplotlib, wordcloud, sklearn

Launching the program:
* python3 maine_ed.py counties
* python3 maine_ed.py state
* python3 maine_ed.py groups
* python3 maine_ed.py tags

Functions within the program include:
* make_call(): This is a reusable function that authenticates the API and makes the calls
* get_convo_ides(): This gets all the conversation IDs for Maine Ed Forum conversations
* get_conversations(): Gets the conversations and parses the data based on the conversation ID passed to it
* preprocess(): This preprocesses the ID for natural language processing and analysis
* create_dataframe(): This converts the conversation data into a Pandas dataframe
* create_wordcloud(): This creates a Wordcloud plot of for the category (county, etc.) that is passed to the function

Data Preprocessing:
The data is cleaned by removing punctuation and timestamps in the text found as [43:20], for example. Stopwords are then removed using the stopwords from nltk.corpus. The stopwords are extended using .extend() to exclude additional words deemed not to have meaningful impact to the analysis. This list could be commented out and/or edited in the future. Finally, the words are lemmatized using WordNetLemmatizer from nltk.stem.

Data Analysis
The data was analyzed using both CountVectorizer and TfIdfVectorizer from sklearn. Data is plotted using both matplotlib.pyplot and Wordcloud.
