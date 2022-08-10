# DS5110 Final Project: Education ACTION Forum of Maine

* Stakeholder: Jennifer Chace (Executive Director)
* Contributors: James Kim and Daryle Lamoureux (The Roux Institute at Northeastern University)

## Disparities Identified in Language Usage in Maine Counties
The Education ACTION Forum of Maine (Maine Ed Forum) has been conducting interviews across the state among youth, educators, community members and representatives from the state's industries. The goal of these conversations is to find insight and trends by groups, by county, etc. to understand better the concerns of the community in regards to education and education-related topics.

## Methodology
This project analyzes the conversations that have taken place to date by accessing these conversations using the Maine Ed Forum's API. The analysis performed data analysis to identify key terms and trends within each conversation, whether by county, group or otherwise. This was primarly done using traditional word count as well as TF/IDF to see if there are differences in the most common words as well as the last common words encountered in these conversations.

## Results
### EDA: one or more figures sufficient to assess project feasibility

![fig1](/figs/Figure_1.png)

#### Description

Simple bar chart of frequency of 5 most occurring words in each text file.


## Technical Implementation
#### Data Source: 

API provided by Maine Ed Forum to access data in real-time

#### Code base: 

Python using the following libraries: os, dotenv, re, requests, nltk, pandas, matplotlib, wordcloud, sklearn

#### Launching the program:
```
python3 maine_ed.py help
python3 maine_ed.py
python3 maine_ed.py counties
python3 maine_ed.py state
python3 maine_ed.py groups
python3 maine_ed.py tags
python3 maine_ed.py (name of county)
```

#### Functions within the program include:
* make_call(): This is a reusable function that authenticates the API and makes the calls
* get_convo_ids(): This gets all the conversation IDs for Maine Ed Forum conversations
* get_conversations(): Gets the conversations and parses the data based on the conversation ID passed to it
* preprocess(): This preprocesses the ID for natural language processing and analysis
* create_dataframe(): This converts the conversation data into a Pandas dataframe
* create_plot(): This plot a bar chart or a wordcloud of given Pandas dataframe 

#### Data Preprocessing:

The data is cleaned by removing punctuation and timestamps in the text found as [43:20], for example. Stopwords are then removed using the stopwords from nltk.corpus. The stopwords are extended using .extend() to exclude additional words deemed not to have meaningful impact to the analysis. This list could be commented out and/or edited in the future. Finally, the words are lemmatized using WordNetLemmatizer from nltk.stem.

#### Data Analysis

The data was analyzed using both CountVectorizer and TfIdfVectorizer from sklearn. Data is plotted using both matplotlib.pyplot and Wordcloud.

## Example
#### Manual
```
You can get bar chart or wordcloud of important word within categories using data by Ed Maine Forums from LVN API
$python3 maine_ed.py
: print help

$python3 maine_ed.py
: print out the number of conversations happened and the number of participant involved total within each county

$python3 maine_ed.py <counties | groups | tags | maine |  name of county | name of group> <bar | cloud>
: counties: plot <bar | cloud> for every county in Maine if it exist in the dataframe
: groups: not implemented yet, (plot <bar | cloud> for every group)
: tags: not implemented yet
: maine: plot <bar | cloud> of most frequent word from every conversation
: name of county: plot <bar | cloud> for specified county if it exist in the dataframe
: name of group: not implemented yet, (plot <bar | cloud> for specified group)
```
Output of below command:
```
$python3 maine_ed.py help
```
#### Data from API
```
          speaker  facilitator                                           sentence  ... conversation_id                  county participant_count
0           Jason         True  [jennifer, gone, ahead, started, recording, ed...  ...            1722  Kennebec County, Maine                 6
1           Jason         True  [end, inviting, different, type, conversation,...  ...            1722  Kennebec County, Maine                 6
2           Jason         True  [participant, conversation, providing, consent...  ...            1722  Kennebec County, Maine                 6
3          Amanda        False                                           [agreed]  ...            1722  Kennebec County, Maine                 6
4         Lindsay        False                                           [agreed]  ...            1722  Kennebec County, Maine                 6
...           ...          ...                                                ...  ...             ...                     ...               ...
6022  Interviewer         True  [receive, invitation, join, next, week, two, w...  ...            2131                 Unknown                 4
6023      Larissa        False                                                 []  ...            2131                 Unknown                 4
6024       Kelsey        False                                         [bye, bye]  ...            2131                 Unknown                 4
6025      Larissa        False                                           [thanks]  ...            2131                 Unknown                 4
6026  Interviewer         True                                            [thank]  ...            2131                 Unknown                 4
```

#### The number of conversations happened and particiapnt invovled total within county

```
                                    title  participant_count
county
Aroostook County, Maine                 3                 14
Bridgton, Cumberland County, Maine      1                  4
Kennebec County, Maine                  5                 22
Knox County, Maine                      1                  4
Lincoln County, Maine                   3                 13
Maine                                   8                 45
Somerset County, Maine                  4                 16
Unknown                                 1                  4
Washington County, Maine                4                 16
York County, Maine                      1                  3
```
Output of below command:
```
$python3 maine_ed.py
```

### Plot of Maine
![fig2](/figs/maine_bar.png)
Output of below command:
```
$python3 maine_ed.py maine bar
```
![fig3](/figs/maine_cloud.png)
Output of below command:
```
$python3 maine_ed.py maine cloud
```
For Maine, TF and TF-IDF are same.

### Plot of Counties
You can get bar chart or wordcloud of every county that is in data by:
```
$python3 maine_ed.py counties bar
```
or
```
$python3 maine_ed.py counties cloud
```

#### Aroostook
![fig4](/figs/aroostook_bar.png)
Output of below command:
```
$python3 maine_ed.py aroostook bar
```
![fig5](/figs/aroostook_cloud.png)
Output of below command:
```
$python3 maine_ed.py aroostook cloud
```

#### Cumberland
![fig6](/figs/cumberland_bar.png)
Output of below command:
```
$python3 maine_ed.py cumberland bar
```
![fig7](/figs/cumberland_cloud.png)
Output of below command:
```
$python3 maine_ed.py cumberland cloud
```

#### Kennebec
![fig8](/figs/kennebec_bar.png)
Output of below command:
```
$python3 maine_ed.py kennebec bar
```
![fig9](/figs/kennebec_cloud.png)
Output of below command:
```
$python3 maine_ed.py kennebec cloud
```

#### Knox
![fig10](/figs/knox_bar.png)
Output of below command:
```
$python3 maine_ed.py knox bar
```
![fig11](/figs/knox_cloud.png)
Output of below command:
```
$python3 maine_ed.py knox cloud
```

#### Lincoln
![fig12](/figs/lincoln_bar.png)
Output of below command:
```
$python3 maine_ed.py lincoln bar
```
![fig13](/figs/lincoln_cloud.png)
Output of below command:
```
$python3 maine_ed.py lincoln cloud
```

#### Somerset
![fig14](/figs/somerset_bar.png)
Output of below command:
```
$python3 maine_ed.py somerset bar
```
![fig15](/figs/somerset_cloud.png)
Output of below command:
```
$python3 maine_ed.py somerset cloud
```

#### Washington
![fig16](/figs/washington_bar.png)
Output of below command:
```
$python3 maine_ed.py washington bar
```
![fig17](/figs/washington_cloud.png)
Output of below command:
```
$python3 maine_ed.py washington cloud
```

#### York
![fig18](/figs/york_bar.png)
Output of below command:
```
$python3 maine_ed.py york bar
```
![fig19](/figs/york_cloud.png)
Output of below command:
```
$python3 maine_ed.py york cloud
```
