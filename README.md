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

Code base: Python using the following libraries: os, dotenv, json, re, requests, nltk, pandas, matplotlib, wordcloud, sklearn

