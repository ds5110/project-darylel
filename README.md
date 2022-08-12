# DS5110 Final Project: Education ACTION Forum of Maine

* Stakeholder: Jennifer Chace (Executive Director)
* Contributors: James Kim and Daryle Lamoureux (The Roux Institute at Northeastern University)

## Disparities Identified in Language Usage in Maine Counties
Conversations about education throughout the state of Maine show the thoughts of concerns among citizens of the state definitely vary from county to county. When looking at the most common words (indicated by TF in the images below), there is overlap but there are also differences in the words encountered in those conversations. The disparities between the counties is much more noticeable in the conversations that focus on less commonly used words (indicated by TF-IDF in the images below) in these conversations. For example, Cumberland County participants used words like prestigious, musical, community, and church, while participants in Lincoln County used words like lobsterman, prevention, stability, troubling, and fundamental. This data could likely be correlated by layering in socioeconmic data as well as graduation rates and other data in these realms to identify possible trends related to the supplementary data.

<p align="center">
  <img src = "./figs/aroostok-cropped.png" width=600 />
</p>
<p align="center">
  <img src = "./figs/cumberland-cropped.png" width=600 />
</p>
<p align="center">
  <img src = "./figs/kennebec-cropped.png" width=600 />
</p>
<p align="center">
  <img src = "./figs/knox-cropped.png" width=600 />
</p>
<p align="center">
  <img src = "./figs/lincoln-cropped.png" width=600 />
</p>
<p align="center">
  <img src = "./figs/somerset_edited.png" width=600 />
</p>
<p align="center">
  <img src = "./figs/washington-cropped.png" width=600 />
</p>
<p align="center">
  <img src = "./figs/york-cropped.png" width=600 />
</p>

## Words Matter
A word may only be spoken once or twice during a lengthy conversation. Oftentimes this results in the word being disregarded. Words that are spoken the least during a conversation, however, may be critical to understanding the subtext or underlying tone of a conversation. This is even more powerful when looking at these less spoken words all together within the context of a conversation. This method used in this project looked at both the words that were spoken the most during a conversation and words that were spoken less often. The analysis of the two different methods side by side in some cases demonstrates what was stated previously -- the less spoken words can paint a very different picture of the conversation that took place. As an example, below is the analysis of Somerset County, which is a rural, northern county in Maine.

<p align="center">
  <img src = "./figs/somerset_edited.png" width=600 />
</p>

The analysis on the left of the most common words shows words that would be expected in a conversation about education, such as school, education, teacher, and community. The analysis on the right of the least common words shows the struggles that the community is experiencing and its impact on education. The words include: pregnant, anxiety, divorce, tension, recover, survival, and resilience. By examining the least common words used, this could provide insight for follow up conversations within Somerset and other counties.

#### Meter

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
