import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_community = pd.read_csv("./csv_james/file_community.csv")
df_educate = pd.read_csv("./csv_james/file_educate.csv")
df_farmers = pd.read_csv("./csv_james/file_farmers.csv")
df_kennebec = pd.read_csv("./csv_james/file_kennebec.csv")
df_presque_isle = pd.read_csv("./csv_james/file_presque_isle.csv")
print(df_community.head())
print(df_educate.head())
print(df_farmers.head())
print(df_kennebec.head())
print(df_presque_isle.head())

fig, ax = plt.subplots(nrows = 1, ncols = 5, figsize=(15,10))
for i, df, title in zip(range(0,5), [df_community, df_educate, df_farmers, df_kennebec, df_presque_isle], ['community', 'educate', 'farmers', 'kennebec', 'presque_isle']):
    ax[i].bar(df['word'][:5],df['frequency'][:5])
    ax[i].set_xticklabels(df['word'][:5], rotation = 45)
    ax[i].set_ylim(0,350)
    ax[i].set_title(title)

plt.show()