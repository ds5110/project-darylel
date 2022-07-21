import pandas as pd
import matplotlib.pyplot as plt

filename = "data/CDC/vaccinations-11-30-2021.csv"
df = pd.read_csv(filename)

x = "Series_Complete_18PlusPop_Pct"
y = "Census2019_18PlusPop"

df.plot.scatter(x, y)
plt.savefig("figs/fig1.png")
plt.show()
