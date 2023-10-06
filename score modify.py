import pandas as pd
from loaders import *
df = pd.read_csv("docs/score.csv")
df2 = load_year2()
print(df2.info())

# add the year column to the score dataframe need 9125 rows
df['year'] = df2['year']
print(df.info())
# make year column an integer
df['year'] = df['year'].astype(int)
# EXPORT IN SCORE2 CSV df
df.to_csv("docs/score2.csv",index=False)
