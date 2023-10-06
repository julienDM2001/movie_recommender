import pandas as pd


# in docs/director, docs/score and docs/budget replace the tmdbId with the movieId get the link file to find the correspondance

link = pd.read_csv("data/small/content/links.csv")
score = pd.read_csv("docs/score.csv")

# replace the tmdbId with the corresponding movieId
score = score.merge(link, how='left', left_on='tmdbId', right_on='tmdbId')
score.drop(["tmdbId"],axis=1,inplace=True)
# if there is an absence of some movieId in the score file, add the movieid with the average score
for movieId in link["movieId"]:
    if movieId not in score["movieId"].values:
        score = score.append({"movieId":movieId,"score":score["score"].mean()},ignore_index=True)
# order the score file by movieId
score = score.sort_values(by=['movieId'])
print(score.info())
# drop the imdbId column
score.drop(["imdbId"],axis=1,inplace=True)
score.to_csv("docs/score.csv",index=False)



# do the same for the budget file
budget = pd.read_csv("docs/budget.csv")
budget = budget.merge(link, how='left', left_on='tmdbId', right_on='tmdbId')
budget.drop(["tmdbId"],axis=1,inplace=True)
for movieId in link["movieId"]:
    if movieId not in budget["movieId"].values:
        budget = budget.append({"movieId":movieId,"budget":budget["budget"].mean()},ignore_index=True)
budget = budget.sort_values(by=['movieId'])
print(budget.info())
budget.drop(["imdbId"],axis=1,inplace=True)
budget.to_csv("docs/budget.csv",index=False)



# do the same for the director file
director = pd.read_csv("docs/director.csv")
# drop the empty columns


director = director.merge(link, how='left', left_on='tmdbId', right_on='tmdbId')
director.drop(["tmdbId"],axis=1,inplace=True)
for movieId in link["movieId"]:
    if movieId not in director["movieId"].values:
        director = director.append({"movieId":movieId,"director":"unknown"},ignore_index=True)
director = director.sort_values(by=['movieId'])
# drop the imdbId column and the empty columns
director.drop(["empty"],axis=1,inplace=True)
director.drop(["imdbId"],axis=1,inplace=True)
print(director.info())
director.to_csv("docs/director.csv",index=False)