import pandas as pd
from constants import Constant as C
import numpy as np
import matplotlib.pyplot as plt

user_based = pd.read_csv(C.EVALUATION_PATH / "evaluation_20230524.csv")
# plot the rmse hit rate and precision in 3 different graphs
# rmse

rmse = user_based[["index", "rmse"]]

rmse.plot(x="index", y="rmse", kind="bar")
plt.title("RMSE of diiferent models", fontsize=18)

plt.xlabel("model",fontsize=18, color="black")
plt.xticks(rotation=0,fontsize=18, color="black",style="oblique")
# range the y axis from 0.9 to 1.1 to make the difference more obvious and add data labels
plt.ylim(1, 1.05,0.005)
plt.yticks(fontsize=18, color="black")
plt.ylabel("RMSE",fontsize=18, color="black")
plt.legend(loc="best")

plt.show()
# hit rate

hit_rate = user_based[["index", "hit rate"]]
hit_rate.plot(x = "index",y= "hit rate",kind="bar")
plt.title("Hit Rate of different models",fontsize=18, color="black")
plt.xlabel("model",fontsize=18, color="black")
plt.xticks(rotation=0,fontsize=18, color="black",style="oblique")
plt.ylabel("hit rate",fontsize=18, color="black")
plt.yticks(fontsize=18, color="black")
plt.legend(loc="best")
plt.show()
# precision

precision = user_based[["index", "precision"]]

precision.plot(x = "index",y= "precision",kind="bar")
plt.xticks(rotation=0,fontsize=18, color="black",style="oblique")

plt.title("Precision of different models",fontsize=16, color="black")
plt.xlabel("model",fontsize=18, color="black")
plt.ylabel("precision",fontsize=18, color="black")
plt.yticks(fontsize=18, color="black")
plt.legend(loc="best")
plt.show()
