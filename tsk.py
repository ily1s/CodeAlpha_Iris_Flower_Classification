import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


iris_data = pd.read_csv("TASK1/iris.csv")

iris_data.drop("Id", axis=1 ,inplace= True)

# print(iris_data.head())

sns.pairplot(iris_data, hue="Species", size=3)
plt.show()