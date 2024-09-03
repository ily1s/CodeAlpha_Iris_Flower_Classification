import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


iris_data = pd.read_csv("TASK1/iris.csv")

iris_data.drop("Id", axis=1 ,inplace= True)

# print(iris_data.head())

sns.pairplot(iris_data, hue="Species", size=3)
plt.show()
