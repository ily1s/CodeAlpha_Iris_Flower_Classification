import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score


# Load the data
data = pd.read_csv("TASK1/iris.csv")

# Drop the 'Id' column as it's not needed
data = data.drop(columns=["Id"])

# Split the data into features (X) and target (y)
x = data.drop("Species", axis=1)
y = data["Species"]

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Initialize the KNN classifier with 1 neighbor
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(x_train, y_train)

# Calculate the accuracy on the test set
accuracy = knn.score(x_test, y_test)
print("Model Accuracy: {:.2f}%".format(accuracy * 100))

y_pred = knn.predict(x_test)
print("Accuracy:" ,accuracy_score(y_test, y_pred) * 100)

print("classification report:" ,classification_report(y_test, y_pred))

# Make a prediction for a new sample
x_new = pd.DataFrame([[5.9, 3, 5.1, 1.8]], columns=x.columns)
prediction = knn.predict(x_new)
print("Prediction for new sample: {}".format(prediction))
