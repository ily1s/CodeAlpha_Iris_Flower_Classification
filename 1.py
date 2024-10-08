import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score ,classification_report, confusion_matrix


df = pd.read_csv("TASK1/iris.csv")


X = df.drop("Species", axis=1)
y = df["Species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

# # Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=123)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Plot the Decision Tree
from sklearn.tree import plot_tree

plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=df["Species"].unique())
plt.title("Decision Tree")
plt.show()


# Decision Tree Insights
#       ** Iris-setosa is perfectly classified by 'petal length'<=2.45
#       **Iris-versicolor - Most instances are correctly classified, with some futher splits based on 'sepal length' and 'petal width'
#       **Iris-virginica is correctly classified with some splits, but with some futher splits misclassified as Iris-versicolor


# Evaluate the model
confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion_mat)


# Confusion Matrix
# True Positives(TP): Diagonal elements, correctly classified for each class
#         for class 0:13
#         for class 1:6
#         for class 2:10

# False Positives(FP): The off-diagonal elements in each column, incorrectly predicted as that class
#        for class 0: 0,0
#         for class 1: 0,1
#         for class 2: 0,0

# False Negatives(FN): The off-diagonal elements in each row, that belong to that class but were predicted as some other class
#         for class 0: 0,0
#         for class 1: 0,0
#         for class 2: 0,1


# Accuracy= total true values/total instances
# Precision= true positives/true positives + false positives
# Recall = true positives/ true positives + false negatives
# F1 Score = 2(Precision*Recall/Precision + Recall)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# # Checking
p = {
    "sepal_length": ["4.7"],
    "sepal_width": ["3.2"],
    "petal_length": ["1.3"],
    "petal_width": ["0.2"],
}

dr = pd.DataFrame(p)

# Make predictions
y_pred = clf.predict(dr)
print("prediction DT: ", y_pred)



# # Logistic Regression
from sklearn.linear_model import LogisticRegression

# Create the logistic regression model
logreg = LogisticRegression(max_iter=200)

# Train the model
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Example new data point
new_data = [[5.1, 3.5, 1.4, 0.2]]

# Predict the class of the new data point
prediction = logreg.predict(new_data)
print("prediction LR: ", prediction)



# # SVM
from sklearn.svm import SVC

# Create the SVM classifier
svm_model = SVC(kernel="linear", random_state=42)

# Train the model
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Example new data point
new_data = [[5.1, 3.5, 1.4, 0.2]]

# Predict the class of the new data point
prediction = svm_model.predict(new_data)
print("prediction SVM: ", prediction)


# # KNN
from sklearn.neighbors import KNeighborsClassifier

# Create the k-NN classifier with k=3 (number of neighbors)
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Example new data point
new_data = [[5.1, 3.5, 1.4, 0.2]]

# Predict the class of the new data point
prediction = knn.predict(new_data)
print("prediction KNN: ", prediction)



# # Random Forest
from sklearn.ensemble import RandomForestClassifier

# Create the Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Example new data point
new_data = [[5.1, 3.5, 1.4, 0.2]]

# Predict the class of the new data point
prediction = rf_model.predict(new_data)
print("prediction RF: ", prediction)
