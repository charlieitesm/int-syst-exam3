import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

gnb = GaussianNB()

# Read the training dataset
training_data = pd.read_csv("../Jupyter/wine.csv", skipinitialspace=True, skip_blank_lines=True)
training_labels = training_data["cultivar"]

training_data.drop("cultivar", inplace=True, axis=1)

gnb.fit(X=training_data, y=training_labels)

test_data = pd.read_csv("../Jupyter/wineTestModel.csv", skipinitialspace=True, skip_blank_lines=True)
test_data.drop("cultivar", inplace=True, axis=1)

prediction = gnb.predict(test_data)
test_data.insert(loc=0, column='cultivar', value=prediction)

print(test_data.to_string())

# Using test splitting

gnb = GaussianNB()

data = pd.read_csv("../Jupyter/wine.csv", skipinitialspace=True, skip_blank_lines=True)
training_data = pd.read_csv("../Jupyter/wine.csv", skipinitialspace=True, skip_blank_lines=True)
training_labels = training_data["cultivar"]

training_data.drop("cultivar", inplace=True, axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    training_data, training_labels, random_state=1988
)

gnb.fit(X_train, y_train)

prediction = gnb.predict(X_test)

print(confusion_matrix(y_test, prediction))
print(f"Accuracy: {accuracy_score(y_test, prediction)}")


