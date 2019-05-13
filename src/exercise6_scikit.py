import pandas as pd
from sklearn.naive_bayes import GaussianNB

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



