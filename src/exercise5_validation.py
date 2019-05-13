import pandas as pd
from sklearn.naive_bayes import GaussianNB

dataset = {
    "Lugar":   [0,0,0,1,0,0,1,1,1,1],
    "Dia":     [0,0,0,0,1,1,1,0,1,1],
    "Arma":    [0,0,0,1,1,1,0,1,1,0],
    "Resulto": [0,0,1,0,0,1,0,0,1,0]
}

test_dataset = {
    "Lugar":   [0],
    "Dia":     [1],
    "Arma":    [0],
}

test_dataset = pd.DataFrame(test_dataset)

data = pd.DataFrame(dataset)
training_labels = data["Resulto"]

data.drop("Resulto", inplace=True, axis=1)

gnb = GaussianNB()

gnb.fit(data, training_labels)

prediction = gnb.predict(test_dataset)

print(prediction)