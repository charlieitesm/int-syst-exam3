import pandas as pd
import math


def calculate_probability(mean: float, stdev: float, x: float):
    pi = 3.14159
    euler = 2.71828

    first_part = 1 / (stdev * math.sqrt((2 * pi)))

    euler_power = -0.5 * math.pow((x - mean) / stdev, 2)
    second_part = math.pow(euler, euler_power)

    probability = first_part * second_part

    return probability


def calculate_posterior(features_posteriors: list, label_posterior: float):
    posterior = 1

    # Multiply all of the features posteriors to in order to calculate Naive Bayes
    for fp in features_posteriors:
        posterior *= fp

    posterior = posterior * label_posterior

    return posterior


male_fp = [
    calculate_probability(5.855, 0.187171935, 6),
    calculate_probability(176.25, 11.08677891, 130),
    calculate_probability(11.25, 0.957427108, 8)
]

posterior_male = calculate_posterior(male_fp, 0.5)

female_fp = [
    calculate_probability(5.4175, 0.311809237, 6),
    calculate_probability(132.5, 23.62907813, 130),
    calculate_probability(7.5, 1.290994449, 8)
]

posterior_female = calculate_posterior(female_fp, 0.5)

# making data frame from csv file
data = pd.read_csv("../Jupyter/wine.csv", skipinitialspace=True, skip_blank_lines=True)

data.sort_values("cultivar", inplace=True)

wine_possible_labels = data["cultivar"].unique()

label_1 = data.loc[data['cultivar'] == 1].drop("cultivar", axis=1)
label_2 = data.loc[data['cultivar'] == 2].drop("cultivar", axis=1)
label_3 = data.loc[data['cultivar'] == 3].drop("cultivar", axis=1)

label_1_means = label_1.mean(axis=0)
label_2_means = label_2.mean(axis=0)
label_3_means = label_3.mean(axis=0)

# We pass ddof to calculate the Population Std Dev, by default Pandas uses a ddof=1 which is for the Sample Std Dev
label_1_stdev = label_1.std(axis=0, ddof=0)
label_2_stdev = label_2.std(axis=0, ddof=0)
label_3_stdev = label_3.std(axis=0, ddof=0)

print("Hello")
