import pandas as pd
import math


def calculate_normal_probability_distribution(mean: float, stdev: float, x: float):
    pi = 3.14159
    euler = 2.71828

    first_part = 1 / (stdev * math.sqrt((2 * pi)))

    euler_power = -0.5 * math.pow((x - mean) / stdev, 2)
    second_part = math.pow(euler, euler_power)

    probability = first_part * second_part

    return probability


def calculate_gaussian_naive_bayes(features_posteriors: list, label_posterior: float):
    posterior = 1

    # Multiply all of the features posteriors to in order to calculate Naive Bayes
    for fp in features_posteriors:
        posterior *= fp

    posterior = posterior * label_posterior

    return posterior


male_fp = [
    calculate_normal_probability_distribution(5.855, 0.187171935, 6),
    calculate_normal_probability_distribution(176.25, 11.08677891, 130),
    calculate_normal_probability_distribution(11.25, 0.957427108, 8)
]

posterior_male = calculate_gaussian_naive_bayes(male_fp, 0.5)

female_fp = [
    calculate_normal_probability_distribution(5.4175, 0.311809237, 6),
    calculate_normal_probability_distribution(132.5, 23.62907813, 130),
    calculate_normal_probability_distribution(7.5, 1.290994449, 8)
]

posterior_female = calculate_gaussian_naive_bayes(female_fp, 0.5)

# Read the training dataset
data = pd.read_csv("../Jupyter/wine.csv", skipinitialspace=True, skip_blank_lines=True)

data.sort_values("cultivar", inplace=True)

wine_possible_labels = data["cultivar"].unique()

label_1 = data.loc[data['cultivar'] == 1].drop("cultivar", axis=1)
label_2 = data.loc[data['cultivar'] == 2].drop("cultivar", axis=1)
label_3 = data.loc[data['cultivar'] == 3].drop("cultivar", axis=1)

label_1_means = label_1.mean(axis=0)
label_2_means = label_2.mean(axis=0)
label_3_means = label_3.mean(axis=0)

# We pass ddof to calculate the Sample Std Dev
label_1_stdev = label_1.std(axis=0, ddof=1)
label_2_stdev = label_2.std(axis=0, ddof=1)
label_3_stdev = label_3.std(axis=0, ddof=1)

# Read the test dataset
test_data = pd.read_csv("../Jupyter/wineTestModel.csv", skipinitialspace=True, skip_blank_lines=True)

# Drop the cultivar label as this is the one we will calculate
test_data.drop("cultivar", axis=1, inplace=True)

# Prepare the variables that we'll use for the calculation and drop the cultivar label
feature_list = list(test_data.columns.values)

# The posteriors, that is P(label_1), is equal to the number of cultivar==1, 2 or 3 divided by the total num of records
posterior_1 = len(label_1) / len(data)
posterior_2 = len(label_2) / len(data)
posterior_3 = len(label_3) / len(data)

# We'll add all of the information in symmetric lists in order to ease the calculation loop structure
cultivar_label_posteriors = [posterior_1, posterior_2, posterior_3]
means = [label_1_means, label_2_means, label_3_means]
stdevs = [label_1_stdev, label_2_stdev, label_3_stdev]

determined_labels = []

# Begin the calculation
for row_index, row in test_data.iterrows():
    print("\n\n***************************")
    print(f"Calculating label for row {row_index + 1}...")
    print("***************************\n")

    cultivar_label_bayes_probabilities = []

    # Check each of the labels
    for label_idx in range(3):

        print(f"\n-------Calculating P(cultivar == {label_idx + 1})...\n")

        features_gauss = []

        for feature in feature_list:
            feature_x = row[feature]
            feature_mean = means[label_idx][feature]
            feature_stdev = stdevs[label_idx][feature]

            feature_gauss = calculate_normal_probability_distribution(mean=feature_mean,
                                                                      stdev=feature_stdev,
                                                                      x=feature_x)

            features_gauss.append(feature_gauss)

            print(f"-------Feature: {feature} -------")
            print(f"X = {feature_x}")
            print(f"Feature mean = {feature_mean}")
            print(f"Feature Std Dev = {feature_stdev}")
            print(f"P({feature}| Cultivar == {label_idx +1}) = {feature_gauss}")

        # Calculate the posterior of the label given all of the features P(
        cultivar_posterior = cultivar_label_posteriors[label_idx]

        label_probability = calculate_gaussian_naive_bayes(features_gauss, cultivar_posterior)

        cultivar_label_bayes_probabilities.append(label_probability)

        print(f"\n >>>>>>>> P(cultivar == {label_idx + 1} | {','.join(feature_list)}): {label_probability}")

    # To get the final value, we just get the index in the label probabilities and offset the value by 1
    final_cultivar = cultivar_label_bayes_probabilities.index(max(cultivar_label_bayes_probabilities)) + 1
    determined_labels.append(final_cultivar)

    print(f"\n ================ RESULTS FOR ROW {row_index + 1} =================")
    print("Given the following Bayes posteriors for each label:")

    for result_idx, result in enumerate(cultivar_label_bayes_probabilities):
        print(f"P(cultivar == {result_idx + 1}): {result}")

    print(f"ROW {row_index + 1} is probably Cultivar {final_cultivar}")

# Adjust the DataFrame to the determined cultivar label
test_data.insert(loc=0, column='cultivar', value=determined_labels)

print("Final Test Dataset:")
print(test_data.to_string())
