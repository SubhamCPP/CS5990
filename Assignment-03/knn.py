#-------------------------------------------------------------------------
# AUTHOR: Subham Panda
# FILENAME: knn.py
# SPECIFICATION: Complete a KNN classification task with discretized temperature classes
# FOR: CS 5990- Assignment #3
# TIME SPENT: 45 minutes
#-------------------------------------------------------------------------

# importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

# defining the discretization classes
classes = [i for i in range(-22, 40, 6)]

# function to discretize the temperature values
def discretize_temperature(value):
    for cl in classes:
        if value < cl + 3: 
            return cl
    return classes[-1]  

# reading the training data
df_training = pd.read_csv('weather_training.csv')
X_training = df_training.drop(columns=['Formatted Date', 'Temperature (C)']).values
y_training = np.array([discretize_temperature(x) for x in df_training['Temperature (C)'].values]).astype(int)

# reading the test data
df_test = pd.read_csv('weather_test.csv')
X_test = df_test.drop(columns=['Formatted Date', 'Temperature (C)']).values
y_test_real = df_test['Temperature (C)'].values 
y_test = np.array([discretize_temperature(x) for x in df_test['Temperature (C)'].values]).astype(int)

# Normalize the feature data
scaler = StandardScaler()
X_training_normalized = scaler.fit_transform(X_training)
X_test_normalized = scaler.transform(X_test)

highest_accuracy = 0
best_params = {}

# loop over the hyperparameter values (k, p, and w) of KNN
for k in k_values:
    for p in p_values:
        for w in w_values:
            # fitting the KNN to the data
            clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
            clf.fit(X_training_normalized, y_training)

            # make the KNN prediction for each test sample and start computing its accuracy
            correct_predictions = 0
            for x_testSample, y_testRealValue in zip(X_test_normalized, y_test_real):
                predicted_class = clf.predict([x_testSample])[0]
                predicted_temperature = predicted_class  
                percentage_difference = 100 * abs(predicted_temperature - y_testRealValue) / abs(y_testRealValue)

                # the prediction is considered correct if within ±15% of the actual value
                if percentage_difference <= 15:
                    correct_predictions += 1

            accuracy = correct_predictions / len(y_test_real)

            # check if the calculated accuracy is higher than the previously one calculated
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                best_params = {'k': k, 'p': p, 'weight': w}
                print(f"Highest KNN accuracy so far: {highest_accuracy:.2f}")
                print(f"Parameters: k = {k}, p = {p}, weight = {w}")

# After completing the grid search, print out the best parameters and the highest accuracy
print(f"Best parameters: k = {best_params['k']}, p = {best_params['p']}, weight = {best_params['weight']}")
print(f"Highest KNN accuracy: {highest_accuracy:.2f}")
