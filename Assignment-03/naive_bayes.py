#-------------------------------------------------------------------------
# AUTHOR: Subham Panda
# FILENAME: naive_bayes.py
# SPECIFICATION: This program will read the weather_training.csv file. Print
# the naive bayes accuracy calculated after all of the predictions.
# FOR: CS 5990- Assignment #3
# TIME SPENT: 40 minutes
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]


#reading the training data
#--> add your Python code here
training_data = pd.read_csv('weather_training.csv')

#update the training class values according to the discretization (11 values only)
def discretizer(data):
    point_ini = -110
    for c in classes:
        if data["Temperature (C)"] > point_ini and data["Temperature (C)"] <= c:
            data["Temperature (C)"] = c
        point_ini = c
    return data

discrete_training = training_data.apply(discretizer, axis = 1)
y_training = np.array(discrete_training["Temperature (C)"])
y_training = y_training.astype(dtype='int')
X_training = np.array(discrete_training.drop(["Temperature (C)","Formatted Date"], axis=1).values)

#reading the test data
#--> add your Python code here
test_data = pd.read_csv('weather_test.csv')
#test_data.dropna(how="all")

#update the test class values according to the discretization (11 values only)
discrete_test = test_data.apply(discretizer, axis = 1)
y_test = discrete_test["Temperature (C)"]
y_test = y_test.astype(dtype='int')
X_test = discrete_test.drop(["Temperature (C)","Formatted Date"], axis=1).values


#fitting the naive_bayes to the data
clf = GaussianNB()
clf = clf.fit(X_training, y_training)

#make the naive_bayes prediction for each test sample and start computing its accuracy
#the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
#to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
accuracy = 0
for (x_testSample, y_testSample) in zip(X_test, y_test):
    prediction = clf.predict(np.array([x_testSample]))
    #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
    #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
    diff = 100*(abs(prediction[0] - y_testSample)/y_testSample)
    if diff >= -15 and diff <= 15:
        accuracy+=1

#--> add your Python code here
result = accuracy/len(y_test)
print(f"naive_bayes accuracy: {result}")