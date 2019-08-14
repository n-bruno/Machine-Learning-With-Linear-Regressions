# Github repo: n-bruno
# Created while watching Tech With Tim's tutorials: https://www.youtube.com/watch?v=45ryDIPHdGg
# Dataset retrieved from https://archive.ics.uci.edu/ml/machine-learning-databases/00320/
from builtins import len

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

# Comma separated file is separated by semicolons
from Constants import Constants

data = pd.read_csv(Constants.filename, sep=Constants.separator)

# Uncomment the below to print the CSV data.
# print(data.head())

data = data[Constants.dataset_member_names]

print("##################################################")
print("# Basic Machine Learning with linear regressions #")
print("##################################################")

print("We're going to predict {0}.".format(Constants.predict))

training_data = np.array(data.drop([Constants.predict], 1))
result_only = np.array(data[Constants.predict])

'''
sklearn.model_selection.train_test_split
Split arrays or matrices into random train and test subsets

Here, we are splitting our training data by 10%.
We're doing this because we don't want to make predictions on all the the data we trained 
our model on.
'''
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(training_data, result_only,
                                                                            test_size=Constants.training_sample_size)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)  # Find the line of best fit
accuracy = linear.score(x_test, y_test)

print("Our accuracy is %s" % accuracy)
print('The %s different coefficients (m):' % (len(Constants.dataset_member_names) - 1), linear.coef_)
print('Y Intercept:', linear.intercept_)

predictions = linear.predict(x_test)

for i in range(len(predictions)):
    print("Data used to make prediction: {0}, Prediction: {1}, Actual result: {2}".format(x_test[i],
                                                                                          int(round(predictions[i])),
                                                                                          y_test[i]))
