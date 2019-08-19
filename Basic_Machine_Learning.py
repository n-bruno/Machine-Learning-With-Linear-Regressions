# Author: n-bruno
# Created while watching Tech With Tim's tutorials: https://www.youtube.com/watch?v=45ryDIPHdGg
# Dataset retrieved from https://archive.ics.uci.edu/ml/machine-learning-databases/00320/
from builtins import len
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from Constants import Constants
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Comma separated file is separated by semicolons
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

# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
if Constants.train:
    best = 0
    for _ in range(1000):
        '''
        Here, we are splitting our training data by 10%.
        We're doing this because we don't want to make predictions on all the the data we trained 
        our model on.
        '''
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(training_data, result_only,
                                                                                    test_size=Constants.training_sample_size)

        linear = linear_model.LinearRegression()

        linear.fit(x_train, y_train)
        acc = linear.score(x_test, y_test)
        print("Accuracy: " + str(acc))

        # If the current model has a better score than one we've already trained then save it
        if acc > best:
            best = acc
            with open(Constants.pickle_dump, "wb") as f:
                pickle.dump(linear, f)


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(training_data, result_only,
                                                                            test_size=Constants.training_sample_size)

pickle_in = open(Constants.pickle_dump, "rb")
linear = pickle.load(pickle_in)

print('The %s different coefficients (m):' % (len(Constants.dataset_member_names) - 1), linear.coef_)
print('Y Intercept:', linear.intercept_)

predictions = linear.predict(x_test)

for i in range(len(predictions)):
    rounded_prediction = int(round(predictions[i]))
    #rounded_prediction = 0 if rounded_prediction < 0 else rounded_prediction

    print("Prediction: {0}, Actual result: {1}, Data used to make prediction: {2}"
        .format(
        str(rounded_prediction).zfill(2),
        str(y_test[i]).zfill(2),
        x_test[i]))


p = "studytime"
style.use("ggplot")
pyplot.scatter(data[p],  data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()