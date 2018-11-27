
"""
Neural Network to predict nhl playoff performance.

NOTE: Set the 'currentYear' as the year used in gather script. Also
make sure you replace Montréal in dataset with Montreal for UTF-8 compatibility
"""

# Data preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# SET THIS
currentYear = '2018'

# Importing the dataset
dataset = pd.read_csv('allnhlstats.csv')
dataset_currentYear = pd.read_csv(currentYear+'nhlstats.csv')
X = dataset.loc[:, ['faceoffWinPctg','goalsAgainst','goalsAgainstPerGame','goalsFor','goalsForPerGame','losses','otLosses','pkPctg','pointPctg','points','ppPctg','regPlusOtWins','shootoutGamesWon','shotsAgainstPerGame','shotsForPerGame','ties','wins']].values
y = dataset.loc[:, 'playoffwins'].values
X_currentYear = dataset_currentYear.loc[:, ['faceoffWinPctg','goalsAgainst','goalsAgainstPerGame','goalsFor','goalsForPerGame','losses','otLosses','pkPctg','pointPctg','points','ppPctg','regPlusOtWins','shootoutGamesWon','shotsAgainstPerGame','shotsForPerGame','ties','wins']].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_test_currentYear = sc.fit_transform(X_currentYear)
X_train_currentYear = sc.transform(X)
y_train = y_train / 16
y_test = y_test / 16

# Import Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
# Kfold Cross Validation
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
def build_classifier():
    adam = Adam(lr=0.001)
    classifier = Sequential()
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 17))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.compile(optimizer = adam, loss = 'mse', metrics = ['accuracy'])
    return classifier

classifier = KerasRegressor(build_fn = build_classifier, batch_size = 10, epochs = 1000)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)

mean = accuracies.mean()
variance = accuracies.std()

# Tuning the ANN
# Import Keras libraries and packages
'''
# Uncomment to run grid search
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    adam = Adam(lr=0.001)
    classifier = Sequential()
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 17))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.compile(optimizer = optimizer, loss = 'mse', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size' : [10, 25, 32],
              'epochs' : [1000, 3000, 5000],
              'optimizer' : ['adam', 'rmsprop', 'nadam']}
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters,
                           scoring = 'accuracy', cv = 10);
grid_search = grid_search.fit(X_train, y_train);
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
'''

'''
# Uncomment to run confusion matrix
# Fitting the ANN to the Training Set
classifier = build_classifier()
classifier.fit(X_train_currentYear, y, epochs = 1000, batch_size = 25)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''

'''
# Uncomment to run prediction of current year
y_pred_currentYear = classifier.predict(X_test_currentYear)
'''