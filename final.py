#David Zamojda
#Data Science Final Project

import pandas as pd
from data_util import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm 
from sklearn import linear_model
from sklearn import ensemble 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn import tree

data = pd.read_csv('./Data/reordered2016crashes.csv')

del data['STREET_NAME']
del data['CRASH_CRN']

data = pd.get_dummies(data, columns=cat_features(data))

#create data set that only has fatalities for graphs
fatalities = data[data.FATAL_COUNT > 0]
fatalities.to_csv('./Data/fatalities.csv')


#-------------- Fatality or Injury ----------------------
features3 = list(data)
features3.remove('FATAL_OR_MAJ_INJ')
features3.remove('FATAL')
features3.remove('INJURY')
features3.remove('FATAL_COUNT')
features3.remove('INJURY_COUNT')
features3.remove('MAJ_INJ_COUNT')
features3.remove('MOD_INJ_COUNT')
features3.remove('MIN_INJ_COUNT')
features3.remove('MODERATE_INJURY')
features3.remove('MAJOR_INJURY')
features3.remove('UNB_DEATH_COUNT')
features3.remove('UNB_MAJ_INJ_COUNT')
features3.remove('BELTED_DEATH_COUNT')
features3.remove('BELTED_MAJ_INJ_COUNT')
features3.remove('MCYCLE_DEATH_COUNT')
features3.remove('MCYCLE_MAJ_INJ_COUNT')
features3.remove('BICYCLE_DEATH_COUNT')
features3.remove('BICYCLE_MAJ_INJ_COUNT')
features3.remove('PED_DEATH_COUNT')
features3.remove('PED_MAJ_INJ_COUNT')
features3.remove('MAX_SEVERITY_LEVEL')
features3.remove('INJURY_OR_FATAL')
features3.remove('PROPERTY_DAMAGE_ONLY')
features3.remove('UNK_INJ_DEG_COUNT')
features3.remove('UNK_INJ_PER_COUNT')
features3.remove('MINOR_INJURY')

data_x = data[features3]
data_y = data['INJURY_OR_FATAL']

imp = preprocessing.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
data_x = imp.fit_transform(data_x)

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

print('----------- DTREE WITH GINI IMPURITY CRITERION ------------------')
dtree_gini_mod = tree.DecisionTreeClassifier(criterion='gini')
dtree_gini_mod.fit(x_train, y_train)
preds_gini = dtree_gini_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds_gini)

n_est = [100]
depth = [None]
for n in n_est:
	for dp in depth:
		# Create model and fit.
		mod = ensemble.RandomForestClassifier(n_estimators=n, max_depth=dp)
		mod.fit(x_train, y_train)

		# Make predictions - both class labels and predicted probabilities.
		preds = mod.predict(x_test)
		print('---------- EVALUATING MODEL: n_estimators = ' + str(n) + ', depth =' + str(dp) + ' -------------------')
		# Look at results.
		print_multiclass_classif_error_report(y_test, preds)
#END-------------- Fatality or Injury ----------------------


#START-------------SPEEED LIMIT -----------------------------
features2 = list(data)
features2.remove('SPEED_LIMIT')

data_x = data[features2]
data_y = data['SPEED_LIMIT']

imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
data_x = imp.fit_transform(data_x)

# Create training and test sets for later use.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)
#Best test for predicting the speed limit area of the accident
print('----------- DTREE WITH GINI IMPURITY CRITERION ------------------')
dtree_gini_mod = tree.DecisionTreeClassifier(criterion='gini')
dtree_gini_mod.fit(x_train, y_train)
preds_gini = dtree_gini_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds_gini)


print('\n----------- DTREE WITH ENTROPY CRITERION -----------------------')
dtree_entropy_mod = tree.DecisionTreeClassifier(criterion='entropy')
dtree_entropy_mod.fit(x_train, y_train)
preds_entropy = dtree_entropy_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds_entropy)

print('\n--------------------- NAIVE BAYES MODEL -----------------------')
gnb_mod = naive_bayes.GaussianNB()
gnb_mod.fit(x_train, y_train)
preds = gnb_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds)

n_est = [70]
depth = [None]
for n in n_est:
	for dp in depth:
		# Create model and fit.
		mod = ensemble.RandomForestClassifier(n_estimators=n, max_depth=dp)
		mod.fit(x_train, y_train)
		# Make predictions - both class labels and predicted probabilities.
		preds = mod.predict(x_test)
		print('---------- EVALUATING MODEL: n_estimators = ' + str(n) + ', depth =' + str(dp) + ' -------------------')
		# Look at results.
		print_multiclass_classif_error_report(y_test, preds)

#END-------------SPEEED LIMIT -----------------------------


