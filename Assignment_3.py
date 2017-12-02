#David Zamojda
#Assignment 3 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import naive_bayes
from sklearn import tree
from sklearn import svm 
from sklearn import ensemble 
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectPercentile, f_classif

from data_util import *

#import data sets
churn = pd.read_csv("./Data/churn_data.csv")
churn_v = pd.read_csv("./Data/churn_validation.csv")


#predictor and response variables for training and validation sets
data_y = churn['Churn']
datav_y = churn_v['Churn']

#perform one hot encoding on data sets
churn = pd.get_dummies(churn, columns=cat_features(churn))
churn_v = pd.get_dummies(churn_v, columns=cat_features(churn_v))

features = list(churn)
features.remove('Churn_Yes')
features.remove('Churn_No')
features.remove('CustID')

data_x = churn[features]
datav_x = churn_v[features]

#Preprocessing
#le = preprocessing.LabelEncoder()
#data_y = le.fit_transform(data_y)

#Use K-Best feature selection 
data_x = SelectKBest(chi2, k=5).fit_transform(data_x, data_y)
datav_x = SelectKBest(chi2, k=5).fit_transform(datav_x, datav_y)
#data_x = SelectPercentile(f_classif, percentile = 50).fit_transform(data_x, data_y)


#Training and test data
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.25, random_state = 4)

print('----------- NAIVE BAYES MODEL ------------------')

#Build and Test the Naive Bayes Model
gnb_mod = naive_bayes.GaussianNB()
gnb_mod.fit(x_train, y_train)
preds = gnb_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds)

#Build D Tree with Gini impurity criteria 
print('----------- DTREE WITH GINI IMPURITY CRITERION ------------------')
dtree_gini_mod = tree.DecisionTreeClassifier(criterion='gini')
dtree_gini_mod.fit(x_train, y_train)
preds_gini = dtree_gini_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds_gini)

print('----------- DTREE WITH ENTROPY CRITERION ------------------')
#Build D Tree with Entropy Criteria
dtree_entropy_mod = tree.DecisionTreeClassifier(criterion='entropy')
dtree_entropy_mod.fit(x_train, y_train)
preds_entropy = dtree_entropy_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds_entropy)

# Build a sequence of models for different n_est and depth values. **NOTE: c=1.0 is equivalent to the default.
cs = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
for c in cs:
	# Create model and fit.
	mod = svm.SVC(C=c)
	mod.fit(x_train, y_train)

	# Make predictions - both class labels and predicted probabilities.
	preds = mod.predict(x_test)
	print('---------- EVALUATING MODEL: C = ' + str(c) + ' -------------------')
	# Look at results.
	print_multiclass_classif_error_report(y_test, preds)
	
	
#Build Random forest Classifier
# Build a sequence of models for different n_est and depth values. **NOTE: nest=10, depth=None is equivalent to the default.
n_est = [5, 10, 50, 100]
depth = [3, 6, None]
for n in n_est:
	for dp in depth:
		# Create model and fit.
		mod = ensemble.RandomForestClassifier(n_estimators=n, max_depth=dp)
		mod.fit(x_train, y_train)

		# Make predictions - both class labels and predicted probabilities.
		preds = mod.predict(x_test)
		print('---------- EVALUATING MODEL: n_estimators = ' + str(n_est) + ', depth =' + str(dp) + ' -------------------')
		# Look at results.
		print_multiclass_classif_error_report(y_test, preds)
	
#highest accuracy Accuracy: 0.897435897436, N_estimator = 5, depth = None , k = 5

#Build model for the validation set

		# Create model and fit.
		mod2 = ensemble.RandomForestClassifier(n_estimators=5, max_depth=None)
		mod2.fit(x_train, y_train)

		# Make predictions - both class labels and predicted probabilities.
		preds = mod2.predict(datav_x)
		print('---------- EVALUATING MODEL: n_estimators = ' + str(5) + ', depth =' + 'None' + ' -------------------')
		#print('Accuracy: ' + str(accuracy_score(y_test, preds)))
		# Look at results.
		print_multiclass_classif_error_report(y_test, preds)





