#David Zamojda
#Assignment 2

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score

#Get a list of the categorical features for a given dataframe. Move to util file for future use!
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))

# Get the indices of the categorical features for a given dataframe. Move to util file for future use!	
def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected)


#Read in Data files
data = pd.read_csv('./data/AmesHousingSetA.csv')
data_test = pd.read_csv('./data/AmesHousingSetB.csv')

#transform into one hot encoding
data = pd.get_dummies(data, columns=cat_features(data))
data_test = pd.get_dummies(data_test, columns = cat_features(data_test))

#create data_x and data_y
features = list(data)
features2 = list(data_test)

features.remove('SalePrice')
features.remove('PID')

# Create a features list that only contains things that both lists have
features_both = []
for f in features:
	if f in features2 :
		features_both.append(f)
#print(features_both)

custom_list = ['Yr.Sold', 'Lot.Frontage','Lot.Area', 'Year.Built', 'Year.Remod.Add', 'Mas.Vnr.Area','BsmtFin.SF.1',
				'BsmtFin.SF.2','Total.Bsmt.SF','X1st.Flr.SF','X2nd.Flr.SF','Gr.Liv.Area','Bsmt.Full.Bath','Bsmt.Half.Bath',
				'Full.Bath', 'Half.Bath','Bedroom.AbvGr','TotRms.AbvGrd','Garage.Area','Wood.Deck.SF','Yr.Sold', 'Mo.Sold']
data_x_custom = data[custom_list]
data_x_custom2 = data_test[custom_list]

#features2 = list(data_test)
features2.remove('SalePrice')

data_x = data[features_both]
data_y = data['SalePrice']

data_x2 = data_test[features_both]
data_y2 = data_test['SalePrice']

#Preprocessing values
imp = preprocessing.Imputer(missing_values = 'NaN' , strategy = 'most_frequent', axis= 0)
data_x = imp.fit_transform(data_x)
data_x_std = preprocessing.scale(data_x)
data_x = data_x_std

data_x_custom = imp.fit_transform(data_x_custom)
data_std = preprocessing.scale(data_x_custom)
data_x_custom = data_std
data_x_custom2 = imp.fit_transform(data_x_custom2)
data_std2 = preprocessing.scale(data_x_custom2)
data_x_custom2 = data_std2

data_x2 = imp.fit_transform(data_x2)
data_x2_std = preprocessing.scale(data_x2)
data_x2 = data_x2_std

#create base linear model
base_model = linear_model.LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state= 4)
base_model.fit(x_train, y_train)
preds = base_model.predict(x_test)

#sns.boxplot(x ='Yr.Sold', y = 'SalePrice', data =data)
#plt.show()

print('Base model R2 score: '+  str(r2_score(y_test, preds)))
print('Base model mean absolute error: ' + str(median_absolute_error(y_test, preds)))
print('Base model mean squared error: ' + str(mean_squared_error(y_test, preds)))
print('Base model mean squared error: ' + str(explained_variance_score(y_test, preds)))

# Create Lasso Linear model with different Alpha values
alpha = [0.1, 0.2, 0.25, 0.5, 1.0, 2.5, 5.0]
for a in alpha:
	lasso_mod = linear_model.Lasso(alpha =a, normalize=True, fit_intercept =True)
	lasso_mod.fit(x_train, y_train)
	preds = lasso_mod.predict(x_test)
	print('R2 Lasso model with alpha = ' + str(a) + ': ' + str(r2_score(y_test, preds)))

# Create linear model based on F-scores, Percentile based Model
selector_f = SelectPercentile(f_regression, percentile=50)
selector_f.fit(x_train, y_train)
xt_train, xt_test = selector_f.transform(x_train), selector_f.transform(x_test)
f_model = linear_model.LinearRegression()
f_model.fit(xt_train, y_train)
preds = f_model.predict(xt_test)
print('R2 Score Percentile Based model: ' + str(r2_score(y_test, preds)))

#My own model that I created 
custom_mod = linear_model.LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(data_x_custom, data_y, test_size = 0.2, random_state= 4)
custom_mod.fit(x_train, y_train)
preds = custom_mod.predict(x_test)
print('R2 Score Custom List model: ' + str(r2_score(y_test, preds)))

#Test the linear regression against the Second data set
print('~~~~~~~~~~~~~~Data set B~~~~~~~~~~~~~')

#base model
#base_model2 = linear_model.LinearRegression()
#x_train, x_test, y_train, y_test = train_test_split(data_x2, data_y2, test_size = 0.2, random_state= 4)
#base_model.fit(x_train, y_train)
preds2 = base_model.predict(data_x2)

print('Base model R2 score: '+  str(r2_score(data_y2, preds2)))
print('Base model mean absolute error: ' + str(median_absolute_error(data_y2, preds2)))
print('Base model mean squared error: ' + str(mean_squared_error(data_y2, preds2)))
print('Base model explained variance score: ' + str(explained_variance_score(data_y2, preds2)))




x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state= 4)
alphas = [0.25, 0.5, 1.0, 2.5, 5.0, 20, 30]
for a in alphas:
	lasso_mod2 = linear_model.Lasso(alpha = a, normalize=True, fit_intercept =True)
#x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state= 4)
	lasso_mod2.fit(x_train, y_train)
	pred = lasso_mod2.predict(data_x2)
	print('R2 Lasso model 2 with alpha: ' + str(a)+ ' '  + str(r2_score(data_y2, pred)))
	print('Median Absolute Error Lasso model 2 with alpha: ' + str(a)+ ' '  + str(median_absolute_error(data_y2, pred)))
	print('Mean Squared Error Lasso model 2 with alpha = 5: ' + str(a)+ ' '  + str(mean_squared_error(data_y2, pred)))
	print('Explained Variance score Lasso model 2 with alpha: ' + str(a)+ ' '  + str(explained_variance_score(data_y2, pred)))

#custom mod

preds = custom_mod.predict(data_x_custom2)
print('R2 Score Custom List model: ' + str(r2_score(data_y2, preds)))



