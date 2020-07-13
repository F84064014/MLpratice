# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 00:09:11 2020

@author: user
"""

import os
import tarfile
import pandas as pd
import numpy as np
from six.moves import  urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

##get Data
fetch_housing_data()
housing = load_housing_data()

#observing data
housing.info()

housing.describe()
housing["ocean_proximity"].value_counts()

import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()

##create test set
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"]<5, 5.0, inplace=True)
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
housing["income_cat"].value_counts()/len(housing)
strat_train_set["income_cat"].value_counts()/len(strat_train_set)
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True) #warning but seems no effect
housing = strat_train_set.copy()

#visualization
housing.plot(kind="scatter",x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))

#Attribute Combination
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]    
housing["population_per_household"] = housing["population"]/housing["households"]
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

##Data Cleaning
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

#Imputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
imputer.statistics_
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

#LabelEncoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded
encoder.classes_

#LabelBinarizer
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer() #sparse_output=True
housing_cat = housing["ocean_proximity"]
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot

#Standarized scale
from sklearn.preprocessing import StandardScaler
encoder = StandardScaler()
housing_num = housing.drop("ocean_proximity", axis=1)
std_housing_num = encoder.fit_transform(housing_num)

##self defined Transformers
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3,4,5,6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): #no *args ro **kargs
        self.add_bedroom_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:,rooms_ix]/X[:,household_ix]
        population_per_household = X[:,population_ix]/X[:,household_ix]
        if self.add_bedroom_per_room:
            bedrooms_per_room = X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        
attr_addr = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_addr.transform(housing.values)

##Transformer for pipeline
from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
      self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

##The original Binarizer would cause some unknown error
from sklearn.preprocessing import LabelBinarizer
class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, X, y=0):
        self.encoder.fit(X)
        return self
    def transform(self, X, y=0):
        return self.encoder.transform(X)

##Pipelines
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer

housing_num = housing.drop("ocean_proximity", axis=1)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
    ])
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', MyLabelBinarizer()),
    ])

from sklearn.pipeline import FeatureUnion
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
    ])    

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared

#linear model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(housing_prepared, housing_labels)

new_data = housing.iloc[:5]
new_labels = housing_labels.iloc[:5]
new_data_prepared = full_pipeline.transform(new_data)
print("Predictions:", lm.predict(new_data_prepared))
print("Labels:", list(new_labels))

#MSE of lm
from sklearn.metrics import mean_squared_error
housing_predictions = lm.predict(housing_prepared)
lm_mse = mean_squared_error(housing_labels, housing_predictions)
lm_rmse = np.sqrt(lm_mse)
lm_rmse

#DecisionTree
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

#cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
print("Scores:", tree_rmse_scores)
print("Mean:", tree_rmse_scores.mean())
print("Standard deviation:", tree_rmse_scores.std())

scores = cross_val_score(lm, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lm_scores = np.sqrt(-scores)
print("Scores:", lm_scores)
print("Mean:", lm_scores.mean())
print("Standard deviation:", lm_scores.std())

#RandomForest
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
forest_scores = np.sqrt(-scores)
print("Scores:", forest_scores)
print("Mean:", forest_scores.mean())
print("Standard deviation:", forest_scores.std())

#GridSearchCV
from sklearn.model_selection import GridSearchCV #the cv stands for cross validation
param_grid = [
        {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
        {'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4]},
        ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                         scoring = 'neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)
grid_search.best_params_
grid_search.best_estimator_

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
#Analyze Best Model
feature_importance = grid_search.best_estimator_.feature_importances_
num_attribs = list(housing.drop("ocean_proximity", axis=1))
extra_attribs = ["rooms_per_household", "pop_per_household", "bedrooms_per_household"]
cat_one_hot_attribs = list(LabelBinarizer().fit(housing["ocean_proximity"]).classes_)
attribs = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importance, attribs), reverse=True)

#final
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

#dump model
import joblib
joblib.dump(final_model, "model713.pkl")
#load model
my_model = joblib.load("model713.pkl")
my_model.predict(X_test_prepared)
