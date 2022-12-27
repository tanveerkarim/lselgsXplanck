#Last updated: 20221226
#This script does hyperparameter tuning for RF 
# good discussion: https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler

from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, classification_report, balanced_accuracy_score, roc_auc_score, precision_score, recall_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

#read in train and test sets
#trn_set = pd.read_pickle("/pscratch/sd/t/tanveerk/final_data_products/elgXplanck/fuji-elg-single-tomo-train-set.pkl")

# need this since pandas can't read these files 
import pickle5 as pickle
with open("/pscratch/sd/t/tanveerk/final_data_products/elgXplanck/fuji-elg-single-tomo-train-set.pkl", "rb") as fh:
    trn_set = pickle.load(fh)

# # feature-label split
y_trn = trn_set['bin_label']
X_trn = trn_set.drop(['bin_label'], axis = 1)
    
# #read in pipeline 
pipeline = load('/pscratch/sd/t/tanveerk/final_data_products/elgXplanck/fuji_RandomForestClassifier_single_tomo.joblib') 

# ##--RUN HYPERPARAMETER SEARCH--##

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Criterion
criterion = ['gini', 'entropy']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 10]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
# Class weights 
class_weights = [{0: (1 - x)/2, 1: x, 2: (1 - x)/2} for x in np.linspace(0.05, 0.95, 10)]
random_grid = {#'scaler': [MinMaxScaler, StandardScaler, MaxAbsScaler],
               'model__n_estimators': n_estimators,
               'model__max_features': max_features,
               'model__max_depth': max_depth,
               'model__criterion': criterion,
               'model__min_samples_split': min_samples_split,
               'model__min_samples_leaf': min_samples_leaf,
               'model__bootstrap': bootstrap,
               'model__class_weight': class_weights}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
#rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(pipeline, param_distributions = random_grid, n_iter = 120, cv = 4, 
                               verbose=10, random_state=20221109)# Fit the random search model
rf_random.fit(X_trn.iloc[:,7:-1], y_trn)

#print out best fits
best_random = rf_random.best_estimator_
print("-----")
print("Best RF model is: ", best_random)
print("-----")
print("Saving best model")
dump(best_random, '/pscratch/sd/t/tanveerk/final_data_products/elgXplanck/fuji_RandomForestClassifier_single_tomo_best_hyperparam.joblib', compress = 1)
print("Saving complete")