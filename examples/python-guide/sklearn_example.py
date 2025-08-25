# -*- coding: utf-8 -*-
"""
Examples on how to use the scikit-learn wrapper interface for the GPBoost
algorithm for combining tree-boosting with random effects and Gaussian
process models

Author: Fabio Sigrist
"""

import numpy as np
import gpboost as gpb
import random
import matplotlib.pyplot as plt

print('Simulating data...')
# Simulate data
n = 5000  # number of samples
m = 500  # number of groups
# Simulate grouped random effects
np.random.seed(1)
# simulate grouped random effects
group = np.arange(n)  # grouping variable
for i in range(m):
    group[int(i * n / m):int((i + 1) * n / m)] = i
b1 = np.random.normal(size=m)  # simulate random effects
eps = b1[group]
# simulate fixed effects
def f1d(x):
    """Non-linear function for simulation"""
    return (1.7 * (1 / (1 + np.exp(-(x - 0.5) * 20)) + 0.75 * x))
X = np.random.rand(n, 2)
f = f1d(X[:, 0])
xi = np.sqrt(0.01) * np.random.normal(size=n)  # simulate error term
y = f + eps + xi  # observed data

#--------------------Training----------------
# define GPModel
gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
# train
bst = gpb.GPBoostRegressor(max_depth=3, learning_rate=0.01, n_estimators=100)
bst.fit(X, y, gp_model=gp_model)
print("Estimated random effects model: ")
gp_model.summary()

#--------------------Prediction----------------
# predict
group_test = np.arange(m)
Xtest = np.zeros((m, 2))
Xtest[:, 0] = np.linspace(0, 1, m)
pred = bst.predict(X=Xtest, group_data_pred=group_test, raw_score = True)
# Compare true and predicted random effects
plt.figure("Comparison of true and predicted random effects")
plt.scatter(b1, pred['random_effect_mean'])
plt.title("Comparison of true and predicted random effects")
plt.xlabel("truth")
plt.ylabel("predicted")
plt.show()
# Fixed effect
plt.figure("Comparison of true and fitted fixed effect")
plt.scatter(Xtest[:, 0], pred['fixed_effect'], linewidth=2, color="b", label="fit")
x = np.linspace(0, 1, 200, endpoint=True)
plt.plot(x, f1d(x), linewidth=2, color="r", label="true")
plt.title("Comparison of true and fitted fixed effect")
plt.legend()
plt.show()

# feature importances
print('Feature importances:', list(bst.feature_importances_))

#--------------------Choosing tuning parameters using the TPESampler from optuna----------------
# Define search space
# Note: if the best combination found below is close to the bounday for a paramter, you might want to extend the corresponding range
search_space = { 'learning_rate': [0.001, 10],
                'min_data_in_leaf': [1, 1000],
                'max_depth': [-1, 10],
                'num_leaves': [2, 1024],
                'lambda_l2': [0, 100],
                'max_bin': [63, np.min([10000,n])],
                'line_search_step_length': [True, False] }
# Define metric
metric = "mse" # Can also use metric = "test_neg_log_likelihood". For more options, see https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst#metric-parameters

gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
# Run parameter optimization using the TPE algorithm and 4-fold CV 
opt_params = gpb.tune_pars_TPE_algorithm_optuna(X=X, y=y, search_space=search_space, 
                                                nfold=4, gp_model=gp_model, metric=metric, tpe_seed=1,
                                                max_num_boost_round=1000, n_trials=100, early_stopping_rounds=20)
print("Best parameters: " + str(opt_params['best_params']))
print("Best number of iterations: " + str(opt_params['best_iter']))
print("Best score: " + str(opt_params['best_score']))

# Alternatively and faster: using manually defined validation data instead of cross-validation
np.random.seed(10)
permute_aux = np.random.permutation(n)
train_tune_idx = permute_aux[0:int(0.8 * n)] # use 20% of the data as validation data
valid_tune_idx = permute_aux[int(0.8 * n):n]
folds = [(train_tune_idx, valid_tune_idx)]
opt_params = gpb.tune_pars_TPE_algorithm_optuna(X=X, y=y, search_space=search_space, 
                                                folds=folds, gp_model=gp_model, metric=metric, tpe_seed=1,
                                                max_num_boost_round=1000, n_trials=100, early_stopping_rounds=20)

#--------------------Choosing tuning parameters using random grid search----------------
# Note: scikit-learn's 'GridSearchCV' is not supported, use the GPBoost internal
#       function 'grid_search_tune_parameters' instead as shown below
# Define parameter search grid
# Note: if the best combination found below is close to the bounday for a paramter, you might want to extend the corresponding range
param_grid = { 'learning_rate': [0.001, 0.01, 0.1, 1, 10], 
              'min_data_in_leaf': [1, 10, 100, 1000],
              'max_depth': [-1, 1, 2, 3, 5, 10],
              'num_leaves': 2**np.arange(1,10),
              'lambda_l2': [0, 1, 10, 100],
              'max_bin': [250, 500, 1000, np.min([10000,n])],
              'line_search_step_length': [True, False]}
other_params = {'verbose': 0}
# Define metric
metric = "mse" # Can also use metric = "test_neg_log_likelihood". For more options, see https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst#metric-parameters
gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
data_train = gpb.Dataset(data=X, label=y)
# Run parameter optimization using random grid search and 4-fold CV
# Note: deterministic grid search can be done by setting 'num_try_random=None'
opt_params = gpb.grid_search_tune_parameters(param_grid=param_grid, params=other_params,
                                             num_try_random=100, nfold=4, seed=1000,
                                             train_set=data_train, gp_model=gp_model,
                                             use_gp_model_for_validation=True, verbose_eval=1,
                                             num_boost_round=1000, early_stopping_rounds=20,
                                             metric=metric)                            
print("Best parameters: " + str(opt_params['best_params']))
print("Best number of iterations: " + str(opt_params['best_iter']))
print("Best score: " + str(opt_params['best_score']))

# Alternatively and faster: using manually defined validation data instead of cross-validation
np.random.seed(10)
permute_aux = np.random.permutation(n)
train_tune_idx = permute_aux[0:int(0.8 * n)] # use 20% of the data as validation data
valid_tune_idx = permute_aux[int(0.8 * n):n]
folds = [(train_tune_idx, valid_tune_idx)]
opt_params = gpb.grid_search_tune_parameters(param_grid=param_grid, params=other_params,
                                             num_try_random=100, folds=folds, seed=1000,
                                             train_set=data_train, gp_model=gp_model,
                                             use_gp_model_for_validation=True, verbose_eval=1,
                                             num_boost_round=1000, early_stopping_rounds=20,
                                             metric=metric)

#--------------------Using a validation set for finding number of iterations----------------
# split into training an test data
train_ind = random.sample(range(n), int(n / 2))
test_ind = [x for x in range(n) if (x not in train_ind)]
X_train = X[train_ind, :]
y_train = y[train_ind]
group_train = group[train_ind]
X_test = X[test_ind, :]
y_test = y[test_ind]
group_test = group[test_ind]
# train
gp_model = gpb.GPModel(group_data=group_train, likelihood="gaussian")
gp_model.set_prediction_data(group_data_pred=group_test)
bst = gpb.GPBoostRegressor(max_depth=3, learning_rate=0.01, n_estimators=100)
bst.fit(X_train, y_train, gp_model=gp_model,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=5)

#--------------------Saving a model and loading it from a file----------------
# Train model and make prediction
gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
bst = gpb.GPBoostRegressor(max_depth=3, learning_rate=0.01, n_estimators=50)
bst.fit(X, y, gp_model=gp_model)
group_test = np.array([1,2,-1])
Xtest = np.random.rand(len(group_test), 2)
pred = bst.predict(X=Xtest, group_data_pred=group_test, 
                   predict_var=True, raw_score = True)
# Save model
import joblib
joblib.dump(bst, 'bst.pkl')
# load model
bst_loaded = joblib.load('bst.pkl')
pred_loaded = bst_loaded.predict(X=Xtest, group_data_pred=group_test, 
                                 predict_var=True, raw_score=True)
# Check equality
print(pred['fixed_effect'] - pred_loaded['fixed_effect'])
print(pred['random_effect_mean'] - pred_loaded['random_effect_mean'])
print(pred['random_effect_cov'] - pred_loaded['random_effect_cov'])

# Alternative way of saving and loading without the use of joblib
# Note: with the following approach, the loaded model is not a "sklearn-API" 
#       'GPBoostRegressor' object anymore but a standard 'Booster' object
model_str = bst.booster_.model_to_string()
bst_loaded = gpb.Booster(model_str = model_str)
pred_loaded = bst_loaded.predict(data=Xtest, group_data_pred=group_test, 
                                 predict_var=True, pred_latent=True)
print(pred['fixed_effect'] - pred_loaded['fixed_effect'])
print(pred['random_effect_mean'] - pred_loaded['random_effect_mean'])
print(pred['random_effect_cov'] - pred_loaded['random_effect_cov'])


#--------------------Classification example----------------
y_bin = ((y - np.mean(y)) > 0) * 1.
gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
# train
bst = gpb.GPBoostClassifier(max_depth=3, learning_rate=0.01, n_estimators=50)
bst.fit(X, y_bin, gp_model=gp_model)
print("Estimated random effects model: ")
gp_model.summary()
# predict
group_test = np.arange(m)
Xtest = np.zeros((m, 2))
Xtest[:, 0] = np.linspace(0, 1, m)
pred = bst.predict(X=Xtest, group_data_pred=group_test, raw_score = True)
