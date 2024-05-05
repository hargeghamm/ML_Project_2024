import numpy as np
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import xlrd
from sklearn.model_selection import train_test_split 
import xgboost as xgb 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



def ucb_selection(rewards, attempts, t, c):
    ucb_values = []
    n = len(rewards)

    for i in range(n):
        if attempts[i] == 0: # Exploration term when no attempts have been made
            ucb_value = float('inf')
        else:
            exploitation = rewards[i] / attempts[i]
            exploration = c * math.sqrt(math.log(t) / attempts[i])
            ucb_value = exploitation + exploration

        ucb_values.append(ucb_value)

    selected_index = max(range(n), key=lambda i: ucb_values[i])

    return selected_index  



def XGBoost_Random(X_train, y_train, X_test, y_test, param_space_XGB):
    rewards = [0] * len(param_space_XGB)
    attempts = [0] * len(param_space_XGB)
    best_config_ucb = None
    best_error_ucb = float('inf')

    best_config_random_XGB = None
    best_error_random_XGB = float('inf')

    max_iterations = 500
    c = 2  # Exploration parameter

    for t in range(1, max_iterations + 1):
        # UCB strategy
        config_idx_ucb = ucb_selection(rewards, attempts, t, c)
        config_ucb = param_space_XGB[config_idx_ucb]
        ucb_model = xgb.XGBClassifier(**config_ucb)

        ucb_model.fit(X_train, y_train)
        y_pred = ucb_model.predict(X_test)

        error_ucb = accuracy_score(y_test, y_pred)
        val_err_ucb = 1 - error_ucb

        rewards[config_idx_ucb] += error_ucb
        attempts[config_idx_ucb] += 1

        if val_err_ucb < best_error_ucb:
            best_config_ucb = config_ucb
            best_error_ucb = val_err_ucb

    # Random Strategy
    config_random_XGB = random.choice(param_space_XGB)
    rand_model_XGB = xgb.XGBClassifier(**config_random_XGB)

    rand_model_XGB.fit(X_train, y_train)
    pred_rand_XGB = rand_model_XGB.predict(X_test)

    error_rand_XGB = accuracy_score(y_test, pred_rand_XGB)
    val_err_rand_XGB = 1 - error_rand_XGB

    if val_err_rand_XGB < best_error_random_XGB:
        best_config_random_XGB = config_random_XGB
        best_error_random_XGB = val_err_rand_XGB

    print("Dataset: Titanic Dataset")

    print("\nUCB Strategy on XGBoosting:")
    print("Best validation error:", best_error_ucb)
    print("Best hyperparameter configuration:", best_config_ucb)

    print("\nRandom Strategy on XGBoosting:")
    print("Best validation error:", best_error_random_XGB)
    print("Best hyperparameter configuration:", best_config_random_XGB)

    return best_config_ucb, best_config_random_XGB, pred_rand_XGB




def SVC_Random(X_train, y_train, X_test, y_test, svc_param_space):
    rewards = [0] * len(svc_param_space)
    attempts = [0] * len(svc_param_space)
    best_config_svc = None
    best_error_svc = float('inf')

    best_config_random_svc = None
    best_error_random_svc = float('inf')

    max_iterations = 500
    c = 2  # Exploration parameter

    for t in range(1, max_iterations + 1):
        config_idx_svc = ucb_selection(rewards, attempts, t, c)
        config_svc = svc_param_space[config_idx_svc]
        svc_model = SVC(**config_svc)

        svc_model.fit(X_train, y_train)
        svc_pred = svc_model.predict(X_test)

        error_svc = accuracy_score(y_test, svc_pred)
        val_err_svc = 1 - error_svc

        rewards[config_idx_svc] += error_svc
        attempts[config_idx_svc] += 1

        if val_err_svc < best_error_svc:
            best_config_svc = config_svc
            best_error_svc = val_err_svc

    # Random Strategy
    config_random_svc = random.choice(svc_param_space)
    rand_model_svc = SVC(**config_random_svc)

    rand_model_svc.fit(X_train, y_train)
    pred_rand_svc = rand_model_svc.predict(X_test)

    error_rand_svc = accuracy_score(y_test, pred_rand_svc)
    val_err_rand_svc = 1 - error_rand_svc

    if val_err_rand_svc < best_error_random_svc:
        best_config_random_svc = config_random_svc
        best_error_random_svc = val_err_rand_svc

    print("Dataset: Titanic Dataset")

    print("\nUCB Strategy for SVM:")
    print("Best validation error:", best_error_svc)
    print("Best hyperparameter configuration:", best_config_svc)

    print("\nRandom Strategy for SVM:")
    print("Best validation error:", best_error_random_svc)
    print("Best hyperparameter configuration:", best_config_random_svc)

    return best_config_svc, best_config_random_svc, pred_rand_svc




def RandomForest_Random(X_train, y_train, X_test, y_test, rf_param_space):
    rewards = [0] * len(rf_param_space)
    attempts = [0] * len(rf_param_space)
    best_config_rf = None
    best_error_rf = float('inf')

    best_config_random_rf = None
    best_error_random_rf = float('inf')

    max_iterations = 500
    c = 2  # Exploration parameter

    for t in range(1, max_iterations + 1):
        config_idx_rf = ucb_selection(rewards, attempts, t, c)
        config_rf = rf_param_space[config_idx_rf]
        rf_model = RandomForestClassifier(**config_rf)

        rf_model.fit(X_train, y_train)
        rf_predict = rf_model.predict(X_test)

        error_rf = accuracy_score(y_test, rf_predict)
        val_err_rf = 1 - error_rf

        rewards[config_idx_rf] += error_rf
        attempts[config_idx_rf] += 1

        if val_err_rf < best_error_rf:
            best_config_rf = config_rf
            best_error_rf = val_err_rf

    # Random Strategy
    config_random_rf = random.choice(rf_param_space)
    rand_model_rf = RandomForestClassifier(**config_random_rf)

    rand_model_rf.fit(X_train, y_train)
    pred_rand_rf = rand_model_rf.predict(X_test)

    error_rand_rf = accuracy_score(y_test, pred_rand_rf)
    val_err_rand_rf = 1 - error_rand_rf

    if val_err_rand_rf < best_error_random_rf:
        best_config_random_rf = config_random_rf
        best_error_random_rf = val_err_rand_rf

    print("Dataset: Titanic Dataset")

    print("\nUCB Strategy for RandomForest:")
    print("Best validation error:", best_error_rf)
    print("Best hyperparameter configuration:", best_config_rf)

    print("\nRandom Strategy for RandomForest:")
    print("Best validation error:", best_error_random_rf)
    print("Best hyperparameter configuration:", best_config_random_rf)

    return best_config_rf, best_config_random_rf, pred_rand_rf


