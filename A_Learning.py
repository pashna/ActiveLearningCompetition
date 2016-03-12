# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import commands
from random import randint
from sklearn.cluster import DBSCAN
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from time import time
from random import random
import datetime

df = pd.read_csv("X_public.csv", sep=",")

def oracle(x):
    query = "java -cp OracleRegression.jar Oracle " + str(x[1]) + " " + str(x[2]) + " " + str(x[3]) + " " + \
            str(x[4]) + " " + str(x[5]) + " " + " " + str(x[6]) + " " + str(x[7]) + " " + \
            str(x[8]) + " " + str(x[9]) + " " + str(x[10])

    return commands.getoutput(query)

y = np.load("y_server.npy")

X = df.as_matrix()
"""
mask = np.invert(np.isnan(y))
X = X[mask]
""";

"""
X = df.as_matrix()
dbscan = DBSCAN(eps=10, min_samples=8)
"""

#y_c = dbscan.fit_predict(X)

mask_unlabeled = np.isnan(y)
mask_labeled = np.invert(mask_unlabeled)

indexes_unlabeled = mask_unlabeled.nonzero()[0]
indexes_labeled = mask_labeled.nonzero()[0]

X_labeled = X[indexes_labeled]
y_labeled = y[indexes_labeled]

X_unlabeled = X[indexes_unlabeled]

def update_X_y():
    # Так будет удобнее, не хочу менять
    global X_labeled
    global y_labeled
    global X_unlabeled
    global indexes_labeled
    global indexes_unlabeled
    
    X_labeled = X[indexes_labeled]
    y_labeled = y[indexes_labeled]
    
    X_unlabeled = X[indexes_unlabeled]

def learn_active_by_error_prediction():
    """
    Функция внутри предсказывает дважды. Сначала обучается на уже размеченных данных.
    Затем, обучается на ошибке предсказаний и возвращает индекс максимального неразмеченного элемента (по unlabeled_indexes)
    
    """
    
    # Учим модель предсказывать то, что уже есть
    gb = GradientBoostingRegressor(n_estimators=5, max_depth=3)
    gb.fit(X_labeled, y_labeled)
    y_predicted = gb.predict(X_labeled)
    
    # Считаем ошибку
    error = np.absolute(y_labeled - y_predicted)
    
    # Учим предсказывать ошибку
    gb = GradientBoostingRegressor(n_estimators=5, max_depth=3)
    gb.fit(X_labeled, error)
    y_predicted_error = gb.predict(X_unlabeled)
    
    max_unlabeled_index = np.argmax(y_predicted_error)
    return max_unlabeled_index

def get_new_value(max_unlabeled_index):
    # Так будет удобнее, не хочу менять
    global indexes_unlabeled
    global indexes_labeled
    
    abs_index = indexes_unlabeled[max_unlabeled_index]

    x_to_oracle = X[abs_index]
    new_y = oracle(x_to_oracle)

    y[abs_index] = new_y

    indexes_unlabeled = np.delete(indexes_unlabeled, max_unlabeled_index)
    indexes_labeled = np.append(indexes_labeled, abs_index)

def learn_active_by_max_variance():
    """
    Строим много деревьев. Предсказываем все. Берем тот объект, для которого получили максимальную дисперсию предсказаний.
    
    """
    rf = RandomForestRegressor(n_estimators=20, max_depth=7, n_jobs=4)
    rf.fit(X_labeled, y_labeled)
    pr = []
    for est in rf.estimators_:
        pr.append(est.predict(X_unlabeled))

    pr = np.asarray(pr)
    variance = np.var(pr, axis=0)

    max_unlabeled_index = np.argmax(variance)
    return max_unlabeled_index

def save_y():
    np.save("ActiveLearning_y", y)

i = 0
while True:
    i += 1
    update_X_y()
    
    if random() < 0.25:
        unlabeled_index = learn_active_by_max_variance()
    else:
        unlabeled_index = learn_active_by_error_prediction()

        
    get_new_value(unlabeled_index)
    
    if i % 200 == 0:
        save_y()
        print "Y was saved at ", datetime.datetime.now()
