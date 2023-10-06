# data science pipelines

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import f1_score

from sklearn.metrics import auc

from sklearn.metrics import average_precision_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import fbeta_score

from sklearn.metrics import make_scorer

from sklearn.metrics import precision_recall_fscore_support


# import warnings filter

from warnings import simplefilter

# data science pipelines

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)

# ignore all warnings

simplefilter(action='ignore', category=Warning)


# define a function to create a data science pipeline

def create_pipeline(model, scaler=None, pca=None):
    """
    Create a data science pipeline
    :param model: model to use
    :param scaler: scaler to use
    :param pca: pca to use
    :return: pipeline
    """
    steps = []

    if scaler is not None:
        steps.append(('scaler', scaler))

    if pca is not None:
        steps.append(('pca', pca))

    steps.append(('model', model))

    pipeline = Pipeline(steps=steps)

    return pipeline


# define a function to create a data science pipeline with grid search

def create_pipeline_with_grid_search(model, scaler=None, pca=None, param_grid=None, cv=None, scoring=None):
    """
    Create a data science pipeline with grid search
    :param model: model to use
    :param scaler: scaler to use
    :param pca: pca to use
    :param param_grid: param grid to use
    :param cv: cv to use
    :param scoring: scoring to use
    :return: pipeline
    """
    steps = []

    if scaler is not None:
        steps.append(('scaler', scaler))

    if pca is not None:
        steps.append(('pca', pca))

    steps.append(('model', model))

    pipeline = Pipeline(steps=steps)

    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring=scoring)

    return grid


# define a function to create a data science pipeline with grid search and cross validation

def create_pipeline_with_grid_search_and_cross_validation(model, scaler=None, pca=None, param_grid=None, cv=None,
                                                            scoring=None):
        """
        Create a data science pipeline with grid search and cross validation
        :param model: model to use
        :param scaler: scaler to use
        :param pca: pca to use
        :param param_grid: param grid to use
        :param cv: cv to use
        :param scoring: scoring to use
        :return: pipeline
        """
        steps = []
    
        if scaler is not None:
            steps.append(('scaler', scaler))
    
        if pca is not None:
            steps.append(('pca', pca))
    
        steps.append(('model', model))
    
        pipeline = Pipeline(steps=steps)
    
        grid = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring=scoring)
    
        return grid


# define a function to create a data science pipeline with grid search and cross validation and scoring

def create_pipeline_with_grid_search_and_cross_validation_and_scoring(model, scaler=None, pca=None, param_grid=None,
                                                                        cv=None, scoring=None):
    """
    Create a data science pipeline with grid search and cross validation and scoring
    :param model: model to use
    :param scaler: scaler to use
    :param pca: pca to use
    :param param_grid: param grid to use
    :param cv: cv to use
    :param scoring: scoring to use
    :return: pipeline
    """
    steps = []

    if scaler is not None:
        steps.append(('scaler', scaler))

    if pca is not None:
        steps.append(('pca', pca))

    steps.append(('model', model))

    pipeline = Pipeline(steps=steps)

    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring=scoring)

    return grid


# define a function to create a data science pipeline with grid search and cross validation and scoring and refit

def create_pipeline_with_grid_search_and_cross_validation_and_scoring_and_refit(model, scaler=None, pca=None,
                                                                                param_grid=None, cv=None, scoring=None):
    """
    Create a data science pipeline with grid search and cross validation and scoring and refit
    :param model: model to use
    :param scaler: scaler to use
    :param pca: pca to use
    :param param_grid: param grid to use
    :param cv: cv to use
    :param scoring: scoring to use
    :return: pipeline
    """
    steps = []

    if scaler is not None:
        steps.append(('scaler', scaler))

    if pca is not None:
        steps.append(('pca', pca))

    steps.append(('model', model))

    pipeline = Pipeline(steps=steps)

    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring=scoring, refit=True)

    return grid


# define a function to create a data science pipeline with grid search and cross validation and scoring and refit and
# verbose

def create_pipeline_with_grid_search_and_cross_validation_and_scoring_and_refit_and_verbose(model, scaler=None,
                                                                                                pca=None,
                                                                                                param_grid=None,
                                                                                                cv=None,
                                                                                                scoring=None):
    """
    Create a data science pipeline with grid search and cross validation and scoring and refit and verbose
    :param model: model to use
    :param scaler: scaler to use
    :param pca: pca to use
    :param param_grid: param grid to use
    :param cv: cv to use
    :param scoring: scoring to use
    :return: pipeline
    """
    steps = []

    if scaler is not None:
        steps.append(('scaler', scaler))

    if pca is not None:
        steps.append(('pca', pca))

    steps.append(('model', model))

    pipeline = Pipeline(steps=steps)

    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring=scoring, refit=True, verbose=1)

    return grid


# applying these pipeline functions in iris dataset

from sklearn.datasets import load_iris

# load the iris dataset

iris = load_iris()

# create X (features) and y (response)

X = iris.data

y = iris.target

# create a train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# create a function to create a data science pipeline with grid search and cross validation and scoring and refit and
# verbose

def create_pipeline_with_grid_search_and_cross_validation_and_scoring_and_refit_and_verbose(model, scaler=None,
                                                                                                pca=None,
                                                                                                param_grid=None,
                                                                                                cv=None,
                                                                                                scoring=None):
    """
    Create a data science pipeline with grid search and cross validation and scoring and refit and verbose
    :param model: model to use
    :param scaler: scaler to use
    :param pca: pca to use
    :param param_grid: param grid to use
    :param cv: cv to use
    :param scoring: scoring to use
    :return: pipeline
    """
    steps = []

    if scaler is not None:
        steps.append(('scaler', scaler))

    if pca is not None:
        steps.append(('pca', pca))

    steps.append(('model', model))

    pipeline = Pipeline(steps=steps)

    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring=scoring, refit=True, verbose=1)

    return grid


# Define the model scaler pca param_grid cv scoring

model = LogisticRegression()

scaler = StandardScaler()

pca = PCA()

param_grid = {'pca__n_components': [1, 2, 3, 4]}

cv = 5

scoring = 'accuracy'

# create a pipeline with grid search and cross validation and scoring and refit and verbose

pipeline = create_pipeline_with_grid_search_and_cross_validation_and_scoring_and_refit_and_verbose(model=model,
                                                                                                    scaler=scaler,
                                                                                                    pca=pca,
                                                                                                    param_grid=param_grid,
                                                                                                    cv=cv,
                                                                                                    scoring=scoring)

# fit the pipeline

pipeline.fit(X_train, y_train)

# predict the response

y_pred = pipeline.predict(X_test)

# evaluate the accuracy

print(accuracy_score(y_test, y_pred))

# print the best parameters

print(pipeline.best_params_)

# print the best score

print(pipeline.best_score_)

# print the best estimator

print(pipeline.best_estimator_)

# print the best index


print(pipeline.best_index_)


# print the cv results

print(pipeline.cv_results_)


# print the refit time


print(pipeline.refit_time_)


# print the scorer

print(pipeline.scorer_)


# print the scoring function

print(pipeline.scoring)






                                                                                            

