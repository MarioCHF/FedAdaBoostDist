#import os
from typing import Any, Tuple, Dict, List, Generator, Optional
from copy import deepcopy
from math import log
#import tqdm
#import pandas as pd
import numpy as np
from numpy.random import choice, permutation
#from sklearn import datasets
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.datasets import load_svmlight_file
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler, LabelEncoder
#import matplotlib.pyplot as plt
#from optparse import OptionParser, Values
#import dload
#import urllib.request as ureq
#import json
#import gzip
#import wandb




class Boosting():
    def __init__(self,
                 n_clf: int=10,
                 clf_class: ClassifierMixin=DecisionTreeClassifier()):
        self.n_clf = n_clf
        self.clf_class = clf_class
        self.clfs: List[ClassifierMixin] = []
        self.alpha: List[ClassifierMixin] = []
    
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            checkpoints: Optional[List[int]]=None,
            seed: int=42):
        raise NotImplementedError()
    
    def num_weak_learners(self):
        return len(self.clfs)
    
    def predict(self,
                X: np.ndarray) -> np.ndarray:
        y_pred = np.zeros(np.shape(X)[0])
        for i, clf in enumerate(self.clfs):
            y_pred += self.alpha[i] * clf.predict(X)
        return 2*(y_pred.flatten() >= 0).astype(int) - 1 


class MulticlassBoosting(Boosting):
    def __init__(self,
                 n_clf: int = 10,
                 clf_class: ClassifierMixin = DecisionTreeClassifier()):
        super(MulticlassBoosting, self).__init__(n_clf, clf_class)
        self.K = None  # to be defined in the fit method

    def predict(self: Boosting,
                X: np.ndarray) -> np.ndarray:
        y_pred = np.zeros((np.shape(X)[0], self.K))
        for i, clf in enumerate(self.clfs):
            pred = clf.predict(X)
            for j, c in enumerate(pred):
                y_pred[j, int(c)] += self.alpha[i]
        return np.argmax(y_pred, axis=1)


# Centralized Multiclass AdaBoost
class Samme(MulticlassBoosting):
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            checkpoints: Optional[List[int]] = None,
            seed: int = 42,
            num_labels = None):

        np.random.seed(seed)
        self.K = len(set(y)) if not num_labels else num_labels # assuming that all classes are in y
        cks = set(checkpoints) if checkpoints is not None else [self.n_clf]
        n_samples = X.shape[0]
        D = np.full(n_samples, (1 / n_samples))
        self.clfs = []
        self.alpha = []
        for t in range(self.n_clf):
            clf = deepcopy(self.clf_class)
            ids = choice(n_samples, size=n_samples, replace=True, p=D)
            X_, y_ = X[ids], y[ids]
            clf.fit(X_, y_)

            predictions = clf.predict(X)
            min_error = np.sum(D[y != predictions]) / np.sum(D)
            # kind of additive smoothing + mi cambio
            if (min_error <1) & (self.K > 1):
                self.alpha.append(log((1.0 - min_error) / (min_error + 0.005)) + log(self.K-1))
            elif (min_error <1) & ((self.K <= 1)):
                self.alpha.append(log((1.0 - min_error) / (min_error + 0.005)))
            else:
                self.alpha.append(log((1.0 - min_error+0.0005) / (min_error + 0.005)))
            D *= np.exp(self.alpha[t] * (y != predictions))
            D /= np.sum(D)
            self.clfs.append(clf)

            if (t+1) in cks:
                yield self


# Support class for Distboost: 
# commitee of classifiers, i.e., weak hypothesis of DistBoost/DistSamme
class Hyp():
    def __init__(self,
                 ht: List[ClassifierMixin],
                 K: int):
        self.ht = ht
        self.K = K

    def predict(self,
                X: np.ndarray) -> np.ndarray:
        y_pred = np.zeros((X.shape[0], self.K))
        for h in self.ht:
            pred = h.predict(X)
            for j, c in enumerate(pred):
                y_pred[j, int(c)] += 1
        return np.argmax(y_pred, axis=1)


class DistSamme(MulticlassBoosting):
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            checkpoints: Optional[List[int]] = None,
            seed: int = 42):

        np.random.seed(seed)
        # assuming that all classes are in y
        self.K = len(set(np.concatenate(y)))
        cks = set(checkpoints) if checkpoints is not None else [self.n_clf]
        n_samples = sum([x.shape[0] for x in X])
        # Distribution over examples for each client. Init: uniform
        D = [np.full(x.shape[0], (1 / n_samples)) for x in X]
        self.clfs = []
        self.alpha = []
        for t in range(self.n_clf):
            ht = []
            # Client-side
            for j, X_ in enumerate(X):
                clf = deepcopy(self.clf_class)
                ids = choice(X_.shape[0], size=X_.shape[0],
                             replace=True, p=D[j]/np.sum(D[j]))
                X__, y__ = X_[ids], y[j][ids]
                clf.fit(X__, y__)
                ht.append(clf)
            
            # Server-side
            H = Hyp(ht, self.K)
            self.clfs.append(H)

            min_error = 0
            predictions = []
            # This is computed partially on the clients and aggregated on the server 
            for j, X_ in enumerate(X):
                predictions.append(H.predict(X_))
                # / np.sum(D[j])
                min_error += np.sum(D[j][y[j] != predictions[j]])
            # kind of additive smoothing + añado yo que el error no sea mayor que self.K (número de clases)
            if (min_error <1):
                self.alpha.append(
                    log((1.0 - min_error) / (min_error + 0.005)) + log(self.K-1))
            else:
                self.alpha.append(log((1.0 - min_error+0.0005) / (min_error + 0.005)))

            for j, X_ in enumerate(X):
                D[j] *= np.exp(self.alpha[t] * (y[j] != predictions[j]))
            Dsum = sum([np.sum(d) for d in D])

            for d in D:
                d /= Dsum

            if (t+1) in cks:
                yield self


class PreweakSamme(MulticlassBoosting):
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            checkpoints: Optional[List[int]] = None,
            seed: int = 42):

        np.random.seed(seed)
        # assuming that all classes are in y
        self.K = len(set(np.concatenate(y)))
        cks = set(checkpoints) if checkpoints is not None else [self.n_clf]
        ht = []
        # Client-side
        for j, X_ in enumerate(X):
            clf = Samme(self.n_clf, self.clf_class)
            for h in clf.fit(X_, y[j], checkpoints):
                pass
            ht.extend(clf.clfs)

        # merge the datasets into one (not possible in a real distributed/federated scenario)
        X_ = np.vstack(X)
        y_ = np.concatenate(y)

        # precompute the predictions so then I simply have to draw according to the sampling
        ht_pred = {h: h.predict(X_) for h in ht}
        n_samples = X_.shape[0]
        D = np.full(n_samples, (1 / n_samples))
        self.clfs = []
        self.alpha = []
        for t in range(self.n_clf):
            ids = choice(X_.shape[0], size=X_.shape[0], replace=True, p=D)
            y__ = y_[ids]

            min_error = 1000
            top_model = None
            # This is computed partially on the clients and aggregated on the server 
            for h, hpred in ht_pred.items():
                err = np.sum(D[y__ != hpred[ids]])  # / np.sum(D)
                if err < min_error:
                    top_model = h
                    min_error = err

            # kind of additive smoothing + lo que yo añado
            if (min_error <1):
                self.alpha.append(log((1.0 - min_error) / (min_error + 0.005)) + log(self.K-1))
            else: 
                self.alpha.append(log((1.0 - min_error+0.0005) / (min_error + 0.005)))
            D *= np.exp(self.alpha[t] * (y_ != ht_pred[top_model]))
            D /= np.sum(D)
            self.clfs.append(top_model)

            if (t+1) in cks:
                yield self


class AdaboostF1(MulticlassBoosting):

    def federated_dist(self,
                       D: np.ndarray,
                       X: List[np.ndarray],
                       j: int) -> np.ndarray:
        min_index = sum([len(X[i]) for i in range(j)])
        D_ = D[min_index: min_index + X[j].shape[0]]
        return D_ / sum(D_)

    def fit(self,
            X: List[np.ndarray],
            y: List[np.ndarray],
            checkpoints: Optional[List[int]] = None,
            seed: int = 42):

        np.random.seed(seed)
        # assuming that all classes are in y
        self.K = len(set(np.concatenate(y)))
        cks = set(checkpoints) if checkpoints is not None else [self.n_clf]

        # merge the datasets into one (not possible in a real distributed/federated scenario)
        X_ = np.vstack(X)
        y_ = np.concatenate(y)

        n_samples = X_.shape[0]
        D = np.full(n_samples, (1 / n_samples))

        self.clfs = []
        self.alpha = []

        for t in range(self.n_clf):
            fed_clfs = []
            # Client-side
            for j, X__ in enumerate(X):
                D_ = self.federated_dist(D, X, j)
                n_samples_ = X__.shape[0]

                clf = deepcopy(self.clf_class)
                ids = choice(n_samples_, size=n_samples_, replace=True, p=D_)
                clf.fit(X__[ids], y[j][ids])
                fed_clfs.append(clf)

            errors = np.array([sum(D[y_ != clf.predict(X_)])
                              for clf in fed_clfs])
            best_clf = fed_clfs[np.argmin(errors)]
            best_error = errors[np.argmin(errors)]

            # This is computed partially on the clients and aggregated on the server 
            predictions = best_clf.predict(X_)

            # kind of additive smoothing + lo que yo añado
            if (best_error <1):
                self.alpha.append(
                    log((1.0 - best_error) / (best_error + 0.005)) + log(self.K-1))
            else:
                self.alpha.append(log((1.0 - best_error+0.0005) / (best_error + 0.005)))
            D *= np.exp(self.alpha[t] * (y_ != predictions))

            D /= np.sum(D)
            self.clfs.append(best_clf)

            if (t+1) in cks:
                yield self


def polato_AdaBoost_eval(train_dict,test_dict, X_test,y_test, seed, model,max_leaf_nodes = None,max_depth=None,n_estimators=10):
    options = deepcopy(locals())
    """Given a dict with the training data of the clients, a dict with the test data of the clients, the global test, seed and model to
    run, it returns the acc and f1 scores of the federated model on the global test and the mean over all local tests"""
    WEAK_LEARNER = DecisionTreeClassifier(random_state=seed, max_leaf_nodes=max_leaf_nodes,max_depth=max_depth)


       
    if model == "distsamme":
        learner = DistSamme(n_estimators, WEAK_LEARNER)
       
    elif model == "preweaksamme":
        learner = PreweakSamme(n_estimators, WEAK_LEARNER)
       
    elif model == "adaboost.f1":
        learner = AdaboostF1(n_estimators, WEAK_LEARNER)
        
    else:
        raise ValueError("Unknown model %s." % model.value)

    X_,y_ = [],[]
    for x_train,y_train in train_dict.values():
        X_.append(x_train)
        y_.append(y_train)
    Classifier = learner.fit(X_, y_)
    strong_learner = next(iter(Classifier))
    y_pred_te = strong_learner.predict(X_test)
    n_clients = len(test_dict)
    f1_scores = np.zeros(n_clients)
    acc_scores = np.zeros(n_clients)
    for i,(X_test_local,y_test_local) in enumerate(test_dict.values()):
        y_pred = strong_learner.predict(X_test_local)
        acc_score = accuracy_score(y_test_local,y_pred)
        f1_score1 = f1_score(y_test_local,y_pred,labels = np.unique(y_test_local),average='weighted',zero_division=0.0)
        acc_scores[i] = acc_score
        f1_scores[i] = f1_score1
    
    acc_local = acc_scores.mean()*100
    f1_local = f1_scores.mean()*100
    acc_global = accuracy_score(y_test, y_pred_te)*100
    f1_global = f1_score(y_test, y_pred_te, labels= np.unique(y_test),average="weighted",zero_division=0.0)*100

    return acc_global,f1_global,acc_local,f1_local

