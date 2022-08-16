import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier
from statistics import mode
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from utils import NlpUtils

AVG_METRIC = "macro"


class TrainClassModel:
    """
    :Date: 2022-08-14
    :Author: Joan Felipe Mendoza
    :Description: TRAIN A TEXT CLASIFICATION MODEL OPTIMIZING HYPERPARAMENTERS
    """
    def __init__(self, input_df, text_col, label_col, clean=True, val_size=0.2, balance=None, drop_duplicates=True):
        """
        :description: initialise process and do training/testing group splitting.
        :param input_df: input table.
               Must include a target variable. and numeric independent variables. (:type: pd.DataFrame)
        :param val_size: percentage size of the test group - between 0 and 1- (:type: float)
        :param balance: undersampling or None (:type: string)
        :param drop_duplicates: -true- duplicate records are removed (:type: boolean)
        """
        self.label = label_col
        df = input_df[[text_col, label_col]]
        if clean:
            df[text_col] = [NlpUtils().clean_text(str(x)) for x in df[text_col]]
        df_train, df_test = self.data_split(df, val_size, balance, drop_duplicates)
        df_train = df_train.drop_duplicates()
        self.X_train = df_train[text_col]
        self.Y_train = df_train[label_col]
        self.X_test = df_test[text_col]
        self.Y_test = df_test[label_col]
        self.vz = val_size
        self.best_params = None

    def fit_best_model(self, n_trials=100, n_splits=5, model_type="logit", metric="f1"):
        """
        :description: train and fit the best* model
        :param n_trials: number of hyperparameter combinations to test (:type: int)
        :param n_splits: number of "folds" for "cross-validation" (:type: int)
        :param metric: metric to optimise - "f1" or "accuracy" - (:type: string)
        :param model_type: name of the model to be trained - "random_forest", "svm", "lgbm"- (:type: string)
        :return: best model obtained, best metric on training, best metric on test
        """
        self.mt = model_type
        self.metric = metric
        model, train_metric = self.train(n_trials=n_trials, n_splits=n_splits)
        print(f"*** MODEL: {self.mt} ***")
        print(f"training metric -{self.metric}-: {train_metric}")
        pred = self.predict(model, self.X_test)
        if self.metric == "f1":
            test_metric = metrics.f1_score(self.Y_test, pred, average=AVG_METRIC)
        else:
            test_metric = metrics.accuracy_score(self.Y_test, pred)
        print(f"test metric -{self.metric}-: {test_metric}")
        print("Confusion Matrix for test set")
        print(metrics.classification_report(self.Y_test, pred))
        print(metrics.confusion_matrix(self.Y_test, pred))
        return model, train_metric, test_metric

    def data_split(self, dataset, test_size, balance=None, drop_duplicates=True):
        """
        :description: separate a dataset into training and test sets
        :param dataset: dataset (:type: pd.DataFrame)
        :param test_size: size of the test set - from 0 to 1- (:type: float)
        :param balance: undersampling or None (:type: string)
        :param drop_duplicates: -true - duplicate records are removed (:type: boolean)
        :return: training set and test set (:type: pandas.DataFrame)
        """
        if drop_duplicates is True:
            df = dataset.drop_duplicates()
            print(f"{len(dataset)-len(df)} duplicated rows removed")
        else:
            df = dataset.copy()
        if balance == "undersampling":
            min_freq = min(pd.value_counts(df[self.label]).to_frame()[self.label].to_list())
            train_freq = int(round((1 - test_size) * min_freq, 0))
            df_train = pd.concat(
                [resample(df[df[self.label] == c], replace=False, n_samples=train_freq) for c in df[self.label].unique()])
            df_test = df.merge(df_train, on=df_train.columns.to_list(), how='left', indicator=True)
            df_test = df_test.loc[df_test["_merge"] == 'left_only', df_test.columns != '_merge']
            print(f"Dataset balanced ONLY for training.")
        else:
            df_train, df_test = train_test_split(df, test_size=test_size, stratify=df[self.label])
        print(f"training set size: {len(df_train)} | test set size: {len(df_test)}")
        print(f"{self.label} for training set:\n"
              f"{df_train[self.label].value_counts()}")
        print(f"{self.label} for test set:\n"
              f"{df_test[self.label].value_counts()}")
        return df_train, df_test

    def train(self, n_trials=50, n_splits=5):
        self.optimize(n_trials)
        cv_model, metric = self.cv_model(n_splits)
        return cv_model, metric

    def optimize(self, n_trials=50):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)
        # Visualize parameter importances.
        optuna.visualization.plot_param_importances(study)
        # Visualize empirical distribution function
        optuna.visualization.plot_edf(study)
        self.best_params = study.best_params

    def objective(self, trial):
        train_x, test_x, train_y, test_y = train_test_split(self.X_train, self.Y_train, test_size=self.vz)
        if self.mt == "random_forest":
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 25, 75, step=10),
                'max_features': 'sqrt',
                'max_depth': trial.suggest_categorical('max_depth', [None, 10, 12, 14]),
                'min_samples_split': trial.suggest_categorical('min_samples_split', [3, 4, 5]),
                'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [2, 3, 4]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
            }
            model = Pipeline([
                ('tfidf', TfidfVectorizer(sublinear_tf=True, min_df=50)),
                ('clf', RandomForestClassifier(**param))
                ])
            model.fit(train_x, train_y)
        elif self.mt == "svm":
            param = {
                'C': trial.suggest_categorical('C', [1, 2, 3, 5, 8, 10, 15, 20, 25, 40, 60, 80, 100]),
                'shrinking': trial.suggest_categorical('shrinking', [True, False]),
                'kernel': trial.suggest_categorical('kernel', ["poly", "rbf", "sigmoid"])
            }
            model = Pipeline([
                ('tfidf', TfidfVectorizer(sublinear_tf=True, min_df=50)),
                ('clf', SVC(**param))
            ])
            model.fit(train_x, train_y)
        elif self.mt == "lgbm":
            param = {
                'learning_rate': trial.suggest_categorical('learning_rate', [.04, .05, .06, .07, .08, .09, .1]),
                'n_estimators': trial.suggest_int('n_estimators', 20, 60),
                'max_depth': trial.suggest_int('max_depth', 10, 20),
                'colsample_bytree': trial.suggest_categorical('colsample_bytree', [.65, .7, .75, .8]),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', .1, .4),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, .5),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 15),
            }
            model = Pipeline([
                ('tfidf', TfidfVectorizer(sublinear_tf=True, min_df=50)),
                ('clf', LGBMClassifier(**param))
            ])
            model.fit(train_x, train_y)
        else:
            param = {
                'penalty': trial.suggest_categorical('penalty', ['none', 'l2']),
            }
            model = Pipeline([
                ('tfidf', TfidfVectorizer(sublinear_tf=True, min_df=50)),
                ('clf', LogisticRegression(**param))
            ])
            model.fit(train_x, train_y)
        preds = model.predict(self.X_test)
        if self.metric == "f1":
            metric = metrics.f1_score(self.Y_test, preds, average=AVG_METRIC)
        else:
            metric = metrics.accuracy_score(self.Y_test, preds)
        return metric

    def cv_model(self, n_splits=5):
        kf = KFold(n_splits=n_splits, random_state=48, shuffle=True)
        cv_model, scores = [], []
        n = 0
        for trn_idx, test_idx in kf.split(self.X_train, self.Y_train):
            X_tr = self.X_train.iloc[trn_idx]
            X_val = self.X_train.iloc[test_idx]
            y_tr = self.Y_train.iloc[trn_idx]
            y_val = self.Y_train.iloc[test_idx]
            if self.mt == "random_forest":
                model = Pipeline([
                    ('tfidf', TfidfVectorizer(sublinear_tf=True, min_df=50)),
                    ('clf', RandomForestClassifier(**self.best_params))
                ])
                model.fit(X_tr, y_tr)
            elif self.mt == "svm":
                model = Pipeline([
                    ('tfidf', TfidfVectorizer(sublinear_tf=True, min_df=50)),
                    ('clf', SVC(**self.best_params))
                ])
                model.fit(X_tr, y_tr)
            elif self.mt == "lgbm":
                model = Pipeline([
                    ('tfidf', TfidfVectorizer(sublinear_tf=True, min_df=50)),
                    ('clf', LGBMClassifier(**self.best_params))
                ])
                model.fit(X_tr, y_tr)
            else:
                model = Pipeline([
                    ('tfidf', TfidfVectorizer(sublinear_tf=True, min_df=50)),
                    ('clf', LogisticRegression(**self.best_params))
                ])
                model.fit(X_tr, y_tr)
            cv_model.append(model)
            preds = model.predict(X_val)
            if self.metric == "f1":
                scores.append(metrics.f1_score(y_val, preds, average=AVG_METRIC))
            else:
                scores.append(metrics.accuracy_score(y_val, preds))
            print(n + 1, scores[n])
            n += 1
        return cv_model, np.mean(scores)

    @staticmethod
    def predict(cv_model, X_data, clean=False):
        if clean:
            X = [NlpUtils().clean_text(str(x)) for x in X_data]
        else:
            X = [str(x) for x in X_data]
        preds = []
        for model in cv_model:
            preds.append(model.predict(X))
        preds = np.array(preds).T.tolist()
        preds = [mode(p) for p in preds]
        return preds
