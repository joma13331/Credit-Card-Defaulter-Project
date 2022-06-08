import os
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix


class CCDModelFinderTrain:
    """
    :Class Name: CCDModelFinderTrain
    :Description: This class will be used to train different models and select the best one
                  amongst them.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self):

        if not os.path.isdir("CCDLogFiles/training/"):
            os.mkdir("CCDLogFiles/training/")

        self.log_path = "CCDLogFiles/training/CCDModelFinderTrain.txt"

        self.ccd_model_finder_logging = logging.getLogger("ccd_model_finder_log")
        self.ccd_model_finder_logging.setLevel(logging.INFO)
        ccd_model_finder_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        ccd_model_finder_handler.setFormatter(formatter)
        self.ccd_model_finder_logging.addHandler(ccd_model_finder_handler)

        self.operation = 'TRAINING'

        self.rfc = RandomForestClassifier(n_jobs=-1, verbose=0)
        self.xgb = XGBClassifier(n_jobs=-1, objective='binary:logistic')
        self.logistic_regression = LogisticRegression(n_jobs=-1, max_iter=10000)
        self.svc = SVC()
        self.kfold = KFold(shuffle=True, random_state=42)

    def ccd_best_logistic_regressor(self, train_x, train_y):
        """
        :Method Name: ccd_best_logistic_regressor
        :Description: This method trains and returns the best model amongst many trained logistic regressors.

        :param train_x: Input training Data
        :param train_y: Input training labels
        :return: The best logistic regressor model
        :On failure: Exception
        """

        try:
            param_grid = {
                'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']
            }
            message = f"{self.operation}: Using GridSearchCV to obtain the optimum parameters({param_grid.keys()})" \
                      f"  of Logistic Regressor"
            self.ccd_model_finder_logging.info(message)

            # GridSearchCV is used as there are only a few combination of parameters.
            grid = GridSearchCV(estimator=self.logistic_regression, param_grid=param_grid,
                                cv=self.kfold, n_jobs=-1,
                                scoring='f1',
                                verbose=0)

            grid.fit(train_x, train_y)

            c = grid.best_params_['C']
            penalty = grid.best_params_['penalty']
            solver = grid.best_params_['solver']
            # l1_ratio = grid.best_params_['l1_ratio']
            score = grid.best_score_

            message = f"{self.operation}: The optimum parameters of Logistic Regressor are C={c}, penalty={penalty}," \
                      f"solver={solver}  with the f1 score of {score}"

            self.ccd_model_finder_logging.info(message)

            self.logistic_regression = LogisticRegression(C=c, penalty=penalty, solver=solver,
                                                          )
            self.logistic_regression.fit(train_x, train_y)

            message = f"{self.operation}: Best Logistic  Regressor trained"
            self.ccd_model_finder_logging.info(message)

            return self.logistic_regression

        except Exception as e:
            message = f"{self.operation}: There was a problem while fitting Logistic Regressor: {str(e)}"
            self.ccd_model_finder_logging.error(message)
            raise e

    def ccd_best_svc(self, train_x, train_y):
        """
        :Method Name: ccd_best_svc
        :Description: This method trains and returns the best model amongst many trained SVC.

        :param train_x: Input training Data
        :param train_y: Input training labels
        :return: The best SVC model
        :On failure: Exception
        """

        try:

            param_grid = {
                'C': [ 0.3, 1, 3, 10],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': [2, 3, 4, 5],
                'gamma': ['scale', 'auto']
            }

            message = f"{self.operation}: Using GridSearchCV to obtain the optimum parameters({param_grid.keys()})" \
                      f" of SVC"
            self.ccd_model_finder_logging.info(message)

            # GridSearchCV is used as there are only a few combination of parameters.
            grid = GridSearchCV(estimator=self.svc, param_grid=param_grid,
                                cv=self.kfold, n_jobs=-1,
                                scoring='f1',
                                verbose=0)

            grid.fit(train_x, train_y)

            kernel = grid.best_params_['kernel']
            gamma = grid.best_params_['gamma']
            c = grid.best_params_['C']
            degree = grid.best_params_['degree']

            score = grid.best_score_

            message = f"{self.operation}: The optimum parameters of SVC are kernel={kernel}, gamma={gamma}, C={c}," \
                      f" degree ={degree} with the f1 score of {score}"
            self.ccd_model_finder_logging.info(message)

            self.svc = SVC(kernel=kernel, gamma=gamma, C=c, degree=degree)
            self.svc.fit(train_x, train_y)

            message = f"{self.operation}: Best SVC trained"
            self.ccd_model_finder_logging.info(message)

            return self.svc

        except Exception as e:
            message = f"{self.operation}: There was a problem while fitting SVC: {str(e)}"
            self.ccd_model_finder_logging.error(message)
            raise e

    def ccd_best_random_forest(self, train_x, train_y):
        """
        :Method Name: ccd_best_random_forest
        :Description: This method trains and returns the best model amongst many trained random forest classifier.

        :param train_x: Input training Data
        :param train_y: Input training labels
        :return: The best random forest classifier model
        :On failure: Exception"""

        try:
            param_grid = {
                'n_estimators': [100, 130, 150, 300],
                'criterion': ['gini', 'entropy'],
                'min_samples_split': [2, 3, 4],
                'max_features': ['auto', 'sqrt', 'log2'],
                'ccp_alpha': np.arange(0, 0.005, 0.001)
            }

            message = f"{self.operation}: Using GridSearchCV to obtain the optimum parameters({param_grid.keys()})" \
                      f" of random forest classifier "
            self.ccd_model_finder_logging.info(message)

            # RandomSearchCV is used as there are a large number combination of parameters.
            grid = RandomizedSearchCV(estimator=self.rfc, param_distributions=param_grid, n_iter=500,
                                      cv=self.kfold, n_jobs=-1,
                                      scoring='f1',
                                      verbose=0)

            grid.fit(train_x, train_y)

            n_estimators = grid.best_params_['n_estimators']
            criterion = grid.best_params_['criterion']
            min_samples_split = grid.best_params_['min_samples_split']
            max_features = grid.best_params_['max_features']
            ccp_alpha = grid.best_params_['ccp_alpha']
            score = grid.best_score_

            message = f"{self.operation}: The optimum parameters of random forrest classifier are " \
                      f"n_estimators={n_estimators}, criterion={criterion}, min_samples_split={min_samples_split}," \
                      f" max_features ={max_features}, ccp_alpha={ccp_alpha} with the adjusted R2 score of {score}"
            self.ccd_model_finder_logging.info(message)

            self.rfc = RandomForestClassifier(n_jobs=-1, verbose=0,
                                              n_estimators=n_estimators, criterion=criterion,
                                              min_samples_split=min_samples_split,
                                              max_features=max_features, ccp_alpha=ccp_alpha
                                              )

            self.rfc.fit(train_x, train_y)

            message = f"{self.operation}: Best random forest classifier trained"
            self.ccd_model_finder_logging.info(message)

            return self.rfc

        except Exception as e:
            message = f"{self.operation}: There was a problem while fitting Random Forest classifiers: {str(e)}"
            self.ccd_model_finder_logging.error(message)
            raise e

    def ccd_best_xgb_classifier(self, train_x, train_y):
        """
        :Method Name: ccd_best_xgb_classifier
        :Description: This method trains and returns the best model amongst many trained xgb classifiers.

        :param train_x: Input training Data
        :param train_y: Input training labels
        :return: The best xgb classifier model
        :On failure: Exception
        """

        try:
            param_grid = {
                'learning_rate': [0.01, 0.03, 0.1],
                'colsample_bytree': [.4, .5, .7, .8],
                'max_depth': [10, 15, 20],
                'n_estimators': [300, 1000, 3000],
                "verbosity": [1]
            }

            message = f"{self.operation}: Using GridSearchCV to obtain the optimum parameters({param_grid.keys()}) " \
                      f"of xgb classifier"
            self.ccd_model_finder_logging.info(message)

            grid = RandomizedSearchCV(estimator=self.xgb, param_distributions=param_grid, n_iter=250,
                                      cv=self.kfold, n_jobs=-1,
                                      scoring='f1', verbose=1)

            grid.fit(train_x, train_y)

            learning_rate = grid.best_params_['learning_rate']
            colsample_bytree = grid.best_params_['colsample_bytree']
            max_depth = grid.best_params_['max_depth']
            n_estimators = grid.best_params_['n_estimators']
            score = grid.best_score_

            message = f"{self.operation}: The optimum parameters of xgb-classifier are learning_rate={learning_rate}, "\
                      f"max_depth={max_depth}, colsample_bytree={colsample_bytree}, n_estimators ={n_estimators} " \
                      f"with the adjusted R2 score of {score}"
            self.ccd_model_finder_logging.info(message)

            self.xgb = XGBClassifier(n_jobs=-1, verbose=0, learning_rate=learning_rate,
                                     colsample_bytree=colsample_bytree,
                                     max_depth=max_depth, n_estimators=n_estimators)

            self.xgb.fit(train_x, train_y)

            message = f"{self.operation}: Best xgb classifier trained"
            self.ccd_model_finder_logging.info(message)
            return self.xgb

        except Exception as e:
            message = f"{self.operation}: There was a problem while fitting Random Forest Regressor: {str(e)}"
            self.ccd_model_finder_logging.error(message)
            raise e

    def ccd_best_model_from_roc_auc(self, roc_auc_scores):
        """
        :Method Name: ccd_best_model_from_roc_auc
        :Description: This method takes in a dictionary with model name as keys and roc_auc score as values,
                      it then returns the best model based on highest roc_auc score.

        :param roc_auc_scores: The dictionary of all roc auc scores
        :return: The best sklearn model for the given dataset
        :On Failure: Exception
        """
        try:
            keys = list(roc_auc_scores.keys())
            values = list(roc_auc_scores.values())
            ind = values.index(max(values))

            if keys[ind] == "logistic":
                message = f"{self.operation}: The best model is logistic regressor"
                self.ccd_model_finder_logging.info(message)
                return keys[ind], self.logistic_regression

            elif keys[ind] == "svc":
                message = f"{self.operation}: The best model is svc"
                self.ccd_model_finder_logging.info(message)
                return keys[ind], self.svc

            elif keys[ind] == "rfc":
                message = f"{self.operation}: The best model is random forest classifier"
                self.ccd_model_finder_logging.info(message)
                return keys[ind], self.rfc

            else:
                message = f"{self.operation}: The best model is xgb classifier"
                self.ccd_model_finder_logging.info(message)
                return keys[ind], self.xgb

        except Exception as e:
            message = f"{self.operation}: There was a problem while obtaining best model from adjusted r2 " \
                      f"dictionary: {str(e)}"
            self.ccd_model_finder_logging.error(message)
            raise e

    def ccd_best_model(self, train_x, train_y, test_x, test_y):
        """
        :Method Name: ccd_best_model
        :Description: This method is used to select the best model from all the best model from all categories.

        :param train_x: the training features
        :param train_y: the training labels
        :param test_x: the test features
        :param test_y: the test labels
        :return: The best sklearn model for the given dataset
        :On Failure: Exception
        """

        try:

            message = f"{self.operation}: Search for best model started"
            self.ccd_model_finder_logging.info(message)

            roc_auc_scores = {}
            f1_scores = {}
            accuracy_scores = {}
            confusion_matrices = {}

            message = f"{self.operation}: Search for best logistic regressor model started"
            self.ccd_model_finder_logging.info(message)

            self.logistic_regression = self.ccd_best_logistic_regressor(train_x, train_y)
            y_pred = self.logistic_regression.predict(test_x)
            roc_auc_scores["logistic"] = roc_auc_score(y_true=test_y, y_score=y_pred)
            f1_scores["logistic"] = f1_score(y_true=test_y, y_pred=y_pred)
            accuracy_scores["logistic"] = accuracy_score(y_true=test_y, y_pred=y_pred)
            confusion_matrices["logistic"] = confusion_matrix(y_true=test_y, y_pred=y_pred)

            message = f"{self.operation}: Search for best ridge model ended"
            self.ccd_model_finder_logging.info(message)

            message = f"{self.operation}: Search for best svc model started"
            self.ccd_model_finder_logging.info(message)

            self.svc = self.ccd_best_svc(train_x, train_y)
            y_pred = self.svc.predict(test_x)
            roc_auc_scores["svc"] = roc_auc_score(y_true=test_y, y_score=y_pred)
            f1_scores["svc"] = f1_score(y_true=test_y, y_pred=y_pred)
            accuracy_scores["svc"] = accuracy_score(y_true=test_y, y_pred=y_pred)
            confusion_matrices["svc"] = confusion_matrix(y_true=test_y, y_pred=y_pred)

            message = f"{self.operation}: Search for best svc model ended"
            self.ccd_model_finder_logging.info(message)

            message = f"{self.operation}: Search for best random forest classifier model started"
            self.ccd_model_finder_logging.info(message)

            self.rfc = self.ccd_best_random_forest(train_x, train_y)
            y_pred = self.rfc.predict(test_x)
            roc_auc_scores["rfc"] = roc_auc_score(y_true=test_y, y_score=y_pred)
            f1_scores["rfc"] = f1_score(y_true=test_y, y_pred=y_pred)
            accuracy_scores["rfc"] = accuracy_score(y_true=test_y, y_pred=y_pred)
            confusion_matrices["rfc"] = confusion_matrix(y_true=test_y, y_pred=y_pred)

            message = f"{self.operation}: Search for best random forest classifier model ended"
            self.ccd_model_finder_logging.info(message)

            message = f"{self.operation}: Search for best xgb classifier model started"
            self.ccd_model_finder_logging.info(message)

            self.xgb = self.ccd_best_xgb_classifier(train_x, train_y)
            y_pred = self.xgb.predict(test_x)
            roc_auc_scores["xgb"] = roc_auc_score(y_true=test_y, y_score=y_pred)
            f1_scores["xgb"] = f1_score(y_true=test_y, y_pred=y_pred)
            accuracy_scores["xgb"] = accuracy_score(y_true=test_y, y_pred=y_pred)
            confusion_matrices["xgb"] = confusion_matrix(y_true=test_y, y_pred=y_pred)

            message = f"{self.operation}: Search for best xgb classifier model ended"
            self.ccd_model_finder_logging.info(message)

            print(f"roc_auc scores: {roc_auc_scores}")
            print(f"f1 scores:{f1_scores}")
            print(f"accuracy scores:{accuracy_scores}")
            print(f"confusion matrices: {confusion_matrices}")

            return self.ccd_best_model_from_roc_auc(roc_auc_scores)

        except Exception as e:
            message = f"{self.operation}: There was a problem while obtaining best model : {str(e)}"
            self.ccd_model_finder_logging.error(message)
            raise e
