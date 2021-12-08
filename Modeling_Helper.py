
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, auc, roc_curve, f1_score
import itertools
import random
import dask_ml.model_selection as dcv
import lightgbm as lgb
import optuna


import warnings
warnings.simplefilter('always', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

class DataPrep:
    def __init__(
        self,
        data_path = './data/merged_data.csv',
        cont_vars = [
            'duration',
            'num_channels',
            'channel_mobile', 
            'channel_social', 
            'channel_web',
            'age',  
            'income',
            'membership_days',
            'difficulty'],
        cat_vars = [
            'offer_id',
            'offer_type',
            'offer_reward',
            'gender', 
            'membership_month', 
            'membership_year'],
        y_var = 'successful_offer'
    
    ):
        """
        Initialises DataPrep
        This class is used to prepare the data

        :param portfolio_path: (str) path to portfolio data
        :param profile_path: (str) path to profile data
        :param transcript_path: (str) path to transcript

        """
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path)
        self.cont_vars = cont_vars
        self.cat_vars = cat_vars
        self.y_var = y_var
    
        self.modeling_data = self.data[[
                'person',
                'offer_id',
                'time_received',
                'offer_type',
                'duration',
                'offer_reward',
                'difficulty',
                'num_channels',
                'channel_email', 
                'channel_mobile', 
                'channel_social', 
                'channel_web',
                'gender', 
                'age',  
                'income',
                'membership_days',
                'membership_month', 
                'membership_year',
                'successful_offer'   
            ]]

        self.features = self.cont_vars+self.cat_vars
    
    def prep_data_logistic(self):
        for i in self.cat_vars:
            y = pd.get_dummies(self.modeling_data[i], prefix=i, drop_first=True)
            y = y.astype('int64')
            self.features = self.features + y.columns.tolist()
            self.features = [x for x in self.features if x != i]
            self.modeling_data = pd.concat([self.modeling_data, y], axis=1)
            
    def prep_data_gbm(self):
        for i in self.cat_vars:
            self.modeling_data.loc[:, i] = self.modeling_data[i].astype('category')
            

class PerformanceAnalysis:
    def __init__(
        self,
        classifier, 
        data, 
        features, 
        y_var,
        which_data,
        prob
    ):
        """
        Initialises DataPrep
        This class is used to prepare the data

        :param portfolio_path: (str) path to portfolio data
        :param profile_path: (str) path to profile data
        :param transcript_path: (str) path to transcript

        """

        self.classifier = classifier
        self.data = data
        self.features = features
        self.y_var = y_var
        self.which_data = which_data
        self.prob = prob
        
    def plot_confusion_matrix(self):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        title='Confusion matrix'
        plt.figure(figsize = (6,6))
        plt.imshow(self.cnf_matrix, interpolation='nearest', cmap=plt.cm.YlGn) #BuGn
        plt.title(title, fontsize = 15)
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks([-0.5,1.5], [0,1], rotation=0)
    #     plt.yticks(tick_marks, [0,1])

        thresh = self.cnf_matrix.max() / 2.
        for i, j in itertools.product(range(self.cnf_matrix.shape[0]), range(self.cnf_matrix.shape[1])):
            plt.text(j, i, self.cnf_matrix[i, j],
                     horizontalalignment="center",
                     color="white" if self.cnf_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label', fontsize = 15)
        plt.xlabel('Predicted label', fontsize = 15)
        
    def perf_analysis(self):
        print('**************', 'Performance:', self.which_data, '****************', '\n')


        y_pred = self.classifier.predict(self.data[self.features])
        y_pred = y_pred.round(0)
        self.cnf_matrix = confusion_matrix(self.data[self.y_var], y_pred)
        self.plot_confusion_matrix()


        print("Accuracy:", round(accuracy_score(self.data[self.y_var], y_pred),2))
        print("Precision:", round(precision_score(self.data[self.y_var], y_pred),2))
        print("Recall:", round(recall_score(self.data[self.y_var], y_pred),2))
        print("F1:", round(f1_score(self.data[self.y_var], y_pred),2))

        if self.prob:

            fpr, tpr, thresholds = roc_curve(self.data[self.y_var], self.classifier.predict_proba(self.data[self.features])[:, 1])

        #     fpr, tpr, thresholds = roc_curve(data[y_var], model.predict(data[features]))
        else:
            fpr, tpr, thresholds = roc_curve(self.data[self.y_var], self.classifier.predict(self.data[self.features]))

        roc_auc = auc(fpr,tpr)

        # Plot ROC
        plt.figure(figsize = (8,6))
        plt.title(self.which_data+':ROC', fontsize = 15)
        plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
        plt.legend(loc='lower right', fontsize =15)
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([-0.1,1.0])
        plt.ylim([-0.1,1.01])
        plt.ylabel('True Positive Rate', fontsize = 15)
        plt.xlabel('False Positive Rate', fontsize = 15)
        plt.show()
        
class DataSplit:
    def __init__(
        self,
        unique_id, 
        data, 
        y_var,
        split_fraq,
        random_seed
    ):
        """
        Initialises DataPrep
        This class is used to prepare the data

        :param portfolio_path: (str) path to portfolio data
        :param profile_path: (str) path to profile data
        :param transcript_path: (str) path to transcript

        """

        self.unique_id = unique_id
        self.data = data
        self.y_var = y_var
        self.split_fraq = split_fraq
        self.random_seed = random_seed
        self.train_df = None
        self.test_df = None
        
    def split_data(self):
        ID_list = self.data[self.unique_id].unique().tolist()
        random.seed(self.random_seed)
        train_IDs = random.sample(ID_list, int(self.split_fraq*len(ID_list)))

        self.train_df = self.data[self.data[self.unique_id].isin(train_IDs)]
        self.test_df = self.data[~self.data[self.unique_id].isin(train_IDs)]
        
        return self.train_df, self.test_df
    
def bayesian_objective(trial, data, features, y_var, scale_pos_weight_val):

    learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.4, log = True)
    num_leaves = trial.suggest_int("num_leaves", 2, 60)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.2,1)
    reg_alpha = trial.suggest_float("reg_alpha", 1e-2, 5, log = True)
    reg_lambda = trial.suggest_float("reg_lambda", 1e-2, 5, log = True)
    max_depth = trial.suggest_int("max_depth", 2, 30)
    min_child_samples = trial.suggest_int("min_child_samples", 50, 600)
    subsample = trial.suggest_float("subsample", 0.2,1)
    n_estimators = trial.suggest_int("n_estimators", 500, 2000) 

    param ={
        'learning_rate': [learning_rate],
        'num_leaves': [num_leaves],
        'colsample_bytree': [colsample_bytree],
        'reg_alpha': [reg_alpha],
        'reg_lambda': [reg_lambda],
        'max_depth':[max_depth],
        'min_child_samples': [min_child_samples], 
        'subsample': [subsample],
        'n_estimators': [n_estimators],
        'verbose': [-1]
            }


    gbm_model = lgb.LGBMClassifier(objective = "binary", 
                                   metric = ["auc", "binary_error"], 
                                   random_state=2021, 
                                   n_jobs=-1, 
                                   scale_pos_weight = scale_pos_weight_val,
                              )
    gbm_gs = dcv.GridSearchCV(
        estimator=gbm_model, param_grid=param, 
        scoring='f1',
        cv=5,
        refit=True,
    )

    gbm_gs.fit(data[features], data[y_var], )

    return gbm_gs.best_score_



class BayesianOpt(object):
    def __init__(
        self,
        train_data,
        features,
        y_var,
        scale_pos_weight_val,
    ):
        """
        Initialises DataPrep
        This class is used to prepare the data

        :param portfolio_path: (str) path to portfolio data
        :param profile_path: (str) path to profile data
        :param transcript_path: (str) path to transcript

        """

        self.train_data = train_data
        self.features = features
        self.y_var = y_var
        self.scale_pos_weight_val = scale_pos_weight_val
        
    def __call__(self, trial):
    
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.4, log = True)
        num_leaves = trial.suggest_int("num_leaves", 2, 60)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.2,1)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-2, 5, log = True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-2, 5, log = True)
        max_depth = trial.suggest_int("max_depth", 2, 30)
        min_child_samples = trial.suggest_int("min_child_samples", 50, 600)
        subsample = trial.suggest_float("subsample", 0.2,1)
        n_estimators = trial.suggest_int("n_estimators", 500, 2000) 

        param ={
            'learning_rate': [learning_rate],
            'num_leaves': [num_leaves],
            'colsample_bytree': [colsample_bytree],
            'reg_alpha': [reg_alpha],
            'reg_lambda': [reg_lambda],
            'max_depth':[max_depth],
            'min_child_samples': [min_child_samples], 
            'subsample': [subsample],
            'n_estimators': [n_estimators],
            'verbose': [-1]
                }


        gbm_model = lgb.LGBMClassifier(objective = "binary", 
                                       metric = ["auc", "binary_error"], 
                                       random_state=2021, 
                                       n_jobs=-1, 
                                       scale_pos_weight = self.scale_pos_weight_val,
                                  )
        gbm_gs = dcv.GridSearchCV(
            estimator=gbm_model, param_grid=param, 
            scoring='f1',
            cv=5,
            refit=True,
        )

        gbm_gs.fit(self.train_data[self.features], self.train_data[self.y_var], )

        return gbm_gs.best_score_
       

