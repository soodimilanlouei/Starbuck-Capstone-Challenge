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

warnings.simplefilter("always", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


class DataPrep:
    def __init__(
            self,
            data_path="./data/merged_data.csv",
            cont_vars=[
                "duration",
                "num_channels",
                "channel_mobile",
                "channel_social",
                "channel_web",
                "age",
                "income",
                "membership_days",
                "difficulty"],
            cat_vars=[
                "offer_id",
                "offer_type",
                "offer_reward",
                "gender",
                "membership_month",
                "membership_year"],
            y_var="successful_offer"

    ):
        """
        Initialises DataPrep
        This class is used to prepare the data

        :param data_path: (str) path to merged data
        :param cont_vars: (list) list of continuous variables
        :param cat_vars: (list) list of categorical variables
        :param y_var: (str) response variable

        """
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path)
        self.cont_vars = cont_vars
        self.cat_vars = cat_vars
        self.y_var = y_var

        self.modeling_data = self.data[[
            "person",
            "offer_id",
            "time_received",
            "offer_type",
            "duration",
            "offer_reward",
            "difficulty",
            "num_channels",
            "channel_email",
            "channel_mobile",
            "channel_social",
            "channel_web",
            "gender",
            "age",
            "income",
            "membership_days",
            "membership_month",
            "membership_year",
            "successful_offer"
        ]]

        self.features = self.cont_vars + self.cat_vars

    def prep_data_logistic(self):
        """
        Create dummy variables

        """
        for i in self.cat_vars:
            y = pd.get_dummies(self.modeling_data[i], prefix=i, drop_first=True)
            y = y.astype("int64")
            self.features = self.features + y.columns.tolist()
            self.features = [x for x in self.features if x != i]
            self.modeling_data = pd.concat([self.modeling_data, y], axis=1)

    def prep_data_gbm(self):
        """
        Change the type of categorical variables to category

        """
        for i in self.cat_vars:
            self.modeling_data.loc[:, i] = self.modeling_data[i].astype("category")


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
        Initialises PerformanceAnalysis
        This class is used to prepare the data

        :param classifier: (Booster) predictive model object
        :param data: (dataframe) data
        :param features: (list) list of independent variables
        :param y_var: (str) response variable
        :param which_data: (str) specify whether train or test data
        :param prob: (boolean) whether predict probabilities or binary outcomes

        """

        self.classifier = classifier
        self.data = data
        self.features = features
        self.y_var = y_var
        self.which_data = which_data
        self.prob = prob

    def plot_confusion_matrix(self):
        """
        Print and plot the confusion matrix.
        """
        title = "Confusion matrix"
        plt.figure(figsize=(6, 6))
        plt.imshow(self.cnf_matrix, interpolation="nearest", cmap=plt.cm.YlGn)  # BuGn
        plt.title(title, fontsize=15)
        plt.colorbar()
        plt.xticks([-0.5, 1.5], [0, 1], rotation=0)

        thresh = self.cnf_matrix.max() / 2.
        for i, j in itertools.product(range(self.cnf_matrix.shape[0]), range(self.cnf_matrix.shape[1])):
            plt.text(j, i, self.cnf_matrix[i, j],
                     horizontalalignment="center",
                     color="white" if self.cnf_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel("True label", fontsize=15)
        plt.xlabel("Predicted label", fontsize=15)

    def perf_analysis(self):
        """
        Print performance metrics and plot the ROC plot.

        """
        print("**************", "Performance:", self.which_data, "****************", "\n")

        y_pred = self.classifier.predict(self.data[self.features])
        y_pred = y_pred.round(0)
        self.cnf_matrix = confusion_matrix(self.data[self.y_var], y_pred)
        self.plot_confusion_matrix()

        print("Accuracy:", round(accuracy_score(self.data[self.y_var], y_pred), 2))
        print("Precision:", round(precision_score(self.data[self.y_var], y_pred), 2))
        print("Recall:", round(recall_score(self.data[self.y_var], y_pred), 2))
        print("F1:", round(f1_score(self.data[self.y_var], y_pred), 2))

        if self.prob:

            fpr, tpr, thresholds = roc_curve(self.data[self.y_var],
                                             self.classifier.predict_proba(self.data[self.features])[:, 1])

        else:
            fpr, tpr, thresholds = roc_curve(self.data[self.y_var], self.classifier.predict(self.data[self.features]))

        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.title(self.which_data + ":ROC", fontsize=15)
        plt.plot(fpr, tpr, "b", label="AUC = %0.3f" % roc_auc)
        plt.legend(loc="lower right", fontsize=15)
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([-0.1, 1.0])
        plt.ylim([-0.1, 1.01])
        plt.ylabel("True Positive Rate", fontsize=15)
        plt.xlabel("False Positive Rate", fontsize=15)
        plt.show()


class DataSplit:
    def __init__(
            self,
            unique_id,
            data,
            y_var,
            split_frac,
            random_seed
    ):
        """
        Initialises DataSplit
        This class is used to prepare the data

        :param unique_id: (str) unique identifier
        :param data: (str) modeling data
        :param y_var: (str) response variable
        :param split_frac: (str) fraction of data to have in train set
        :param random_seed: (int) random seed

        """

        self.unique_id = unique_id
        self.data = data
        self.y_var = y_var
        self.split_frac = split_frac
        self.random_seed = random_seed
        self.train_df = None
        self.test_df = None

    def split_data(self):
        """
        Split data randomly into train and test

        """
        ID_list = self.data[self.unique_id].unique().tolist()
        random.seed(self.random_seed)
        train_IDs = random.sample(ID_list, int(self.split_fraq * len(ID_list)))

        self.train_df = self.data[self.data[self.unique_id].isin(train_IDs)]
        self.test_df = self.data[~self.data[self.unique_id].isin(train_IDs)]

        return self.train_df, self.test_df


class BayesianOpt(object):
    def __init__(
            self,
            train_data,
            features,
            y_var,
            scale_pos_weight_val,
    ):
        """
        Initialises BayesianOpt
        This class is used to prepare the data

        :param train_data: (dataframe) train data
        :param features: (list) list of independent variables
        :param y_var: (str) response variable
        :param scale_pos_weight_val: (float) value for scale_pos_weight parameter

        """

        self.train_data = train_data
        self.features = features
        self.y_var = y_var
        self.scale_pos_weight_val = scale_pos_weight_val

    def __call__(self, trial):
        """
        Run an iteration of bayesian optimization

        """
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.4, log=True)
        num_leaves = trial.suggest_int("num_leaves", 2, 60)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.2, 1)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-2, 5, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-2, 5, log=True)
        max_depth = trial.suggest_int("max_depth", 2, 30)
        min_child_samples = trial.suggest_int("min_child_samples", 50, 600)
        subsample = trial.suggest_float("subsample", 0.2, 1)
        n_estimators = trial.suggest_int("n_estimators", 500, 2000)

        param = {
            "learning_rate": [learning_rate],
            "num_leaves": [num_leaves],
            "colsample_bytree": [colsample_bytree],
            "reg_alpha": [reg_alpha],
            "reg_lambda": [reg_lambda],
            "max_depth": [max_depth],
            "min_child_samples": [min_child_samples],
            "subsample": [subsample],
            "n_estimators": [n_estimators],
            "verbose": [-1]
        }

        gbm_model = lgb.LGBMClassifier(objective="binary",
                                       metric=["auc", "binary_error"],
                                       random_state=2021,
                                       n_jobs=-1,
                                       scale_pos_weight=self.scale_pos_weight_val,
                                       )
        gbm_gs = dcv.GridSearchCV(
            estimator=gbm_model, param_grid=param,
            scoring="f1",
            cv=5,
            refit=True,
        )

        gbm_gs.fit(self.train_data[self.features], self.train_data[self.y_var], )

        return gbm_gs.best_score_


def performance_comparison(performance_dict):
    """
    Compare the performance of two predictive models (logistic regression and gbm)

    """
    print("Improvement in AUC: ", round(
        (performance_dict["gbm"]["auc"] - performance_dict["logistic"]["auc"]) * 100 / performance_dict["logistic"][
            "auc"], 2), "%")
    print("Improvement in Accuracy: ", round(
        (performance_dict["gbm"]["accuracy"] - performance_dict["logistic"]["accuracy"]) * 100 /
        performance_dict["logistic"]["accuracy"], 2), "%")
    print("Improvement in Precision: ", round(
        (performance_dict["gbm"]["precision"] - performance_dict["logistic"]["precision"]) * 100 /
        performance_dict["logistic"]["precision"], 2), "%")
    print("Improvement in Recall: ", round(
        (performance_dict["gbm"]["recall"] - performance_dict["logistic"]["recall"]) * 100 /
        performance_dict["logistic"]["recall"], 2), "%")
    print("Improvement in F1 Score: ", round(
        (performance_dict["gbm"]["f1_score"] - performance_dict["logistic"]["f1_score"]) * 100 /
        performance_dict["logistic"]["f1_score"], 2), "%")


class FindUplift:
    def __init__(
            self,
            model,
            data,
            features=[
                "duration",
                "num_channels",
                "channel_email",
                "channel_mobile",
                "channel_social",
                "channel_web",
                "age",
                "income",
                "membership_days",
                "num_offers",
                "membership_days_X_income",
                "num_channels_X_income",
                "age_X_income",
                "membership_month_sin",
                "membership_month_cos",
                "difficulty",
                "offer_id",
                "offer_type",
                "offer_reward",
                "gender",
                "membership_year"
            ],
            cat_vars=[
                "offer_id",
                "offer_type",
                "offer_reward",
                "gender",
                "membership_year"
            ],
            y_var="successful_offer",
    ):
        """
        Initialises FindUplift
        This class is used to prepare the data

        :param model: (booster) uplift model
        :param data: (dataframe) data
        :param features: (list) list of continuous variables
        :param cat_vars: (list) list of categorical variables
        :param y_var: (str) response variable

        """

        self.model = model
        self.data = data
        self.features = features
        self.cat_vars = cat_vars
        self.y_var = y_var
        self.model_auuc = None
        self.random_auuc = None
        data_prep = Processing.DataPrep()
        self.portfolio = data_prep.portfolio_prep()
        self.data["original_offer_id"] = self.data["offer_id"]
        self.data["original_pred"] = gbm_model.predict(self.data[self.features])
        self.offer_list = [
            "bogo_5_10",
            "bogo_5_5",
            "discount_10_2",
            "discount_10_5",
            "bogo_7_10",
            "discount_7_2",
            "discount_7_3",
            "bogo_7_5"
        ]

    def find_uplift(self):
        """
        Predict the uplift using the model

        """
        for i in self.portfolio["offer_id"].unique().tolist():
            temp = self.portfolio[self.portfolio["offer_id"] == i].reset_index(drop=True)
            self.data["offer_id"] = i
            self.data = self.data[[x for x in self.data.columns.tolist() if x not in ["offer_type",
                                                                                      "duration",
                                                                                      "offer_reward",
                                                                                      "difficulty",
                                                                                      "num_channels",
                                                                                      "channel_email",
                                                                                      "channel_mobile",
                                                                                      "channel_social",
                                                                                      "channel_web"]]]
            self.data = self.data.merge(temp, on=["offer_id"], how="left")
            self.data["num_channels_X_income"] = self.data["num_channels"] * self.data["income"]

            for j in self.cat_vars:
                self.data.loc[:, j] = self.data[j].astype("category")
            self.data[i] = self.model.predict(self.data[self.features])
        self.data["informational"] = self.data[["informational_3", "informational_4"]].max(axis=1)

        for i in self.offer_list:
            self.data[i] = self.data[i] - self.data["informational"]
        self.data["informational"] = 0

        self.data["recom_offer"] = self.data[["bogo_5_10",
                                              "bogo_5_5",
                                              "discount_10_2",
                                              "discount_10_5",
                                              "bogo_7_10",
                                              "discount_7_2",
                                              "discount_7_3",
                                              "bogo_7_5",
                                              "informational"]].idxmax(axis=1)
        self.data["best_uplift"] = self.data[["bogo_5_10",
                                              "bogo_5_5",
                                              "discount_10_2",
                                              "discount_10_5",
                                              "bogo_7_10",
                                              "discount_7_2",
                                              "discount_7_3",
                                              "bogo_7_5"]].max(axis=1)

    def find_baseline_uplift(self):
        """
        Predict the baseline uplift.

        """
        for i in ["informational_3", "informational_4"]:
            temp = self.portfolio[self.portfolio["offer_id"] == i].reset_index(drop=True)
            self.data["offer_id"] = i
            self.data = self.data[[x for x in self.data.columns.tolist() if x not in ["offer_type",
                                                                                      "duration",
                                                                                      "offer_reward",
                                                                                      "difficulty",
                                                                                      "num_channels",
                                                                                      "channel_email",
                                                                                      "channel_mobile",
                                                                                      "channel_social",
                                                                                      "channel_web"]]]
            self.data = self.data.merge(temp, on=["offer_id"], how="left")
            self.data["num_channels_X_income"] = self.data["num_channels"] * self.data["income"]

            for j in self.cat_vars:
                self.data.loc[:, j] = self.data[j].astype("category")
            self.data[i] = self.model.predict(self.data[self.features])
        self.data["informational"] = self.data[["informational_3", "informational_4"]].max(axis=1)
        self.data["original_uplift"] = self.data["original_pred"] - self.data["informational"]

    def calculate_auuc(self, original_uplift=False):
        """
        Plot the uplift curve and calculate the AUUC
        :param original_uplift: (boolean) whether the provided predicted uplifts come from the baseline or model

        """
        treat_var = "original_offer_id"
        model_names = ["Model"] + ["Rand_" + str(x) for x in range(1, 41)]

        if original_uplift:
            self.data["best_uplift"] = self.data["original_uplift"]

        df_preds = self.data[["best_uplift", treat_var, self.y_var]]
        df_preds["is_treated"] = np.where(self.data[treat_var].isin(["informational_3", "informational_4"]), 0, 1)
        df_preds = df_preds.sort_values("best_uplift", ascending=False).reset_index(drop=True)
        df_preds.index = df_preds.index + 1
        df_preds["cumsum_tr"] = df_preds["is_treated"].cumsum()
        df_preds["cumsum_ct"] = df_preds.index.values - df_preds["cumsum_tr"]
        df_preds["cumsum_y_tr"] = (df_preds[self.y_var] * df_preds["is_treated"]).cumsum()
        df_preds["cumsum_y_ct"] = (df_preds[self.y_var] * (1 - df_preds["is_treated"])).cumsum()
        df_preds["lift"] = df_preds["cumsum_y_tr"] / df_preds["cumsum_tr"] - df_preds["cumsum_y_ct"] / df_preds[
            "cumsum_ct"]
        lift = []
        lift.append(df_preds["cumsum_y_tr"] / df_preds["cumsum_tr"] - df_preds["cumsum_y_ct"] / df_preds["cumsum_ct"])

        np.random.seed(2021)

        for i in range(40):
            df_preds = self.data[["best_uplift", treat_var, self.y_var]]
            df_preds["best_uplift"] = np.random.rand(df_preds.shape[0])
            df_preds["is_treated"] = np.where(self.data[treat_var].isin(["informational_3", "informational_4"]), 0, 1)

            df_preds = df_preds.sort_values("best_uplift", ascending=False).reset_index(drop=True)
            df_preds.index = df_preds.index + 1
            df_preds["cumsum_tr"] = df_preds["is_treated"].cumsum()
            df_preds["cumsum_ct"] = df_preds.index.values - df_preds["cumsum_tr"]
            df_preds["cumsum_y_tr"] = (df_preds[self.y_var] * df_preds["is_treated"]).cumsum()
            df_preds["cumsum_y_ct"] = (df_preds[self.y_var] * (1 - df_preds["is_treated"])).cumsum()
            df_preds["lift"] = df_preds["cumsum_y_tr"] / df_preds["cumsum_tr"] - df_preds["cumsum_y_ct"] / df_preds[
                "cumsum_ct"]
            lift.append(
                df_preds["cumsum_y_tr"] / df_preds["cumsum_tr"] - df_preds["cumsum_y_ct"] / df_preds["cumsum_ct"])

        lift = pd.concat(lift, join="inner", axis=1)
        lift.loc[0] = np.zeros((lift.shape[1],))
        lift = lift.sort_index().interpolate()

        lift.columns = model_names
        lift["RANDOM"] = lift[model_names[1:]].mean(axis=1)
        lift.drop(model_names[1:], axis=1, inplace=True)
        gain = lift.mul(lift.index.values, axis=0)

        gain = gain.iloc[np.linspace(0, gain.index[-1], 100, endpoint=True)]

        plt.figure(figsize=(7, 6))
        pp = plt.plot(gain)
        plt.xlabel("Population")
        plt.ylabel("Gain")
        if original_uplift:
            name = "Baseline"
        else:
            name = "Model"

        plt.legend([pp[0], pp[1]], [name, "Random"])
        plt.show()

        self.model_auuc = gain["Model"].sum() / gain["Model"].shape[0]
        self.random_auuc = gain["RANDOM"].sum() / gain["RANDOM"].shape[0]
        print(name, "AUUC:", self.model_auuc, "   Random AUUC:", self.random_auuc)
