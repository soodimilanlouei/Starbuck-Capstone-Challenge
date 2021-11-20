import pandas as pd
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from pandas import DataFrame

import warnings

warnings.simplefilter("always", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


class DataPrep():
    def __init__(
            self,
            portfolio_path: str = "data/portfolio.json",
            profile_path: str = "data/profile.json",
            transcript_path: str = "data/transcript.json"
    ):
        """
        Initialises DataPrep
        This class is used to prepare the data

        :param portfolio_path: (str) path to portfolio data
        :param profile_path: (str) path to profile data
        :param transcript_path: (str) path to transcript

        """

        # read in the json files
        self.portfolio = pd.read_json(portfolio_path, orient="records", lines=True)
        self.profile = pd.read_json(profile_path, orient="records", lines=True)
        self.transcript = pd.read_json(transcript_path, orient="records", lines=True)
        self.new_ids = {}

    def portfolio_prep(self):
        self.portfolio = pd.concat([self.portfolio[["reward", "difficulty", "duration", "offer_type", "id"]],
                                    pd.get_dummies(self.portfolio["channels"].apply(pd.Series),
                                                   prefix="channel")], axis=1)
        self.portfolio = self.portfolio.groupby(self.portfolio.columns, axis=1).sum()
        self.portfolio["num_channels"] = self.portfolio[
            ["channel_email", "channel_mobile", "channel_social", "channel_web"]].sum(axis=1)

        self.portfolio.rename(columns={"id": "offer_id", "reward": "offer_reward"}, inplace=True)

        self.portfolio["offer_id"].replace(
            {
                "ae264e3637204a6fb9bb56bc8210ddfd": "bogo_7_10",
                "4d5c57ea9a6940dd891ad53e9dbe8da0": "bogo_5_10",
                "3f207df678b143eea3cee63160fa8bed": "informational_4",
                "9b98b8c7a33c4b65b9aebfe6a799e6d9": "bogo_7_5",
                "0b1e1539f2cc45b7b9fa7c272da2e1d7": "discount_10_5",
                "2298d6c36e964ae4a3e7e9706d1fb8c2": "discount_7_3",
                "fafdcd668e3743c1bb461111dcafc2a4": "discount_10_2",
                "5a8bc65990b245e5a138643cd4eb9837": "informational_3",
                "f19421c1d4aa40978ebb69ca19b0e20d": "bogo_5_5",
                "2906b810c7d4411798c6938adc9daaa5": "discount_7_2"

            },
            inplace=True,
        )
        self.portfolio = self.portfolio.sort_values(by="offer_id").reset_index(drop=True)

        self.portfolio = self.portfolio[["offer_id", "offer_type", "num_channels", "duration", "offer_reward",
                                         "difficulty", "channel_email", "channel_mobile", "channel_social",
                                         "channel_web"
                                         ]]

        return self.portfolio

    def profile_prep(self):
        original_ids = self.profile["id"].unique()
        for counter in range(len(original_ids)):
            self.new_ids[original_ids[counter]] = "user_" + str(counter + 1)

        self.profile["id"] = self.profile["id"].map(self.new_ids)

        self.profile.rename(columns={"id": "person"}, inplace=True)

        self.profile["became_member_on"] = pd.to_datetime(self.profile["became_member_on"], format="%Y%m%d")
        self.profile["membership_days"] = datetime.datetime.today().date() - pd.to_datetime(
            self.profile["became_member_on"], format="%Y%m%d").dt.date
        self.profile["membership_days"] = self.profile["membership_days"].dt.days
        self.profile["membership_month"] = self.profile["became_member_on"].dt.month
        self.profile["membership_year"] = self.profile["became_member_on"].dt.year
        self.profile["age"] = np.where(self.profile["age"] == 118, np.nan, self.profile["age"])
        self.profile["gender"] = np.where(self.profile["gender"] == None, np.nan, self.profile["gender"])
        self.profile["gender"].replace([None], np.nan, inplace=True)
        self.profile = self.profile.dropna().reset_index(drop=True)

        return self.profile

    def transcript_prep(self):
        self.transcript = pd.concat(
            [self.transcript.drop(["value"], axis=1), self.transcript["value"].apply(pd.Series)], axis=1)
        self.transcript["offer_id"] = np.where(self.transcript["offer id"].isnull(), self.transcript["offer_id"],
                                               self.transcript["offer id"])
        del self.transcript["offer id"]
        self.transcript["event"].replace(
            {
                "offer received": "offer_received",
                "offer viewed": "offer_viewed",
                "offer completed": "offer_completed"
            },
            inplace=True,
        )

        self.transcript["offer_id"].replace(
            {
                "ae264e3637204a6fb9bb56bc8210ddfd": "bogo_7_10",
                "4d5c57ea9a6940dd891ad53e9dbe8da0": "bogo_5_10",
                "3f207df678b143eea3cee63160fa8bed": "informational_4",
                "9b98b8c7a33c4b65b9aebfe6a799e6d9": "bogo_7_5",
                "0b1e1539f2cc45b7b9fa7c272da2e1d7": "discount_10_5",
                "2298d6c36e964ae4a3e7e9706d1fb8c2": "discount_7_3",
                "fafdcd668e3743c1bb461111dcafc2a4": "discount_10_2",
                "5a8bc65990b245e5a138643cd4eb9837": "informational_3",
                "f19421c1d4aa40978ebb69ca19b0e20d": "bogo_5_5",
                "2906b810c7d4411798c6938adc9daaa5": "discount_7_2"

            },
            inplace=True,
        )
        self.transcript["person"] = self.transcript["person"].map(self.new_ids)

        return self.transcript


class DataDescription():
    def __init__(
            self,
            portfolio: DataFrame,
            profile: DataFrame,
            transcript: DataFrame
    ):
        """
        Initialises PrepTrainingData
        This class is used to build the model features

        :param portfolio: (dataframe) portfolio data
        :param profile: (dataframe) profile data
        :param transcript: (dataframe) transcript data
        """

        self.portfolio = portfolio
        self.profile = profile
        self.transcript = transcript

    def describe_portfolio(self):
        describe_portfolio_plot(self.portfolio)

    def describe_profile(self):
        describe_profile_categorical(self.profile)
        describe_profile_continuous(self.profile)

    def describe_transcript(self):
        describe_transcript_plot(self.transcript)


def describe_portfolio_plot(portfolio):
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.rc("axes", axisbelow=True)
    plt.bar(portfolio["offer_id"], portfolio["num_channels"], color="teal")
    plt.xticks(rotation=90)
    plt.yticks(np.arange(0, 5, 1))
    plt.ylabel("Number of Channels", fontsize=15)
    plt.grid()
    plt.subplot(1, 3, 2)
    plt.rc("axes", axisbelow=True)
    plt.bar(portfolio["offer_id"], portfolio["difficulty"], color="teal")
    plt.xticks(rotation=90)
    plt.yticks(np.arange(0, 25, 5))
    plt.ylabel("Difficulty", fontsize=15)
    plt.grid()
    plt.subplot(1, 3, 3)
    plt.rc("axes", axisbelow=True)
    plt.bar(portfolio["offer_id"], portfolio["duration"], color="teal")
    plt.xticks(rotation=90)
    plt.yticks(np.arange(0, 12, 2))
    plt.ylabel("Duration", fontsize=15)
    plt.grid()
    plt.show()


def describe_profile_categorical(profile):
    plt.figure(figsize=(20, 5))
    plt.subplots_adjust(wspace=0.3)
    plt.subplot(1, 3, 1)
    plt.rc("axes", axisbelow=True)
    membership_month = pd.DataFrame(profile["membership_month"].value_counts())
    membership_month = membership_month.sort_index()
    plt.bar(membership_month.index, membership_month["membership_month"], color=["green", "green", "green",
                                                                                 "yellow", "yellow", "yellow",
                                                                                 "orange", "orange", "orange",
                                                                                 "grey", "grey", "grey"])
    plt.xticks(np.arange(0, 13, 1))
    plt.xlabel("Registration Month", fontsize=15)
    plt.ylabel("Number of Users", fontsize=15)
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.rc("axes", axisbelow=True)
    membership_year = pd.DataFrame(profile["membership_year"].value_counts())
    membership_year = membership_year.sort_index()
    plt.bar(membership_year.index, membership_year["membership_year"], color="teal")
    plt.plot(membership_year.index, membership_year["membership_year"], "-o", color="orange")
    plt.xlabel("Registration Year", fontsize=15)
    plt.ylabel("Number of Users", fontsize=15)
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.rc("axes", axisbelow=True)
    gender = pd.DataFrame(profile["gender"].value_counts())
    bar_chart = plt.bar(gender.index, gender["gender"], color="teal")
    plt.xlabel("Gender", fontsize=15)
    plt.ylabel("Number of Users", fontsize=15)

    for p in bar_chart:
        height = p.get_height()
        plt.text(x=p.get_x() + p.get_width() / 2, y=height + 80,
                 s="{}%".format(round(height * 100 / gender["gender"].sum(), 2)),
                 ha="center")
    plt.grid()
    plt.show()


def describe_profile_continuous(profile):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=False, figsize=(20, 6))

    p = sns.kdeplot(data=profile, x="age", shade=True, color="teal", ax=ax1, bw=0.3)
    p.set_xlabel("Age", fontsize=15)
    p.set_ylabel("Density", fontsize=15)

    p = sns.kdeplot(data=profile, x="income", shade=True, color="teal", ax=ax2, bw=0.3)
    p.set_xlabel("Income", fontsize=15)
    p.set_ylabel("Density", fontsize=15)

    p = sns.kdeplot(data=profile, x="membership_days", shade=True, color="teal", ax=ax3, bw=0.3)
    p.set_xlabel("Membership Days", fontsize=15)
    p.set_ylabel("Density", fontsize=15)
    plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=False, figsize=(20, 6))

    p = sns.kdeplot(data=profile, x="age", hue="gender", shade=True, ax=ax1, bw=0.3)
    p.set_xlabel("Age", fontsize=15)
    p.set_ylabel("Density", fontsize=15)

    p = sns.kdeplot(data=profile, x="income", hue="gender", shade=True, ax=ax2, bw=0.3)
    p.set_xlabel("Income", fontsize=15)
    p.set_ylabel("Density", fontsize=15)

    p = sns.kdeplot(data=profile, x="membership_days", hue="gender", shade=True, ax=ax3, bw=0.3)
    p.set_xlabel("Membership Day", fontsize=15)
    p.set_ylabel("Density", fontsize=15)
    plt.show()


def describe_transcript_plot(transcript):
    plt.figure(figsize=(20, 5))
    plt.subplots_adjust(wspace=0.3)
    plt.subplot(1, 3, 1)
    plt.rc("axes", axisbelow=True)

    transcript["counter"] = 1
    offers_sent = transcript[transcript["event"] == "offer_received"].groupby(["person"])[
        ["counter"]].sum().reset_index()
    offers_sent["Frequency"] = 1
    offers_sent = offers_sent.groupby(["counter"])[["Frequency"]].sum().reset_index()
    bar_chart = plt.bar(offers_sent["counter"], offers_sent["Frequency"], color="teal")
    plt.xlabel("Number of Received Offers", fontsize=15)
    plt.ylabel("Frequency", fontsize=15)
    plt.grid()
    for p in bar_chart:
        height = p.get_height()
        plt.text(x=p.get_x() + p.get_width() / 2, y=height + 80,
                 s="{}%".format(round(height * 100 / offers_sent["Frequency"].sum(), 2)),
                 ha="center")
    plt.subplot(1, 3, 2)

    offers_viewed = transcript[transcript["event"] == "offer_viewed"].groupby(["person"])[
        ["counter"]].sum().reset_index()
    offers_viewed["Frequency"] = 1
    offers_viewed = offers_viewed.groupby(["counter"])[["Frequency"]].sum().reset_index()
    bar_chart = plt.bar(offers_viewed["counter"], offers_viewed["Frequency"], color="teal")
    plt.xlabel("Number of Viewed Offers", fontsize=15)
    plt.ylabel("Frequency", fontsize=15)
    plt.grid()
    for p in bar_chart:
        height = p.get_height()
        plt.text(x=p.get_x() + p.get_width() / 2, y=height + 80,
                 s="{}%".format(round(height * 100 / offers_sent["Frequency"].sum(), 2)),
                 ha="center")
    plt.subplot(1, 3, 3)

    offers_frequency = transcript[transcript["event"] == "offer_received"].groupby(["offer_id"])[
        ["counter"]].sum().reset_index()
    bar_chart = plt.bar(offers_frequency["offer_id"], offers_frequency["counter"], color="teal")
    for p in bar_chart:
        height = p.get_height()
        plt.text(x=p.get_x() + p.get_width() / 2, y=height - 1000,
                 s="{}%".format(round(height * 100 / offers_frequency["counter"].sum(), 2)), rotation=90, color="white",
                 ha="center")
    plt.xticks(rotation=90)
    plt.ylabel("Frequency", fontsize=15)
    plt.show()


class DataMerge():
    def __init__(
            self,
            portfolio: DataFrame,
            profile: DataFrame,
            transcript: DataFrame
    ):
        """
        Initialises PrepTrainingData
        This class is used to build the model features

        :param portfolio: (dataframe) portfolio data
        :param profile: (dataframe) profile data
        :param transcript: (dataframe) transcript data
        """

        self.portfolio = portfolio
        self.profile = profile
        self.transcript = transcript

    def data_merge(self):
        offer_received = self.transcript[self.transcript["event"] == "offer_received"].copy(deep=True).reset_index(drop=True)
        offer_received = offer_received.drop(["amount", "event"], axis=1)
        offer_received.rename(columns={"time": "time_received", "reward": "original_reward"}, inplace=True)

        offer_viewed = self.transcript[self.transcript["event"] == "offer_viewed"].copy(deep=True).reset_index(drop=True)
        offer_viewed = offer_viewed.drop(["amount", "event", "reward"], axis=1)
        offer_viewed.rename(columns={"time": "time_viewed"}, inplace=True)

        transaction = self.transcript[self.transcript["event"] == "transaction"].copy(deep=True).reset_index(drop=True)
        transaction = transaction.drop(["event", "reward", "offer_id"], axis=1)
        transaction.rename(columns={"time": "time_transaction"}, inplace=True)

        offer_completed = self.transcript[self.transcript["event"] == "offer_completed"].copy(deep=True).reset_index(drop=True)
        offer_completed = offer_completed.drop(["event", "amount"], axis=1)
        offer_completed.rename(columns={"time": "time_completed"}, inplace=True)

        self.merged_data = offer_received.merge(offer_viewed, on=["person", "offer_id"], how="outer")
        self.merged_data = self.merged_data.merge(transaction, on=["person"], how="outer")
        self.merged_data = self.merged_data.merge(offer_completed, on=["person", "offer_id"], how="outer")

        self.merged_data = self.merged_data.merge(self.portfolio[["offer_id", "duration"]], on=["offer_id"], how="left")
        self.merged_data = self.merged_data[
            ["person", "offer_id", "original_reward", "time_received", "duration", "time_viewed", "time_transaction",
             "amount", "time_completed", "reward"]]

        self.merged_data.loc[:, "original_reward"] = self.merged_data["original_reward"].fillna(0)

        return self.merged_data

    def remove_redundant_data(self):
        self.merged_data = self.merged_data[
            ((self.merged_data["time_viewed"].isnull()) & (self.merged_data["time_completed"].isnull())) |
            ((self.merged_data["time_viewed"] >= self.merged_data["time_received"]) & (
                    self.merged_data["time_completed"] >= self.merged_data["time_viewed"])) |
            ((self.merged_data["time_viewed"] >= self.merged_data["time_received"]) & (
                    self.merged_data["time_completed"] >= self.merged_data["time_received"])) |
            ((self.merged_data["time_viewed"] >= self.merged_data["time_received"]) & (
                self.merged_data["time_completed"].isnull()))].copy(deep=True)

        self.merged_data.loc[:, "offer_id"] = self.merged_data["offer_id"].fillna("no_offer")
        self.merged_data = self.merged_data[self.merged_data["offer_id"] != "no_offer"].copy(deep=True).reset_index(
            drop=True)

        return self.merged_data

    def find_success_tried_offers(self):
        not_viewed_data = self.merged_data[self.merged_data["time_viewed"].isnull()].reset_index(drop=True)
        not_viewed_data = not_viewed_data.sort_values(
            by=["person", "offer_id", "time_received", "time_transaction"]).reset_index(drop=True)
        not_viewed_data["successful_offer"] = 0
        not_viewed_data["tried_offer"] = np.where(
            (not_viewed_data["offer_id"].isin(["informational_4", "informational_3"])) &
            (not_viewed_data["time_transaction"] <= not_viewed_data["time_received"] + not_viewed_data["duration"]) &
            (not_viewed_data["time_transaction"] >= not_viewed_data["time_received"]),
            1, np.where(
                (not_viewed_data["time_completed"] <= not_viewed_data["time_received"] + not_viewed_data["duration"]) &
                (not_viewed_data["time_completed"] >= not_viewed_data["time_received"]), 1, 0))

        not_viewed_data = not_viewed_data.drop(["time_transaction", "amount"], axis=1)
        not_viewed_data = not_viewed_data.drop_duplicates().reset_index(drop=True)

        viewed_data = self.merged_data[~self.merged_data["time_viewed"].isnull()].reset_index(drop=True)
        viewed_data["successful_offer"] = np.where(
            (viewed_data["offer_id"].isin(["informational_4", "informational_3"])) & (
                viewed_data["time_transaction"].isnull()),
            0,
            np.where((viewed_data["time_completed"] <= viewed_data["time_received"] + viewed_data["duration"]) & (
                    viewed_data["time_completed"] >= viewed_data["time_received"]) &
                     (viewed_data["time_viewed"] <= viewed_data["time_completed"]), 1, 0))

        viewed_data["successful_offer"] = np.where(
            (viewed_data["offer_id"].isin(["informational_4", "informational_3"])) &
            (viewed_data["time_transaction"] >= viewed_data["time_viewed"]) &
            (viewed_data["time_transaction"] <= viewed_data["time_received"] + viewed_data["duration"]),
            1, viewed_data["successful_offer"])

        viewed_data["tried_offer"] = np.where(
            (viewed_data["time_transaction"] <= viewed_data["time_received"] + viewed_data["duration"]) & (
                    viewed_data["time_transaction"] >= viewed_data["time_received"]), 1, 0)
        viewed_data = viewed_data.drop(["time_transaction", "amount"], axis=1)
        viewed_data = viewed_data.drop_duplicates().reset_index(drop=True)

        viewed_data = viewed_data.sort_values(by=["person", "offer_id", "time_received"]).reset_index(drop=True)
        viewed_data = viewed_data.sort_values(by=["successful_offer", "tried_offer"], ascending=False).reset_index(
            drop=True)

        viewed_data["time_completed"] = viewed_data["time_completed"].fillna(-1)
        viewed_data["reward"] = viewed_data["reward"].fillna(0)

        viewed_data = viewed_data.groupby(
            ["person", "offer_id", "original_reward", "time_received", "duration", "time_completed", "reward",
             "successful_offer", "tried_offer",
             ]).first().reset_index()

        viewed_data = viewed_data.groupby(
            ["person", "offer_id", "original_reward", "duration", "time_received", "reward", "time_viewed",
             ])[["successful_offer", "tried_offer"]].max().reset_index()

        self.merged_data = pd.concat([not_viewed_data, viewed_data], ignore_index=True)

        return self.merged_data

    def append_other_data(self):
        self.merged_data = self.merged_data.merge(self.profile, on="person", how="left")
        self.merged_data = self.merged_data.merge(self.portfolio, on=["offer_id", "duration"], how="left")

        return self.merged_data

    def save_data(self):
        self.merged_data.to_csv("./data/merged_data.csv")


def offer_performance(merged_data):
    offers = merged_data.groupby(["offer_id"])["tried_offer", "successful_offer"].mean().reset_index()

    plt.figure(figsize=(20, 5))
    plt.subplots_adjust(wspace=0.3)
    plt.subplot(1, 2, 1)
    plt.rc("axes", axisbelow=True)

    plt.bar(offers["offer_id"], offers["tried_offer"], color="teal")
    plt.xticks(rotation=90)
    plt.ylabel("Trying Rate", fontsize=15)
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.bar(offers["offer_id"], offers["successful_offer"], color="teal")
    plt.xticks(rotation=90)
    plt.ylabel("Success Rate", fontsize=15)
    plt.grid()
    plt.show()
