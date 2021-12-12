# Starbucks Capstone Challenge
### Machine Learning Engineer Nanodegree
## Project Organization

    ├── README.md                       <- The top-level README for developers using this project.
    │
    ├── data
    │   ├── portfolio.json              <- portfolio data
    │   │
    │   ├── profile.json                <- profile data
    │   │
    │   ├── transcript.json             <- transcript data
    │   │
    │   └── merged_data.csv             <- combined data
    │
    ├── images
    │   │
    │   └── uplift.png                  <- uplift quadrant                                  
    │
    ├── model
    │   │
    │   └── gbm_model.pkl               <- predictive model object
    │
    ├── EDA.ipynb                       <- Exploratory Data Analysis notebook
    │
    └── Predictive_Modeling.ipynb       <- Predictive Modeling notebook
    │
    └── Uplift_Modeling.ipynb           <- Uplift Modeling notebook
    │
    └── Processing.py                   <- Processing module
    │
    └── Modeling_Helper.py              <- Modeling Helper module
    │
    └── Project_Report.pdf              <- Project report file

## Project Overview
<p align="justify"> In the Starbucks Capstone Challenge of Udacity's Machine Learning Engineer Nanodegree, we are given a number of simulated datasets which emulate customer behavior on the Starbucks rewards mobile app. This app is mainly used for sending either informational messages or promotional offers. A customer might be targeted by (1) informational advertisement, (2) discount offer, or (3) buy one get one free (BOGO) offer. The data provided includes the attributes of all offers available, the demographics of each customer, and the features of each transaction made. While it is not possible to send all the offers to all customers, the goal of this project is to extract insights from the data provided and identify customer segmentation and particular offers that they react to better. Additionally, the aim is to design a new recommendation system which specifies which offer (if any) should be given to an individual customer. </p>

In order to address the questions laid out above, we relied on three main approaches: 
- Exploratory Data Analysis 
- Predictive Modeling 
- Uplift Modeling

<p align="justify"> <b>Exploratory Data Analysis</b>: This approach helped us to summarize the main characteristics pertinent to our data visually detect patterns in customer behavior and identify particular demographics who react favorably to the offers that the mobile app is providing. This was our initial investigation towards the data to gather insights for the modeling work. </p>

<p align="justify"> <b>Predictive Modeling</b>: the initial hypothesis was that customer characteristics and offer attributes are associated with the likelihood of successfully completing an offer. We built a model to find these potential associations. It is worth noting that while this model can provide insights about the purchasing likelihood, it cannot be directly used as a recommendation model. </p>

<p align="justify"> <b>Uplift Modeling</b>: One of the goals of this project was to design a recommendation system which can assist the mobile app in intelligently sending offers to customers. The logic here is each offer acts as an intervention and an offer should be sent to a customer whose likelihood of purchasing increases if he/she receives the offer. Uplift models are  mainly designed to find the right treatments for the right customers and we  used this approach in this project. </p>

Below is a short guide to get you up and running as quick as possible.

## Language

Python 3.8

## Package Versions

- pandas==1.1.4
- numpy==1.21.4
- matplotlib==3.3.3
- sklearn==1.0.1
- dask_ml==2021.11.16
- statsmodels==0.12.1
- lightgbm==3.0.0 
- imblearn==0.7.0
- optuna==2.10.0
- shap==0.39.0

## Codes
<p align="justify"> In this project, there are three jupyter notebooks, along with two modules. The EDA notebook which needs to be run first contains the code for running the Exploratory Data Analysis and for preparing the data for the modeling steps. The final dataset ready for modeling is saved through this notebook. The Processing module contains classes and functions that are mainly imported in the EDA notebook and it helps with visualization and data wrangling. The second notebook is called Predictive_Modeling.ipynb in which we build a baseline model to compare our final GBM model with. It also contains the steps taken for refining the GBM model, including some of the feature engineering and hyper-parameter tuning. The third notebook called Uplift_Modeling.ipynb is created to build an uplift model and evaluate its performance, comparing it with random assignment and the baseline strategy. The Modeling_Helper module contains classes and functions to help with modeling work. </p>
