# Deep Learning and NLP in Cryptocurrency Forecasting: Integrating Financial, Blockchain, and Social Media Data

This repository encompasses the code for the acquisition and processing of the data, exploratory data analysis, model training and evaluation as well as related utilities and documentation for the forecasting of cryptocurrency prices using various deep learning and natural language processing techniques.

## Structure

The repository contains the following directories:

- <code>1_data_acquisition</code>: Interfacing with APIs to collect and preprocess data.

- <code>2_data_processing</code>: Preprocessing and cleaning of the data acquired in the previous step.

- <code>3_nlp_models</code>: Applying our NLP approaches and post-processing the acquired scores.

- <code>4_eda</code>: Exploratory analysis of numeric and textual data, as well as Granger causality analysis.

- <code>5_time_series_models</code>: Training and evaluation of the time series models.

- <code>6_feature_importance</code>: Assessment of the feature importances using an XGBoost model trained on daily price fluctuations.

- <code>utils</code>: Utility scripts and helper functions used throughout the project.

## Requirements

The code in this repository was developed using Python 3.9.14. The required Python packages are listed in the <code>requirements.txt</code> file.
