# Forecasting Cryptocurrency Prices Using Deep Learning: Integrating Financial, Blockchain, and Text Data

<em>Master's Thesis</em>

Author: Vincent Gurgul<br>
Submission date: September 14, 2023

This repository encompasses the code for the processing, exploratory data analysis, model training and evaluation as well as related utilities and documentation for the forecasting of cryptocurrency prices using various deep learning and natural language processing techniques. Furthermore, it contains the master thesis itself and the presentation slides from the European Conference on Data Analysis (DSSV-ECDA 2023) where the results were presented.

Note: The data themselves are missing due to the GitHub storage limit.

The resulting publication can be accessed on ArXiv: https://arxiv.org/abs/2311.14759


## Structure

The repository contains the following directories:

- <code>1_data_acquisition</code>: Scripts used to acquire the data used in the thesis. This folder is not shared publicly since the code contains passwords and API keys.

- <code>2_data_processing</code>: Scripts for preprocessing and cleaning of the data acquired in the previous step. This includes stationarity analysis and target variable engineering.

- <code>3_nlp_models</code>: Scripts for fine-tuning and applying the large language models used for NLP, as well as post-processing the acquired scores.

- <code>4_eda</code>: Directory containing EDA of numeric and textual data, as well as the Granger causality analysis.

- <code>5_time_series_models</code>: Scripts used to train and evaluate the time series models.

- <code>6_feature_importance</code>: Assessment of the feature importance using an XGBoost model trained on daily price fluctuations.

- <code>documentation</code>: Documentation complementing the dataset and the code. This includes an overview and explanation of all features and a complete, disaggregated presentation of the results.

- <code>utils</code>: Utility scripts and helper functions used throughout the project.


## Requirements

The code in this repository was developed using Python 3.9.14. The required Python packages are listed in the <code>requirements.txt</code> file.
