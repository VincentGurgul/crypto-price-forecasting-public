# Forecasting Cryptocurrency Prices Using Deep Learning: Integrating Financial, Blockchain, and Text Data

<em>Master's Thesis</em>

Author: Vincent Gurgul<br>
Submission date: September 14, 2023

This repository encompasses the code for the processing, exploratory data analysis, model training and evaluation as well as related utilities and documentation for the forecasting of cryptocurrency prices using various deep learning and natural language processing techniques. Furthermore, it contains the master thesis itself and the presentation slides from the European Conference on Data Analysis (DSSV-ECDA 2023) where the results were presented.

Note: The data themselves are missing due to GitHub storage limit.


## Structure

```
├── 1_data_acquisition: Scripts used to acquire the data used in the thesis. This folder is not shared publicly since the code contains passwords and API keys.
├── 2_data_processing: Scripts for preprocessing and cleaning of the data acquired in the previous step. This includes stationarity analysis and target variable engineering.
├── 3_nlp_models: Scripts for fine-tuning and applying the large language models used for NLP, as well as post-processing the acquired scores.
├── 4_eda: Directory containing EDA of numeric and textual data, as well as the Granger causality analysis.
├── 5_time_series_models: Scripts used to train and evaluate the time series models.
├── 6_feature_importance: Assessment of the feature importance using an XGBoost model trained on daily price fluctuations.
├── documentation: Documentation complementing the dataset and the code. This includes an overview and explanation of all features and a complete, disaggregated presentation of the results.
├── utils: Utility scripts and helper functions used throughout the project.
├── dssv_slides.pdf: Presentation slides from the European Conference on Data Analysis (DSSV-ECDA 2023).
├── master_thesis.pdf: The master thesis that was submitted.
├── README.md: This readme file.
└── requirements.txt: The required Python packages.
```

## Requirements

The code in this repository was developed using Python 3.9.14. The required Python packages are listed in the <code>requirements.txt</code> file.
