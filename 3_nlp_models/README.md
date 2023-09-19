# 3. NLP Models

This directory contains the code for the three NLP models utilised in the master thesis. Each subdirectory represents a specific model or approach, with associated code and post-processing.

## Structure

```
└── 1_twitter_roberta_pretrained
    └── main.py: Code for extracting sentiment scores with Twitter-RoBERTa-Base (version from 25.01.2023)
└── 2_bart_zero_shot
    └── main.py: Code for extracting bullishness scores with the BART-Large MNLI zero-shot classifier
└── 3_roberta_finetuned
    ├── config.py: Configuration settings for the RoBERTa model.
    ├── data_generation.ipynb: Processing text data for train-test splitting.
    ├── functions.py: Utility functions for the RoBERTa fine-tuning.
    ├── hyperparam_opt.ipynb: Notebook for RoBERTa hyperparameter optimization.
    └── main.ipynb: Main code for fine-tuning a RoBERTa-Base model on our target.
└── 4_processing
    └── main.ipynb: Aggregates NLP scores by day and checks stationarity.
```
