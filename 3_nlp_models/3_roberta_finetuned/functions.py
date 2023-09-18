''' Functions for RoBERTa finetuning and prediction. '''

import torch
import numpy as np

from scipy.special import softmax

import wandb
import pandas as pd

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


def get_first_half(list):
    return list[:len(list)//2]


def get_second_half(list):
    return list[len(list)//2:]


def get_score(text: str, tokenizer, model, discrete: bool = False) -> float:
    ''' This function outputs a sentiment score for a given input text.
    
    Args:
        text (str): Input string.
        tokenizer: Huggingface tokenizer
        model: Huggingface model
        discrete (bool, optional): Whether to output discrete prediction
            (0 or 1) or score, e.g. 0.765.
        
    Returns:
        float: Value in the range (-infty,infty) or [-1,1], depending on
            softmax setting, that reflects the sentiment of the input string.
            1 or infty := positive, -1 or -infty := negative.
        
    Raises:
        ValueError: If input is not of type `str` or `List[str]` or
            `List[List[str]]`.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    encoded_input = (tokenizer(text,
                               return_tensors='pt',
                               truncation=True,
                               max_length=512)
                     .to(device))
    
    logits = model(**encoded_input)[0][0].detach().cpu().numpy()

    if discrete:
        output = np.argmax(logits, axis=-1)
    else:
        output = softmax(logits)[1]

    return output


def run_and_log_finetuning(dataset_name: str,
                           project_name: str,
                           **kwargs) -> float:
    ''' Runs finetuning of RoBERTa on a given dataset with a given set of
    hyperparameters and logs the results to Weights & Biases.
    
    Args:
        dataset_name (str): Name of the dataset to be finetuned on
        project_name (str): Name of W&B project to log results to
        **kwargs: keyword arguments for the Huggingface TrainingArguments class

    Returns:
        float: ROC AUC of the finetuned RoBERTa model
    '''
    data = (pd.read_parquet(f'./data_merged/{dataset_name}_merged.parquet.gzip')
            .dropna())

    data['train'] = data.text.apply(get_first_half)
    data['test'] = data.text.apply(get_second_half)
    train = (
        data.drop(columns=['text', 'test'])
        .rename(columns={'train': 'text'})
        .explode('text')
        .dropna()
    )
    test = (
        data.drop(columns=['text', 'train'])
        .rename(columns={'test': 'text'})
        .explode('text')
        .dropna()
    )
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train[['label', 'text']]),
        'test': Dataset.from_pandas(test[['label', 'text']]),
    })

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    train_dataset = tokenized_datasets['train'].shuffle(seed=42)

    model = AutoModelForSequenceClassification.from_pretrained('roberta-base')

    wandb.init(
        project=project_name,
        group=dataset_name,
        reinit=True,
    )
    training_args = TrainingArguments(
        output_dir='./trainer',
        **kwargs,
        seed=42,
        report_to='wandb',
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()
    model.eval()
    
    test['roberta_finetuned_score'] = test.text.apply(
        lambda x: get_score(x, tokenizer, model)
    )
    accuracy = accuracy_score(test.label, test.roberta_finetuned_score > 0.5)
    pr_auc = average_precision_score(test.label, test.roberta_finetuned_score)
    roc_auc = roc_auc_score(test.label, test.roberta_finetuned_score)

    print(f'''\nAccuracy: {accuracy:.4f}
PR AUC: {pr_auc:.4f}
ROC AUC: {roc_auc:.4f}''')

    wandb.log(data={'accuracy': accuracy,
                    'pr_auc': pr_auc,
                    'roc_auc': roc_auc})
    wandb.finish()

    return roc_auc


def get_roberta_finetuned_results(dataset_name: str, **kwargs) -> pd.DataFrame:
    ''' Runs finetuning of RoBERTa on a given dataset with a given set of
    hyperparameters and returns the predictions on the test data.
    
    Args:
        dataset_name (str): name of the dataset to be finetuned on
        **kwargs: keyword arguments for the Huggingface TrainingArguments class

    Returns:
        pd.DataFrame: dataframe of predictions on the test data
    '''
    data = (pd.read_parquet(f'./data_merged/{dataset_name}_merged.parquet.gzip')
            .dropna())

    data['train'] = data.text.apply(get_first_half)
    data['test'] = data.text.apply(get_second_half)
    train = (
        data.drop(columns=['text', 'test'])
        .rename(columns={'train': 'text'})
        .explode('text')
        .dropna()
    )
    test = (
        data.drop(columns=['text', 'train'])
        .rename(columns={'test': 'text'})
        .explode('text')
        .dropna()
    )
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train[['label', 'text']]),
        'test': Dataset.from_pandas(test[['label', 'text']]),
    })

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    train_dataset = tokenized_datasets['train'].shuffle(seed=42)

    model = AutoModelForSequenceClassification.from_pretrained('roberta-base')

    training_args = TrainingArguments(
        output_dir='./trainer',
        **kwargs,
        seed=42,
        report_to='none',
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()
    model.eval()

    test['roberta_finetuned_score'] = test.text.apply(
        lambda x: get_score(x, tokenizer, model)
    )
    accuracy = accuracy_score(test.label, test.roberta_finetuned_score > 0.5)
    pr_auc = average_precision_score(test.label, test.roberta_finetuned_score)
    roc_auc = roc_auc_score(test.label, test.roberta_finetuned_score)

    print(f'''\nAccuracy: {accuracy:.4f}
PR AUC: {pr_auc:.4f}
ROC AUC: {roc_auc:.4f}''')
    
    return test.drop(columns=['label']).reset_index(drop=True)
