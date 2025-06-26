#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from datasets import load_dataset, Features, Value
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification, 
                          Trainer, 
                          TrainingArguments)
import numpy as np
from sklearn.metrics import accuracy_score

model_name = "/root/autodl-tmp/OpenPrompt/bert-base-uncased"
output_dir = "/root/autodl-tmp/OpenPrompt/my_nlp_project/outputs/finetuning_baseline"
num_train_epochs = 3
per_device_train_batch_size = 16
learning_rate = 2e-5

data_files = {
    "train": "/root/autodl-tmp/OpenPrompt/my_nlp_project/data/train.tsv",
    "validation": "/root/autodl-tmp/OpenPrompt/my_nlp_project/data/dev.tsv"
}

features = Features({
    'sentence': Value('string'),
    'label': Value('int64')
})

raw_datasets = load_dataset(
    "csv", 
    data_files=data_files, 
    delimiter="\t", 
    features=features,
    column_names=["sentence", "label"]
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    learning_rate=learning_rate,
    evaluation_strategy="epoch",
    logging_dir=f"{output_dir}/logs",
    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

eval_results = trainer.evaluate()