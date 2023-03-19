#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import os
import numpy as np
from utils.data_utils import return_kmer, val_datasets_generator, HF_dataset
from utils.model_utils import load_model, compute_metrics
from transformers import Trainer, TrainingArguments
import wandb

wandb.finish()
KMER = 3  # The length of the K-mers to be used by the model and tokenizer


df_training = pd.read_csv("data/Trainingdata.csv")
df_training["CLASS"] = df_training["CLASS"].apply(lambda x: x - 1)
y_train = df_training["CLASS"].values

df_training.head(3)


NUM_CLASSES = len(np.unique(y_train))


model_config = {
    "model_HF_path": f"zhihan1996/DNA_bert_{KMER}",
    "num_classes": NUM_CLASSES,
}

model, tokenizer, device = load_model(model_config, return_model=True)


train_kmers = []
for seq in df_training["SEQ"]:
    kmer_seq = return_kmer(seq, K=KMER)
    train_kmers.append(kmer_seq)


train_encodings = tokenizer.batch_encode_plus(
    train_kmers,
    max_length=512,  # max len of BERT
    padding=True,
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt",
)

train_dataset = HF_dataset(
    train_encodings["input_ids"], train_encodings["attention_mask"], y_train
)


df_val = pd.read_csv(
    f"data/TestData/Testdata-2.csv"
)  # i use the Testdata-2 as the validation set
val_kmers, labels_val = [], []
for seq, label in zip(df_val["SEQ"], df_val["CLASS"]):
    kmer_seq = return_kmer(seq, K=KMER)
    val_kmers.append(kmer_seq)
    labels_val.append(label - 1)


val_encodings = tokenizer.batch_encode_plus(
    val_kmers,
    max_length=512,
    padding=True,
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt",
)
val_encodings = tokenizer(val_kmers, truncation=True, padding=True)
val_dataset = HF_dataset(
    val_encodings["input_ids"], val_encodings["attention_mask"], labels_val
)


results_dir = f'./results/classification/{model_config["model_HF_path"]}/'
EPOCHS = 10
bs = 8

wandb.init(project="DNA_bert", name=model_config["model_HF_path"])
wandb.config.update(model_config)


training_args = TrainingArguments(
    output_dir=results_dir + "training_results",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=60,
    load_best_model_at_end=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

trainer.train()


eval_results = []
for val_dataset in val_datasets_generator(tokenizer, KMER, val_dir="data/TestData/"):
    res = trainer.evaluate(val_dataset)
    eval_results.append(res)

# average over the eval_accuracy and eval_f1 from the dic items in eval_results
avg_acc = np.mean([res["eval_accuracy"] for res in eval_results])
avg_f1 = np.mean([res["eval_f1"] for res in eval_results])

wandb.log({"avg_acc": avg_acc, "avg_f1": avg_f1})
wandb.finish()
