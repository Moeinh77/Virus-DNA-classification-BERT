#!/usr/bin/env python
# coding: utf-8

############################################################################################################################
# This is the main script for training the DNA_bert model on the training data and evaluating on the test data             #
# The model is trained and evaluated using the HuggingFace Trainer API                                                     #
# The model is saved in the "saved_models" directory                                                                       #
# The model is logged to Weights and Biases using the WandbLogger                                                          #
############################################################################################################################

#########################################
### Importing the necessary libraries ###
#########################################

import wandb
import pandas as pd
import numpy as np
from pathlib import Path
from utils.data_utils import return_kmer, val_dataset_generator, HF_dataset
from utils.model_utils import load_model, compute_metrics
from utils.viz_utils import count_plot
from transformers import Trainer, TrainingArguments

############################################
### Reading the training and test data ####
############################################

KMER = 3  # The length of the K-mers to be used by the model and tokenizer

training_data_path = Path("data/TrainingData/Trainingdata.csv")
eval_data_path = Path("data/TestData/Testdata-2.csv")

df_training = pd.read_csv(training_data_path)

train_kmers, labels_train = [], []
for seq, label in zip(df_training["SEQ"], df_training["CLASS"]):
    kmer_seq = return_kmer(seq, K=KMER)
    train_kmers.append(kmer_seq)
    labels_train.append(label - 1)

NUM_CLASSES = len(np.unique(labels_train))

count_plot(labels_train, "Training Class Distribution")

model_config = {
    "model_path": f"zhihan1996/DNA_bert_{KMER}",
    "num_classes": NUM_CLASSES,
}

model, tokenizer, device = load_model(model_config, return_model=True)

SEQ_MAX_LEN = 512  # max len of BERT

train_encodings = tokenizer.batch_encode_plus(
    train_kmers,
    max_length=SEQ_MAX_LEN,
    padding=True,  # pad to max len
    truncation=True,  # truncate to max len
    return_attention_mask=True,
    return_tensors="pt",  # return pytorch tensors
)
train_dataset = HF_dataset(
    train_encodings["input_ids"], train_encodings["attention_mask"], labels_train
)


df_val = pd.read_csv(eval_data_path)  # i use the Testdata-2 as the validation set

val_kmers, labels_val = [], []
for seq, label in zip(df_val["SEQ"], df_val["CLASS"]):
    kmer_seq = return_kmer(seq, K=KMER)
    val_kmers.append(kmer_seq)
    labels_val.append(label - 1)

count_plot(labels_val, "Validation Class Distribution")

val_encodings = tokenizer.batch_encode_plus(
    val_kmers,
    max_length=SEQ_MAX_LEN,
    padding=True,  # pad to max len
    truncation=True,  # truncate to max len
    return_attention_mask=True,
    return_tensors="pt",  # return pytorch tensors
)
val_dataset = HF_dataset(
    val_encodings["input_ids"], val_encodings["attention_mask"], labels_val
)

############################################
### Training and evaluating the model #####
############################################

results_dir = Path("./results/classification/")
results_dir.mkdir(parents=True, exist_ok=True)
EPOCHS = 10
BATCH_SIZE = 8

# initialize wandb for logging the training process
wandb.init(project="DNA_bert", name=model_config["model_path"])
wandb.config.update(model_config)

training_args = TrainingArguments(
    output_dir=results_dir / "checkpoints",  # output directory
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # L2 regularization lambda value
    logging_steps=60,  # log metrics every 60 steps
    load_best_model_at_end=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,  # compute metrics function is used for evaluation at the end of each epoch
    tokenizer=tokenizer,
)

trainer.train()

# save the model and tokenizer
model_path = results_dir / "model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# evaluate on all the test datasets
eval_results = []
for val_dataset in val_dataset_generator(
    tokenizer, kmer_size=KMER, val_dir="data/TestData/"
):
    res = trainer.evaluate(val_dataset)
    eval_results.append(res)

# average over the eval_accuracy and eval_f1 from the dic items in eval_results
avg_acc = np.mean([res["eval_accuracy"] for res in eval_results])
avg_f1 = np.mean([res["eval_f1"] for res in eval_results])

print(f"Average accuracy: {avg_acc}")
print(f"Average F1: {avg_f1}")

wandb.log({"avg_acc": avg_acc, "avg_f1": avg_f1})
wandb.finish()
