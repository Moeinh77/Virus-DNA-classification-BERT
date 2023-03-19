import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_model(model_config, return_model=False):
    """
    Load the model based on the input configuration

    Parameters
    ----------
    model_config : dict
        model configuration

    return_model : bool, optional
        return model, tokenizer, device, by default False

    Returns
    -------
    model, tokenizer, device: optional

    """

    global model, device, tokenizer

    if torch.cuda.is_available():
        # for CUDA
        torch.cuda.empty_cache()
        device = torch.device("cuda:0")
        print("Running the model on CUDA")

    elif torch.backends.mps.is_available():
        # for M1
        device = torch.device("mps")
        print("Running the model on M1 CPU")

    else:
        print("Running the model on CPU")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_HF_path"], do_lower_case=False
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_config["model_HF_path"], num_labels=model_config["num_classes"]
    )

    print(f'{ model_config["model_HF_path"]} loaded')

    model.to(device)
    # model.eval()

    if return_model:
        return model, tokenizer, device


def compute_metrics(eval_preds):
    """
    Compute the metrics for the model

    Parameters
    ----------
    eval_preds : tuple
        tuple of predictions and labels

    Returns
    -------
    :dict
        dictionary of metrics
    """

    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
