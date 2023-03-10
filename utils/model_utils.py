import torch 
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

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

    tokenizer = AutoTokenizer.from_pretrained(model_config["model_path"] , do_lower_case=False)

    model = AutoModelForSequenceClassification.from_pretrained(model_config["model_path"] ,)

    print(f'{ model_config["model_path"]} loaded')

    model.to(device)
    # model.eval()

    if return_model:
        return model, tokenizer, device


def tokenize(batch):
    return tokenizer(batch, padding="max_length", truncation=True)
