import torch
from datasets import Dataset


class SeqDataset_HF(torch.utils.data.Dataset):
    """
    This class is used to create a dataset for the HuggingFace transformers.
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return {
            'input_ids': torch.tensor(self.input_ids[index]),
            'attention_mask': torch.tensor(self.attention_masks[index]),
            'labels': torch.tensor(self.labels[index])
        }

class SeqDataset(torch.utils.data.Dataset):
    """
    This class is used to create a dataset in format of pytorch datasets.
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return self.encodings[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


def return_kmer(seq, K=6):
    """
    This function outputs the K-mers of a sequence
    
    Parameters
    ----------
    seq : str
        A single sequence to be split into K-mers
    K : int, optional
        The length of the K-mers, by default 6
    
    Returns
    -------
    kmer_seq : str
        A string of K-mers separated by spaces
    """
    
    kmer_list = []
    for x in range(len(seq) - K + 1):
        kmer_list.append(seq[x : x + K])

    kmer_seq = " ".join(kmer_list)
    return kmer_seq
