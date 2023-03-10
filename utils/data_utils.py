import torch
from datasets import Dataset

class HF_dataset(torch.utils.data.Dataset):
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

def val_datasets_generator(val_dir='data/TestData'):
    for file in os.listdir(val_dir):
        df_test = pd.read_csv(f'{val_dir}/{file}')
        print(file, len(df_test))
        val_kmers, labels_val = [], []
        
        cls = 'CLASS' if 'CLASS' in df_test.columns else 'Class'

        for seq, label in zip(df_test['SEQ'], df_test[cls]):
                kmer_seq = return_kmer(seq, K=KMER)
                val_kmers.append(kmer_seq)
                labels_val.append(label-1)
        val_encodings = tokenizer.batch_encode_plus(
            val_kmers,
            max_length=512,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        val_dataset = HF_dataset(val_encodings['input_ids'], val_encodings['attention_mask'], labels_val)
        yield val_dataset

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
