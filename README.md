## 2023 version with DNA-BERT:
I decided to re-do this project but with a pre-trained model (**DNA-BERT**) found at this repository: https://github.com/jerryji1993/DNABERT. I have used **HuggingFace** library for loading the model, training, and evaluating it. The pre-trained easily and quickly beats my last year's implementation which was using the Tensorflow MultiHeadAttention module.
The experiments from the last year are in the old_code folder now.

I used the **Weights and Biases** library for logging the results of the training and as shown below in only 2 epochs the model reaches an f1 score of 0.99. 

![](data/eval_logs.png)

In order to use fine-tune the DNA-BERT model using the HuggingFace API you need to load the model and prepare the sequences like the following:

```python
NUM_CLASSES = # number of the classes in your data

from utils.data_utils import return_kmer, HF_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
tokenizer = AutoTokenizer.from_pretrained(
   zhihan1996/DNA_bert_6, do_lower_case=False
)

model = AutoModelForSequenceClassification.from_pretrained(
     zhihan1996/DNA_bert_6, num_labels=NUM_CLASSES
)

sequences = # your DNA sequences 

train_kmers = [return_kmer(seq) for seq in sequences]

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
```
Now you can feed the train_dataset to transformers.Trainer.

I have also implemented a UI in a form of a Flask Web application for the trained model. If you are interested in the code checkout app.py.
In order to run the web app locally you need to install the Flask.

![](UI.png)

------------------------------------------------------------------------------------------------------------------------------
## 2022 version with Transformer Module from TF:
### Introduction
This project aims to classify the DNA sequences of six different pathogens including COVID-19. I compare the performance of Transformer networks and a simple Convolutional Neural Network as the baseline. The dataset includes a training set and five test sets, I experiment with full lengthed and median lengthed sequences and utilized the F1 score as my metric. 

### Dataset:
There are 6 classes of pathogens in this dataset: SARS-CoV-1, MERS-CoV, SARS-CoV-2 Ebola, Dengue, and Influenza. The dataset is gathered by the authors of the paper [1] and was available for download on their website. The dataset includes 18324 samples, the original paper has used only 1500 of this data for training, and the rest is divided into 5 test files. I will use the same number of samples for the training and test as the original paper. The dataset that is made available by the authors of [1] is in the CSV format and each CSV file includes the class number (1-6), class name, and the DNA sequence.  [dataset](http://www.nitttrkol.ac.in/indrajit/projects/COVID-DeepPredictor/)

#### Input data visualization:
I have used T-SNE and PCA methods to reduce the dimensionality of my data so I can visualize the training and validation data in a 2D space. As you can observe, the sequences related to SARS-COV-1, SARS-COV-2, and MERS seem to be more closely related to each other both in the T-SNE and PCA-produced figures. This is because these three pathogens are in the same genus Betacoronavirus, and we can see that it is reflected in the data as well. I think differentiating the mentioned three pathogens from each other would be the more challenging part of the classification task. The visualization of sequences in the validation set also demonstrates the same pattern, with the mentioned three pathogens being closer to each other and the other three being less closely related.

![](data/tsne_train.png)

![](data/tsne_val.png)

### Experimental settings:
I have used the TensorFlow framework for implementing the models in this work and the models are trained on an NVIDIA GTX 1080Ti GPU. Due to the limitation of resources, the batch size has been set to 4 and all models have been trained for 20 epochs. The global average pooling method has been used instead of flattening in all the models.

### Transformer settings:
The number of attention heads is set to 2 for the transformer models in my project. Due to the long length of the input sequences, I was not able to feed them directly to the transformer model, therefore I have used a convolution block that includes a conv1d layer and a max-pooling layer to reduce the dimensions of the input. I experiment with 32, 64 and 128 conv filters for the transformers. The positional encoded embeddings are fed into the conv block and the result of the convolution is then passed on to the attention block. The results of the attention block are fed to a global average pooling layer and then to a feed-forward layer of 20 nodes before the final softmax layer.

![](CNN_Transformer_32_architecture.png)

### Results:
The baseline achieves a higher score when using full sequences, however, transformers work better when sequences are resized to the median length. When using the whole sequences, the baseline can generalize better than the more complex transformer models, however when we remove some of the data in resizing the sequences to the median length, transformer models work better. This project demonstrates that the biggest model doesn’t always yield the best result. Perhaps, it is better to always start with simpler models such as a simple CNN, and then try more complex models like transformers. Additionally, it's good to remember that we can always benefit from CNNs as feature extractors. The input sequences are downsized by a scale of 4 due to the conv layers and the follow-up max-pooling layers, however, the attention heads can still learn the data well and achieve high scores on test files as well. 
The table below demonstrates the average accuracy of each model on all five test sets of the data. Individual scores are available in results folders.

| Model  | Maximum Length Sequences |Median Length Seqeunces|
| ------------- | ------------- |------------- |
| Baseline CNN | **0.996**  | 0.906 |
| CNN_Transformer_32  | 0.985 | **0.986**  |
| CNN_Transformer_64   | 0.914 | 0.959|
| CNN_Transformer_128   | 0.950 | 0.896 |

### References:
[1] Indrajit, Saha, et al. “COVID-DeepPredictor: Recurrent Neural Network to Predict SARS-CoV-2 and Other Pathogenic Viruses”, Journal of Frontiers in genetics, volume 12,83,2021
