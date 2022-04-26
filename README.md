### Abstract
This project aims to classify the DNA sequences of six differentpathogens including COVID-19. I compare the performance ofTransformer networks and a simple Convolutional Neural Network as the baseline. The dataset includes a training set andfive test sets, I experiment with full lengthed and median lengthed sequences and utilized the F1 score as my metric. 

### Dataset:
There are 6 classes of pathogens in this dataset: SARS-CoV-1, MERS-CoV, SARS-CoV-2 Ebola, Dengue, and Influenza. The dataset is gathered by the authors of the paper [1] and was available for download on their website. The dataset includes 18324 samples, the original paper has used only 1500 of this data for training, and the rest is divided into 5 test files. I will use the same number of samples for the training and test as the original paper. The dataset that is made available by the authors of [1] is in the CSV format and each CSV file includes the class number (1-6), class name, and the DNA sequence. 

#### Input data visualization:
I have used T-SNE and PCA methods to reduce the dimensionality of my data so I can visualize the training and validation data in a 2D space. As you can observe, the sequences related to SARS-COV-1, SARS-COV-2 and MERS seem to be more closely related to each other both in the T-SNE and PCA-produced figures. This is because these three pathogens are in the same genus Betacoronavirus, and we can see that it is reflected in the data as well. I think differentiating the mentioned three pathogens from each other would be the more challenging part of the classification task. The visualization of sequences in the validation set also demonstrates the same pattern, with the mentioned three pathogens being closer to each other and the other three being less closely related.

![](data/tsne_train.png)

![](data/tsne_val.png)

### Experimental settings:
I have used the TensorFlow framework for implementing the models in this work and the models are trained on an NVIDIA GTX 1080Ti GPU. Due to the limitation of resources, the batch size has been set to 4 and all models have been trained for 20 epochs. The global average pooling method has been used instead of flattening in all the models.

### Results:
The baseline achieves a higher score when using full sequences, however,transformers work better when sequences are resized to the median length. When using the whole sequences, the baseline cangeneralize better than the more complex transformers models, however when we remove some of the data in resizing the seqeunces to the median length, transformer models work better. The reuslts of testing the models are available in results folder.
|          |F1 score|
| Model  | Maximum Length Sequences |Median Length Seqeunces|
| ------------- | ------------- |
|Baseline | 0.996  | 0.906 |
|CNN_Transformer_32  | 0.985 | 0.986  |
| CNN_Transformer_64   | 0.914 | 0.959|
| CNN_Transformer_128   | 0.950 | 0.896 |

### Refrences:
[1] Indrajit, Saha, et al. “COVID-DeepPredictor: Recurrent Neural Network to Predict SARS-CoV-2 and Other Pathogenic Viruses”, Journal of Frontiers in genetics, volume 12,83,2021
