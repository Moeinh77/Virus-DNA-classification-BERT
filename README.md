# Virus-DNA-Classification

### Dataset:

#### Dataset importance: 
DNA sequence classification has been a topic of interest in Bioinformatics research for many years. Recognizing different sequences and classifying them is a step closer to the treatment of many genetic disorders. After the recent pandemic, the importance of developing our knowledge about pathogens such as SARS and MERS families has shown its importance.  It's critical to analyze the data for a better taxonomic classification of the viruses so that strategic planning for controlling and treating the virus can continue with more knowledge about this virus’ nature. My project involves the classification of six different pathogens based on their DNA sequences. These pathogens are SARS-CoV-1, MERS-CoV, SARS-CoV-2 Ebola, Dengue, and Influenza. 

#### Dataset structure and preprocessing:
The dataset is gathered by the authors of the paper [1] and was available for download on their website. The dataset includes 18324 samples, the original paper has used only 1500 of this data for training, and the rest is divided into 5 test files. I will use the same number of samples for the training and test as the original paper. The dataset that is made available by the authors of [1] is in the CSV format and each CSV file includes the class number (1-6), class name, and the DNA sequence. 

#### Class Distribution:

In the training set each of the six classes has 250 samples and this makes the training set very balanced. However, the test files are imbalanced and do not include the same number of samples for each class. I use test set 2 for the validation purpose during the training of my models. Test set 2 includes 200 samples for classes 2 to 6 and 91 samples for class 1. I have used the F1 score to measure the performance of my models on each of the test sets.

#### Input data visualization:
I have used T-SNE and PCA methods to reduce the dimensionality of my data so I can visualize the training and validation data in a 2D space. As you can observe, the sequences related to SARS-COV-1, SARS-COV-2 and MERS seem to be more closely related to each other both in the T-SNE and PCA-produced figures. This is because these three pathogens are in the same genus Betacoronavirus, and we can see that it is reflected in the data as well. I think differentiating the mentioned three pathogens from each other would be the more challenging part of the classification task.
The visualization of sequences in the validation set also demonstrates the same pattern, with the mentioned three pathogens being closer to each other and the other three being less closely related.
[]
[]

### Refrences:
[1] Indrajit, Saha, et al. “COVID-DeepPredictor: Recurrent Neural Network to Predict SARS-CoV-2 and Other Pathogenic Viruses”, Journal of Frontiers in genetics, volume 12,83,2021
