import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

labels = ['SARS-COV-1', 'MERS', 'SARS-COV-2', 'Ebola', 'Dengue', 'Influenza']

def count_plot(Y, label):
    """
    Plot the class distribution of the data set set

    Parameters
    ----------
    Y : numpy array
        The class labels of the data set
    label : str
        The name of the data

    Returns
    -------
    None
    """
    sns.countplot(Y)
    plt.title(f'Class dist in {} set')
    plt.savefig(f'{label}.png')
    plt.show()

def plot_tsne(X, class_num, label):
    X_embedded_tsne = TSNE(n_components=2, n_iter = 2000 , init='random').fit_transform(X)

    lower_bound = 0
    for k in range(1,class_num):
        upper_bound= 250*k
        plt.scatter(X_embedded_tsne[lower_bound:upper_bound,0],
                    X_embedded_tsne[lower_bound:upper_bound,1]
                    )
        lower_bound = upper_bound

    plt.title(f'TSNE - {label} data 2D')
    plt.legend(labels)
    plt.savefig(f'{label}.png')    

def plot_pca(X, class_num, label):
    X_embedded_pca = PCA(n_components=2).fit_transform(X)
    lower_bound = 0
    for k in range(1,class_num):
        upper_bound= 250*k
        plt.scatter(X_embedded_pca[lower_bound:upper_bound,0],
                    X_embedded_pca[lower_bound:upper_bound,1]
                    )
        lower_bound = upper_bound

    plt.title('PCA - Training data 2D')
    plt.legend(labels, loc = 'upper left')
    plt.savefig(f'{label}.png')