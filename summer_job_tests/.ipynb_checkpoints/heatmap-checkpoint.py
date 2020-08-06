from train_cross_val_example import train_classifier
from sklearn.datasets import load_iris
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#retrieve data (using iris as example)
features, labels = load_iris(return_X_y=True)
results = train_classifier(features, labels, num_iterations=3)
print(results['all_rocs'])
print(results['all_precisions'])
#file_path = 'significance.tsv'


def create_heatmap_iris(results, features, labels):
    """
    significance = pd.read_csv(file_path, sep='\s', index_col=0)
    print(significance.head())
    print(significance.dtypes)
    """
    #Get aveage probability
    probas = results['all_probas']
    probas_sum = [0 for i in range(len(probas[0]))]
    num_arrays = 0
    for array in probas:
        #sns.set()
        #sns.heatmap(pd.DataFrame(array).T)
        #plt.show
        num_elements = 0
        num_arrays += 1
        for element in array:
            probas_sum[num_elements] += element
            num_elements += 1
    average_probas = np.array(probas_sum) / num_arrays

    df_probas = pd.DataFrame(average_probas)
    df_probas.columns = np.unique(labels)
    print(df_probas.head())
    """"""
    sns.set()
    sns.heatmap(df_probas.T)
    plt.show

def create_heatmap_tsv(file_path):
    significance = pd.read_csv(file_path, sep='\s', index_col=0)
    print(significance.head())
    print(significance.dtypes)

    sns.set()
    fig, ax = plt.subplots(figsize=(15, 4))
    sns.heatmap(significance.T)
    plt.show

#create_heatmap_iris(results, features, labels)
create_heatmap_tsv('Significance.tsv')
