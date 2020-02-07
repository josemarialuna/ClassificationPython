from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

np.random.seed(0)
sns.set(font_scale=0.5)

df_main = pd.read_csv('data/sum_autoencoder.txt', sep='\t', index_col=0)


data = df_main

methods = ['average']
# methods = ['ward', 'median', 'centroid', 'single', 'average', 'complete', 'weighted']

for method in methods:
    sns.clustermap(data, method=method, xticklabels=True, yticklabels=True,
                   cmap="Blues")
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.title(method)
    plt.savefig('heatmap_sum-' + method + '.png', dpi=300)
    plt.savefig('heatmap_sum-' + method + '.eps', format='eps', dpi=300)
# plt.show()
