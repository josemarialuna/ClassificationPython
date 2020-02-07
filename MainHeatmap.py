from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

np.random.seed(0)
sns.set(font_scale=0.5)

df_main = pd.read_csv('data/jorge.csv', sep='\t', index_col=0)
df_main['Class'] = df_main['Class'].map({'YES': 1, 'NO': 0})
# df_class = df_main.pop('Class')
# df_main.pop('Time')


scaler = MinMaxScaler()
df_main[df_main.columns] = scaler.fit_transform(df_main[df_main.columns])

n_clusters = (2, 2)

my_palette = dict(zip(df_main.Class.unique(), ['orange', 'brown']))
row_colors = df_main.Class.map(my_palette)
df_main.pop('Class')

# my_palette2 = dict(zip(df_main.Time.unique(), sns.color_palette("RdBu_r", 7)))
# row_colors2 = df_main.Time.map(my_palette2)
# df_main.pop('Time')

data = df_main

methods = ['average']
# methods = ['ward', 'median', 'centroid', 'single', 'average', 'complete', 'weighted']

for method in methods:
    sns.clustermap(data, method=method, xticklabels=True, yticklabels=True, row_colors=[row_colors],
                   cmap="Blues")
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.title(method)
    plt.savefig('heatmap-' + method + '.png', dpi=300)
    plt.savefig('heatmap-' + method + '.eps', format='eps')
# plt.show()
