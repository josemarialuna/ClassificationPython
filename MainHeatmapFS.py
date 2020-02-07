from docutils.nodes import thead
from sklearn.preprocessing import MinMaxScaler, binarize
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

np.random.seed(0)
sns.set(font_scale=0.5)

df_main = pd.read_csv('data/101280.txt', sep='\t', index_col=0)
df_main['Class'] = df_main['Class'].map({'YES': 1, 'NO': 0})
# df_class = df_main.pop('Class')
df_main.pop('Time')

scaler = MinMaxScaler()
df_main[df_main.columns] = scaler.fit_transform(df_main[df_main.columns])


my_palette = dict(zip(df_main.Class.unique(), ['orange', 'brown']))
row_colors = df_main.Class.map(my_palette)
y = df_main['Class']
#
# binarize(df_main, copy=False, threshold=0.75)
# # df_main[df_main.columns] = binarizer.fit_transform(df_main[df_main.columns])

# # Create and fit selector
# selector = SelectKBest(f_classif, k=15)
# selector.fit(df_main, y)
# # Get columns to keep
# cols = selector.get_support(indices=True)
# # Create new dataframe with only desired columns, or overwrite existing
# features_df_new = df_main.iloc[:, cols]

# my_palette2 = dict(zip(df_main.Time.unique(), sns.color_palette("RdBu_r", 7)))
# row_colors2 = df_main.Time.map(my_palette2)
# df_main.pop('Time')

data = df_main

print(data.head(50))

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
