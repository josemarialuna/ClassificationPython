from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, f1_score, cohen_kappa_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

separator = '\t'
brcaFile = 'data/101280.txt'

df_main = pd.read_csv(brcaFile, sep=separator, index_col=0)
df_main['Class'] = df_main['Class'].map({'YES': 1, 'NO': 0})

instances_training = len(df_main)
scaler = MinMaxScaler()
df_main.pop('Time')
df_main.pop('c13')
df_main.pop('c27')
df_main.pop('c30')
df_main.pop('c60')
df_main.pop('c62')
df_main.pop('c66')
df_main.pop('c76')
df_main.pop('c99')


y = df_main.pop('Class').values
X = scaler.fit_transform(df_main)

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

mcc_scores = []
# Logging for Visual Comparison
log_cols = ["Classifier", "Accuracy", "Log Loss", "F1-Score", "Kappa", "Recall"]
log = pd.DataFrame(columns=log_cols)
for idx, (train_index, test_index) in enumerate(skfold.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="rbf", C=0.025, probability=True),
        NuSVC(probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        GaussianNB()]
    # LinearDiscriminantAnalysis(),
    # QuadraticDiscriminantAnalysis()]

    for clf in classifiers:
        clf.fit(X_train, y_train)
        name = clf.__class__.__name__

        print("=" * 30)
        print(name)

        print('****Results****')
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        print("Accuracy: {:.4%}".format(acc))

        ll = log_loss(y_test, train_predictions)
        print("Log Loss: {}".format(ll))

        f1 = f1_score(y_test, train_predictions)
        print("F1-Score: {}".format(f1))

        kappa = cohen_kappa_score(y_test, train_predictions)
        print("Kappa: {}".format(kappa))

        recall = recall_score(y_test, train_predictions)
        print("Recall: {}".format(recall))

        log_entry = pd.DataFrame([[name, acc * 100, ll, f1* 100, kappa* 100, recall* 100]], columns=log_cols)
        log = log.append(log_entry)

log = log.groupby('Classifier').mean().reset_index()
log.to_csv('Classification_results.csv', sep='\t')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

plt.xlabel('Accuracy %')
# plt.title('Classifier Accuracy')
plt.tight_layout()
figure = plt.gcf()  # get current figure
figure.set_size_inches(8, 8)
plt.savefig('Classifier Accuracy' + '.png', dpi=100)
figure.clear()

sns.set_color_codes("muted")
figure = sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")
figure = figure.get_figure()
plt.tight_layout()
figure.set_size_inches(8, 8)
figure.savefig('Classifier Log Loss' + '.png', dpi=100)

# plt.xlabel('Log Loss')
# plt.title('Classifier Log Loss')
# plt.tight_layout()
# figure = plt.gcf()  # get current figure
# figure.set_size_inches(10, 8)
# plt.savefig('Classifier Log Loss' + '.png', dpi=100)
