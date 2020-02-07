from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

separator = '\t'
brcaFile = r"C:\datasets\brca\brca_autoencoded.csv"

df_main = pd.read_csv(brcaFile, sep=separator, index_col=0)
df_main['Class'] = df_main['Class'].map({'YES': 1, 'NO': 0})

instances_training = len(df_main)
scaler = MinMaxScaler()

X = scaler.fit_transform(df_main)
y = df_main.pop('Class').values

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

mcc_scores = []
for idx, (train_index, test_index) in enumerate(skfold.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    rf = RandomForestClassifier(n_estimators=100, random_state=None,
                                min_samples_leaf=int(0.03 * instances_training))
    rf.fit(X_train, y_train)
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    print("=== cohen_kappa_score ===")
    kappa = cohen_kappa_score(y_test, y_pred_test)
    print(kappa)
    print('\n')

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred_test))
    print('\n')

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred_test))
    print('\n')

    print("Train Accuracy : ", accuracy_score(y_train, y_pred_train))
    accuracy = accuracy_score(y_test, y_pred_test)
    print("Test Accuracy  : ", accuracy)

    print("\n\n")

    mcc_scores.append(accuracy)
    # mcc = matthews_corrcoef(y_test, y_pred_test)
    # mcc_scores.append(mcc)
