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
import seaborn as sns
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os

separator = ','
brcaFile = 'C:\\datasets\\brca\\brca - copia.csv'
# brcaFile = 'C:\\datasets\\brca\\test.csv'

print("Loading file...")
df_main = pd.read_csv(brcaFile, sep=separator, index_col=0)
print("Loaded!")
print(df_main.head())
df_main['class'] = df_main['class'].map({'YES': 1, 'NO': 0})

instances_training = len(df_main)
scaler = MinMaxScaler()
# df_main.pop('Time')

print("Scaling data...")
X = scaler.fit_transform(df_main)
print("Scaled!")
y = df_main.pop('class').values

for i in range(0, 10):
    print("Starting Iteration" + str(i))
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)

    mcc_scores = []
    # Logging for Visual Comparison
    log_cols = ["Classifier", "Accuracy", "Log Loss", "F1-Score", "Kappa", "Recall"]
    log = pd.DataFrame(columns=log_cols)
    for idx, (train_index, test_index) in enumerate(skfold.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print("Saving files" + str(i) + "...")

        os.mkdir('./' + str(i))
        np.savetxt('./' +str(i) + '/X_train.csv', X_train, delimiter=",")
        np.savetxt('./' +str(i) + '/X_test.csv', X_test, delimiter=",")
        np.savetxt('./' +str(i) + '/y_train.csv', y_train, delimiter=",")
        np.savetxt('./' +str(i) + '/y_test.csv', y_test, delimiter=",")

        ##### AUTOENCODER #####

        with tf.device('/device:gpu:1'):
            def plot_training_history(history, keys):
                for k in keys:
                    plt.figure()
                    plt.plot(history.history[k])
                    plt.title('model ' + k)
                    plt.ylabel(k)
                    plt.xlabel('epoch')
                    plt.legend(['train'], loc='upper left')
                    plt.show()

        print("Initializing...")

        num_instances, num_attr = X_train.shape[0], X_train.shape[1]

        # this is the size of our encoded representations
        encoding_dim = 100  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
        encoding_dim2 = 200
        encoding_dim3 = 100

        # this is our input placeholder
        input = Input(shape=(num_attr,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(encoding_dim, activation='relu')(input)
        decoded = Dense(num_attr, activation='sigmoid')(encoded)
        encoder = Model(input, encoded)

        autoencoder = Model(input, decoded)
        autoencoder.summary()

        autoencoder.compile(optimizer='adadelta', loss='mse')

        ## Ajustar número de epochs y tamaño batch
        history = autoencoder.fit(X_train, X_train,
                                  epochs=20,
                                  batch_size=10,
                                  shuffle=True,
                                  )

        plot_training_history(history, ["loss"])

        X_train_encoded = encoder.predict(X_train)
        X_test_encoded = encoder.predict(X_test)

        np.savetxt('./' + str(i) + '/X_train_encoded.csv', X_train_encoded, delimiter=",")
        np.savetxt('./' + str(i) + '/X_test_encoded.csv', X_test_encoded, delimiter=",")

        # weights = encoder.get_weights()[0]
        # print(type(weights))
        # np.save("weights", weights)
        #
        # weights_transpose = weights.transpose()
        #
        # for peso in pen[1]:
        #     # for peso in range(1, 101):
        #     np.savetxt(str(pen[0]) + "_weights_c" + str(peso) + ".csv", weights_transpose[peso - 1], delimiter='\t')

        #######################

        print("Saved all files from iteration " + str(i) + "succesfully!")

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
            clf.fit(X_train_encoded, y_train)
            name = clf.__class__.__name__

            print("=" * 30)
            print(name)

            print('****Results****')
            train_predictions = clf.predict(X_test_encoded)
            np.savetxt('./' + str(i) + '/y_test_predict.csv', train_predictions, delimiter=",")
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

            log_entry = pd.DataFrame([[name, acc * 100, ll, f1 * 100, kappa * 100, recall * 100]], columns=log_cols)
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

    plt.xlabel('Log Loss')
    plt.title('Classifier Log Loss')
    plt.tight_layout()
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(10, 8)
    plt.savefig('Classifier Log Loss' + '.png', dpi=100)
