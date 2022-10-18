import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

stemmer = PorterStemmer()
stopwords_english = stopwords.words('english')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# added files
import NeuralNetwork
import Preprocessing
import sklearnModels
import Exploration
import lstm_model

# tfidf transformer to use after loading the best model
transformer = None


def read_file(path, file_type):
    """
    Reads tsv file from the given path. According to the file_type set the column names.
    :param path: path of the file
    :param file_type: train or test
    :return: Dataframe
    """
    feat = {"train": ['tweet_id', 'user_handle', 'tweet_text', 'time_stamp', 'device'],
            "test": ['user_handle', 'tweet_text', 'time_stamp']}
    df = pd.read_csv(path, sep='\\t', names=feat[file_type])
    return df


def get_train_test(data, train_size):
    """
    Split data into train and test sets. While the size of the train set equals to train_size (percent).
    :param data: data to split
    :param train_size: percent of train set
    :return: tuple of X_train, X_test, y_train, y_test
    """
    y = data.pop('label')
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)
    return X_train, X_test, y_train, y_test


def save_best_model(all_models):
    """
    Save the best model using pickle. The last best model is logistic regression with tfidf vectorizer and C=10.0
    In order to save the best model and use it on the whole train set, we loaded the train data and fitted the tfidf vectorizer
    again (in the training we split the data into train and test - so the best model didn't train on the whole data).
    :return: None
    """
    results = sorted(all_models, key=lambda x: all_models[x], reverse=True)
    best_model = results[0]

    train_path = 'trump_train.tsv'
    train_data = read_file(train_path, 'train')
    train_data = Preprocessing.pre_process(train_data, 'train')
    x_train = train_data['clean_text']
    y_train = train_data['label']
    transformer = TfidfVectorizer(ngram_range=[1, 3])
    x_vec = transformer.fit_transform(x_train)
    best_model = LogisticRegression(C=10.0, max_iter=5000)
    best_model.fit(x_vec, y_train)
    pickle.dump(best_model, open('best_model.sav', 'wb'))


def load_best_model():
    """
    Load the best model from a pickle file. But in order to use the best model we also need to fit a tfidf vectorizer
    on the train data, because we need to transform the test data according to the train data.
    :return:  best_model
    """
    global transformer
    train_path = 'trump_train.tsv'
    train_data = read_file(train_path, 'train')
    train_data = Preprocessing.pre_process(train_data, 'train')
    x_train = train_data['clean_text']
    transformer = TfidfVectorizer(ngram_range=[1, 3])
    transformer.fit_transform(x_train)
    loaded_model = pickle.load(open('best_model.sav', 'rb'))
    return loaded_model


def train_best_model():
    """
    Training the best model. First we load the train data, preprocess it, initiate an instance of the best model,
    train a tfidf vectorizer and train the model with the vectors we got from the vectorizer.
    :return: trained best model
    """
    global transformer
    train_path = 'trump_train.tsv'
    train_data = read_file(train_path, 'train')
    train_data = Preprocessing.pre_process(train_data, 'train')

    logistic_model = LogisticRegression(max_iter=5000, C=10.0)

    x_train = train_data['clean_text']
    y_train = train_data['label']
    transformer = TfidfVectorizer(ngram_range=[1, 3])
    train_vectors_tf = transformer.fit_transform(x_train)

    logistic_model.fit(train_vectors_tf, y_train)

    return logistic_model


def predict(m, fn):
    """
    Predict the labels of the test set. Read the test file, preproecss it, use the tfidf vectorizer
    which was trained previously, and use those tfidf vectors to predict the label of the test records using the
    m model.

    :param m: trained model
    :param fn: path
    :return: list of labels (0 & 1)
    """
    test_df = read_file(fn, 'test')
    test_df_pre = Preprocessing.pre_process(test_df, 'test')
    x_test = test_df_pre['clean_text']
    test_vectors_tf = transformer.transform(x_test)
    result = m.predict(test_vectors_tf)
    return list(result)


#   Running example of the whole assigment
def main():
    # # # # # # # # # # # # # # # # # #
    # How to train all models - Example #
    # # # # # # # # # # # # # # # # # #

    # set the train path
    train_path = 'trump_train.tsv'

    # load the train data and preprocess it
    train_data = read_file(train_path, 'train')
    train_data = Preprocessing.pre_process(train_data, 'train')

    # explore the train data
    Exploration.data_exploration(train_data)

    # split the train data into train and test
    X_train, X_test, y_train, y_test = get_train_test(train_data, 0.8)

    # train 5 types of models - logistic regression, SVM, RandomForest, NN, LSTM.
    logistic_model, logistic_score = sklearnModels.run_logistic_regression(X_train, X_test, y_train, y_test)
    svm_model, svm_score = sklearnModels.run_svc(X_train, X_test, y_train, y_test)
    rf_model, rf_score = sklearnModels.run_random_forest(X_train, X_test, y_train, y_test)
    nn_model, nn_score = NeuralNetwork.NeuralNetwork().fit(X_train, X_test, y_train, y_test)
    lstm_model_, lstm_score = lstm_model.fit(X_train, X_test, y_train, y_test)

    # get the results of all models
    all_models = {logistic_model: logistic_score,
                  svm_model: svm_score,
                  rf_model: rf_score,
                  nn_model: nn_score,
                  lstm_model_: lstm_score}

    # save the best model according to the all_models scores
    save_best_model(all_models)

    # # # # # # # # # # # # # # # # # #
    # How to use the driver - Example #
    # # # # # # # # # # # # # # # # # #

    # set the test file path
    test_path = 'trump_test.tsv'
    # load the best model (be sure the pickle file is on the same directory)
    model = load_best_model()
    # if you want you can train the same best model but from scratch
    model_trained = train_best_model()
    # use the model or model_trained in order to predict the test record labels
    results = predict(model, test_path)
    # convert int to str
    results = [str(x) for x in results]
    # write results into a txt file
    with open('results.txt', 'w') as f:
        f.write(" ".join(results))