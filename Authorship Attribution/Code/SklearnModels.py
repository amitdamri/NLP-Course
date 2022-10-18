from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def get_meta_features(x_train, x_test):
    """
    Get the meta features only. Scale the features using MinMaxScaler.
    :param x_train: train data
    :param x_test: test data
    :return: train mata data, test meta data
    """
    train_features = ['month', 'day', 'hour', '#capital_letters', '#num_hashtags',
                      '#num_retweet', '#num_mentions', '#num_stopwords', 'num_words', 'num_chars']

    scaler = MinMaxScaler()
    meta_train = scaler.fit_transform(x_train[train_features])
    meta_test = scaler.transform(x_test[train_features])

    return meta_train, meta_test


def run_grid_search(train_x, train_y, test_x, test_y, name, param_grid, model):
    """
    Run grid search using the given model and param grid on the given train and test sets.
    We use 5 fold cross validation.
    :param train_x: train data
    :param train_y: train labels
    :param test_x: test data
    :param test_y: test labels
    :param name: name of the model
    :param param_grid: parameters to search
    :param model: model to fit
    :return: best estimator, score on the test set
    """
    grid_search_classifier = GridSearchCV(model, cv=5, param_grid=param_grid)

    grid_search_classifier.fit(train_x, train_y)
    print("*" * 80, "\n", name, "\n", "*" * 80)
    print("Best score: ", grid_search_classifier.best_score_)
    print("Best params: ", grid_search_classifier.best_params_)
    test_score = grid_search_classifier.best_estimator_.score(test_x, test_y)
    print("Test accuracy: ", test_score)
    return grid_search_classifier.best_estimator_, test_score


def run_logistic_regression(x_train, x_test, y_train, y_test):
    """
    Run grid search on 3 different logistic regression models - tfidf, ngram, meta features
    and return the best model and its result
    :param x_train: train data
    :param x_test: test data
    :param y_train: train label
    :param y_test: test label
    :return: best logistic model, best result
    """
    logistic_model, logistic_score = logistic_regression_model(x_train, x_test, y_train, y_test)
    logistic_ngram_model, ngram_score = logistic_regression_ngrams_model(x_train, x_test, y_train, y_test)
    logistic_tf_idf_model, tfidf_score = logistic_regression_tfidf_model(x_train, x_test, y_train, y_test)
    results = {logistic_model: logistic_score, logistic_ngram_model: ngram_score, logistic_tf_idf_model: tfidf_score}
    results_sorted = sorted(results, key=lambda x: results[x], reverse=True)
    return results_sorted[0], results[results_sorted[0]]


def logistic_regression_model(x_train, x_test, y_train, y_test):
    """
    Run grid search with logistic regressing model with meta features only.
    You can see the meta features in get_meta_features function.
    :param x_train: train data
    :param x_test: test data
    :param y_train: train label
    :param y_test: test label
    :return: logistic model, score
    """
    param_grid_ = {'C': [1e-5, 1e-3, 1e-1, 1e0, 1e1, 1e2]}
    logistic_model = LogisticRegression(max_iter=5000)
    x_train_meta, x_test_meta = get_meta_features(x_train, x_test)
    best_logistic_model, score = run_grid_search(x_train_meta, y_train, x_test_meta, y_test,
                                                 'Logistic Regression Model', param_grid_, logistic_model)

    return best_logistic_model, score


def logistic_regression_ngrams_model(x_train, x_test, y_train, y_test):
    """
    Run grid search with logistic regressing model with clean tweets only. Each tweet transformed into counter using
    CountVectorizer.
    :param x_train: train data
    :param x_test: test data
    :param y_train: train label
    :param y_test: test label
    :return: logistic model, score
    """
    param_grid_ = {'C': [1e-5, 1e-3, 1e-1, 1e0, 1e1, 1e2]}
    logistic_model = LogisticRegression(max_iter=5000)

    x_bow_train = x_train['clean_text']
    x_bow_test = x_test['clean_text']

    count_vectorizer = feature_extraction.text.CountVectorizer(ngram_range=[1, 3])
    train_vectors_bow = count_vectorizer.fit_transform(x_bow_train)
    test_vectors_bow = count_vectorizer.transform(x_bow_test)

    best_bow_logistic_model, score = run_grid_search(train_vectors_bow, y_train, test_vectors_bow, y_test,
                                                     'Logistic Regression NGRAM Model', param_grid_, logistic_model)

    return best_bow_logistic_model, score


def logistic_regression_tfidf_model(x_train, x_test, y_train, y_test):
    """
    Run grid search with logistic regressing model with clean tweets only. Each tweet transformed into tfidf value using
    TfidfVectorizer.
    :param x_train: train data
    :param x_test: test data
    :param y_train: train label
    :param y_test: test label
    :return: logistic model, score
    """
    param_grid_ = {'C': [1e-5, 1e-3, 1e-1, 1e0, 1e1, 1e2]}
    logistic_model = LogisticRegression(max_iter=5000)

    x_tf_train = x_train['clean_text']
    x_tf_test = x_test['clean_text']

    transformer = TfidfVectorizer(ngram_range=[1, 3])
    train_vectors_tf = transformer.fit_transform(x_tf_train)
    test_vectors_tf = transformer.transform(x_tf_test)

    best_tfidf_logistic_model, score = run_grid_search(train_vectors_tf, y_train, test_vectors_tf, y_test,
                                                       'Tfidf token Model',
                                                       param_grid_, logistic_model)

    return best_tfidf_logistic_model, score


def run_random_forest(x_train, x_test, y_train, y_test):
    """
    Run grid search on RandomForest Model and return its result.
    :param x_train: train data
    :param x_test: test data
    :param y_train: train label
    :param y_test: test label
    :return: random forest model, result
    """
    rf, score = random_forest_model(x_train, x_test, y_train, y_test)
    return rf, score


def random_forest_model(x_train, x_test, y_train, y_test):
    """
    Run grid search on RandomForest Model, the parameter we check is max_depth.
    We ran the model only on the meta features given by get_meta_features function.
    :param x_train: train data
    :param x_test: test data
    :param y_train: train label
    :param y_test: test label
    :return: random forest model, result
    """
    param_grid_ = {'max_depth': range(1, 4)}
    rf_model = RandomForestClassifier()
    x_train_meta, x_test_meta = get_meta_features(x_train, x_test)
    best_rf_model, score = run_grid_search(x_train_meta, y_train, x_test_meta, y_test,
                                           'RandomForest Model', param_grid_, rf_model)
    return best_rf_model, score


def get_train_features_1gram(X_train, X_test):
    """
    get 1gram features
    :param X_train: dataframe of X features of train set
    :param X_test: dataframe of X features of test set
    :return: 1gram dataframe of train, 1gram dataframe of test and a CountVectorizer fitted object
    """
    vectorizer = CountVectorizer(analyzer='word', stop_words='english', ngram_range=(1, 1))
    ngrams_train = vectorizer.fit_transform(X_train["clean_text"])
    ngrams_test = vectorizer.transform(X_test["clean_text"])

    # arr = ngrams.toarray()
    return ngrams_train, ngrams_test, vectorizer


def get_train_features_2gram(X_train, X_test):
    """
    get 2gram features
    :param X_train: dataframe of X features of train set
    :param X_test: dataframe of X features of test set
    :return: 2gram dataframe of train, 2gram dataframe of test and a CountVectorizer fitted object
    """
    vectorizer = CountVectorizer(analyzer='word', stop_words='english', ngram_range=(2, 2))
    ngrams_train = vectorizer.fit_transform(X_train["clean_text"])
    ngrams_test = vectorizer.transform(X_test["clean_text"])

    # arr = ngrams.toarray()
    return ngrams_train, ngrams_test, vectorizer


def get_train_features_tfidf(X_train, X_test):
    """
    get tfidf features
    :param X_train: dataframe of X features of train set
    :param X_test: dataframe of X features of test set
    :return: tfidf dataframe of train, tfidf dataframe of test and a CountVectorizer fitted object
    """
    vectorizer = TfidfVectorizer()
    tfidf_train = vectorizer.fit_transform(X_train["clean_text"])
    tfidf_test = vectorizer.transform(X_test["clean_text"])

    return tfidf_train, tfidf_test, vectorizer


def run_svc(x_train, x_test, y_train, y_test):
    """
    train and evaluate (with cross validation) svm models with different kernels and different set of features
    :param x_train: the x features in the train set
    :param x_test: the x features in the test set
    :param y_train: the y features in the train set
    :param y_test: the y features in the test set
    :return: the best model and its score
    """
    print(80 * "=")
    print("Running SVC:")
    print(80 * "_")
    param_grid_ = {'C': [1e-5, 1e-3, 1e-1, 1e0, 1e1, 1e2]}

    print("Training SVC classifier with linear kernel and tfidf:")
    train, test, vectorizer_tfidf = get_train_features_tfidf(x_train, x_test)
    clf = SVC(kernel='linear', random_state=42)
    svc_tfidf_linear_model, svc_tfidf_linear_score = run_grid_search(train, y_train, test, y_test,
                                                                     'SVC tfidf features with linear kernel',
                                                                     param_grid_, clf)

    print("Training SVC classifier with rdf kernel and tfidf:")
    train, test, vectorizer_tfidf = get_train_features_tfidf(x_train, x_test)
    clf = SVC(kernel='rbf', random_state=42)
    svc_tfidf_rbf_model, svc_tfidf_rbf_score = run_grid_search(train, y_train, test, y_test,
                                                               'SVC tfidf features with rbf kernel', param_grid_, clf)

    print("Training SVC classifier with linear kernel and 1 grams:")
    train, test, vectorizer_1gram = get_train_features_1gram(x_train, x_test)
    clf = SVC(kernel='linear', random_state=42)
    svc_1gram_linear_model, svc_1gram_linear_score = run_grid_search(train, y_train, test, y_test,
                                                                     'SVC 1gram features with linear kernel',
                                                                     param_grid_, clf)

    print("Training SVC classifier with rbf kernel and 1 grams features:")
    train, test, vectorizer_1gram = get_train_features_1gram(x_train, x_test)
    clf = SVC(kernel='rbf', random_state=42)
    svc_1gram_rbf_model, svc_1gram_rbf_score = run_grid_search(train, y_train, test, y_test,
                                                               'SVC 1gram features with rbf kernel', param_grid_, clf)

    print("Training SVC classifier with linear kernel and 2 grams:")
    train, test, vectorizer_2gram = get_train_features_2gram(x_train, x_test)
    clf = SVC(kernel='linear', random_state=42)
    svc_2gram_linear_model, svc_2gram_linear_score = run_grid_search(train, y_train, test, y_test,
                                                                     'SVC 2gram features with linear kernel',
                                                                     param_grid_, clf)

    print("Training SVC classifier with rbf kernel and 2 grams features:")
    train, test, vectorizer_2gram = get_train_features_2gram(x_train, x_test)
    clf = SVC(kernel='rbf', random_state=42)
    svc_2gram_rbf_model, svc_2gram_rbf_score = run_grid_search(train, y_train, test, y_test,
                                                               'SVC 2gram features with rbf kernel', param_grid_, clf)

    print(80 * "=")

    results = {svc_tfidf_linear_model: svc_tfidf_linear_score,
               svc_tfidf_rbf_model: svc_tfidf_rbf_score,
               svc_1gram_linear_model: svc_1gram_linear_score,
               svc_1gram_rbf_model: svc_1gram_rbf_score,
               svc_2gram_linear_model: svc_2gram_linear_score,
               svc_2gram_rbf_model: svc_2gram_rbf_score}

    results_sorted = sorted(results, key=lambda x: results[x], reverse=True)
    return results_sorted[0], results[results_sorted[0]]
