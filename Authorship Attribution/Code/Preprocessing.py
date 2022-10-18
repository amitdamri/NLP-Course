import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

nltk.download('stopwords')

stemmer = PorterStemmer()
stopwords_english = stopwords.words('english')


def drop_na(df):
    """
    Remove NA records.
    :return: df without na's
    """
    df_copy = df.copy()
    df_copy.dropna(inplace=True)
    return df_copy


def add_label(df):
    """
    Add label according to the device - 'android' = 0 (Trump), 'iphone' = 1 (Not trump)
    :return: df with labels
    """
    df_copy = df.copy()
    df_copy['label'] = np.where(df_copy['device'] == 'android', 1, 0)
    return df_copy


def timestamp_to_datetime(df):
    """
    Convert time_stamp column to datetime.
    :return: dataframe
    """
    df['time_stamp'] = pd.to_datetime(df['time_stamp'], format="%Y-%m-%d %H:%M:%S")
    return df


def count_capital(tweet):
    """
    Count number of capital letters on the given tweet.
    :return: int - number of capital letters
    """
    capital_letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    counter = 0
    for char in tweet:
        if char in capital_letters:
            counter += 1
    return counter


def count_hashtags(tweet):
    """
    Count number of hashtags.
    :param tweet:
    :return: int - number of hashtags
    """
    return tweet.count('#')


def count_retweet(tweet):
    """
    Count number of retweets.
    :param tweet:
    :return: int - number of retweets.
    """
    return len(re.findall(r'^RT[\s]+', tweet))


def count_mentions(tweet):
    """
    Count number of mentions.
    :param tweet:
    :return: int - number of mentions.
    """
    return len(re.findall(r'^@\w+', tweet))


def count_stop_words(tweet):
    """
    Count number of stopwords.
    :param tweet:
    :return: int - number of stopwords.
    """
    tweet = tweet.split(' ')
    counter = 0
    for word in tweet:
        if word in stopwords_english:
            counter += 1
    return counter


def re_process(tweet):
    """
    use regular expression to remove retweets, hashtags, mentions, urls, numbers, and double spaces.
    we remove only the sign not the text.
    :param tweet
    :return: clean tween
    """
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'\d+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r' +', r' ', tweet)
    return tweet


def tokenize_tweet(tweet):
    """
    Transform tweet into tokens using TweetTokenizer of nltk library. Remove stop wrods and use stemming.
    :param tweet:
    :return: full clean_tweet
    """
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    # if the word is a stop word remove it, otherwise stem the word and
    # if its length less than or equal to 1 remove it.
    tweets_clean = []
    for word in tweet_tokens:
        if word not in stopwords_english:
            stem_word = stemmer.stem(word)  # stemming word
            if len(word) > 1:
                tweets_clean.append(stem_word)

    full_clean_tweet = " ".join(tweets_clean)
    return full_clean_tweet


def pre_process(data, data_type):
    """
    Run the full process of pre processing on the data.
    :param data: data to preprocess
    :param data_type: train or test
    :return: pre processed data
    """
    # drop na values
    preprocess_data = drop_na(data)

    # add label only for train set
    if data_type == 'train':
        preprocess_data = add_label(preprocess_data)

    # get year, month, day and hour from timestamp
    preprocess_data = timestamp_to_datetime(preprocess_data)
    preprocess_data['year'] = preprocess_data['time_stamp'].apply(lambda time: time.year)
    preprocess_data['month'] = preprocess_data['time_stamp'].apply(lambda time: time.month)
    preprocess_data['day'] = preprocess_data['time_stamp'].apply(lambda time: time.day)
    preprocess_data['hour'] = preprocess_data['time_stamp'].apply(lambda time: time.hour)

    # add meta features
    preprocess_data['#capital_letters'] = preprocess_data['tweet_text'].apply(
        lambda tweet: count_capital(tweet))
    preprocess_data['#num_hashtags'] = preprocess_data['tweet_text'].apply(lambda tweet: count_hashtags(tweet))
    preprocess_data['#num_retweet'] = preprocess_data['tweet_text'].apply(lambda tweet: count_retweet(tweet))
    preprocess_data['#num_mentions'] = preprocess_data['tweet_text'].apply(lambda tweet: count_mentions(tweet))
    preprocess_data['#num_stopwords'] = preprocess_data['tweet_text'].apply(
        lambda tweet: count_stop_words(tweet))

    # remove unnecessary chars and lower the words
    preprocess_data['tweet_text'] = preprocess_data['tweet_text'].apply(lambda tweet: re_process(tweet))
    preprocess_data['tweet_text'] = preprocess_data['tweet_text'].apply(lambda tweet: tweet.lower())

    # Stemming and stopwords
    preprocess_data['clean_text'] = preprocess_data['tweet_text'].apply(
        lambda tweet: tokenize_tweet(tweet))

    # count number of words and chars
    preprocess_data["num_words"] = preprocess_data['clean_text'].apply(lambda tweet: len(tweet.split()))
    preprocess_data["num_chars"] = preprocess_data['clean_text'].apply(lambda tweet: len(list(tweet)))

    # if train set remove empty tweets
    if data_type != 'test':
        preprocess_data = preprocess_data.replace(r'^\s*$', np.nan, regex=True).dropna()

    return preprocess_data
