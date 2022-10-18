import pandas as pd

def data_exploration(train_df):
    """
    Exploring the given train dataframe to get insights about the data and how to work with it
    :param train_df: dataframe of Trump's and non-Trump's tweets
    """
    print(80 * "=")
    print("Data Exploration")
    print(80 * "-")

    print("Data first rows:")
    print(train_df.head())
    print()

    print(f"Data shape: {train_df.shape}")
    print()

    print("Possible device field values:")
    print(train_df["device"].unique())
    print()

    print("Data time:")
    print(f"Starting: {train_df['time_stamp'].min()}")
    print(f"Ending: {train_df['time_stamp'].max()}")
    print()

    print("Device used per user_handle")
    print(train_df.groupby(["user_handle", "device"])["device"].count())
    # We can see that POTUS user handle has 1 tweet from iphone - then we will treat it as not trump's tweet
    # We can see that PressSec has tweets from iphone and web clients - we would assume it's because the
    # staff that work from the office posted those tweets, thus they are not Trump's tweets.
    # For the account realDonaldTrump, we can see that the majority 92005) of the tweets were from android.
    # we would assume that all of those tweets are Trump's. All the rest are from an iPhone, Blackberry, iPad or
    # different web clients. We would then assume that all of these were tweeted by the staff.
    # Then, all tweets on all accounts that arefrom android are Trump's and all the rest are not Trump's
    print()

    print("Label count")
    print(train_df.groupby("label")["tweet_id"].count())
    print()

    longest_tweet_chars = max([len(list(tweet)) for tweet in train_df['tweet_text']])
    print("The length of the chars' longest tweet is: ", longest_tweet_chars)

    longest_tweet_words = max([len(tweet.split(' ')) for tweet in train_df['tweet_text']])
    print("The length of the words' longest tweet is: ", longest_tweet_words)

    unique_chars = set([tweet.lower()[i] for tweet in train_df['tweet_text'] for i in range(len(tweet))])
    print("Unique chars: ", unique_chars)

    alpha_chars = list('abcdefghijklmnopqrstuvwxyz')
    non_alpha_chars = [char for char in unique_chars if char not in alpha_chars]
    print("Non alpha chars: ", non_alpha_chars)

    print(80 * "=")
