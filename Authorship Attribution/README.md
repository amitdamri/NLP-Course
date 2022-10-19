# Authorship Attribution

<p align="center">
	<img height=300 width=500 src="https://user-images.githubusercontent.com/49988048/196622766-12611d7c-21d2-4d0f-a6c5-bc8be0eba5ee.png">
</p>

In this project I used a number of supervised machine learning classifier, in order to perform an authorship attribution task on Donald Trump’s tweets.
I used Python’s nltk, sklearn and Pytorch packages/libraries for preprocessing, training and testing the classifiers.

### Classifiers
1. Logistic Regression
2. SVC (linear and nonlinear kernels)
3. Neural Network (Pytorch)
4. Recurrent Neural Network (LSTM - Pytorch)
5. Random Forest

### Classification: Who Controls this Account
Politicians, as well as other public figures, usually have assistants and staffers that manage most of their social media presence.
However, like many other norm defying actions, Donald Trump, the 45th President of the United States is taking pride in his untamed use of Twitter.
At times, during the presidential campaign, it was <a href="https://www.theatlantic.com/politics/archive/2016/08/donald-trump-twitter-iphone-android/495239/">hypothesized</a>
that Donald Trump is being kept away from his Twitter account in order to avoid unnecessary PR calamities. 
Trump’s tweets are not explicitly labeled (Hillary Clinton, for example, used to sign tweets composed by her by an addition of ‘-H’ at the end of the tweet 
while unsigned tweets were posted by her staffers). It is known, however, that Trump was using an android phone  while the staffers were most likely to use an iPhone.
Luckily, the device information is part of the data available via the Twitter API, hence the device used can be used as an authorship label. 
</br><strong>Note</strong>: Trump switched to a secured iPhone in April 2017, hence, building an accurate authorship model on older data can be used for authorship attribution of newer tweets.

### Data 
- trump_train.tsv - A small dataset of a couple of thousands tweets from Trump’s account posted between early 2015 and mid 2017.
The file is in a tab separated format, each tweet in a new line. The fields in the file correspond to:
<tweet id> <user handle> <tweet text> <time stamp> <device> 
1.	The handle field: the handle field can take one of the following three user names: realDonaldTrump (this is Trump’s account), POTUS (stands for President of the United States, this is the official presidential account, thus not Trump before the election) and PressSec - the official twitter account of the president’s Press Secretary. 
2.	The device field: the device field can take various values ranging from ‘android’, iphone’,  instagram, a web client among other possibilities.
3.	The format of the timestamp field is '%Y-%m-%d %H:%M:%S'.

- trump_test.tsv - An unlabeled test set with 390 tweets. This file lacks the <tweet id> and <device> fields.  
  
