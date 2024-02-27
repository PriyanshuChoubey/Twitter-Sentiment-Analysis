import nltk
from nltk.corpus import twitter_samples
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier

#nltk.download('twitter_samples')
#nltk.download('stopwords')
#nltk.download('punkt')

# Load positive and negative tweets from NLTK's twitter_samples corpus
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# Preprocess the tweets``  
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_tweet(tweet):
    tweet = tweet.lower()
    tweet = word_tokenize(tweet)
    tweet = [word for word in tweet if word.isalpha()]  # Remove non-alphabetic characters
    tweet = [word for word in tweet if word not in stop_words]
    tweet = [stemmer.stem(word) for word in tweet]
    return tweet

# Build a list of tuples where each tuple contains the preprocessed tweet and its sentiment label
positive_tweets_processed = [(preprocess_tweet(tweet), 'positive') for tweet in positive_tweets]
negative_tweets_processed = [(preprocess_tweet(tweet), 'negative') for tweet in negative_tweets]

# Combine positive and negative tweets, and split into training and testing sets
all_tweets = positive_tweets_processed + negative_tweets_processed
split_ratio = 0.8
split_idx = int(len(all_tweets) * split_ratio)

train_set = all_tweets[:split_idx]
test_set = all_tweets[split_idx:]

# Build a feature set using the most common words as features
all_words = [word for (tweet, sentiment) in all_tweets for word in tweet]
fdist = FreqDist(all_words)
common_words = fdist.most_common(3000)
features = [word for (word, freq) in common_words]

def extract_features(tweet):
    tweet_words = set(tweet)
    features_dict = {}
    for word in features:
        features_dict[f'contains({word})'] = (word in tweet_words)
    return features_dict

train_set_features = [(extract_features(tweet), sentiment) for (tweet, sentiment) in train_set]
test_set_features = [(extract_features(tweet), sentiment) for (tweet, sentiment) in test_set]

# Train a Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_set_features)

# Evaluate the classifier
accuracy = classify.accuracy(classifier, test_set_features)
print(f'Accuracy: {accuracy}')

# Example predictions
example_tweets = ["I am mad at you","I will happily sad forever","I reached late"]
for tweet in example_tweets:
    processed_tweet = preprocess_tweet(tweet)
    feature_set = extract_features(processed_tweet)
    sentiment = classifier.classify(feature_set)
    print(f'Tweet: {tweet}, Sentiment: {sentiment}')
    
