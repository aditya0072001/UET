# Sentimental Analysis

# Importing the libraries

import nltk

nltk.download('movie_reviews')


import random
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from statistics import mode

# Importing the dataset

documents = [(list(movie_reviews.words(fileid)), category)
                for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)]

# Shuffling the dataset

random.shuffle(documents)

# Creating a list of all words

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

# Creating a frequency distribution of all words

all_words = nltk.FreqDist(all_words)

# Creating a list of the top 3000 most common words

word_features = list(all_words.keys())[:3000]

# Function to find the features in a document

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words) # Returns true if the word is present in the document
    return features


# Creating a feature set for each document

featuresets = [(find_features(rev), category) for (rev, category) in documents]

# Splitting the dataset into the Training set and Test set

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# Creating a class to find the most voted classification

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        
        return conf

# Training the classifier

classifier = nltk.NaiveBayesClassifier.train(training_set)

# Testing the classifier

print("Naive Bayes Accuracy: ", (nltk.classify.accuracy(classifier, testing_set)) * 100)

classifier.show_most_informative_features(15)

# Training the classifier using the SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)


print("MNB Accuracy: ", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)

# Sentimental Analysis

print("Classification: ", MNB_classifier.classify(testing_set[0][0]))

# Sentimental analysis on sample sentences

sample_text = "This movie is awesome! The acting was great, plot was wonderful and there were pythons...so yea!"

print("Classification: ", MNB_classifier.classify(find_features(sample_text.split())))

sample_text = "This movie is utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, 0/10"

print("Classification: ", MNB_classifier.classify(find_features(sample_text.split())))







