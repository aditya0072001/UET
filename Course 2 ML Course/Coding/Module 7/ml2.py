# Text Representation using Bag of Words and TF IDF

# Importing necessary libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Sample text data
text = ["I love NLP and I will learn NLP in 2month ",
        "I love Deep Learning and I will learn Deep Learning in 2month"]

# COnverting text to lower case

text = [txt.lower() for txt in text]

# Removing stopwords
stop_words = set(stopwords.words('english'))
text = [word_tokenize(txt) for txt in text]
text = [[word for word in txt if word not in stop_words] for txt in text]

# Tokenize words

text = [" ".join(txt) for txt in text]

# Bag of Words

cv = CountVectorizer()
text_bow = cv.fit_transform(text).toarray()
print(text_bow)

# TF IDF

tfidf = TfidfVectorizer()
text_tfidf = tfidf.fit_transform(text).toarray()

print(text_tfidf)