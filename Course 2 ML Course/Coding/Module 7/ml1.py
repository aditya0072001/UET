# Text Preprocessing techniques

# Importing libraries

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Sample text for performing preprocessing

text = """Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.
The sky is pinkish-blue. You shouldn't eat cardboard"""

# Converting text to lowercase

lower_case = text.lower()

# Printing text after converting to lowercase

print(lower_case)

# Removing punctuations

cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

# Printing text after removing punctuations

print(cleaned_text)

# Tokenizing words

tokenized_words = word_tokenize(cleaned_text, "english")

# Printing tokenized words

print(tokenized_words)

# Removing stop words

final_words = []

for word in tokenized_words:
    if word not in stopwords.words('english'):
        final_words.append(word)

# Printing final words

print(final_words)

# Lemmatization - From plural to single + Base form of a word (example better-> good)

lemma_words = []

for word in final_words:
    word = WordNetLemmatizer().lemmatize(word)
    lemma_words.append(word)

# Printing lemma words

print(lemma_words)

# print the preprocessing text

print(' '.join(lemma_words))

