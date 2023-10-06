# word2vec and bert example

# word2vec

# importing libraries

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize

# Sample text data

text = "Word2Vec is NLP technique. Its very effective"

# Tokenize the text into sentences and words

sentences = [word_tokenize(sentence) for sentence in sent_tokenize(text)]

# model word2vec

model = Word2Vec(sentences, vector_size = 100, window = 5, min_count = 1, sg = 0)

# find word embeddings
word_embeddings = model.wv

# get vector representation of a word
vector = word_embeddings['Word2Vec']
print("Vector for 'Word2Vec",vector)

# BERT model

# importing libraries

from transformers import BertTokenizer, BertModel
import torch

# Sample text

text = "Hellow this is for BERT model"

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize and encode the text

input_ids = tokenizer(text,return_tensors = "pt")['input_ids']

# Output from BERT

outputs = model(input_ids)

# Contextual embeddings
contextual_embeddings = outputs.last_hidden_state

print(contextual_embeddings)