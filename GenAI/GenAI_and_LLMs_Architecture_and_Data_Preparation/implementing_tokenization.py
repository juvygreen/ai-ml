# Installing required libraries
# The following required libraries are not pre-installed in the Skills Network Labs environment. You will need to run the following cell to install them:
# !pip install nltk
# !pip install transformers==4.42.1
# !pip install sentencepiece
# !pip install spacy
# !python -m spacy download en_core_web_sm
# !python -m spacy download de_core_news_sm
# !pip install scikit-learn
# !pip install torch==2.2.2
# !pip install torchtext==0.17.2
# !pip install numpy==1.26.0

import nltk
nltk.download("punkt")
nltk.download('punkt_tab')
import spacy
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.util import ngrams
from transformers import BertTokenizer
from transformers import XLNetTokenizer

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')


# Word-based tokenizer
# nltk
# As the name suggests, this is the splitting of text based on words.
# There are different rules for word-based tokenizers, such as splitting on spaces or splitting on punctuation.
# Each option assigns a specific ID to the split word. Here you use nltk's word_tokenize
text = "This is a sample sentence for word tokenization."
tokens = word_tokenize(text)
print(tokens)

# This showcases word_tokenize from nltk library
text = "I couldn't help the dog. Can't you do it? Don't be afraid if you are."
tokens = word_tokenize(text)
print(tokens)


# This showcases the use of the 'spaCy' tokenizer with torchtext's get_tokenizer function
text = "I couldn't help the dog. Can't you do it? Don't be afraid if you are."
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

# Making a list of the tokens and priting the list
token_list = [token.text for token in doc]
print("Tokens:", token_list)

# Showing token details
for token in doc:
    print(token.text, token.pos_, token.dep_)


#Each word is split into a token, leading to a significant increase in the model's overall vocabulary.
# Each token is mapped to a large vector containing the word's meanings, resulting in large model parameters.

text = "Unicorns are real. I saw a unicorn yesterday."
token = word_tokenize(text)
print(token)


#Subword-based tokenizer
#The subword-based tokenizer allows frequently used words to remain unsplit while breaking down
# infrequent words into meaningful subwords. Techniques such as SentencePiece,
# or WordPiece are commonly used for subword tokenization.

#WordPiece
#Initially, WordPiece initializes its vocabulary to include every character present
# in the training data and progressively learns a specified number of merge rules.
# WordPiece doesn't select the most frequent symbol pair but rather the one that
# maximizes the likelihood of the training data when added to the vocabulary.
# In essence, WordPiece evaluates what it sacrifices by merging two symbols to ensure it's a worthwhile endeavor.
#Now, the WordPiece tokenizer is implemented in BertTokenizer.
# Note that BertTokenizer treats composite words as separate tokens.
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.tokenize("IBM taught me tokenization.")


#Unigram and SentencePiece
#Unigram is a method for breaking words or text into smaller pieces. It accomplishes this by starting with a large list
# of possibilities and gradually narrowing it down based on how frequently those pieces appear in the text.
# This approach aids in efficient text tokenization.
#SentencePiece is a tool that takes text, divides it into smaller, more manageable parts, assigns IDs to these segments,
# and ensures that it does so consistently. Consequently, if you use SentencePiece on the same text repeatedly,
# you will consistently obtain the same subwords and IDs.
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
tokenizer.tokenize("IBM taught me tokenization.")


# Tokenization with PyTorch
# In PyTorch, especially with the torchtext library, the tokenizer breaks down text from a data set into individual
# words or subwords, facilitating their conversion into numerical format
dataset = [
    (1,"Introduction to NLP"),
    (2,"Basics of PyTorch"),
    (1,"NLP Techniques for Text Classification"),
    (3,"Named Entity Recognition with PyTorch"),
    (3,"Sentiment Analysis using PyTorch"),
    (3,"Machine Translation with PyTorch"),
    (1," NLP Named Entity,Sentiment Analysis,Machine Translation "),
    (1," Machine Translation with NLP "),
    (1," Named Entity vs Sentiment Analysis  NLP ")]

#This next line imports the get_tokenizer function from the torchtext.data.utils module.
# In the torchtext library, the get_tokenizer function is utilized to fetch a tokenizer by name. It provides support
# for a range of tokenization methods, including basic string splitting, and returns various tokenizers based on the
# argument passed to it.
from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer("basic_english")
#You apply the tokenizer to the dataset. Note: If basic_english is selected, it returns the _basic_english_normalize()
# function, which normalizes the string first and then splits it by space.
tokenizer(dataset[0][1])


#Token indices
#You would represent words as numbers as NLP algorithms can process and manipulate numbers more efficiently and quickly
# than raw text. You use the function build_vocab_from_iterator, the output is typically referred to as 'token indices'
# or simply 'indices.' These indices represent the numeric representations of the tokens in the vocabulary.
def yield_tokens(data_iter):
    for  _,text in data_iter:
        yield tokenizer(text)
my_iterator = yield_tokens(dataset)
#This creates an iterator called my_iterator using the generator. To begin the evaluation of the generator and retrieve
# the values, you can iterate over my_iterator using a for loop or retrieve values from it using the next() function.
next(my_iterator)


#Out-of-vocabulary (OOV)
#When text data is tokenized, there may be words that are not present in the vocabulary because they are rare or unseen
# during the vocabulary building process. When encountering such OOV words during actual language processing tasks like
# text generation or language modeling, the model can use the <unk> token to represent them.
vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

#This code demonstrates how to fetch a tokenized sentence from an iterator, convert its tokens into indices using a
# provided vocabulary, and then print both the original sentence and its corresponding indices.
def get_tokenized_sentence_and_indices(iterator):
    tokenized_sentence = next(iterator)  # Get the next tokenized sentence
    token_indices = [vocab[token] for token in tokenized_sentence]  # Get token indices
    return tokenized_sentence, token_indices

tokenized_sentence, token_indices = get_tokenized_sentence_and_indices(my_iterator)
next(my_iterator)

print("Tokenized Sentence:", tokenized_sentence)
print("Token Indices:", token_indices)


#Using the lines of code provided above in a simple example, demonstrate tokenization and the building of
# vocabulary in PyTorch.
lines = ["IBM taught me tokenization",
         "Special tokenizers are ready and they will blow your mind",
         "just saying hi!"]

special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')

tokens = []
max_length = 0

for line in lines:
    tokenized_line = tokenizer_en(line)
    tokenized_line = ['<bos>'] + tokenized_line + ['<eos>']
    tokens.append(tokenized_line)
    max_length = max(max_length, len(tokenized_line))

for i in range(len(tokens)):
    tokens[i] = tokens[i] + ['<pad>'] * (max_length - len(tokens[i]))

print("Lines after adding special tokens:\n", tokens)

# Build vocabulary without unk_init
vocab = build_vocab_from_iterator(tokens, specials=['<unk>'])
vocab.set_default_index(vocab["<unk>"])

# Vocabulary and Token Ids
print("Vocabulary:", vocab.get_itos())
print("Token IDs for 'tokenization':", vocab.get_stoi())


new_line = "I learned about embeddings and attention mechanisms."

# Tokenize the new line
tokenized_new_line = tokenizer_en(new_line)
tokenized_new_line = ['<bos>'] + tokenized_new_line + ['<eos>']

# Pad the new line to match the maximum length of previous lines
new_line_padded = tokenized_new_line + ['<pad>'] * (max_length - len(tokenized_new_line))

# Convert tokens to IDs and handle unknown words
new_line_ids = [vocab[token] if token in vocab else vocab['<unk>'] for token in new_line_padded]

# Example usage
print("Token IDs for new line:", new_line_ids)


# Comparative text tokenization and performance analysis
text = """
Going through the world of tokenization has been like walking through a huge maze made of words, symbols, and meanings. 
Each turn shows a bit more about the cool ways computers learn to understand our language. 
And while I'm still finding my way through it, the journey’s been enlightening and, honestly, a bunch of fun.
Eager to see where this learning path takes me next!"
"""

# Counting and displaying tokens and their frequency
from collections import Counter


def show_frequencies(tokens, method_name):
    print(f"{method_name} Token Frequencies: {dict(Counter(tokens))}\n")


import nltk
import spacy
from transformers import BertTokenizer, XLNetTokenizer
from datetime import datetime

# NLTK Tokenization
start_time = datetime.now()
nltk_tokens = nltk.word_tokenize(text)
nltk_time = datetime.now() - start_time

# SpaCy Tokenization
nlp = spacy.load("en_core_web_sm")
start_time = datetime.now()
spacy_tokens = [token.text for token in nlp(text)]
spacy_time = datetime.now() - start_time

# BertTokenizer Tokenization
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
start_time = datetime.now()
bert_tokens = bert_tokenizer.tokenize(text)
bert_time = datetime.now() - start_time

# XLNetTokenizer Tokenization
xlnet_tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
start_time = datetime.now()
xlnet_tokens = xlnet_tokenizer.tokenize(text)
xlnet_time = datetime.now() - start_time

# Display tokens, time taken for each tokenizer, and token frequencies
print(f"NLTK Tokens: {nltk_tokens}\nTime Taken: {nltk_time} seconds\n")
show_frequencies(nltk_tokens, "NLTK")

print(f"SpaCy Tokens: {spacy_tokens}\nTime Taken: {spacy_time} seconds\n")
show_frequencies(spacy_tokens, "SpaCy")

print(f"Bert Tokens: {bert_tokens}\nTime Taken: {bert_time} seconds\n")
show_frequencies(bert_tokens, "Bert")

print(f"XLNet Tokens: {xlnet_tokens}\nTime Taken: {xlnet_time} seconds\n")
show_frequencies(xlnet_tokens, "XLNet")
