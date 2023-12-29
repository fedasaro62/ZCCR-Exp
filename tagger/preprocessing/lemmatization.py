import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
import spacy

nlp                 = spacy.load("en_core_web_sm")
stemmer             = SnowballStemmer(language='english')
wordnet_lemmatizer  = WordNetLemmatizer()

text = "studies studying cries cry driving driver asleep sleepy"
tokenization = nltk.word_tokenize(text)
print(text)

for token in nlp(text):
    print("Spacy Lemma {}".format(token.lemma_))

for w in tokenization:
    print("NLTK Lemma {}".format(wordnet_lemmatizer.lemmatize(w)))  

for w in tokenization: #" ".split(text):
    print("NLTK Stemmer ", stemmer.stem(w))








