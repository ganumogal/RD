
import nltk
import matplotlib.pyplot as plt

nltk.download("punkt")
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import sent_tokenize

text="Hello Mr. Smith, how are you doing today  The weather is great, and city is awesome. The sky is pinkish - blue. You should not eat cardboard"

tokenized_text=sent_tokenize(text)
print(tokenized_text)
print("\n")

from nltk.tokenize import word_tokenize

tokenized_word=word_tokenize(text)
print(tokenized_word)
print("\n")

from nltk.probability import FreqDist

fdist = FreqDist(tokenized_word)
print(fdist)
print("\n")

fdist.plot(30,cumulative=False)
plt.show()

sent = "Albert Einstein was born in Ulm, Germany in 1879."

tokens=nltk.word_tokenize(sent)
print(tokens)
print("\n")

nltk.download('averaged_perceptron_tagger') 
nltk.pos_tag(tokens)

from nltk.corpus import stopwords
stop_words=set(stopwords.words("english")) 
print(stop_words)
print("\n")

filtered_sent=[]
for w in tokenized_text:
 if w not in stop_words:
  filtered_sent.append(w)
  print("Tokenized Sentence:",tokenized_text)
  print("\n")
  print("Filtered Sentence:",filtered_sent)
  print("\n")

print("**********")
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()
stemmed_words=[]
for w in filtered_sent:
 stemmed_words.append(ps.stem(w))
 print("Filtered Sentence:",filtered_sent)
 print("\n")
 print("Stemmed Sentence:",stemmed_words)
 print("\n")


print("Lexicon Normalization")

#performing stemming and Lemmatization
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()
word = "flying"
print("Lemmatized Word:",lem.lemmatize(word,"v"))
print("\n")
print("Stemmed Word:",stem.stem(word))
print("\n")
