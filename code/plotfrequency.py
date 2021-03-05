import nltk
from nltk.corpus import PlaintextCorpusReader


#load in corpus of reuters newspaper texts. This is built into NLTK.
reuters = nltk.corpus.reuters.words()
kaledoscope = PlaintextCorpusReader('C:\\Users\\Karl\\Desktop\\nltk demo\\kaleidoscope.txt', '.*')
kwords = kaledoscope.words()


#NLTK has a list of common words we will remove from our graph
stopwords = nltk.corpus.stopwords.words('english')

#Using a for loop, we will remove all special characters, and all stopwords.
reuters_filtered = [word for word in reuters if ((word not in stopwords) and (word.isalpha()))]
k_filtered = [word for word in kwords if ((word not in stopwords) and (word.isalpha()))]

#We use built in plotting functionality to get a list of most common words
reuters_freq = nltk.FreqDist(reuters_filtered)
k_freq = nltk.FreqDist(k_filtered)

#Number of times the word 'oil' is in 
print(reuters_freq['oil'])
print(k_freq['student'])

#print most frequent words for corpus
reuters_freq.plot(50)
k_freq.plot(50)

