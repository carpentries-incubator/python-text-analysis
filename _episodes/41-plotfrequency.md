---
title: "Plotting word frequency"
teaching: 0
exercises: 0
questions:
- ""
objectives:
- "Learn how to import corpora into NLTK."
- "Learn how to use stopwords in NLTK."
- "Learn how to count words in NLTK and graph them."
keypoints:
- "FIXME"
---

#Importing libraries
Python has a variety of libraries which have prewritten functions useful for text analysis. The first library we will use will be the Natural Language Toolkit, or NLTK.

~~~
import nltk
~~~
{: .python}

#Importing Documents and Corpora
Text analysis is often done over a collection of articles, books or texts. In text analysis, these are all called documents. A collection of documents is called a corpus.
Suppose we have a selection of text files that we would like to treat as a corpus, stored within a directory on our computer. To get this corpus into Python, we can use a function in NLTK. 
First, we will call the corpus module, and then PlaintextCorpusReader function from within that module. 

~~~
k = nltk.corpus.PlaintextCorpusReader('testcorpus', '.*')
~~~
{: .python}


Now the corpus of files gets read into python and stored as a NLTK corpus object.
NLTK also has a number of built in corpora which we can use for the purposes of training models and doing demonstrations. These need to be installed in python, and a full list is available at: http://www.nltk.org/nltk_data/
We will download a set of articles from Reuters news service.

~~~
nltk.download('reuters')
~~~
{: .python}

Once downloaded, the corpus will remain on your computer for future use. These example corpora can be called as part of the corpus module. Let's load the reuters corpus into our program and save it to a variable.

~~~
reuters = nltk.corpus.reuters
~~~
{: .python}

Note that both the reuters corpus, and our custom corpus are of the same datatype, so methods that work on one will work on the other.

#Creating a list of words
Many Natural Language Processing techniques treat corpora as a so-called "bag of words." 
NLTK has a function that allows us to put all of the words in a corpus object into a list. To do this, we simply call the words function on the corpus object.
We can treat these as regular lists in python and can do things like iterate over them. Let's print the first 10 words in the reuters corpus.

~~~
r_words = reuters.words()
k_words = kaledoscope.words()
for i in range(1, 10):
	print rwords[i]
~~~
{: .python}

#Removing stopwords, special characters, and capital letters
Stopwords are words that are very common, such as "the", "and", "a" and so on.
Frequently, the most common words are words such as these, which are not informative about the contents of the corpus.
To get a better visual of 'interesting words' let's remove these words from our lists. NLTK has a list of common stopwords we will use.
We also want to remove special characters like dollar signs, periods, and so on. The isalpha function is part of built in Python string functionality that tells us if a string is alphanumeric.
Finally, we will make all words lower case using the python built in string function lower.
Using a for loop, we remove all special characters and all stopwords.

~~~
stopwords = nltk.corpus.stopwords.words('english')
r_filtered = [word.lower() for word in r_words if ((word not in stopwords) and (word.isalpha()))]
k_filtered = [word.lower() for word in k_words if ((word not in stopwords) and (word.isalpha()))]
~~~
{: .python}

#Frequency Distributions
NLTK has built in functionality to count words and store how many times they occur in a dictionary. Let's use this on our collection of words.

~~~
r_freq = nltk.FreqDist(r_filtered)
k_freq = nltk.FreqDist(k_filtered)
~~~
{: .python}

This creates a dictionary in python. To look up how many times a word occurs, we simply look up the word in the dictionary. 

~~~
print(r_freq['oil'])
print(k_freq['student'])
~~~
{: .python}

#NLTK Plotting Most Common Words
Finally, NLTK has a function that allows us to easily plot the top occuring words. 
Let's print most 50 frequent words for corpus.

~~~
reuters_freq.plot(50)
k_freq.plot(50)
~~~
{: .python}
