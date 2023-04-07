---
title: "Document Embeddings and TF-IDF"
teaching: 20
exercises: 20
questions:
- "todo"
objectives:
- "todo"
keypoints:
- "todo"
---
## Document embeddings 
The method of using word counts is just one way we might embed a document in vector space.  
Let’s talk about more complex and representational ways of constructing document embeddings.  
To start, imagine we want to represent each word in our model individually, instead of considering an entire document. 
How individual words are represented in vector space is something called "word embeddings" and they are an important concept in NLP.

##One hot encoding 
How would we make word embeddings for a simple document such as “Feed the duck”? 
Let’s imagine we have a vector space with a million different words in our corpus, and we are just looking at part of the vector space below.

|      | dodge | duck | ... | farm | feather | feed | ... | tan | the |
|------|-------|------|-----|------|---------|------|-----|-----|-----|
| feed | 0     | 0    |     | 0    | 0       | 1    |     | 0   | 0   |
| the  | 0     | 0    |     | 0    | 0       | 0    |     | 0   | 1   |
| duck | 0     | 1    |     | 0    | 0       | 0    |     | 0   | 0   |
|------|-------|------|-----|------|---------|------|-----|-----|-----|
| Document | 0     | 1    |     | 0    | 0       | 1    |     | 0   | 1   |

We can see that each word embedding gives a 1 for a dimension corresponding to the word, and a zero for every other dimension. 
This kind of encoding is known as “one hot” encoding, where a single value is 1 and all others are 0.  

Once we have all the word embeddings for each word in the document, we sum them all up to get the document embedding. 
This is the simplest and most intuitive way to construct a document embedding from a set of word embeddings. 
But does it accurately represent the importance of each word? 
Our next model, TF-IDF, will embed words with different values rather than just 0 or 1.

## TF-IDF 

Currently our model assumes all words are created equal and are all equally important. However, in the real world we know that certain words are more important than others. 
For example, in a set of novels, knowing one novel contains the word “the” 100 times does not tell us much about it. 
However, if the novel contains a rarer word such as “whale” 100 times, that may tell us quite a bit about its content. 
A more accurate model would weigh these rarer words more heavily, and more common words less heavily, so that their relative importance is part of our model.  

However, rare is a relative term. In a corpus of documents about blue whales, the term ‘whale’ may be present in nearly every document. 
In that case, other words may be rarer and more informative. How do we determine how these weights are done? One method for constructing more advanced word embeddings is a model called TF-IDF. 

Tf-idf stands for term frequency-inverse document frequency. The model consists of two parts- term frequency and inverse document frequency. We multiply the two terms to get the TF-IDF value. 

TF stands for term frequency, and measures how frequently a term occurs in a document. 
The simplest way to calculate term frequency is by simply adding up the number of times a term t occurs in a document d, and dividing by the total word count in the corpus. 

IDF, or inverse document frequency, measures a term’s importance. 
Document frequency is the number of documents a term occurs in, so inverse document frequency gives higher scores to words that occur in fewer documents. 
This is represented by the equation: 

idf_i = ln[(N +1) / df_i] + 1 

N represents the total number of documents in the corpus, and df_i represents document frequency for a particular word i.
 If the equation seems too complex, the key thing to understand is that words that occur in more documents get weighted less heavily. 

Because tf-idf scores words by their frequency, it’s a useful tool for extracting terms from text. 
We can also embed documents in vector space using TF-IDF scores rather than simple word counts. 
Tf-idf can be used to weaken the impact of stop-words, since due to their common nature, they have very low scores. 

Now that we’ve seen how TF-IDF works, let’s put it into practice. 

We will be downloading some documents from project Gutenberg to use as a test corpus. This data is available from the link below:
https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip

We will also reuse the methods we created in the preprocessing lesson, which are reproduced below:

~~~
import glob
import os
from pathlib import Path
def create_file_list(directory, filter_str='*'):
    #find files matching certain pattern. if no pattern, use star.
    files = Path(directory).glob(filter_str)
	#create an array of file paths as strings.
    files_to_analyze = []
    for f in files:
        files_to_analyze.append(str(f))
    return files_to_analyze
	
class Our_Tokenizer:
    def __init__(self):
        #import spacy tokenizer/language model
        self.nlp = en_core_web_sm.load()
        self.nlp.max_length = 4500000 # increase max number of characters that spacy can process (default = 1,000,000)

	def tokenize(self, document):
		tokens = self.nlp(document)
		tokens = str.lower(token.lemma_) for token in tokens if (
			token.is_stop == False and
			token.is_punct == False)
		return tokens
	def tokenize(self, document):
		tokens = self.nlp(document)
        tokens = [token.lemma_ for token in tokens if (
            token.is_stop == False and # filter out stop words
            token.is_punct == False and # filter out punct
            token.is_space == False and #filter newlines
            token.pos_ != 'PROPN')] #remove all proper nouns such as names
		return tokens

corpus_dir = 'C:\\Users\\Desktop\\documents\\'
corpus_file_list = create_file_list(corpus_dir)
~~~
{: .language-python}

We used basic, one-hot encoding type of vectorizer in the previous lesson. But now we will use an alternative vectorizer, one that uses TF-IDF to calculate the value of different cells.

~~~
#create instance of our tokenizer
our_tok = Our_Tokenizer()
# import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#create vectorizer, using our special tokenizer
tfidf = TfidfVectorizer(input = 'filename', tokenizer=our_tok.tokenize )
#calculate tf-idf for entire corpus in one line of code :)
result = tfidf.fit_transform(corpus_file_list)
~~~
{: .language-python}

Let's run a piece of code that will tell us the dimensions of the matrix this just created.

~~~
result.shape
~~~
{: .language-python}

We have a huge number of dimensions in the columns of our matrix, each one of which represents a word. We also have a number of documents, each represented as a row.

Let's take a look at some of the words in our documents. Each of these represents a dimension in our model.

~~~
tfidf.get_feature_names_out()[0:1000]
~~~

How many values do we have?

~~~
# print preview of IDF(t) values 
df_idf = pd.DataFrame(tfidf.idf_, index=tfidf.get_feature_names_out(),columns=["idf_weights"]) 
# sort ascending IDF(t) scores
# - recall that IDF(t) = N/DF(t), where N is the number of documents and DF(t) = number of times a term occurs across all documents
# - the rarer a word is, the higher the IDF(t) value
df_idf=df_idf.sort_values(by=['idf_weights'],ascending=False) 
df_idf.iloc[0:20,:]
df_idf
~~~
{: .language-python}
~~~
 	idf_weights
000 	1.916291
omitted 	1.916291
oration 	1.916291
oracle 	1.916291
opulent 	1.916291
... 	...
pale 	1.000000
painting 	1.000000
swift 	1.000000
painted 	1.000000
boy 	1.000000

25811 rows × 1 columns
~~~
{: .language-python}

Values are no longer just whole numbers such as 0, 1 or 2. Instead, they are weighted according to how often they occur. More common words have lower weights, and less common words have higher weights.

In this lesson, we learned about document embeddings and how they could be done in multiple ways. While one hot encoding is a simple way of doing embeddings, it may not be the best representation. 
TF-IDF is another way of performing these embeddings that improves the representation of words in our model by weighting them. TF-IDF is often used as an intermediate step in some of the more advanced models we will construct later. 

