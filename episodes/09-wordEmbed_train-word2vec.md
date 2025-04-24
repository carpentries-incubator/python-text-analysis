---
teaching: 45
exercises: 20
keypoints:
- "As an alternative to using a pre-trained model, training a Word2Vec model on a specific dataset allows you use Word2Vec for NER-related tasks."
objectives:
- "Understand the benefits of training a Word2Vec model on your own data rather than using a pre-trained model"
questions:
- "How can we train a Word2Vec model?"
- "When is it beneficial to train a Word2Vec model on a specific dataset?"

title: Training Word2Vec
---


## Colab Setup

Run this code to enable helper functions and read data back in.


```python
# Run this cell to mount your Google Drive.
from google.colab import drive
drive.mount('/content/drive')

# Show existing colab notebooks and helpers.py file
from os import listdir
wksp_dir = '/content/drive/My Drive/Colab Notebooks/text-analysis/code'
print(listdir(wksp_dir))

# Add folder to colab's path so we can import the helper functions
import sys
sys.path.insert(0, wksp_dir)
```

```python
# pip install necessary to access parse module (called from helpers.py)
!pip install parse
```

### Load in the data

```python
# Read the data back in.
from pandas import read_csv
data = read_csv("/content/drive/My Drive/Colab Notebooks/text-analysis/data/data.csv")
```

Create list of files we'll use for our analysis. We'll start by fitting a word2vec model to just one of the books in our list — Moby Dick.

```python
single_file = data.loc[data['Title'] == 'moby_dick','File'].item()
single_file
```

Let's preview the file contents to make sure our code and directory setup is working correctly.

```python
# open and read file
f = open(single_file,'r')
file_contents = f.read()
f.close()

# preview file contents
preview_len = 500
print(file_contents[0:preview_len])
```

```python
file_contents[0:preview_len] # Note that \n are still present in actual string (print() processes these as new lines)
```

## Preprocessing steps
1. Split text into sentences
2. Tokenize the text
3. Lemmatize and lowercase all tokens
4. Remove stop words

### 1. Convert text to list of sentences
Remember that we are using the sequence of words in a sentence to learn meaningful word embeddings. The last word of one sentence does not always relate to the first word of the next sentence. For this reason, we will split the text into individual sentences before going further.

#### Punkt Sentence Tokenizer
NLTK's sentence tokenizer ('punkt') works well in most cases, but it may not correctly detect sentences when there is a complex paragraph that contains many punctuation marks, exclamation marks, abbreviations, or repetitive symbols. It is not possible to define a standard way to overcome these issues. If you want to ensure every "sentence" you use to train the Word2Vec is truly a sentence, you would need to write some additional (and highly data-dependent) code that uses regex and string manipulation to overcome rare errors.

For our purposes, we're willing to overlook a few sentence tokenization errors. If this work were being published, it would be worthwhile to double-check the work of punkt.

```python
import nltk
nltk.download('punkt') # dependency of sent_tokenize function
sentences = nltk.sent_tokenize(file_contents)
```

```python
sentences[300:305]
```

### 2-4: Tokenize, lemmatize, and remove stop words
 Pull up preprocess text helper function and unpack the code...
* We'll run this function on each sentence
* Lemmatization, tokenization, lowercase and stopwords are all review
* For the lemmatization step, we'll use NLTK's lemmatizer which runs very quickly
* We'll also use NLTK's stop word lists and its tokenization function. Recall that stop words are usually thought of as the most common words in a language. By removing them, we can let the Word2Vec model focus on sequences of meaningful words, only. 

```python
from helpers import preprocess_text
```

```python
# test function
string = 'It is not down on any map; true places never are.'
tokens = preprocess_text(string, 
                         remove_stopwords=True,
                         verbose=True)
print('Result', tokens)
```


```python
# convert list of sentences to pandas series so we can use the apply functionality
import pandas as pd
sentences_series = pd.Series(sentences)
```


```python
tokens_cleaned = sentences_series.apply(preprocess_text, 
                                        remove_stopwords=True, 
                                        verbose=False)
```


```python
# view sentences before clearning
sentences[300:305]
```


```python
# view sentences after cleaning
tokens_cleaned[300:305]
```

```python
tokens_cleaned.shape # 9852 sentences
```

```python
# remove empty sentences and 1-word sentences (all stop words)
tokens_cleaned = tokens_cleaned[tokens_cleaned.apply(len) > 1]
tokens_cleaned.shape
```

### Train Word2Vec model using tokenized text
We can now use this data to train a word2vec model. We'll start by importing the Word2Vec module from gensim. We'll then hand the Word2Vec function our list of tokenized sentences and set sg=0 ("skip-gram") to use the continuous bag of words (CBOW) training method. 

**Set seed and workers for a fully deterministic run**: Next we'll set some parameters for reproducibility. We'll set the seed so that our vectors get randomly initialized the same way each time this code is run. For a fully deterministically-reproducible run, we'll also limit the model to a single worker thread (workers=1), to eliminate ordering jitter from OS thread scheduling — noted in [gensim's documentation](https://radimrehurek.com/gensim/models/word2vec.html)


```python
# import gensim's Word2Vec module
from gensim.models import Word2Vec

# train the word2vec model with our cleaned data
model = Word2Vec(sentences=tokens_cleaned, seed=0, workers=1, sg=0)
```

Gensim's implementation is based on the original [Tomas Mikolov's original model of word2vec]("https://arxiv.org/pdf/1301.3781.pdf"), which downsamples all frequent words automatically based on frequency. The downsampling saves time when training the model.

### Next steps: word embedding use-cases
We now have a vector representation for all the (lemmatized and non-stop words) words referenced throughout Moby Dick. Let's see how we can use these vectors to gain insights from our text data.


### Most similar words
Just like with the pretrained word2vec models, we can use the most_similar function to find words that meaningfully relate to a queried word.


```python
# default
model.wv.most_similar(positive=['whale'], topn=10)
```

### Vocabulary limits
Note that Word2Vec can only produce vector representations for words encountered in the data used to train the model. 

```python
model.wv.most_similar(positive=['orca'],topn=30) # KeyError: "Key 'orca' not present in vocabulary"
``` 

### fastText solves OOV issue
If you need to obtain word vectors for out of vocabulary (OOV) words, you can use the fastText word embedding model, instead (also provided from Gensim). 
The fastText model can obtain vectors even for out-of-vocabulary (OOV) words, by summing up vectors for its component char-ngrams, provided at least one of the char-ngrams was present in the training data.

### Word2Vec for Named Entity Recognition
What can we do with this most similar functionality? One way we can use it is to construct a list of similar words to represent some sort of category. For example, maybe we want to know what other sea creatures are referenced throughout Moby Dick. We can use gensim's most_smilar function to begin constructing a list of words that, on average, represent a "sea creature" category.  

We'll use the following procedure:
1. Initialize a small list of words that represent the category, sea creatures.
2. Calculate the average vector representation of this list of words
3. Use this average vector to find the top N most similar vectors (words)
4. Review similar words and update the sea creatures list
5. Repeat steps 1-4 until no additional sea creatures can be found


```python
# start with a small list of words that represent sea creatures 
sea_creatures = ['whale','fish','creature','animal']

# The below code will calculate an average vector of the words in our list, 
# and find the vectors/words that are most similar to this average vector
model.wv.most_similar(positive=sea_creatures, topn=30)
```


```python
# we can add shark to our list
model.wv.most_similar(positive=['whale','fish','creature','animal','shark'],topn=30) 
```


```python
# add leviathan (sea serpent) to our list
model.wv.most_similar(positive=['whale','fish','creature','animal','shark','leviathan'],topn=30) 
```

No additional sea creatures. It appears we have our list of sea creatures recovered using Word2Vec

#### Limitations 
There is at least one sea creature missing from our list — a giant squid. The giant squid is only mentioned a handful of times throughout Moby Dick, and therefore it could be that our word2vec model was not able to train a good representation of the word "squid". Neural networks only work well when you have lots of data

> ## Exploring the skip-gram algorithm 
>
> The skip-gram algoritmm sometimes performs better in terms of its ability to capture meaning of rarer words encountered in the training data. Train a new Word2Vec model using the skip-gram algorithm, and see if you can repeat the above categorical search task to find the word, "squid".
> 
> > ## Solution
> > 
> > ```python
> > # import gensim's Word2Vec module
> > from gensim.models import Word2Vec
> > 
> > # train the word2vec model with our cleaned data
> > model = Word2Vec(sentences=tokens_cleaned, seed=0, workers=1, sg=1)
> > model.wv.most_similar(positive=['whale','fish','creature','animal','shark','leviathan'],topn=100) # still no sight of squid 
> > ```
> > 
> > ~~~
> > [('whalemen', 0.9931729435920715),
> >  ('specie', 0.9919217824935913),
> >  ('bulk', 0.9917919635772705),
> >  ('ground', 0.9913252592086792),
> >  ('skeleton', 0.9905602931976318),
> >  ('among', 0.9898401498794556),
> >  ('small', 0.9887762665748596),
> >  ('full', 0.9885162115097046),
> >  ('captured', 0.9883950352668762),
> >  ('found', 0.9883666634559631),
> >  ('sometimes', 0.9882548451423645),
> >  ('snow', 0.9880553483963013),
> >  ('magnitude', 0.9880378842353821),
> >  ('various', 0.9878063201904297),
> >  ('hump', 0.9876748919487),
> >  ('cuvier', 0.9875931739807129),
> >  ('fisherman', 0.9874721765518188),
> >  ('general', 0.9873012900352478),
> >  ('living', 0.9872495532035828),
> >  ('wholly', 0.9872384667396545),
> >  ('bone', 0.987160861492157),
> >  ('mouth', 0.9867696762084961),
> >  ('natural', 0.9867129921913147),
> >  ('monster', 0.9865870475769043),
> >  ('blubber', 0.9865683317184448),
> >  ('indeed', 0.9864518046379089),
> >  ('teeth', 0.9862186908721924),
> >  ('entire', 0.9861844182014465),
> >  ('latter', 0.9859246015548706),
> >  ('book', 0.9858523607254028)]
> > ~~~
> > {: .output}
> > 
> > 
> > **Discuss Exercise Result**: When using Word2Vec to reveal items from a category, you risk missing items that are rarely mentioned. This is true even when we use the Skip-gram training method, which has been found to have better performance on rarer words. For this reason, it's sometimes better to save this task for larger text corpuses. In a later lesson, we will explore how large language models (LLMs) can yield better performance on Named Entity Recognition related tasks.
> > 
> {:.solution}
{:.challenge}



> ## Entity Recognition Applications
> 
> How else might you exploit this kind of analysis in your research? Share your ideas with the group.
> 
> > ## Solution
> > 
> > **Example**: Train a model on newspaper articles from the 19th century, and collect a list of foods (the topic chosen) referenced throughout the corpus. Do the same for 20th century newspaper articles and compare to see how popular foods have changed over time.
> {:.solution}
{:.challenge}

> ## Comparing Vector Representations Across Authors
> 
> Recall that the Word2Vec model learns to encode a word's meaning/representation based on that word's most common surrounding context words. By training two separate Word2Vec models on, e.g., books collected from two different authors (one model for each author), we can compare how the different authors tend to use words differently. What are some research questions or words that we could investigate with this kind of approach?
> 
> > ## Solution
> > 
> > As one possible approach, we could compare how authors tend to represent different genders. It could be that older (outdated!) books tend to produce word vectors for man and women that are further apart from one another than newer books due to historic gender norms.
> > 
> {:.solution}
{:.challenge}

### Other word embedding models
While Word2Vec is a famous model that is still used throughout many NLP applications today, there are a few other word embedding models that you might also want to consider exploring. GloVe and fastText are among the two most popular choices to date.

```python
# Preview other word embedding models available
print(list(api.info()['models'].keys()))
```

#### Similarities
* All three algorithms generate vector representations of words in a high-dimensional space.
* They can be used to solve a wide range of natural language processing tasks.
* They are all open-source and widely used in the research community.

#### Differences
* Word2Vec focuses on generating embeddings by predicting the context of a word given its surrounding words in a sentence, while GloVe and fastText generate embeddings by predicting the probability of a word co-occurring with another word in a corpus.
* fastText also includes character n-grams, allowing it to generate embeddings for words not seen during training, making it particularly useful for handling out-of-vocabulary words.
* In general, fastText is considered to be the fastest to train among the three embedding techniques (GloVe, fastText, and Word2Vec). This is because fastText uses subword information, which reduces the vocabulary size and allows the model to handle out-of-vocabulary words. Additionally, fastText uses a hierarchical softmax for training, which is faster than the traditional softmax used by Word2Vec. Finally, fastText can be trained on multiple threads, further speeding up the training process.
