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

Run this code to enable helper functions.


```python
# Run this cell to mount your Google Drive.
from google.colab import drive
drive.mount('/content/drive')

# Set workshop directory
from os import listdir
wksp_dir = '/content/drive/My Drive/Colab Notebooks/text-analysis'

# Add helper functions to colab's path
import sys
helper_path = wksp_dir + '/code'
sys.path.insert(0, helper_path)

# Check that helper directory is correct
listdir(helper_path)
```
~~~
Mounted at /content/drive
['analysis.py',
 'pyldavis.py',
 '.gitkeep',
 'helpers.py',
 'preprocessing.py',
 'attentionviz.py',
 'mit_restaurants.py',
 'plotfrequency.py',
 '__pycache__']
~~~
{: .output}

### Load in the data
Create list of files we'll use for our analysis. We'll start by fitting a word2vec model to just one of the books in our list — Moby Dick.

```python
# pip install necessary to access parse module (called from helpers.py)
!pip install parse
```

Get list of files available to analyze

```python
from helpers import create_file_list 
data_dir = wksp_dir + '/data/books/'
corpus_file_list = create_file_list(data_dir, "*.txt")
corpus_file_list[0:5]
```

~~~
['/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dickens-bleakhouse.txt',
 '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dumas-blacktulip.txt',
 '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/austen-northanger.txt',
 '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dickens-christmascarol.txt',
 '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/austen-persuasion.txt']
~~~
{: .output}

Parse filelist into a dataframe. Make sure you don't have any extra forward slashes in the pattern — this will cause an error in the helper function.

```python
pattern = data_dir + "{Author}-{Title}.txt"
pattern
```

~~~
'/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/{Author}-{Title}.txt'
~~~
{: .output}

```python
from helpers import parse_into_dataframe 
data = parse_into_dataframe(data_dir + "{Author}-{Title}.txt", corpus_file_list)
data.head()
```

```python
single_file = data.loc[data['Title'] == 'moby_dick','File'].item()
single_file
```

~~~
'/content/drive/My Drive/Colab Notebooks/text-analysis/data/melville-moby_dick.txt'
~~~
{: .output}

Let's preview the file contents to make sure our code so far is working correctly.

```python
# open and read file
f = open(single_file,'r')
file_contents = f.read()
f.close()

# preview file contents
preview_len = 500
print(file_contents[0:preview_len])
```

~~~
[Moby Dick by Herman Melville 1851]


ETYMOLOGY.

(Supplied by a Late Consumptive Usher to a Grammar School)

The pale Usher--threadbare in coat, heart, body, and brain; I see him
now.  He was ever dusting his old lexicons and grammars, with a queer
handkerchief, mockingly embellished with all the gay flags of all the
known nations of the world.  He loved to dust his old grammars; it
somehow mildly reminded him of his mortality.

"While you take in hand to school others, and to teach them by wha
~~~
{: .output}

```python
file_contents[0:preview_len] # Note that \n are still present in actual string (print() processes these as new lines)
```

~~~
'[Moby Dick by Herman Melville 1851]\n\n\nETYMOLOGY.\n\n(Supplied by a Late Consumptive Usher to a Grammar School)\n\nThe pale Usher--threadbare in coat, heart, body, and brain; I see him\nnow.  He was ever dusting his old lexicons and grammars, with a queer\nhandkerchief, mockingly embellished with all the gay flags of all the\nknown nations of the world.  He loved to dust his old grammars; it\nsomehow mildly reminded him of his mortality.\n\n"While you take in hand to school others, and to teach them by wha'
~~~
{: .output}

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

~~~
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
~~~
{: .output}


```python
sentences[300:305]
```


~~~
['How then is this?',
 'Are the green fields gone?',
 'What do they\nhere?',
 'But look!',
 'here come more crowds, pacing straight for the water, and\nseemingly bound for a dive.']
~~~
{: .output}


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
tokens
```

~~~
Tokens ['It', 'is', 'not', 'down', 'on', 'any', 'map', 'true', 'places', 'never', 'are']
Lowercase ['it', 'is', 'not', 'down', 'on', 'any', 'map', 'true', 'places', 'never', 'are']
Lemmas ['it', 'is', 'not', 'down', 'on', 'any', 'map', 'true', 'place', 'never', 'are']
StopRemoved ['map', 'true', 'place', 'never']

['map', 'true', 'place', 'never']
~~~
{: .output}



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



~~~
['How then is this?',
 'Are the green fields gone?',
 'What do they\nhere?',
 'But look!',
 'here come more crowds, pacing straight for the water, and\nseemingly bound for a dive.']
~~~
{: .output}



```python
# view sentences after cleaning
tokens_cleaned[300:305]
```

~~~
    300                                                   []
    301                                 [green, field, gone]
    302                                                   []
    303                                               [look]
    304    [come, crowd, pacing, straight, water, seeming...
    dtype: object
~~~
{: .output}


```python
tokens_cleaned.shape # 9852 sentences
```

~~~
(9852,)
~~~
{: .output}


```python
# remove empty sentences and 1-word sentences (all stop words)
tokens_cleaned = tokens_cleaned[tokens_cleaned.apply(len) > 1]
tokens_cleaned.shape
```


~~~
(9007,)
~~~
{: .output}

### Train Word2Vec model using tokenized text
We can now use this data to train a word2vec model. We'll start by importing the Word2Vec module from gensim. We'll then hand the Word2Vec function our list of tokenized sentences and set sg=0 to use the continuous bag of words (CBOW) training method. 

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

~~~
[('great', 0.9986481070518494),
 ('white', 0.9984517097473145),
 ('fishery', 0.9984385371208191),
 ('sperm', 0.9984176158905029),
 ('among', 0.9983417987823486),
 ('right', 0.9983320832252502),
 ('three', 0.9983301758766174),
 ('day', 0.9983181357383728),
 ('length', 0.9983041882514954),
 ('seen', 0.998255729675293)]
~~~
{: .output}

### Vocabulary limits
Note that Word2Vec can only produce vector representations for words encountered in the data used to train the model. 

```python
model.wv.most_similar(positive=['orca'],topn=30) 
```

    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-25-9dc7ea336470> in <cell line: 1>()
    ----> 1 model.wv.most_similar(positive=['orca'],topn=30)
    

    /usr/local/lib/python3.9/dist-packages/gensim/models/keyedvectors.py in most_similar(self, positive, negative, topn, clip_start, clip_end, restrict_vocab, indexer)
        839 
        840         # compute the weighted average of all keys
    --> 841         mean = self.get_mean_vector(keys, weight, pre_normalize=True, post_normalize=True, ignore_missing=False)
        842         all_keys = [
        843             self.get_index(key) for key in keys if isinstance(key, _KEY_TYPES) and self.has_index_for(key)


    /usr/local/lib/python3.9/dist-packages/gensim/models/keyedvectors.py in get_mean_vector(self, keys, weights, pre_normalize, post_normalize, ignore_missing)
        516                 total_weight += abs(weights[idx])
        517             elif not ignore_missing:
    --> 518                 raise KeyError(f"Key '{key}' not present in vocabulary")
        519 
        520         if total_weight > 0:


    KeyError: "Key 'orca' not present in vocabulary"

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


~~~
[('great', 0.9997826814651489),
 ('part', 0.9997532963752747),
 ('though', 0.9997507333755493),
 ('full', 0.999735951423645),
 ('small', 0.9997267127037048),
 ('among', 0.9997209906578064),
 ('case', 0.9997204542160034),
 ('like', 0.9997190833091736),
 ('many', 0.9997131824493408),
 ('fishery', 0.9997081756591797),
 ('present', 0.9997068643569946),
 ('body', 0.9997056722640991),
 ('almost', 0.9997050166130066),
 ('found', 0.9997038245201111),
 ('whole', 0.9997023940086365),
 ('water', 0.9996949434280396),
 ('even', 0.9996913075447083),
 ('time', 0.9996898174285889),
 ('two', 0.9996897578239441),
 ('air', 0.9996871948242188),
 ('length', 0.9996850490570068),
 ('vast', 0.9996834397315979),
 ('line', 0.9996828436851501),
 ('made', 0.9996813535690308),
 ('upon', 0.9996812343597412),
 ('large', 0.9996775984764099),
 ('known', 0.9996767640113831),
 ('harpooneer', 0.9996761679649353),
 ('sea', 0.9996750354766846),
 ('shark', 0.9996744990348816)]
~~~
{: .output}

```python
# we can add shark to our list
model.wv.most_similar(positive=['whale','fish','creature','animal','shark'],topn=30) 
```

~~~
[('great', 0.9997999668121338),
 ('though', 0.9997922778129578),
 ('part', 0.999788761138916),
 ('full', 0.999781608581543),
 ('small', 0.9997766017913818),
 ('like', 0.9997683763504028),
 ('among', 0.9997652769088745),
 ('many', 0.9997631311416626),
 ('case', 0.9997614622116089),
 ('even', 0.9997515678405762),
 ('body', 0.9997514486312866),
 ('almost', 0.9997509717941284),
 ('present', 0.9997479319572449),
 ('found', 0.999747633934021),
 ('water', 0.9997465014457703),
 ('made', 0.9997431635856628),
 ('air', 0.9997406601905823),
 ('whole', 0.9997400641441345),
 ('fishery', 0.9997299909591675),
 ('harpooneer', 0.9997295141220093),
 ('time', 0.9997290372848511),
 ('two', 0.9997289776802063),
 ('sea', 0.9997265934944153),
 ('strange', 0.9997244477272034),
 ('large', 0.999722421169281),
 ('place', 0.9997209906578064),
 ('dead', 0.9997198581695557),
 ('leviathan', 0.9997192025184631),
 ('sometimes', 0.9997178316116333),
 ('high', 0.9997177720069885)]
~~~
{: .output}


```python
# add leviathan (sea serpent) to our list
model.wv.most_similar(positive=['whale','fish','creature','animal','shark','leviathan'],topn=30) 
```


~~~
[('though', 0.9998274445533752),
 ('part', 0.9998168349266052),
 ('full', 0.9998133182525635),
 ('small', 0.9998107552528381),
 ('great', 0.9998067021369934),
 ('like', 0.9998064041137695),
 ('even', 0.9997999668121338),
 ('many', 0.9997966885566711),
 ('body', 0.9997950196266174),
 ('among', 0.999794602394104),
 ('found', 0.9997929334640503),
 ('case', 0.9997885823249817),
 ('almost', 0.9997871518135071),
 ('made', 0.9997868537902832),
 ('air', 0.999786376953125),
 ('water', 0.9997802972793579),
 ('whole', 0.9997780919075012),
 ('present', 0.9997757077217102),
 ('harpooneer', 0.999768853187561),
 ('place', 0.9997684955596924),
 ('much', 0.9997658729553223),
 ('time', 0.999765157699585),
 ('sea', 0.999765157699585),
 ('dead', 0.999764621257782),
 ('strange', 0.9997624158859253),
 ('high', 0.9997615218162537),
 ('two', 0.999760091304779),
 ('sometimes', 0.9997592568397522),
 ('half', 0.9997562170028687),
 ('vast', 0.9997541904449463)]
~~~
{: .output}


No additional sea creatures. It appears we have our list of sea creatures recovered using Word2Vec

#### Limitations 
There is at least one sea creature missing from our list — a giant squid. The giant squid is only mentioned a handful of times throughout Moby Dick, and therefore it could be that our word2vec model was not able to train a good representation of the word "squid". Neural networks only work well when you have lots of data

#### Exercise: Exploring the skip-gram algorithm 
The skip-gram algoritmm sometimes performs better in terms of its ability to capture meaning of rarer words encountered in the training data. Train a new Word2Vec model using the skip-gram algorithm, and see if you can repeat the above categorical search task to find the word, "squid".


```python
# import gensim's Word2Vec module
from gensim.models import Word2Vec

# train the word2vec model with our cleaned data
model = Word2Vec(sentences=tokens_cleaned, seed=0, workers=1, sg=1)
model.wv.most_similar(positive=['whale','fish','creature','animal','shark','leviathan'],topn=100) # still no sight of squid 
```


~~~
[('whalemen', 0.9931729435920715),
 ('specie', 0.9919217824935913),
 ('bulk', 0.9917919635772705),
 ('ground', 0.9913252592086792),
 ('skeleton', 0.9905602931976318),
 ('among', 0.9898401498794556),
 ('small', 0.9887762665748596),
 ('full', 0.9885162115097046),
 ('captured', 0.9883950352668762),
 ('found', 0.9883666634559631),
 ('sometimes', 0.9882548451423645),
 ('snow', 0.9880553483963013),
 ('magnitude', 0.9880378842353821),
 ('various', 0.9878063201904297),
 ('hump', 0.9876748919487),
 ('cuvier', 0.9875931739807129),
 ('fisherman', 0.9874721765518188),
 ('general', 0.9873012900352478),
 ('living', 0.9872495532035828),
 ('wholly', 0.9872384667396545),
 ('bone', 0.987160861492157),
 ('mouth', 0.9867696762084961),
 ('natural', 0.9867129921913147),
 ('monster', 0.9865870475769043),
 ('blubber', 0.9865683317184448),
 ('indeed', 0.9864518046379089),
 ('teeth', 0.9862186908721924),
 ('entire', 0.9861844182014465),
 ('latter', 0.9859246015548706),
 ('book', 0.9858523607254028)]
~~~
{: .output}


#### Discuss Exercise Result
When using Word2Vec to reveal items from a category, you risk missing items that are rarely mentioned. This is true even when we use the Skip-gram training method, which has been found to have better performance on rarer words. For this reason, it's sometimes better to save this task for larger text corpuses. In a later lesson, we will explore a task called Named Entity Recognition, which will handle this type of task in a more robust and systematic way.

#### Exercise: Categorical Search Applications
How else might you exploit this kind of analysis? Share your ideas with the group.

* **Example**: Train a model on newspaper articles from the 19th century, and collect a list of foods (the topic chosen) referenced throughout the corpus. Do the same for 20th century newspaper articles and compare.





### Other word embedding models

While Word2Vec is a famous model that is still used throughout many NLP applications today, there are a few other word embedding models that you might also want to consider exploring. GloVe and fastText are among the two most popular choices to date.

#### GloVe
"[GloVe](https://nlp.stanford.edu/projects/glove/)" to generate word embeddings. GloVe, coined from Global Vectors, is a model based on the *distribtuional hypothesis*. The GloVe model is trained on the non-zero entries of a global word-word co-occurrence matrix, which tabulates how frequently words co-occur with one another in a given corpus (the GloVe model in spaCy comes pretrained on the [Gigaword dataset](https://catalog.ldc.upenn.edu/LDC2011T07)). The main intuition underlying the model is the observation that ratios of word-word co-occurrence probabilities have the potential for encoding some form of meaning.

Populating this matrix requires a single pass through the entire corpus to collect the statistics. For large corpora, this pass can be computationally expensive, but it is a one-time up-front cost."

#### fastText


```python
# Preview other word embedding models available
print(list(api.info()['models'].keys()))
```

~~~
['fasttext-wiki-news-subwords-300', 'conceptnet-numberbatch-17-06-300', 'word2vec-ruscorpora-300', 'word2vec-google-news-300', 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100', 'glove-wiki-gigaword-200', 'glove-wiki-gigaword-300', 'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200', '__testing_word2vec-matrix-synopsis']
~~~
{: .output}

