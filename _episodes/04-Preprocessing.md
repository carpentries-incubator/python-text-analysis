---
exercises: 20
keypoints:
- TODO
objectives:
- TODO
questions:
- TODO
teaching: 20
title: Preprocessing
---

##Colab Setup

We'll start by mounting our google drive so that colab can read the helper functions. We'll also go through how many of these functions are written in this lesson.


```python
# Run this cell to mount your Google Drive.
from google.colab import drive
drive.mount('/content/drive')

# show existing colab notebooks and helpers.py file
from os import listdir
wksp_dir = '/content/drive/My Drive/Colab Notebooks/text-analysis'
listdir(wksp_dir)

# add folder to colab's path so we can import the helper functions
import sys
sys.path.insert(0, wksp_dir)
```

    Mounted at /content/drive


##Curse of Dimensionality

Last time, we discussed how every time we added a new word, we would simply add a new dimension to our vector space model. But there is a drawback to simply adding more dimensions to our model. As the number of dimensions increases, the amount of data needed to make good generalizations about that model also goes up. This is sometimes referred to as the __curse of dimensionality__.

This lesson, we will be focusing on how we can load our data into a document-term matrix, while employing various strategies to keep the number of unique words in our model down, which will allow our model to perform better.

##Loading text files

To start, we'll say you have a corpus of text files we want to analyze. Let's create a method to list the files we want to analyze. To make this method more flexible, we will also use glob to allow us to put in regular expressions so we can filter the files if so desired.



```python
!pip install pathlib
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: pathlib in /usr/local/lib/python3.9/dist-packages (1.0.1)



```python
import glob
import os
from pathlib import Path
```


```python
def create_file_list(directory, filter_str='*'):
    """
    Example:
        corpus_file_list = create_file_list("python-text-analysis/data", "*.txt")
    """
    files = Path(directory).glob(filter_str)
    files_to_analyze = list(map(str, files))
    return files_to_analyze
```


```python
corpus_dir = '/content/drive/My Drive/Colab Notebooks/text-analysis/data'
corpus_file_list = create_file_list(corpus_dir)
print(corpus_file_list)
```

    ['/content/drive/My Drive/Colab Notebooks/text-analysis/data/dickens-olivertwist.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/chesterton-knewtoomuch.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/dumas-tenyearslater.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/.gitkeep', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/dumas-twentyyearsafter.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/austen-pride.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/dickens-taleoftwocities.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/chesterton-whitehorse.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/dickens-hardtimes.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/austen-emma.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/chesterton-thursday.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/dumas-threemusketeers.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/chesterton-ball.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/austen-ladysusan.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/austen-persuasion.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/melville-conman.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/chesterton-napoleon.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/chesterton-brown.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/dumas-maninironmask.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/dumas-blacktulip.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/dickens-greatexpectations.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/dickens-ourmutualfriend.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/austen-sense.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/dickens-christmascarol.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/dickens-davidcopperfield.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/dickens-pickwickpapers.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/melville-bartleby.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/dickens-bleakhouse.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/dumas-montecristo.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/lemmas.zip', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/austen-northanger.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/melville-moby_dick.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/shakespeare-twelfthnight.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/melville-typee.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/mit_restaurant.zip', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/shakespeare-romeo.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/melville-omoo.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/melville-piazzatales.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/shakespeare-muchado.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/shakespeare-midsummer.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/shakespeare-lear.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/melville-pierre.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/shakespeare-caesar.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/shakespeare-othello.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/.ipynb_checkpoints']


We will use the full corpus, but it might be useful to filter if we have multiple file types in our directory. We can filter our list using a regular expression as well. If I want just documents written by Austen, I can filter on part of the file path name.



```python
austen_list = create_file_list(corpus_dir, '*austen*')
print(austen_list)
```

    ['/content/drive/My Drive/Colab Notebooks/text-analysis/data/austen-pride.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/austen-emma.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/austen-ladysusan.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/austen-persuasion.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/austen-sense.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/austen-northanger.txt']


Let's take a closer look at Emma. We are looking at the first full sentence, which begins with character 50 and ends at character 290.


```python
preview_len = 290
print(austen_list[1])
sentence = ""
with open(austen_list[1], 'r') as f:
	sentence = f.read(preview_len)[50:preview_len]
print(sentence)
```

    /content/drive/My Drive/Colab Notebooks/text-analysis/data/austen-emma.txt
    Emma Woodhouse, handsome, clever, and rich, with a comfortable home
    and happy disposition, seemed to unite some of the best blessings
    of existence; and had lived nearly twenty-one years in the world
    with very little to distress or vex her.
    


##Tokenization

Tokenization is the process of breaking down texts (strings of characters) into words, groups of words, and sentences. Humans automatically understand words and sentences as discrete units of meaning. However, for computers, we have to break up documents. A string of characters needs to be understood by a program as smaller units so that it can be embedded. These are called __tokens__.  

While our tokens will be words, this will not always be the case! Different models may have different ways of tokenizing strings. The strings may be broken down into multiple word tokens, single word tokens, or even components of words like letters or by its morphology. Punctuation may or may not be included. We will be using a tokenizer that breaks documents into single words for this lesson. Now let's load our test sentence into our tokenizer.


```python
import spacy 
import en_core_web_sm

spacyt = spacy.load("en_core_web_sm")
```


```python
class Our_Tokenizer:
  def __init__(self):
    #import spacy tokenizer/language model
    self.nlp = en_core_web_sm.load()
    self.nlp.max_length = 4500000 # increase max number of characters that spacy can process (default = 1,000,000)
  def tokenize(self, document):
    tokens = self.nlp(document)
    return tokens
```

This will load spacy and its preprocessing pipeline for English. __Pipelines__ are a series of interrelated tasks, where the output of one task is used as an input for another. Different languages may have different rulesets, and therefore require different preprocessing pipelines. Running the document we created through the NLP model we loaded performs a variety of tasks for us. Let's look at these in greater detail.


```python
tokens = spacyt(sentence)
for t in tokens:
	print(t.text)
```

    Emma
    Woodhouse
    ,
    handsome
    ,
    clever
    ,
    and
    rich
    ,
    with
    a
    comfortable
    home
    
    
    and
    happy
    disposition
    ,
    seemed
    to
    unite
    some
    of
    the
    best
    blessings
    
    
    of
    existence
    ;
    and
    had
    lived
    nearly
    twenty
    -
    one
    years
    in
    the
    world
    
    
    with
    very
    little
    to
    distress
    or
    vex
    her
    .
    
    


The single sentence has been broken down into a set of tokens. Tokens in spacy aren't just strings- they're python objects with a variety of attributes. Full documentation for these attributes can be found at: https://spacy.io/api/token

##Lemmas (Stemming and Lemmatization)

Think about similar words, such as running, ran, and runs. All of these words have a similar root, but a computer does not know this. Without preprocessing, each of these words would be a new token. Stemming and Lemmatization are used to group together words that are similar or forms of the same word. Stemming is removing the conjugation and pluralized endings for words. For example; words like “digitization”, and “digitizing” might chopped down to "digitiz." Lemmatization is the more sophisticated of the two, and looks for the linguistic base of a word. Lemmatization can group words that mean the same thing but may not be grouped through simple “stemming,” such as [bring, brought…]

Similarly, capital letters are considered different from non-capital letters, meaning that capitalized versions of words are considered different from non-capitalized versions. Converting all words to lower case ensures that capitalized and non-capitalized versions of words are considered the same.

These steps are taken to reduce the number of dimensions in our model, and allow us to learn more from less data.

Spacy also created a lemmatized version of our document. Let's try accessing this by typing the following:



```python
for t in tokens:
  print(t.lemma)
```

    14931068470291635495
    17859265536816163747
    2593208677638477497
    7792995567492812500
    2593208677638477497
    5763234570816168059
    2593208677638477497
    2283656566040971221
    10580761479554314246
    2593208677638477497
    12510949447758279278
    11901859001352538922
    2973437733319511985
    12006852138382633966
    962983613142996970
    2283656566040971221
    244022080605231780
    3083117615156646091
    2593208677638477497
    15203660437495798636
    3791531372978436496
    1872149278863210280
    7000492816108906599
    886050111519832510
    7425985699627899538
    5711639017775284443
    451024245859800093
    962983613142996970
    886050111519832510
    4708766880135230039
    631425121691394544
    2283656566040971221
    14692702688101715474
    13874798850131827181
    16179521462386381682
    8304598090389628520
    9153284864653046197
    17454115351911680600
    14889849580704678361
    3002984154512732771
    7425985699627899538
    1703489418272052182
    962983613142996970
    12510949447758279278
    9548244504980166557
    9778055143417507723
    3791531372978436496
    14526277127440575953
    3740602843040177340
    14980716871601793913
    6740321247510922449
    12646065887601541794
    962983613142996970


Spacy stores words by an ID number, and not as a full string, to save space in memory. Many spacy functions will return numbers and not words as you might expect. Fortunately, adding an underscore for spacy will return text representations instead. We will also add in the lower case function so that all words are lower case.



```python
for token in tokens:
	print(str.lower(token.lemma_))
```

    emma
    woodhouse
    ,
    handsome
    ,
    clever
    ,
    and
    rich
    ,
    with
    a
    comfortable
    home
    
    
    and
    happy
    disposition
    ,
    seem
    to
    unite
    some
    of
    the
    good
    blessing
    
    
    of
    existence
    ;
    and
    have
    live
    nearly
    twenty
    -
    one
    year
    in
    the
    world
    
    
    with
    very
    little
    to
    distress
    or
    vex
    she
    .
    
    


Notice how words like "best" and "her" have been changed to their root words such as "good" and "she". Let's change our tokenizer to save the lower cased, lemmatized versions of words instead of the original words.




```python
class Our_Tokenizer:
  def __init__(self):
    #import spacy tokenizer/language model
    self.nlp = en_core_web_sm.load()
    self.nlp.max_length = 4500000 # increase max number of characters that spacy can process (default = 1,000,000)
  def tokenize(self, document):
    tokens = self.nlp(document)
    for token in tokens:
      token = str.lower(token.lemma_)
    return tokens
```

##Stop-Words and Punctuation

Stop-words are common words that are often filtered out for more efficient natural language data processing. Words such as “the,” “an,” “a,” “of,” “and/or,” “many” don't necessarily tell us a lot about a document's content and are often removed in simpler models. Stop lists (groups of stop words) are curated by sorting terms by their collection frequency, or the total number of times that they appear in a document or corpus. Punctuation also is something we are not interested in, since we have a "bag of words" assumption which does not care about the position of words or punctuation. Many open-source software packages for language processing, such as Spacy, include stop lists. Let's look at Spacy's stopword list.




```python
from spacy.lang.en.stop_words import STOP_WORDS
print(STOP_WORDS)

```

    {'’s', 'must', 'again', 'had', 'much', 'a', 'becomes', 'mostly', 'once', 'should', 'anyway', 'call', 'front', 'whence', '’ll', 'whereas', 'therein', 'himself', 'within', 'ourselves', 'than', 'they', 'toward', 'latterly', 'may', 'what', 'her', 'nowhere', 'so', 'whenever', 'herself', 'other', 'get', 'become', 'namely', 'done', 'could', 'although', 'which', 'fifteen', 'seems', 'hereafter', 'whereafter', 'two', "'ve", 'to', 'his', 'one', '‘d', 'forty', 'being', 'i', 'four', 'whoever', 'somehow', 'indeed', 'that', 'afterwards', 'us', 'she', "'d", 'herein', '‘ll', 'keep', 'latter', 'onto', 'just', 'too', "'m", '’re', 'you', 'no', 'thereby', 'various', 'enough', 'go', 'myself', 'first', 'seemed', 'up', 'until', 'yourselves', 'while', 'ours', 'can', 'am', 'throughout', 'hereupon', 'whereupon', 'somewhere', 'fifty', 'those', 'quite', 'together', 'wherein', 'because', 'itself', 'hundred', 'neither', 'give', 'alone', 'them', 'nor', 'as', 'hers', 'into', 'is', 'several', 'thus', 'whom', 'why', 'over', 'thence', 'doing', 'own', 'amongst', 'thereupon', 'otherwise', 'sometime', 'for', 'full', 'anyhow', 'nine', 'even', 'never', 'your', 'who', 'others', 'whole', 'hereby', 'ever', 'or', 'and', 'side', 'though', 'except', 'him', 'now', 'mine', 'none', 'sixty', "n't", 'nobody', '‘m', 'well', "'s", 'then', 'part', 'someone', 'me', 'six', 'less', 'however', 'make', 'upon', '‘s', '‘re', 'back', 'did', 'during', 'when', '’d', 'perhaps', "'re", 'we', 'hence', 'any', 'our', 'cannot', 'moreover', 'along', 'whither', 'by', 'such', 'via', 'against', 'the', 'most', 'but', 'often', 'where', 'each', 'further', 'whereby', 'ca', 'here', 'he', 'regarding', 'every', 'always', 'are', 'anywhere', 'wherever', 'using', 'there', 'anyone', 'been', 'would', 'with', 'name', 'some', 'might', 'yours', 'becoming', 'seeming', 'former', 'only', 'it', 'became', 'since', 'also', 'beside', 'their', 'else', 'around', 're', 'five', 'an', 'anything', 'please', 'elsewhere', 'themselves', 'everyone', 'next', 'will', 'yourself', 'twelve', 'few', 'behind', 'nothing', 'seem', 'bottom', 'both', 'say', 'out', 'take', 'all', 'used', 'therefore', 'below', 'almost', 'towards', 'many', 'sometimes', 'put', 'were', 'ten', 'of', 'last', 'its', 'under', 'nevertheless', 'whatever', 'something', 'off', 'does', 'top', 'meanwhile', 'how', 'already', 'per', 'beyond', 'everything', 'not', 'thereafter', 'eleven', 'n’t', 'above', 'eight', 'before', 'noone', 'besides', 'twenty', 'do', 'everywhere', 'due', 'empty', 'least', 'between', 'down', 'either', 'across', 'see', 'three', 'on', 'formerly', 'be', 'very', 'rather', 'made', 'has', 'this', 'move', 'beforehand', 'if', 'my', 'n‘t', "'ll", 'third', 'without', '’m', 'yet', 'after', 'still', 'same', 'show', 'in', 'more', 'unless', 'from', 'really', 'whether', '‘ve', 'serious', 'these', 'was', 'amount', 'whose', 'have', 'through', 'thru', '’ve', 'about', 'among', 'another', 'at'}


It's possible to add and remove words as well. Let's add the name of a character, since knowing that "Emma Woodhouse" is more likely to show up in a Jane Austin novel is not particularly insightful analysis.


```python
#remember, we need to tokenize things in order for our model to analyze them.
z = spacyt("zebra")[0]
print(z.is_stop)
```

    False



```python
#add zebra to our stopword list
STOP_WORDS.add("zebra")
spacyt = spacy.load("en_core_web_sm")
z = spacyt("zebra")[0]
print(z.is_stop)


```

    True



```python
#remove zebra from our list.
STOP_WORDS.remove("zebra")
spacyt = spacy.load("en_core_web_sm")
z = spacyt("zebra")[0]
print(z.is_stop)
```

    False


This will only adjust the stopwords for the current session, but it is possible to save them if desired. More information about how to do this can be found in the Spacy documentation. You might use this stopword list to filter words from documents using spacy, or just by manually iterating through it like a list.

Let's add "Emma" to our list of stopwords, since knowing that the name "Emma" is often in Jane Austin does not tell us anything interesting.

Let's see what our example looks like without stopwords and punctuation.


```python
STOP_WORDS.add("emma")
#Have to reload and retokenize our sentence!
spacyt = spacy.load("en_core_web_sm")
tokens = spacyt(sentence)

for token in tokens:
    if token.is_stop == False and token.is_punct == False:
	       print(str.lower(token.lemma_))

```

    woodhouse
    handsome
    clever
    rich
    comfortable
    home
    
    
    happy
    disposition
    unite
    good
    blessing
    
    
    existence
    live
    nearly
    year
    world
    
    
    little
    distress
    vex
    
    


Notice that because we added "emma" to our stopwords, she is not in our preprocessed sentence any more. Let's filter out stopwords and punctuation from our custom tokenizer now as well.



```python
class Our_Tokenizer:
  def __init__(self):
    #import spacy tokenizer/language model
    self.nlp = en_core_web_sm.load()
    self.nlp.max_length = 4500000 # increase max number of characters that spacy can process (default = 1,000,000)
  def tokenize(self, document):
    tokens = self.nlp(document)
    for token in tokens:
      if token.is_stop == False and token.is_punct == False:
        token = str.lower(token.lemma_)     
    return tokens
```

##POS Tagging

While we can manually add Emma to our stopword list, it may occur to you that novels are filled with characters with unique and unpredictable names. We've already missed the word "Woodhouse" from our list. Creating an enumerated list of all of the possible character names seems impossible.

One way we might address this problem is by using __Parts of speech (POS)__ tagging. POS are things such as nouns, verbs and adjectives. POS tags often prove useful, so some tokenizers also have built in POS tagging done. Spacy is one such library. These tags are not 100% accurate, but they are better than nothing. Spacy's POS tags can be used by accessing the pos_ method for each token.


```python
for token in tokens:
    if token.is_stop == False and token.is_punct == False:
	       print(str.lower(token.lemma_)+" "+token.pos_)
```

    woodhouse PROPN
    handsome ADJ
    clever ADJ
    rich ADJ
    comfortable ADJ
    home NOUN
    
     SPACE
    happy ADJ
    disposition NOUN
    unite VERB
    good ADJ
    blessing NOUN
    
     SPACE
    existence NOUN
    live VERB
    nearly ADV
    year NOUN
    world NOUN
    
     SPACE
    little ADJ
    distress VERB
    vex VERB
    
     SPACE


Because our dataset is relatively small, we may find that character names and places weigh very heavily in our early models. We also have a number of blank or white space tokens, which we will also want to remove. We will finish our special tokenizer by removing punctuation and proper nouns from our documents.



```python
class Our_Tokenizer:
  def __init__(self):
    #import spacy tokenizer/language model
    self.nlp = en_core_web_sm.load()
    self.nlp.max_length = 4500000 # increase max number of characters that spacy can process (default = 1,000,000)
  def tokenize(self, document):
    tokens = self.nlp(document)
    tokens = [token.lemma_ for token in tokens if (
      token.is_stop == False and # filter out stop words
      token.is_punct == False and # filter out punct
      token.is_space == False and #filter newlines
      token.pos_ != 'PROPN')] #remove all proper nouns such as names
    return tokens
```

Let's test our custom tokenizer on this selection of text to see how it works.



```python
our_tok = Our_Tokenizer()
tokenset1= our_tok.tokenize(sentence)
print(tokenset1)

```

    ['handsome', 'clever', 'rich', 'comfortable', 'home', 'happy', 'disposition', 'unite', 'good', 'blessing', 'existence', 'live', 'nearly', 'year', 'world', 'little', 'distress', 'vex']


Let's add two more sentences to our corpus, and then put this representation in vector space. We'll do this using scikit learn. We can specify a tokenizer with sci-kit learn, so we will use the tokenizer we just defined. Then, we will take a look at all the different terms in our dictionary, which contains a list of all the words that occur in our corpus.


```python
s2 = "Happy holidays! Have a happy new year!"
s3 = "What a handsome, happy, healthy little baby!"
corp = [sentence, s2, s3] 

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(tokenizer= our_tok.tokenize)
dtm = vectorizer.fit_transform(corp)
vectorizer.get_feature_names_out()

```

    /usr/local/lib/python3.9/dist-packages/sklearn/feature_extraction/text.py:528: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
      warnings.warn(





    array(['baby', 'blessing', 'clever', 'comfortable', 'disposition',
           'distress', 'existence', 'good', 'handsome', 'happy', 'healthy',
           'holiday', 'home', 'little', 'live', 'nearly', 'new', 'rich',
           'unite', 'vex', 'world', 'year'], dtype=object)



Finally, lets take a look a the term-document matrix. Each document is a row, and each column is a dimension that represents a word. The values in each cell are simple word counts.


```python
print(dtm.toarray())
```

    [[0 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 0 1 1 1 1 1]
     [0 0 0 0 0 0 0 0 0 2 0 1 0 0 0 0 1 0 0 0 0 1]
     [1 0 0 0 0 0 0 0 1 1 1 0 0 1 0 0 0 0 0 0 0 0]]


If desired, we could calculate cosine similarity between different documents as well.


```python
from sklearn.metrics.pairwise import cosine_similarity as cs
print(cs(dtm[0], dtm[1]))
print(cs(dtm[0], dtm[2]))

```

    [[0.26726124]]
    [[0.31622777]]


According to this model, our third sentence is closer to our original sentence than the second one. We could conduct similar analysis over larger groups of text, such as all the documents in our corpus. However, running this method over everything would take a considerable amount of time. For this reason, we've provided pre-lemmatized versions of our texts for our next lesson.

This lesson has covered a number of preprocessing steps. We created a list of our files in our corpus, which we can use in future lessons. We customized a tokenizer from Spacy, to better suit the needs of our corpus, which we can also use moving forward. Finally, we put our sample sentences in a term-document matrix for the first time and calculated cosine similarity scores between the two. Next we will use a more complex model called TF-IDF.
