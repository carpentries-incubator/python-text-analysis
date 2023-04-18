---
exercises: 20
keypoints:
- TODO
objectives:
- TODO
questions:
- TODO
teaching: 20
title: TrainingWordEmbeddings
---

## Word Embeddings Outline
#### Notebook1 - Word Embeddings Intro
1. Recap of embeddings covered so far

2. Distributional hypothesis

    a. intro word2vec

3. show word2vec capability

    a. man/king example

4. Explain how it works using NN

    a. weights associated with each word

    b. explain two mechanisms: CBOW and SG

#### Notebook2 - Training Word2Vec From Scratch

5. Train model from scratch to show additional use-cases (start in new notebook)

    a. categorical search using word2vec

    b. exercise: compare CBOW and SG

6. Additional research questions

#### Notebook3 - Limitations of Word2Vec and Alternative Approaches

7. Limitations & alternative word embedding methods

    a. Can't store multiple vectors/meanings per word

    b. It doesnâ€™t have a good technique to deal with ambiguity. Two exact words but in two different contexts will have too close vectors.)

    c. OOV (fasttext solves)

8. Summary


##Colab Setup

Run this code to enable helper functions.


```python
# Run this cell to mount your Google Drive.
from google.colab import drive
drive.mount('/content/drive')

# show existing colab notebooks and helpers.py file
from os import listdir
wksp_dir = '/content/drive/My Drive/Colab Notebooks/text-analysis'

# add folder to colab's path so we can import the helper functions
import sys
sys.path.insert(0, wksp_dir)
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).



```python
# check that wksp_dir is correct
listdir(wksp_dir)
```




    ['.ipynb_checkpoints', '__pycache__', 'data', 'helpers.py']




```python
# # text preprocessing libraries
# import re
# import string
# import nltk
```

### Load in the data
Create list of files we'll use for our analysis. We'll start by fitting a word2vec model to just one of the books in our list -- Moby Dick.


```python
# pip install necessary to access parse module (called from helpers.py)
!pip install parse

# test that functions can now be imported from helpers.py
from helpers import create_file_list 

# get list of files to analyze
data_dir = wksp_dir + '/data/'
corpus_file_list = create_file_list(data_dir, "*.txt")

# parse filelist into a dataframe
from helpers import parse_into_dataframe 
data = parse_into_dataframe(data_dir + "{Author}-{Title}.txt", corpus_file_list)
data
```

    Mounted at /content/drive
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting parse
      Downloading parse-1.19.0.tar.gz (30 kB)
      Preparing metadata (setup.py) ... [?25l[?25hdone
    Building wheels for collected packages: parse
      Building wheel for parse (setup.py) ... [?25l[?25hdone
      Created wheel for parse: filename=parse-1.19.0-py3-none-any.whl size=24589 sha256=20b14514845a775ab8e06cdd166fd4675b907c794634c8a90c8fa8f9c225b5f7
      Stored in directory: /root/.cache/pip/wheels/d6/9c/58/ee3ba36897e890f3ad81e9b730791a153fce20caa4a8a474df
    Successfully built parse
    Installing collected packages: parse
    Successfully installed parse-1.19.0






  <div id="df-f3971340-b039-46c6-885a-454cb1bd996b">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Author</th>
      <th>Title</th>
      <th>Item</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>austen</td>
      <td>emma</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>austen</td>
      <td>sense</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>austen</td>
      <td>pride</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>austen</td>
      <td>northanger</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>austen</td>
      <td>persuasion</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>austen</td>
      <td>ladysusan</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>chesterton</td>
      <td>brown</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>chesterton</td>
      <td>thursday</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>chesterton</td>
      <td>whitehorse</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>chesterton</td>
      <td>ball</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>chesterton</td>
      <td>napoleon</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>chesterton</td>
      <td>knewtoomuch</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>dickens</td>
      <td>olivertwist</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>dickens</td>
      <td>ourmutualfriend</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>dickens</td>
      <td>pickwickpapers</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>dickens</td>
      <td>greatexpectations</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>dickens</td>
      <td>bleakhouse</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>0</th>
      <td>dickens</td>
      <td>taleoftwocities</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>dickens</td>
      <td>davidcopperfield</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>dickens</td>
      <td>christmascarol</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>dickens</td>
      <td>hardtimes</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>dumas</td>
      <td>montecristo</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>dumas</td>
      <td>maninironmask</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>dumas</td>
      <td>twentyyearsafter</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>dumas</td>
      <td>blacktulip</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>dumas</td>
      <td>tenyearslater</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>dumas</td>
      <td>threemusketeers</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>litbank</td>
      <td>conll</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>melville</td>
      <td>moby_dick</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>30</th>
      <td>melville</td>
      <td>piazzatales</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>33</th>
      <td>melville</td>
      <td>pierre</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>melville</td>
      <td>conman</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>36</th>
      <td>melville</td>
      <td>omoo</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>38</th>
      <td>melville</td>
      <td>bartleby</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>melville</td>
      <td>typee</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>40</th>
      <td>shakespeare</td>
      <td>lear</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>31</th>
      <td>shakespeare</td>
      <td>romeo</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>32</th>
      <td>shakespeare</td>
      <td>twelfthnight</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>35</th>
      <td>shakespeare</td>
      <td>othello</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>37</th>
      <td>shakespeare</td>
      <td>muchado</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>39</th>
      <td>shakespeare</td>
      <td>caesar</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
    <tr>
      <th>41</th>
      <td>shakespeare</td>
      <td>midsummer</td>
      <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-f3971340-b039-46c6-885a-454cb1bd996b')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-f3971340-b039-46c6-885a-454cb1bd996b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-f3971340-b039-46c6-885a-454cb1bd996b');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
single_file = data.loc[data['Title'] == 'moby_dick','Item'].item()
# single_file = data.loc[data['Title'] == 'romeo','Item'].item()
single_file

```




    '/content/drive/My Drive/Colab Notebooks/text-analysis/data/melville-moby_dick.txt'




```python
f = open(single_file,'r')
# ['Item'][0], 'r')
file_contents = f.read()
print(type(file_contents))
f.close()
# # preview doc
preview_len = 500
print(file_contents[0:preview_len])
```

    <class 'str'>
    [Moby Dick by Herman Melville 1851]
    
    
    ETYMOLOGY.
    
    (Supplied by a Late Consumptive Usher to a Grammar School)
    
    The pale Usher--threadbare in coat, heart, body, and brain; I see him
    now.  He was ever dusting his old lexicons and grammars, with a queer
    handkerchief, mockingly embellished with all the gay flags of all the
    known nations of the world.  He loved to dust his old grammars; it
    somehow mildly reminded him of his mortality.
    
    "While you take in hand to school others, and to teach them by wha



```python
file_contents[0:preview_len] # \n are still present in actual string (print() processes these as new lines)
```




    '[Moby Dick by Herman Melville 1851]\n\n\nETYMOLOGY.\n\n(Supplied by a Late Consumptive Usher to a Grammar School)\n\nThe pale Usher--threadbare in coat, heart, body, and brain; I see him\nnow.  He was ever dusting his old lexicons and grammars, with a queer\nhandkerchief, mockingly embellished with all the gay flags of all the\nknown nations of the world.  He loved to dust his old grammars; it\nsomehow mildly reminded him of his mortality.\n\n"While you take in hand to school others, and to teach them by wha'



## Preprocessing steps
1. Split text into sentences
2. Tokenize the text
3. lemmatize and lowercase all tokens
4. remove stop words?

Stop words are usually thought of as the most common words in a language.
"By removing these words, we remove the low-level information from our text in order to give more focus to the important information. In order words, we can say that the removal of such words does not show any negative consequences on the model we train for our task."

Let's review the stop words included with nltk.

### Convert text to list of sentences
"The built-in Punkt sentence tokenizer works well if you want to tokenize simple paragraphs. After importing the NLTK module, all you need to do is use the â€œsent_tokenize()â€� method on a large text corpus. However, the Punkt sentence tokenizer may not correctly detect sentences when there is a complex paragraph that contains many punctuation marks, exclamation marks, abbreviations, or repetitive symbols. It is not possible to define a standard way to overcome these issues. You will have to write custom code for tackling these issues using regex, string manipulation, or by training your own data model instead of using the built-in Punkt data model."


```python
import nltk
nltk.download('punkt') # dependency of sent_tokenize function
sentences = nltk.sent_tokenize(file_contents)
type(sentences)
```

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt.zip.





    list




```python
sentences[300:305]
```




    ['How then is this?',
     'Are the green fields gone?',
     'What do they\nhere?',
     'But look!',
     'here come more crowds, pacing straight for the water, and\nseemingly bound for a dive.']



### Pull up preprocess text helper function and unpack steps 
* We'll run this function on each sentence
* Lemmatization, tokenization, lowercase are all review
* Stop words were removed for TD-IDF. Here, we make it optional. You don't always want to remove stop words before fitting a word2vec model. 

### Stopwords
"It clearly makes sense to consider 'not' as a stop word if your task is based on word frequencies (e.g. tfâ€“idf analysis for document classification).

If you're concerned with the context (e.g. sentiment analysis) of the text it might make sense to treat negation words differently. Negation changes the so-called valence of a text. This needs to be treated carefully and is usually not trivial. One example would be the Twitter negation corpus. An explanation of the approach is given in this paper."

"Do we always remove stop words? Are they always useless for us? ğŸ™‹â€�â™€ï¸�
The answer is no! ğŸ™…â€�â™‚ï¸�

We do not always remove the stop words. The removal of stop words is highly dependent on the task we are performing and the goal we want to achieve. For example, if we are training a model that can perform the sentiment analysis task, we might not remove the stop words.

Movie review: â€œThe movie was not good at all.â€�

Text after removal of stop words: â€œmovie goodâ€�"




```python
# prep stopwords
from nltk.corpus import stopwords
nltk.download('stopwords')


```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.





    True




```python
# review list of stopwords
print(stopwords.words("english")) # note negatives and additional context cue words, e.g., before/during/after, above/below, 'not/nor/no
```

    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]



```python
# prep lemmatizer
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

def preprocess_text(text: str, remove_stopwords: bool, verbose: bool) -> list:
    """Function that cleans the input text by going to:
    - remove numbers
    - remove special characters
    - remove stopwords
    - convert to lowercase
    - remove excessive white spaces
    Arguments:
        text (str): text to clean
        remove_stopwords (bool): whether to remove stopwords
    Returns:
        str: cleaned text
    """
    # remove numbers and special characters
    text = re.sub("[^A-Za-z]+", " ", text)

    # 1. create tokens with help from nltk
    tokens = nltk.word_tokenize(text)
    if verbose:
      print('Tokens', tokens)
    # 2. lowercase and strip out any extra whitepaces with .strip()
    tokens = [w.lower().strip() for w in tokens]
    if verbose:
      print('Lowercase', tokens)

    # 3. convert tokens to lemmatized versions
    wnl = WordNetLemmatizer()
    tokens = [wnl.lemmatize(w) for w in tokens]
    if verbose:
      print('Lemmas', tokens)
        
    # remove stopwords
    if remove_stopwords:
        # 2. check if it's a stopword
        tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
        if verbose:
          print('StopRemoved', tokens)

    # return a list of cleaned tokens
    return tokens


```

    [nltk_data] Downloading package wordnet to /root/nltk_data...



```python
# test function
string = 'It is not down on any map; true places never are.'
tokens = preprocess_text(string, 
                         remove_stopwords=True,
                         verbose=True)
tokens
```

    Tokens ['It', 'is', 'not', 'down', 'on', 'any', 'map', 'true', 'places', 'never', 'are']
    Lowercase ['it', 'is', 'not', 'down', 'on', 'any', 'map', 'true', 'places', 'never', 'are']
    Lemmas ['it', 'is', 'not', 'down', 'on', 'any', 'map', 'true', 'place', 'never', 'are']
    StopRemoved ['map', 'true', 'place', 'never']





    ['map', 'true', 'place', 'never']




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
sentences[300:305]
```




    ['How then is this?',
     'Are the green fields gone?',
     'What do they\nhere?',
     'But look!',
     'here come more crowds, pacing straight for the water, and\nseemingly bound for a dive.']




```python
tokens_cleaned[300:305]
```




    300                                                   []
    301                                 [green, field, gone]
    302                                                   []
    303                                               [look]
    304    [come, crowd, pacing, straight, water, seeming...
    dtype: object




```python
tokens_cleaned.shape # 9852 sentences
```




    (9852,)




```python
# remove empty sentences and 1-word sentences (all stop words)
tokens_cleaned = tokens_cleaned[tokens_cleaned.apply(len) > 1]
tokens_cleaned.shape
```




    (9007,)




```python
tokens_cleaned[300:305]
```




    359                     [contrary, passenger, must, pay]
    360                    [difference, world, paying, paid]
    361    [act, paying, perhaps, uncomfortable, inflicti...
    362                                      [paid, compare]
    363    [urbane, activity, man, receives, money, reall...
    dtype: object



### Train Word2Vec model using tokenized text
We can now use this data to train a word2vec model. We'll start by importing the Word2Vec module from gensim. 

**Set seed and workers for a fully deterministic run**: Next we'll set some parameters for reproducibility. We'll set the seed so that our vectors get randomly initialized the same way each time this code is run. For a fully deterministically-reproducible run, we'll also limit the model to a single worker thread (workers=1), to eliminate ordering jitter from OS thread scheduling â€” noted in [gensim's documentation](https://radimrehurek.com/gensim/models/word2vec.html)


```python
# import gensim's Word2Vec module
from gensim.models import Word2Vec

# train the word2vec model with our cleaned data
model = Word2Vec(sentences=tokens_cleaned, seed=0, workers=1, sg=1)
```

Gensim's implementation is based on the original [Tomas Mikolov's original model of word2vec]("https://arxiv.org/pdf/1301.3781.pdf"), which downsamples all frequent words automatically based on frequency. The downsampling saves time when training the model.

### Next steps: word embedding use-cases
We now have a vector representation for all of the (lemmatized) words referenced throughout Moby Dick. Let's see how we can use these vectors to gain insights from our text data.


### Most similar words
Just like with the pretrained word2vec models, we can use the most_similar function to find words that meaningfully relate to a queried word.


```python
# default
model.wv.most_similar(positive=['whale'], topn=10)
```




    [('great', 0.8898752927780151),
     ('sperm', 0.8600963950157166),
     ('fishery', 0.8341314196586609),
     ('right', 0.8274235725402832),
     ('greenland', 0.8232520222663879),
     ('whalemen', 0.818642795085907),
     ('white', 0.8153451085090637),
     ('bone', 0.8102072477340698),
     ('known', 0.8068004250526428),
     ('present', 0.806235671043396)]



### Vocabulary limits
Note that Word2Vec can only produce vector representations for words encountered in the data used to train the model. 


```python
model.wv.most_similar(positive=['orca'],topn=30) 
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-116-9dc7ea336470> in <cell line: 1>()
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


### Categorical Search
What can we do with this most similar functionality? One way we can use it is to construct a list of similar words to represent some sort of category. For example, maybe we want to know what other sea creatures are referenced throughout the book. We can use gensim's most_smilar function to begin constructing a list of words that, on average, represent a "sea creature" category.  

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
# and find the vectors/words that are most similar to it
model.wv.most_similar(positive=sea_creatures, topn=30)
```




    [('whalemen', 0.9905402660369873),
     ('leviathan', 0.9896253347396851),
     ('ground', 0.9871560335159302),
     ('specie', 0.9861741065979004),
     ('bulk', 0.9844054579734802),
     ('skeleton', 0.98426353931427),
     ('bone', 0.984119176864624),
     ('among', 0.98386549949646),
     ('vast', 0.9826066493988037),
     ('entire', 0.982082724571228),
     ('present', 0.9819111227989197),
     ('found', 0.9817379117012024),
     ('full', 0.9812783598899841),
     ('hump', 0.9808592200279236),
     ('large', 0.9806221127510071),
     ('small', 0.9802840948104858),
     ('fisherman', 0.9801744222640991),
     ('various', 0.9800621271133423),
     ('captured', 0.9800336360931396),
     ('snow', 0.9797706604003906),
     ('part', 0.9797096848487854),
     ('whole', 0.9795398712158203),
     ('monster', 0.9787752628326416),
     ('surface', 0.9787067174911499),
     ('sometimes', 0.9786787033081055),
     ('magnitude', 0.9785758852958679),
     ('mouth', 0.9783716201782227),
     ('instance', 0.97798091173172),
     ('cuvier', 0.9777255654335022),
     ('general', 0.9775925874710083)]




```python
# we can add shark to our list
model.wv.most_similar(positive=['whale','fish','creature','animal','shark'],topn=30) 
```




    [('leviathan', 0.9933432340621948),
     ('whalemen', 0.9930416345596313),
     ('specie', 0.9909929633140564),
     ('ground', 0.9908096194267273),
     ('bulk', 0.9906870126724243),
     ('skeleton', 0.9897971749305725),
     ('among', 0.9892410039901733),
     ('small', 0.9876346588134766),
     ('captured', 0.9875791668891907),
     ('full', 0.9875747561454773),
     ('snow', 0.9875345230102539),
     ('found', 0.9874730110168457),
     ('various', 0.9872222542762756),
     ('hump', 0.9870041608810425),
     ('magnitude', 0.9869523048400879),
     ('sometimes', 0.9868108034133911),
     ('bone', 0.9867365956306458),
     ('cuvier', 0.9865687489509583),
     ('fisherman', 0.9862841367721558),
     ('wholly', 0.9862143993377686),
     ('mouth', 0.9861882328987122),
     ('entire', 0.9858185052871704),
     ('general', 0.9857311844825745),
     ('monster', 0.9856332540512085),
     ('teeth', 0.9855717420578003),
     ('surface', 0.985281765460968),
     ('living', 0.9852625131607056),
     ('natural', 0.9852477312088013),
     ('blubber', 0.9851930737495422),
     ('present', 0.9849874377250671)]




```python
# add leviathan (sea serpent) to list
model.wv.most_similar(positive=['whale','fish','creature','animal','shark','leviathan'],topn=30) 
```




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



No additional sea creatures. It appears we have our list of sea creatures recovered using Word2Vec

#### Limitations 
There is at least one sea creature missing from our list â€” a giant squid. The giant squid is only mentioned briefly in Moby Dick, and therefore it could be that our word2vec model was not able to train a good representation of the word "squid".

Think about how rarely occuring or novel entities such as these might be found. In a later lesson, we will explore a task called Named Entity Recognition, which will handle this type of task in a more robust and systematic way.

When using word2vec to reveal items from a category, you risk missing items that are rarely mentioned. For this reason, it's sometimes better to save this task for larger text corpuses, or more widely pretrained models. 

#### Exercise? Additional questions you could explore using this method...
* **Example**: Train a model on newspaper articles from the 19th century, and collect a list of foods (the topic chosen) referenced throughout the corpus. Do the same for 20th century newspaper articles and compare.


**How else might you exploit this kind of analysis? Share your ideas with the group.**

### Other word embeddings with Gensim 

#### GloVe
"[GloVe](https://nlp.stanford.edu/projects/glove/)" to generate word embeddings. GloVe, coined from Global Vectors, is a model based on the *distribtuional hypothesis*. The GloVe model is trained on the non-zero entries of a global word-word co-occurrence matrix, which tabulates how frequently words co-occur with one another in a given corpus (the GloVe model in spaCy comes pretrained on the [Gigaword dataset](https://catalog.ldc.upenn.edu/LDC2011T07)). The main intuition underlying the model is the observation that ratios of word-word co-occurrence probabilities have the potential for encoding some form of meaning.

Populating this matrix requires a single pass through the entire corpus to collect the statistics. For large corpora, this pass can be computationally expensive, but it is a one-time up-front cost."


```python
# Preview other word embedding models available
print(list(api.info()['models'].keys()))
```

    ['fasttext-wiki-news-subwords-300', 'conceptnet-numberbatch-17-06-300', 'word2vec-ruscorpora-300', 'word2vec-google-news-300', 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100', 'glove-wiki-gigaword-200', 'glove-wiki-gigaword-300', 'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200', '__testing_word2vec-matrix-synopsis']



```python
# make reference to NER to segue into Karl's section

```
