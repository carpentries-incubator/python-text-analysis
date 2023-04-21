---
title: "Preprocessing"
teaching: 10
exercises: 10
questions:
- "How can I prepare data for NLP?"
objectives:
- "Load a test document into Spacy."
- "Learn preprocessing tasks."
keypoints:
- "Learn tokenization"
- "Learn lemmatization"
- "Learn stopwords"
- "Learn about casing"
---

## Preprocessing

Currently, our data is still in a format that is best for humans to read. Humans, without having to think too consciously about it, understand how words and sentences group up and divide into discrete units of meaning. We also understand that the words *run*, *ran*, and *running* are just different grammatical forms of the same underlying concept. Finally, not only do we understand how punctuation affects the meaning of a text, we also can make sense of texts that have odd amounts or odd placements of punctuation.

For example, Darcie Wilder's [*literally show me a healthy person*](https://www.mtv.com/news/1vw892/read-an-excerpt-of-darcie-wilders-literally-show-me-a-healthy-person) has very little capitalization or punctuation:

> in the unauthorized biography of britney spears she says her advice is to lift 1 lb weights and always sing in elevators every time i left to skateboard in the schoolyard i would sing in the elevator i would sing britney spears really loud and once the door opened and there were so many people they heard everything so i never sang again

Across the texts in our corpus, our authors write with different styles, preferring different dictions, punctuation, and so on.

To prepare our data to be more uniformly understood by our algorithms later, we need to (a) break it into smaller units, (b) replace words with their roots, and (c) remove unwanted common or unhelpful words and punctuation.

### Tokenization

Tokenization is the process of breaking down texts (strings of characters) into words, groups of words, and sentences. A string of characters needs to be understood by a program as smaller units so that it can be embedded. These are called **tokens**.  

While our tokens will be words, this will not always be the case. Different models may have different ways of tokenizing strings. The strings may be broken down into multiple word tokens, single word tokens, or even components of words like letters or morphology. Punctuation may or may not be included.

We will be using a tokenizer that breaks documents into single words for this lesson.

Let's load our tokenizer and test it with the first sentence of Emma:

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

This will load spacy and its preprocessing pipeline for English. **Pipelines** are a series of interrelated tasks, where the output of one task is used as an input for another. Different languages may have different rulesets, and therefore require different preprocessing pipelines. Running the document we created through the NLP model we loaded performs a variety of tasks for us. Let's look at these in greater detail.

```python
tokens = spacyt(sentence)
for t in tokens:
 print(t.text)
```

```text
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
```

The single sentence has been broken down into a set of tokens. Tokens in spacy aren't just strings: They're python objects with a variety of attributes. Full documentation for these attributes can be found at: <https://spacy.io/api/token>

### Stems and Lemmas

Think about similar words, such as *running*, *ran*, and *runs*. All of these words have a similar root, but a computer does not know this. Without preprocessing, each of these words would be a new token.

Stemming and Lemmatization are used to group together words that are similar or forms of the same word.

Stemming is removing the conjugation and pluralized endings for words. For example, words like *digitization*, and *digitizing* might chopped down to *digitiz*.

Lemmatization is the more sophisticated of the two, and looks for the linguistic base of a word. Lemmatization can group words that mean the same thing but may not be grouped through simple stemming, such as irregular verbs like *bring* and *brought*.

Similarly, in naive tokenization, capital letters are considered different from non-capital letters, meaning that capitalized versions of words are considered different from non-capitalized versions. Converting all words to lower case ensures that capitalized and non-capitalized versions of words are considered the same.

These steps are taken to reduce the complexities of our NLP models and to allow us to train them from less data.

When we tokenized the first sentence of Emma above, Spacy also created a lemmatized version of itt. Let's try accessing this by typing the following:

```python
for t in tokens:
  print(t.lemma)
```

```txt
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
```

Spacy stores words by an ID number, and not as a full string, to save space in memory. Many spacy functions will return numbers and not words as you might expect. Fortunately, adding an underscore for spacy will return text representations instead. We will also add in the lower case function so that all words are lower case.

```python
for token in tokens:
 print(str.lower(token.lemma_))
```

```txt
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
```

Notice how words like *best* and *her* have been changed to their root words like *good* and *she*. Let's change our tokenizer to save the lower cased, lemmatized versions of words instead of the original words.

```python
class Our_Tokenizer:
  def __init__(self):
    # import spacy tokenizer/language model
    self.nlp = en_core_web_sm.load()
    self.nlp.max_length = 4500000 # increase max number of characters that spacy can process (default = 1,000,000)
  def tokenize(self, document):
    tokens = self.nlp(document)
    simplified_tokens = [str.lower(token.lemma_) for token in tokens]
    return simplified_tokens
```

### Stop-Words and Punctuation

Stop-words are common words that are often filtered out for more efficient natural language data processing. Words such as *the* and *and* don't necessarily tell us a lot about a document's content and are often removed in simpler models. Stop lists (groups of stop words) are curated by sorting terms by their collection frequency, or the total number of times that they appear in a document or corpus. Punctuation also is something we are not interested in, at least not until we get to more complex models. Many open-source software packages for language processing, such as Spacy, include stop lists. Let's look at Spacy's stopword list.

```python
from spacy.lang.en.stop_words import STOP_WORDS
print(STOP_WORDS)
```

```txt
{''s', 'must', 'again', 'had', 'much', 'a', 'becomes', 'mostly', 'once', 'should', 'anyway', 'call', 'front', 'whence', ''ll', 'whereas', 'therein', 'himself', 'within', 'ourselves', 'than', 'they', 'toward', 'latterly', 'may', 'what', 'her', 'nowhere', 'so', 'whenever', 'herself', 'other', 'get', 'become', 'namely', 'done', 'could', 'although', 'which', 'fifteen', 'seems', 'hereafter', 'whereafter', 'two', "'ve", 'to', 'his', 'one', ''d', 'forty', 'being', 'i', 'four', 'whoever', 'somehow', 'indeed', 'that', 'afterwards', 'us', 'she', "'d", 'herein', ''ll', 'keep', 'latter', 'onto', 'just', 'too', "'m", ''re', 'you', 'no', 'thereby', 'various', 'enough', 'go', 'myself', 'first', 'seemed', 'up', 'until', 'yourselves', 'while', 'ours', 'can', 'am', 'throughout', 'hereupon', 'whereupon', 'somewhere', 'fifty', 'those', 'quite', 'together', 'wherein', 'because', 'itself', 'hundred', 'neither', 'give', 'alone', 'them', 'nor', 'as', 'hers', 'into', 'is', 'several', 'thus', 'whom', 'why', 'over', 'thence', 'doing', 'own', 'amongst', 'thereupon', 'otherwise', 'sometime', 'for', 'full', 'anyhow', 'nine', 'even', 'never', 'your', 'who', 'others', 'whole', 'hereby', 'ever', 'or', 'and', 'side', 'though', 'except', 'him', 'now', 'mine', 'none', 'sixty', "n't", 'nobody', ''m', 'well', "'s", 'then', 'part', 'someone', 'me', 'six', 'less', 'however', 'make', 'upon', ''s', ''re', 'back', 'did', 'during', 'when', ''d', 'perhaps', "'re", 'we', 'hence', 'any', 'our', 'cannot', 'moreover', 'along', 'whither', 'by', 'such', 'via', 'against', 'the', 'most', 'but', 'often', 'where', 'each', 'further', 'whereby', 'ca', 'here', 'he', 'regarding', 'every', 'always', 'are', 'anywhere', 'wherever', 'using', 'there', 'anyone', 'been', 'would', 'with', 'name', 'some', 'might', 'yours', 'becoming', 'seeming', 'former', 'only', 'it', 'became', 'since', 'also', 'beside', 'their', 'else', 'around', 're', 'five', 'an', 'anything', 'please', 'elsewhere', 'themselves', 'everyone', 'next', 'will', 'yourself', 'twelve', 'few', 'behind', 'nothing', 'seem', 'bottom', 'both', 'say', 'out', 'take', 'all', 'used', 'therefore', 'below', 'almost', 'towards', 'many', 'sometimes', 'put', 'were', 'ten', 'of', 'last', 'its', 'under', 'nevertheless', 'whatever', 'something', 'off', 'does', 'top', 'meanwhile', 'how', 'already', 'per', 'beyond', 'everything', 'not', 'thereafter', 'eleven', 'n't', 'above', 'eight', 'before', 'noone', 'besides', 'twenty', 'do', 'everywhere', 'due', 'empty', 'least', 'between', 'down', 'either', 'across', 'see', 'three', 'on', 'formerly', 'be', 'very', 'rather', 'made', 'has', 'this', 'move', 'beforehand', 'if', 'my', 'n't', "'ll", 'third', 'without', ''m', 'yet', 'after', 'still', 'same', 'show', 'in', 'more', 'unless', 'from', 'really', 'whether', ''ve', 'serious', 'these', 'was', 'amount', 'whose', 'have', 'through', 'thru', ''ve', 'about', 'among', 'another', 'at'}
```

It's possible to add and remove words as well, for example, *zebra*:

```python
# remember, we need to tokenize things in order for our model to analyze them.
z = spacyt("zebra")[0]
print(z.is_stop) # False

# add zebra to our stopword list
STOP_WORDS.add("zebra")
spacyt = spacy.load("en_core_web_sm")
z = spacyt("zebra")[0]
print(z.is_stop) # True

# remove zebra from our list.
STOP_WORDS.remove("zebra")
spacyt = spacy.load("en_core_web_sm")
z = spacyt("zebra")[0]
print(z.is_stop) # False
```

Let's add "Emma" to our list of stopwords, since knowing that the name "Emma" is often in Jane Austin does not tell us anything interesting.

This will only adjust the stopwords for the current session, but it is possible to save them if desired. More information about how to do this can be found in the Spacy documentation. You might use this stopword list to filter words from documents using spacy, or just by manually iterating through it like a list.

Let's see what our example looks like without stopwords and punctuation:

```python
# add emma to our stopword list
STOP_WORDS.add("emma")
spacyt = spacy.load("en_core_web_sm")

# retokenize our sentence
tokens = spacyt(sentence)

for token in tokens:
  if not token.is_stop and not token.is_punct:
    print(str.lower(token.lemma_))
```

```txt
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
```

Notice that because we added *emma* to our stopwords, she is not in our preprocessed sentence any more.

Let's filter out stopwords and punctuation from our custom tokenizer now as well:

```python
class Our_Tokenizer:
  def __init__(self):
    # import spacy tokenizer/language model
    self.nlp = en_core_web_sm.load()
    self.nlp.max_length = 4500000 # increase max number of characters that spacy can process (default = 1,000,000)
  def tokenize(self, document):
    tokens = self.nlp(document)
    simplified_tokens = [
      str.lower(token.lemma_)
      for token in tokens
      if not token.is_stop and not token.is_punct
    ]

    return simplified_tokens
```

### Parts of Speech

While we can manually add Emma to our stopword list, it may occur to you that novels are filled with characters with unique and unpredictable names. We've already missed the word "Woodhouse" from our list. Creating an enumerated list of all of the possible character names seems impossible.

One way we might address this problem is by using **Parts of speech (POS)** tagging. POS are things such as nouns, verbs, and adjectives. POS tags often prove useful, so some tokenizers also have built in POS tagging done. Spacy is one such library. These tags are not 100% accurate, but they are a great place to start. Spacy's POS tags can be used by accessing the ```pos_``` method for each token.

```python
for token in tokens:
  if token.is_stop == False and token.is_punct == False:
    print(str.lower(token.lemma_)+" "+token.pos_)
```

```txt
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
```

Because our dataset is relatively small, we may find that character names and places weigh very heavily in our early models. We also have a number of blank or white space tokens, which we will also want to remove.

We will finish our special tokenizer by removing punctuation and proper nouns from our documents:

```python
class Our_Tokenizer:
  def __init__(self):
    # import spacy tokenizer/language model
    self.nlp = en_core_web_sm.load()
    self.nlp.max_length = 4500000 # increase max number of characters that spacy can process (default = 1,000,000)
  def tokenize(self, document):
    tokens = self.nlp(document)
    simplified_tokens = [
      str.lower(token.lemma_)
      for token in tokens
      if not token.is_stop
      and not token.is_punct
      and token.pos_ != "PROPN"
    ]

    return simplified_tokens
```

Alternative, instead of "blacklisting" all of the parts of speech we don't want to include, we can "whitelist" just the few that we want, based on what they information they might contribute to the meaning of a text:

```python
class Our_Tokenizer:
  def __init__(self):
    # import spacy tokenizer/language model
    self.nlp = en_core_web_sm.load()
    self.nlp.max_length = 4500000 # increase max number of characters that spacy can process (default = 1,000,000)
  def tokenize(self, document):
    tokens = self.nlp(document)
    simplified_tokens = [
      str.lower(token.lemma_)
      for token in tokens
      if not token.is_stop
      and not token.is_punct
      and token.pos_ in {"ADJ", "ADV", "INTJ", "NOUN", "VERB"}
    ]

    return simplified_tokens
```

Either way, let's test our custom tokenizer on this selection of text to see how it works.

```python
tokenizer = Our_Tokenizer()
tokens = tokenizer.tokenize(sentence)
print(tokens)
```

```txt
['handsome', 'clever', 'rich', 'comfortable', 'home', 'happy', 'disposition', 'unite', 'good', 'blessing', 'existence', 'live', 'nearly', 'year', 'world', 'little', 'distress', 'vex']
```

## Putting it All Together

Now that we've built a tokenizer we're happy with, lets use it to create lemmatized versions of all the books in our corpus.

That is, we want to turn this:

```txt
Emma Woodhouse, handsome, clever, and rich, with a comfortable home
and happy disposition, seemed to unite some of the best blessings
of existence; and had lived nearly twenty-one years in the world
with very little to distress or vex her.
```

into this:

```txt
handsome
clever
rich
comfortable
home
happy
disposition
seem
unite
good
blessing
existence
live
nearly
year
world
very
little
distress
vex
```

To help make this quick for all the text in all our books, we'll use a helper function we prepared for learners:

```python
from helpers import lemmatize_files
lemma_file_list = lemmatize_files(tokenizer, corpus_file_list)
```

```txt
['/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dickens-olivertwist.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/chesterton-knewtoomuch.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dumas-tenyearslater.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dumas-twentyyearsafter.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/austen-pride.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dickens-taleoftwocities.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/chesterton-whitehorse.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dickens-hardtimes.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/austen-emma.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/chesterton-thursday.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dumas-threemusketeers.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/chesterton-ball.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/austen-ladysusan.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/austen-persuasion.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/melville-conman.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/chesterton-napoleon.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/chesterton-brown.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dumas-maninironmask.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dumas-blacktulip.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dickens-greatexpectations.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dickens-ourmutualfriend.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/austen-sense.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dickens-christmascarol.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dickens-davidcopperfield.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dickens-pickwickpapers.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/melville-bartleby.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dickens-bleakhouse.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dumas-montecristo.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/austen-northanger.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/melville-moby_dick.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/shakespeare-twelfthnight.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/melville-typee.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/shakespeare-romeo.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/melville-omoo.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/melville-piazzatales.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/shakespeare-muchado.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/shakespeare-midsummer.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/shakespeare-lear.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/melville-pierre.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/shakespeare-caesar.txt.lemmas', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/shakespeare-othello.txt.lemmas']
```

This process may take several minutes to run. Doing this preprocessing now however will save us much, much time later.

## Saving Our Progress

Let's save our progress by storing a spreadsheet (```*.csv``` or ```*.xlsx``` file) that lists all our authors, books, and associated filenames, both the original and lemmatized copies.

We'll use another helper we prepared to make this easy:

```python
pattern = "/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/{author}-{title}.txt"
data = parse_into_dataframe(pattern, corpus_file_list, col_name="File")
data["Lemma_File"] = lemma_file_list
```

<div id="df-ad561d05-40ed-4c1f-97be-3b2928b28a1d">
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
            <th>0</th>
            <td>austen</td>
            <td>sense</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>21</th>
            <td>austen</td>
            <td>persuasion</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>12</th>
            <td>austen</td>
            <td>pride</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>22</th>
            <td>austen</td>
            <td>northanger</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>9</th>
            <td>austen</td>
            <td>emma</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>7</th>
            <td>austen</td>
            <td>ladysusan</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>10</th>
            <td>chesterton</td>
            <td>thursday</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>8</th>
            <td>chesterton</td>
            <td>ball</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>5</th>
            <td>chesterton</td>
            <td>brown</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>30</th>
            <td>chesterton</td>
            <td>knewtoomuch</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>39</th>
            <td>chesterton</td>
            <td>whitehorse</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>27</th>
            <td>chesterton</td>
            <td>napoleon</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>18</th>
            <td>dickens</td>
            <td>hardtimes</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>28</th>
            <td>dickens</td>
            <td>bleakhouse</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>38</th>
            <td>dickens</td>
            <td>davidcopperfield</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>40</th>
            <td>dickens</td>
            <td>taleoftwocities</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>17</th>
            <td>dickens</td>
            <td>christmascarol</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>20</th>
            <td>dickens</td>
            <td>greatexpectations</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>41</th>
            <td>dickens</td>
            <td>pickwickpapers</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>2</th>
            <td>dickens</td>
            <td>ourmutualfriend</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>13</th>
            <td>dickens</td>
            <td>olivertwist</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>37</th>
            <td>dumas</td>
            <td>threemusketeers</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>33</th>
            <td>dumas</td>
            <td>montecristo</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>32</th>
            <td>dumas</td>
            <td>twentyyearsafter</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>3</th>
            <td>dumas</td>
            <td>tenyearslater</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>29</th>
            <td>dumas</td>
            <td>maninironmask</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>14</th>
            <td>dumas</td>
            <td>blacktulip</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>34</th>
            <td>litbank</td>
            <td>conll</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>4</th>
            <td>melville</td>
            <td>moby_dick</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>24</th>
            <td>melville</td>
            <td>typee</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>23</th>
            <td>melville</td>
            <td>pierre</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>11</th>
            <td>melville</td>
            <td>piazzatales</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>19</th>
            <td>melville</td>
            <td>conman</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>1</th>
            <td>melville</td>
            <td>omoo</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>15</th>
            <td>melville</td>
            <td>bartleby</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>26</th>
            <td>shakespeare</td>
            <td>othello</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>6</th>
            <td>shakespeare</td>
            <td>midsummer</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>16</th>
            <td>shakespeare</td>
            <td>muchado</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>31</th>
            <td>shakespeare</td>
            <td>caesar</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>35</th>
            <td>shakespeare</td>
            <td>lear</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>36</th>
            <td>shakespeare</td>
            <td>romeo</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
          <tr>
            <th>25</th>
            <td>shakespeare</td>
            <td>twelfthnight</td>
            <td>/content/drive/My Drive/Colab Notebooks/text-a...</td>
          </tr>
        </tbody>
      </table>
      </div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ad561d05-40ed-4c1f-97be-3b2928b28a1d')"
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
          document.querySelector('#df-ad561d05-40ed-4c1f-97be-3b2928b28a1d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ad561d05-40ed-4c1f-97be-3b2928b28a1d');
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
</div>

Finally, we'll save this table to a file:

```python
data.to_csv("/content/drive/My Drive/Colab Notebooks/text-analysis/data/data.csv", index=False)
data.to_xlsx("/content/drive/My Drive/Colab Notebooks/text-analysis/data/data.xlsx", index=False)
```

#Outro and Conclusion

This lesson has covered a number of preprocessing steps. We created a list of our files in our corpus, which we can use in future lessons. We customized a tokenizer from Spacy, to better suit the needs of our corpus, which we can also use moving forward.

Next lesson, we will start talking about the concepts behind our model.
