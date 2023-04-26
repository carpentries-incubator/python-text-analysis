---
title: "Preparing and Preprocessing Your Data"
teaching: 10
exercises: 10
questions:
- "How can I prepare data for NLP?"
- "What are tokenization, casing and lemmatization?"
objectives:
- "Load a test document into Spacy."
- "Learn preprocessing tasks."
keypoints:
- "Tokenization breaks strings into smaller parts for analysis."
- "Casing removes capital letters."
- "Stopwords are common words that do not contain much useful information."
- "Lemmatization reduces words to their root form."
---
# Preparing and Preprocessing Your Data

## Collection

The first step to preparing your data is to collect it. Whether you use API's to gather your material or some other method depends on your research interests. For this workshop, we'll use pre-gathered data.

During the setup instructions, we asked you to download a number of files. These included about forty texts downloaded from [Project Gutenberg](https://www.gutenberg.org/), which will make up our corpus of texts for our hands on lessons in this course.

Take a moment to orient and familiarize yourself with them:

- Austen
  - Emma - [record](https://gutenberg.org/ebooks/158#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/Emma_(novel))
  - Lady Susan - [record](https://gutenberg.org/ebooks/946#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/Lady_Susan)
  - Northanger Abbey - [record](https://gutenberg.org/ebooks/121#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/Northanger_Abbey)
  - Persuasion - [record](https://www.gutenberg.org/ebooks/105#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/Persuasion_(novel))
  - Pride and Prejudice - [record](https://gutenberg.org/ebooks/1342#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/Pride_and_Prejudice)
  - Sense and Sensibility - [record](https://gutenberg.org/ebooks/21839#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/Sense_and_Sensibility)
- Chesteron
  - The Ball and the Cross - [record](https://gutenberg.org/ebooks/5265#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/The_Ball_and_the_Cross)
  - The Innocence of Father Brown - [record](https://gutenberg.org/ebooks/204#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/Father_Brown)
  - The Man Who Knew Too Much - [record](https://gutenberg.org/ebooks/1720#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/The_Man_Who_Knew_Too_Much_(book))
  - The Napoleon of Notting Hill - [record](https://gutenberg.org/ebooks/20058#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/The_Napoleon_of_Notting_Hill)
  - The Man Who was Thursday - [record](https://gutenberg.org/ebooks/1695#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/The_Man_Who_Was_Thursday)
  - The Ballad of the White Horse - [record](https://gutenberg.org/ebooks/1719#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/The_Ballad_of_the_White_Horse)
- Dickens
  - Bleak House - [record](https://www.gutenberg.org/ebooks/1023#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/Bleak_House)
  - A Christmas Carol - [record](https://gutenberg.org/ebooks/24022#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/A_Christmas_Carol)
  - David Copperfield - [record](https://gutenberg.org/ebooks/766#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/David_Copperfield)
  - Great Expectations - [record](https://gutenberg.org/ebooks/1400#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/Great_Expectations)
  - Hard Times - [record](https://gutenberg.org/ebooks/786#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/Hard_Times_(novel))
  - Oliver Twist - [record](https://gutenberg.org/ebooks/730#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/Oliver_Twist)
  - Our Mutual Friend - [record](https://gutenberg.org/ebooks/883#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/Our_Mutual_Friend)
  - The Pickwick Papers - [record](https://gutenberg.org/ebooks/580#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/The_Pickwick_Papers)
  - A Tale of Two Cities - [record](https://gutenberg.org/ebooks/98#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/A_Tale_of_Two_Cities)
- Dumas
  - The Black Tulip - [record](https://gutenberg.org/ebooks/965#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/The_Black_Tulip)
  - The Man in the Iron Mask - [record](https://gutenberg.org/ebooks/2759#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/The_Vicomte_of_Bragelonne:_Ten_Years_Later#Part_Three:_The_Man_in_the_Iron_Mask_(Chapters_181%E2%80%93269))
  - The Count of Monte Cristo - [record](https://www.gutenberg.org/ebooks/1184#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/The_Count_of_Monte_Cristo)
  - Ten Years Later - [record](https://gutenberg.org/ebooks/2681#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/The_Vicomte_of_Bragelonne:_Ten_Years_Later)
  - The Three Musketeers - [record](https://gutenberg.org/ebooks/1257#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/The_Three_Musketeers)
  - Twenty Years After - [record](https://gutenberg.org/ebooks/1259#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/Twenty_Years_After)
- Melville
  - Bartleby, the Scrivener - [record](https://gutenberg.org/ebooks/11231#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/Bartleby,_the_Scrivener)
  - The Confidence-Man - [record](https://www.gutenberg.org/ebooks/21816) &middot; [wiki](https://en.wikipedia.org/wiki/The_Confidence-Man)
  - Moby Dick - [record](https://gutenberg.org/ebooks/2701#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/Moby-Dick)
  - Omoo - [record](https://gutenberg.org/ebooks/4045#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/Omoo)
  - The Piazza Tales - [record](https://gutenberg.org/ebooks/15859#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/The_Piazza_Tales)
  - Pierre - [record](https://gutenberg.org/ebooks/34970#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/Pierre;_or,_The_Ambiguities)
  - Typee - [record](https://gutenberg.org/ebooks/1900#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/Typee)
- Shakespeare
  - The Trajedy of Julius Caesar - [record](https://gutenberg.org/ebooks/1120#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/Julius_Caesar_(play))
  - The Trajedy of King Lear - [record](https://gutenberg.org/ebooks/1532#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/King_Lear)
  - A Midsummer Night's Dream - [record](https://gutenberg.org/ebooks/1514#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/A_Midsummer_Night%27s_Dream)
  - Much Ado about Nothing - [record](https://gutenberg.org/ebooks/1519#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/Much_Ado_About_Nothing)
  - Othello, the Moor of Venice - [record](https://www.gutenberg.org/ebooks/1531) &middot; [wiki](https://en.wikipedia.org/wiki/Othello)
  - Romeo and Juliet - [record](https://gutenberg.org/ebooks/1513#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/Romeo_and_Juliet)
  - Twelfth Night - [record](https://gutenberg.org/ebooks/1526#bibrec) &middot; [wiki](https://en.wikipedia.org/wiki/Twelfth_Night)

While a full-sized corpus can include thousands of texts, these forty-odd texts will be enough for our illustrative purposes.

## Loading Data into Python

We'll start by mounting our Google Drive so that Colab can read the helper functions. We'll also go through how many of these functions are written in this lesson.

```python
# Run this cell to mount your Google Drive.
from google.colab import drive
drive.mount('/content/drive')

# Show existing colab notebooks and helpers.py file
from os import listdir
wksp_dir = '/content/drive/My Drive/Colab Notebooks/text-analysis'
listdir(wksp_dir)

# Add folder to colab's path so we can import the helper functions
import sys
sys.path.insert(0, wksp_dir)
```

Next, we have a corpus of text files we want to analyze. Let's create a method to list those files. To make this method more flexible, we will also use ```glob``` to allow us to put in regular expressions so we can filter the files if so desired. ```glob``` is a tool for listing files in a directory whose file names match some pattern, like all files ending in ```*.txt```.

```python
!pip install pathlib parse
```

```python
import glob
import os
from pathlib import Path
```

```python
def create_file_list(directory, filter_str='*'):
  files = Path(directory).glob(filter_str)
  files_to_analyze = list(map(str, files))
  return files_to_analyze
```

Alternatively, we can load this function from the ```helpers.py``` file we provided for learners in this course:

```python
from helpers import create_file_list
```

Either way, now we can use that function to list the books in our corpus:

```python
corpus_dir = '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books'
corpus_file_list = create_file_list(corpus_dir)
print(corpus_file_list)
```

```txt
['/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dickens-olivertwist.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/chesterton-knewtoomuch.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dumas-tenyearslater.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dumas-twentyyearsafter.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/austen-pride.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dickens-taleoftwocities.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/chesterton-whitehorse.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dickens-hardtimes.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/austen-emma.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/chesterton-thursday.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dumas-threemusketeers.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/chesterton-ball.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/austen-ladysusan.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/austen-persuasion.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/melville-conman.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/chesterton-napoleon.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/chesterton-brown.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dumas-maninironmask.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dumas-blacktulip.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dickens-greatexpectations.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dickens-ourmutualfriend.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/austen-sense.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dickens-christmascarol.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dickens-davidcopperfield.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dickens-pickwickpapers.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/melville-bartleby.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dickens-bleakhouse.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/dumas-montecristo.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/austen-northanger.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/melville-moby_dick.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/shakespeare-twelfthnight.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/melville-typee.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/shakespeare-romeo.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/melville-omoo.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/melville-piazzatales.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/shakespeare-muchado.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/shakespeare-midsummer.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/shakespeare-lear.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/melville-pierre.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/shakespeare-caesar.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/shakespeare-othello.txt']
```

We will use the full corpus later, but it might be useful to filter to just a few specific files. For example, if I want just documents written by Austen, I can filter on part of the file path name:

```python
austen_list = create_file_list(corpus_dir, 'austen*')
print(austen_list)
```

```txt
['/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/austen-pride.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/austen-emma.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/austen-ladysusan.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/austen-persuasion.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/austen-sense.txt', '/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/austen-northanger.txt']
```

Let's take a closer look at Emma. We are looking at the first full sentence, which begins with character 50 and ends at character 290.

```python
preview_len = 290
emmapath = create_file_list(corpus_dir, 'austen-emma*')[0]
print(emmapath)
sentence = ""
with open(emmapath, 'r') as f:
  sentence = f.read(preview_len)[50:preview_len]

print(sentence)
```

```txt
/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/austen-emma.txt
Emma Woodhouse, handsome, clever, and rich, with a comfortable home
and happy disposition, seemed to unite some of the best blessings
of existence; and had lived nearly twenty-one years in the world
with very little to distress or vex her.
```

## Preprocessing

Currently, our data is still in a format that is best for humans to read. Humans, without having to think too consciously about it, understand how words and sentences group up and divide into discrete units of meaning. We also understand that the words *run*, *ran*, and *running* are just different grammatical forms of the same underlying concept. Finally, not only do we understand how punctuation affects the meaning of a text, we also can make sense of texts that have odd amounts or odd placements of punctuation.

For example, Darcie Wilder's [*literally show me a healthy person*](https://www.mtv.com/news/1vw892/read-an-excerpt-of-darcie-wilders-literally-show-me-a-healthy-person) has very little capitalization or punctuation:

> in the unauthorized biography of britney spears she says her advice is to lift 1 lb weights and always sing in elevators every time i left to skateboard in the schoolyard i would sing in the elevator i would sing britney spears really loud and once the door opened and there were so many people they heard everything so i never sang again

Across the texts in our corpus, our authors write with different styles, preferring different dictions, punctuation, and so on.

To prepare our data to be more uniformly understood by our early models, we need to (a) break it into smaller units, (b) replace words with their roots, and (c) remove unwanted common or unhelpful words and punctuation.

### Tokenization

Tokenization is the process of breaking down texts (strings of characters) into words, groups of words, and sentences. A string of characters needs to be understood by a program as smaller units so that it can be embedded. These are called **tokens**.  

While our tokens will be single words for now, this will not always be the case. Different models have different ways of tokenizing strings. The strings may be broken down into multiple word tokens, single word tokens, or even components of words like letters or morphology. Punctuation may or may not be included.

We will be using a tokenizer that breaks documents into single words for this lesson.

Let's load our tokenizer and test it with the first sentence of Emma:

```python
import spacy
import en_core_web_sm
spacyt = spacy.load("en_core_web_sm")
```

We will define a tokenizer method with the text editor. Keep this open so we can add to it throughout the lesson.

```python
class Our_Tokenizer:
  def __init__(self):
    #import spacy tokenizer/language model
    self.nlp = en_core_web_sm.load()
    self.nlp.max_length = 4500000 # increase max number of characters that spacy can process (default = 1,000,000)
  def __call__(self, document):
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
for t in tokens:
 print(str.lower(t.lemma_))
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
  def __call__(self, document):
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

Notice that because we added *emma* to our stopwords, she is not in our preprocessed sentence any more. Other stopwords are also missing such as numbers.

Let's filter out stopwords and punctuation from our custom tokenizer now as well:

```python
class Our_Tokenizer:
  def __init__(self):
    # import spacy tokenizer/language model
    self.nlp = en_core_web_sm.load()
    self.nlp.max_length = 4500000 # increase max number of characters that spacy can process (default = 1,000,000)
  def __call__(self, document):
    tokens = self.nlp(document)
    simplified_tokens = []    
    for token in tokens:
        if not token.is_stop and not token.is_punct:
            simplified_tokens.append(str.lower(token.lemma_))
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
  def __call__(self, document):
    tokens = self.nlp(document)
    simplified_tokens = [
      #our helper function expects spacy tokens. It will take care of making them lowercase lemmas.
      token for token in tokens
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
  def __call__(self, document):
    tokens = self.nlp(document)
    simplified_tokens = [
      #our helper function expects spacy tokens. It will take care of making them lowercase lemmas.
      token for token in tokens
      if not token.is_stop
      and not token.is_punct
      and token.pos_ in {"ADJ", "ADV", "INTJ", "NOUN", "VERB"}
    ]
    return simplified_tokens
```

Either way, let's test our custom tokenizer on this selection of text to see how it works.

```python
tokenizer = Our_Tokenizer()
tokens = tokenizer(sentence)
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

To help make this quick for all the text in all our books, we'll use a helper function we prepared for learners to use our tokenizer, do the casing and lemmatization we discussed earlier, and write the results to a file:

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
from helpers import parse_into_dataframe
pattern = "/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/{author}-{title}.txt"
data = parse_into_dataframe(pattern, corpus_file_list)
data["Lemma_File"] = lemma_file_list
```

Finally, we'll save this table to a file:

```python
data.to_csv("/content/drive/My Drive/Colab Notebooks/text-analysis/data/data.csv", index=False)
```

## Outro and Conclusion

This lesson has covered a number of preprocessing steps. We created a list of our files in our corpus, which we can use in future lessons. We customized a tokenizer from Spacy, to better suit the needs of our corpus, which we can also use moving forward.

Next lesson, we will start talking about the concepts behind our model.
