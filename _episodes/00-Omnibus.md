# Introduction

## Context for Digital Humanists

The humanities involves a wide variety of fields. Each of those fields brings a variety of research interests and methods to focus on a wide variety of questions.

For this reason, this course will touch on a variety of techniques and tasks which natural language processing can help accomplish

We will discuss different metrics which can be used to evaluate how well we accomplished those tasks.

And we will also discuss models commonly used and how they work, so that researchers can understand the underlying approach used by these tools.

These methods are not infallible or without bias. They are simply another tool you can use to analyze texts and should be critically considered in the same way any other tool would be. The goal of this workshop is not to replace or discredit existing humanist methods, but to help humanists learn new tools to help them accomplish their research.

## What is Natural Language Processing?

Natural Language Processing (NLP) attempts to process human languages using computer models.

### What does NLP do?

There are many possible uses for NLP. Machine Learning and Artificial Intelligence can be thought of as a set of computer algorithms used to take data as an input and produce a desired output. What distinguishes NLP from other types of machine learning is that text and human language is the main input for NLP tasks

A model is a mathematical construct designed to turn our text input into a desired output,
which will vary based on the task. We can think of the various tasks NLP can do as different types
of desired outputs, which may require different models.

### What can I do with NLP

Some of the many functions of NLP include topic modelling and categorization, named entity recognition, search, summarization and more.

We're going to explore some of these tasks in this lesson. We will start by using looking at some of the tasks achievable using the popular "HuggingFace" library.

Navigating to <https://huggingface.co/tasks>, we can see examples of many of the tasks achievable using NLP.

What do these different tasks mean? Let's take a look at an example. Conversational tasks are also known as chatbots. A user engages in conversation with a bot. Let's click on this task now.

<https://huggingface.co/tasks/conversational>

Huggingface usefully provides an online demo as well as a description of the task. On the right, we can see there is a demo of a particular model that does this task. Give conversing with the chatbot a try.

If we scroll down, there is also a link to sample models and datasets HuggingFace has made available that can do variations of this task.  Documentation on how to use the model is available by scrolling down the page. Model specific information is available by clicking on the model. Let's go back to the <https://huggingface.co/tasks>

#### Worked Example: Chatbot in Python

We've got an overview of what different tasks we can accomplish. Now let's try getting started with doing these tasks in Python. We won't worry too much about how this model works for the time being, but will instead just focusing trying it out. We'll start by running a chatbot, just like the one we used online.

```python
!pip install transformers
```

NLP tasks often need to be broken down into simpler subtasks to be executed in a particular order. These are called "pipelines" since the output from one subtask is used as the input to the next subtask. We will define a "pipeline" in Python. Feel free to prompt the chatbot as you wish.

```python
from transformers import pipeline, Conversation
converse = pipeline("conversational", model="microsoft/DialoGPT-medium")

conversation_1 = Conversation("Going to the movies tonight - any suggestions?")
conversation_2 = Conversation("What's the last book you have read?")
converse([conversation_1, conversation_2])
```

```txt
[Conversation id: 91dc8c91-cec7-4826-8a26-2d6c06298696 
  user >> Going to the movies tonight - any suggestions? 
  bot >> The Big Lebowski ,
  Conversation id: f7b2a7b4-a941-4f0f-88a3-3153626278e8 
  user >> What's the last book you have read? 
  bot >> The Last Question ]
```

#### Group Activity and Discussion

Break out into groups and look at a couple of tasks for HuggingFace. The groups will be based on general categories for each task. Discuss possible applications of this type of model to your field of research. Try to brainstorm possible applications for now, don't worry about technical implementation.

1. Tasks that seek to convert non-text into text
    - <https://huggingface.co/tasks/image-to-text>
    - <https://huggingface.co/tasks/text-to-image>
    - <https://huggingface.co/tasks/automatic-speech-recognition>
    - <https://huggingface.co/tasks/image-to-text>
2. Searching and classifying documents as a whole
    - <https://huggingface.co/tasks/text-classification>
    - <https://huggingface.co/tasks/sentence-similarity>
3. Classifying individual words- Sequence based tasks
    - <https://huggingface.co/tasks/token-classification>
    - <https://huggingface.co/tasks/translation>
4. Interactive and generative tasks such as conversation and question answering
    - <https://huggingface.co/tasks/conversational>
    - <https://huggingface.co/tasks/question-answering>

Briefly present a summary of some of the tasks you explored. What types of applications could you see this type of task used in? How might this be relevant to a research question you have? Summarize these tasks and present your findings to the group.

{: .discussion}

#### Topic Modeling

Topic modeling is a type of analysis that attempts to categorize texts.
Documents might be made to match categories defined by the user, in a process called supervised learning.
For example, in a process called authorship identification, we might set a number of authors as "categories" and try to identify which author wrote a text.
Alternatively, the computer might be asked to come up with a set number of topics, and create categories without precoded documents,
in a process called unsupervised learning. Supervised learning requires human labelling and intervention, where
unsupervised learning does not.

![Topic Modeling Graph](../images/01-topicmodelling.png)

#### Named Entity Recognition

The task of Named Entity Recognition is trying to label words belonging to a certain group.
The entities we are looking to recognize may be proper nouns, quantities, or even just words belonging to a certain category, such as animals.
A possible application of this would be to track co-occurrence of characters in different chapters in a book.

![Named Entity Recognition](../images/01-ner.png)

#### Search

Search attempts to retrieve documents that are similar to your query.
In order to do this, there must be some way to compute the similarity between documents.
A search query can be thought of as a small input document, and the outputs could be relevant documents stored in the corpus.

![Search and Document Summarization](../images/01-search.png)

#### Document Summarization

Document summarization takes documents which are longer, and attempts to output a document with the same meaning by finding
relevant snippets or by generating a smaller document that conveys the meaning of the first document.

#### Text Prediction

Text prediction attempts to predict future text inputs from a user based on previous text inputs. Predictive text is used in search engines and also on smartphones to help correct inputs and speed up the process of text input.

## The Interpretive Loop

![The Interpretive Loop](../images/01-task-tool.md)

Despite the range of tasks we'll talk about, many NLP tasks, tools, and models have the same or related underlying data, techniques, and thought process.

Throughout this course, we will talk about an "interpretive loop" between your humanities research tasks and your NLP research tools. Along this loop are a number of common tasks we'll see again and again:

1. Preprocessing the data so it can be processed by the machine
2. Representing the processed data as mathematical constructs that bridge (a) our human intuition on how we might solve a task and (b) the algorithms we're using to help with that task
3. Outputting the result of our algorithms in a human readable format
4. Interpreting the results as it relates to our research tasks, our interests, our stakeholders, and so on

Our next lesson will discuss some of the steps of preprocessing in greater detail.

## Summary and Outro

We've looked at a variety of different tasks you can accomplish with NLP, and used Python to generate text based on one of the models available through HuggingFace.

In the lessons that follow, we will be working on better understanding what is happening in these models.

# Preparing and Preprocessing your Data

## Collection

The first step to preparing your data is to collect it.

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

## OCR and Speech Transcription

In this course, we assume that all of your documents are in text format, ie. a file format that can be copied and pasted into a notepad file. The texts we collected above provided in this format to us by Project Gutenberg.

Not all data is of this type, for example image, sound, PDF, and DOC files.

Fortunately, there exists tools to convert file types like these into text. While these tools are beyond the scope of our lesson, they are still worth mentioning.
Optical Character Recognition (OCR) is a field of study that converts images to text. Tools such as Tesseract, Amazon Textract, or Google's Document AI can perform OCR tasks.
Speech transcription will take audio files and convert them to text as well. Google's Speech-to-Text and Amazon Transcribe are two cloud solutions for speech transcription.

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
!pip install pathlib
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
print(austen_list[1])
sentence = ""
with open(austen_list[1], 'r') as f:
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

# Vector Space

Now that we've preprocessed our data, let's move to the next step of the interpretative loop: representation.

Many NLP models make use of a concept called Vector Space. The concept works like this:

1. We create **embeddings**, or mathematical surrogates, of words and documents in vector space. These embeddings can be represented as sets of coordinates in multidimensional space, or as multi-dimensional matrices.
2. These embeddings should be based on some sort of **feature extraction**, meaning that meaningful features from our original documents are somehow represented in our embedding. This will make it so that relationships between embeddings in this vector space will correspond to relationships in the actual documents.

## Bags of Words

In the models we'll look at today, we have a **"bag of words"** assumption as well. We will not consider the placement of words in sentences, their context, or their conjugation into different forms (run vs ran), not until later in this course.

A "bag of words" model is like putting all words from a sentence in a bag and just being concerned with how many of each word you have, not their order or context.

### Worked Example: Bag of Words

Let's suppose we want to model a small, simple set of toy documents. Our entire corpus of documents will only have two words, *to* and *be*. We have four documents, A, B, C and D:

- A: be be be be be be be be be be to
- B: to be to be to be to be to be to be to be to be
- C: to to be be
- D: to be to be

We will start by embedding words using a "one hot" embedding algorithm. Each document is a new row in our table. Every time word 'to' shows up in a document, we add one to our value for the 'to' dimension for that row, and zero to every other dimension. Every time 'be' shows up in our document, we will add one to our value for the 'be' dimension for that row, and zero to every other dimension.

How does this corpus look in vector space? We can display our model using a **document-term matrix**, which looks like the following:

| Document   | to      | be |
| ---------- | ----------- | ----------- |
| Document A | 1 | 10 |
| Document B | 8 | 8 |
| Document C | 2 | 2 |
| Document D | 2 | 2 |

Notice that documents C and D are represented exactly the same. This is unavoidable right now because of our "bag of words" assumption, but much later on we will try to represent positions of words in our models as well. Let's visualize this using Python.

```python
import numpy as np
import matplotlib.pyplot as plt
```

```python
corpus = np.array([[1,10],[8,8],[2,2],[2,2]])
print(corpus)
```

```txt
[[ 1 10]
  [ 8  8]
  [ 2  2]
  [ 2  2]]
```

### Graphing our model

We don't just have to think of our words as columns. We can also think of them as dimensions, and the values as coordinates for each document.

```python
# matplotlib expects a list of values by column, not by row.
# We can simply turn our table on its edge so rows become columns and vice versa.
corpusT = np.transpose(corpus)
print(corpusT)
```

```txt
[[ 1  8  2  2]
  [10  8  2  2]]
```

```python
X = corpusT[0]
Y = corpusT[1]
# define some colors for each point. Since points A and B are the same, we'll have them as the same color.
mycolors = ['r','g','b','b']

# display our visualization
plt.scatter(X,Y, c=mycolors)
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.show()
```

![png](VectorSpace_files/VectorSpace_5_0.png)

### Distance and Similarity

What can we do with this simple model? At the heart of many research tasks is **distance** or **similarity**, in some sense. When we classify or search for documents, we are asking for documents that are "close to" some known examples or search terms. When we explore the topics in our documents, we are asking for a small set of concepts that capture and help explain as much as the ways our documents might differ from one another. And so on.

There are two measures of distance/similarity we'll consider here: **Euclidean distance** and **cosine similarity**.

#### Euclidean Distance

The Euclidian distance formula makes use of the Pythagorean theorem, where $a^2 + b^2 = c^2$. We can draw a triangle between two points, and calculate the hypotenuse to find the distance. This distance formula works in two dimensions, but can also be generalized over as many dimensions as we want. Let's use distance to compare A to B, C and D. We'll say the closer two points are, the smaller their distance, so the more similar they are.

```python
from sklearn.metrics.pairwise import euclidean_distances as dist

#What is closest to document D?
D = [corpus[3]]
print(D)
```

```txt
[array([2, 2])]
```

```python
dist(corpus, D)
```

```txt
array([[8.06225775],
       [8.48528137],
       [0.        ],
       [0.        ]])
```

Distance may seem like a decent metric at first. Certainly, it makes sense that document D has zero distance from itself. C and D are also similar, which makes sense given our bag of words assumption. But take a closer look at documents B and D. Document B is just document D copy and pasted 4 times! How can it be less similar to document D than document B?

Distance is highly sensitive to document length. Because document A is shorter than document B, it is closer to document D. While distance may be an intuitive measure of similarity, it is actually highly dependent on document length.

We need a different metric that will better represent similarity. This is where vectors come in. Vectors are geometric objects with both length and direction. They can be thought of as a ray or an arrow pointing from one point to another.

Vectors can be added, subtracted, or multiplied together, just like regular numbers can. Our model will consider documents as vectors instead of points, going from the origin at $(0,0)$ to each document. Let's visualize this.

```python
# we need the point of origin in order to draw a vector. Numpy has a function to create an array full of zeroes.
origin = np.zeros([1,4])
print(origin)
```

```txt
[[0. 0. 0. 0.]]
```

```python
# draw our vectors
plt.quiver(origin, origin, X, Y, color=mycolors, angles='xy', scale_units='xy', scale=1)
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.show()
```

![png](VectorSpace_files/VectorSpace_9_1.png)

Document A and document D are headed in exactly the same direction, which matches our intution that both documents are in some way similar to each other, even though they differ in length.

#### Cosine Similarity

**Cosine Similarity** is a metric which is only concerned with the direction of the vector, not its length. This means the length of a document will no longer factor into our similarity metric. The more similar two vectors are in direction, the closer the cosine similarity score gets to 1. The more orthogonal two vectors get (the more at a right angle they are), the closer it gets to 0. And as the more they point in opposite directions, the closer it gets to -1.

You can think of cosine similarity between vectors as signposts aimed out into multidimensional space. Two similar documents going in the same direction have a high cosine similarity, even if one of them is much further away in that direction.

Now that we know what cosine similarity is, how does this metric compare our documents?

```python
from sklearn.metrics.pairwise import cosine_similarity as cs
cs(corpus, D)
```

```txt
array([[0.7739573],
       [1.       ],
       [1.       ],
       [1.       ]])
```

Both A and D are considered similar by this metric. Cosine similarity is used by many models as a measure of similarity between documents and words.

### Generalizing over more dimensions

If we want to add another word to our model, we can add another dimension, which we can represent as another column in our table. Let's add more documents with new words in them.

- E: be or not be
- F: to be or not to be

| Document | to | be | or | not |
| ---------- | ----------- | ----------- | ----------- | ----------- |
| Document A | 1 | 10 | 0 | 0
| Document B | 8 | 8 | 0 | 0
| Document C | 2 | 2 | 0 | 0
| Document D | 2 | 2 | 0 | 0
| Document E | 0 | 2 | 1 | 1
| Document F | 2 | 2 | 1 | 1

We can keep adding dimensions for however many words we want to add. It's easy to imagine vector space with two or three dimensions, but visualizing this mentally will rapidly become downright impossible as we add more and more words. Vocabularies for natural languages can easily reach tens of thousands of words.

Keep in mind, it's not necessary to visualize how a high dimensional vector space looks. These relationships and formulae work over an arbitrary number of dimensions. Our methods for how to measure similarity will carry over, even if drawing a graph is no longer possible.

```python
# add two new dimensions to our corpus
corpus = np.hstack((corpus, np.zeros((4,2))))
print(corpus)
```

```txt
[[ 1. 10.  0.  0.]
  [ 8.  8.  0.  0.]
  [ 2.  2.  0.  0.]
  [ 2.  2.  0.  0.]]
```

```python
E = np.array([[0,2,1,1]])
F = np.array([[2,2,1,1]])

#add document E to our corpus
corpus = np.vstack((corpus, E))
print(corpus)
```

```txt
[[ 1. 10.  0.  0.]
  [ 8.  8.  0.  0.]
  [ 2.  2.  0.  0.]
  [ 2.  2.  0.  0.]
  [ 0.  2.  1.  1.]]
```

What do you think the most similar document is to document F?

```python
cs(corpus, F)
```

```txt
array([[0.69224845],
        [0.89442719],
        [0.89442719],
        [0.89442719],
        [0.77459667]])
```

This new document seems most similar to the documents B,C and D.

This principle of using vector space will hold up over an arbitrary number of dimensions, and therefore over a vocabulary of arbitrary size.

This is the essence of vector space modeling: documents are embedded as vectors in very high dimensional space.

How we define these dimensions and the methods for feature extraction may change and become more complex, but the essential idea remains the same.

Next, we will discuss TF-IDF, which balances the above "bag of words" approach against the fact that some words are more or less interesting: *whale* conveys more useful information than *the*, for example.

# TF-IDF

The method of using word counts is just one way we might embed a document in vector space.  
Let's talk about more complex and representational ways of constructing document embeddings.  
To start, imagine we want to represent each word in our model individually, instead of considering an entire document.
How individual words are represented in vector space is something called "word embeddings" and they are an important concept in NLP.

## One hot encoding: Limitations

How would we make word embeddings for a simple document such as "Feed the duck"?

Let's imagine we have a vector space with a million different words in our corpus, and we are just looking at part of the vector space below.

|      | dodge | duck | ... | farm | feather | feed | ... | tan | the |
|------|-------|------|-----|------|---------|------|-----|-----|-----|
| feed | 0     | 0    |     | 0    | 0       | 1    |     | 0   | 0   |
| the  | 0     | 0    |     | 0    | 0       | 0    |     | 0   | 1   |
| duck | 0     | 1    |     | 0    | 0       | 0    |     | 0   | 0   |
|------|-------|------|-----|------|---------|------|-----|-----|-----|
| Document | 0     | 1    |     | 0    | 0       | 1    |     | 0   | 1   |

Similar to what we did in the previous lesson, we can see that each word embedding gives a 1 for a dimension corresponding to the word, and a zero for every other dimension.
This kind of encoding is known as "one hot" encoding, where a single value is 1 and all others are 0.

Once we have all the word embeddings for each word in the document, we sum them all up to get the document embedding.
This is the simplest and most intuitive way to construct a document embedding from a set of word embeddings.

But does it accurately represent the importance of each word?

Our next model, TF-IDF, will embed words with different values rather than just 0 or 1.

## TF-IDF Basics

Currently our model assumes all words are created equal and are all equally important. However, in the real world we know that certain words are more important than others.

For example, in a set of novels, knowing one novel contains the word *the* 100 times does not tell us much about it. However, if the novel contains a rarer word such as *whale* 100 times, that may tell us quite a bit about its content.

A more accurate model would weigh these rarer words more heavily, and more common words less heavily, so that their relative importance is part of our model.  

However, rare is a relative term. In a corpus of documents about blue whales, the term *whale* may be present in nearly every document. In that case, other words may be rarer and more informative. How do we determine how these weights are done?

One method for constructing more advanced word embeddings is a model called TF-IDF.

TF-IDF stands for term frequency-inverse document frequency. The model consists of two parts: term frequency and inverse document frequency. We multiply the two terms to get the TF-IDF value.

Term frequency is a measure how frequently a term occurs in a document. The simplest way to calculate term frequency is by simply adding up the number of times a term occurs in a document, and dividing by the total word count in the corpus.

Inverse document frequency measures a term's importance. Document frequency is the number of documents a term occurs in, so inverse document frequency gives higher scores to words that occur in fewer documents.
This is represented by the equation:

$idf_i = ln[(N+1) / df_i] + 1$

where $N$ represents the total number of documents in the corpus, and $df_i$ represents document frequency for a particular word i. The key thing to understand is that words that occur in more documents get weighted less heavily.

We can also embed documents in vector space using TF-IDF scores rather than simple word counts. This also weakens the impact of stop-words, since due to their common nature, they have very low scores.

Now that we've seen how TF-IDF works, let's put it into practice.

### Worked Example: TD-IDF

Earlier, we preprocessed our data to lemmatize each file in our corpus, then saved our results for later.

Let's load our data back in to continue where we left off:

```python
from pandas import read_csv
data = read_csv("/content/drive/My Drive/Colab Notebooks/text-analysis/data/data.csv")
```

#### TD-IDF Vectorizer

Next, let's load a vectorizer from ```sklearn``` that will help represent our corpus in TF-IDF vector space for us.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(input='filename', max_df=.6, min_df=.1)
```

Here, `max_df=.6` removes terms that appear in more than 60% of our documents (overly common words like the, a, an) and `min_df=.1` removes terms that appear in less than 10% of our documents (overly rare words like specific character names, typos, or punctuation the tokenizer doesn't understand). We're looking for that sweet spot where terms are frequent enough for us to build theoretical understanding of what they mean for our corpus, but not so frequent that they can't help us tell our documents apart.

Now that we have our vectorizer loaded, let's used it to represent our data.

```python
tfidf = vectorizer.fit_transform(list(data["Lemma_File"]))
print(tfidf.shape)
```

```txt
(41, 9879)
```

#### Inspecting Results

We have a huge number of dimensions in the columns of our matrix (just shy of 10,000), where each one of which represents a word. We also have a number of documents (about forty), each represented as a row.

Try different values for `max_df` and `min_df` to see how the number of columns change as more or fewer words are included in the model.

Let's take a look at some of the words in our documents. Each of these represents a dimension in our model.

```python
tfidf.get_feature_names_out()[0:1000]
```

How many values do we have?

```python
# print preview of IDF(t) values 
df_idf = pd.DataFrame(tfidf.idf_, index=tfidf.get_feature_names_out(),columns=["idf_weights"]) 
# sort ascending IDF(t) scores
# - recall that IDF(t) = N/DF(t), where N is the number of documents and DF(t) = number of times a term occurs across all documents
# - the rarer a word is, the higher the IDF(t) value
df_idf=df_idf.sort_values(by=['idf_weights'],ascending=False) 
df_idf.iloc[0:20,:]
df_idf
```

(The result below will differ based on how you configured your tokenizer and vectorizer earlier.)

```txt
  idf_weights
000  1.916291
omitted  1.916291
oration  1.916291
oracle  1.916291
opulent  1.916291
...  ...
pale  1.000000
painting  1.000000
swift  1.000000
painted  1.000000
boy  1.000000

25811 rows × 1 columns
```

Values are no longer just whole numbers such as 0, 1 or 2. Instead, they are weighted according to how often they occur. More common words have lower weights, and less common words have higher weights.

## TF-IDF Summary

In this lesson, we learned about document embeddings and how they could be done in multiple ways. While one hot encoding is a simple way of doing embeddings, it may not be the best representation.
TF-IDF is another way of performing these embeddings that improves the representation of words in our model by weighting them. TF-IDF is often used as an intermediate step in some of the more advanced models we will construct later.

# Topic Modeling with LSA

So far, we've learned the kinds of task NLP can be used for, preprocessed our data, and represented it as a TF-IDF vector space.

Now, we begin to close the loop with Topic Modeling.

Topic Modeling is a frequent goal of text analysis. Topics are the things that a document is about, by some sense of "about." We could think of topics as:

- discrete categories that your documents belong to, such as fiction vs. non-fiction
- or spectra of subject matter that your documents contain in differing amounts, such as being about politics, cooking, racing, dragons, ...

In the first case, we could use machine learning to predict discrete categories, such as [trying to determine the author of the Federalist Papers](https://towardsdatascience.com/hamilton-a-text-analysis-of-the-federalist-papers-e64cb1764fbf).

In the second case, we could try to determine the least number of topics that provides the most information about how our documents differ from one another, then use those concepts to gain insight about the "stuff" or "story" of our corpus as a whole.

In this lesson we'll focus on this second case, where topics are treated as spectra of subject matter. There are a variety of ways of doing this, and not all of them use the vector space model we have learned. For example:

- Vector-space models:
  - Principle Component Analysis (PCA)
  - Epistemic Network Analysis (ENA)
  - Linear Discriminant Analysis (LDA)
  - Latent Semantic Analysis (LSA)
- Probability models:
  - Latent Dirichlet Allocation (LDA)

Specifically, we will be discussing Latent Semantic Analysis (LSA). We're narrowing our focus to LSA because it introduces us to concepts and workflows that we will use in the future.

The assumption behind LSA is that underlying the thousands of words in our vocabulary are a smaller number of hidden ("latent") topics, and that those topics help explain the distribution of the words we see across our documents. In all our models so far, each dimension has corresponded to a single word. But in LSA, each dimension now corresponds to a hidden topic, and each of those in turn corresponds to the words that are most strongly associated with it.

For example, a hidden topic might be [the lasting influence of the Battle of Hastings on the English language](https://museumhack.com/english-language-changed/), with some documents using more words with Anglo-Saxon roots and other documents using more words with Latin roots. This dimension is "hidden" because authors don't usually stamp a label on their books with a summary of the linguistic histories of their words. Still, we can imagine a spectrum between words that are strongly indicative of authors with more Anglo-Saxon diction vs. words strongly indicative of authors with more Latin diction. Once we have that spectrum, we can place our documents along it, then move on to the next hidden topic, then the next, and so on, until we've discussed the fewest, strongest hidden topics that capture the most "story" about our corpus.

## How LSA Works: SVD of TF-IDF Matrix

Mathematically, these "latent semantic" dimensions are derived from our TF-IDF matrix, so let's begin there. From the previous lesson:

```python
tfidf = vectorizer.fit_transform(list(data["Lemma_File"]))
print(tfidf.shape)
```

```txt
(41, 9879)
```

What do these dimensions mean? We have 41 documents, which we can think of as rows. And we have several thousands of tokens, which is like a dictionary of all the types of words we have in our documents, and which we represent as columns.

Now we want to reduce the number of dimensions used to represent our documents.

Sure, we *could* talk through each of the thousands of words in our dictionary and how each document uses them more or less. And we *could* talk through each of our individual 41 documents and what words it tends to use or not. Qualitative researchers are capable of great things.

But those research strategies won't take advantage of our model, they require a HUGE page burden to walk a reader through, and they put all the pressure on you to notice cross-cutting themes on your own.

Instead, we can use a technique from linear algebra, Singlar Value Decomposition (SVD). to reduce those thousands of words or 41 documents to a smaller set of more cross-cutting dimensions. The idea is to choose the fewest dimensions that capture the most variance.

### SVD

We won't deeply dive into all the mathematics of SVD, but we will discuss what happens in abstract.

The mathematical technique we are using is called "SVD" because we are "decomposing" our original TF-IDF matrix and creating a special matrix with "singular values." Any matrix M of arbitrary size can always be split or decomposed into three matrices that multiply together to make M. There are often many non-unique ways to do this decomposition.

The three resulting matrices are called U, Σ, and Vt.

![Image of SVD. Visualisation of the priciple of SVD by CMG Lee.](images/05-svd.png)

The U matrix is a matrix where there are documents as rows, and different topics as columns. The scores in each cell show how much each document is "about" each topic.

The Vt matrix is a matrix where there are a set of terms as columns, and different topics as rows. Again, the values in each cell correspond to how much a given word indicates a given topic.

The Σ matrix is special, and the one from which SVD gets its name. Nearly every cell in the matrix is zero. Only the diagonal cells are filled in: there are singular values in each row and column. Each singular value represent the amount of variation in our data explained by each topic--how much of the "stuff" of the story that topic covers.

A good deal of variation can often be explained by a relatively small number of topics, and often the variation each topic describes shrinks with each new topic.  Because of this, we can truncate or remove individual rows with the lowest singular values, since they provide the least amount of information.

Once this truncation happens, we can multiply together our three matrices and end up with a smaller matrix with topics instead of words as dimensions.

![Image of singular value decomposition with columns and rows truncated. ](images/05-truncatedsvd.png)

### Information Loss

This allows us to focus our account of our documents on a narrower set of cross-cutting topics.

This does come at a price though.

When we reduce dimensionality, our model loses information about our dataset. Our hope is that the information that was lost was unimportant. But "importance" depends on your moral theoretical stances. Because of this, it is important to carefully inspect the results of your model, carefully interpret the "topics" it identifies, and check all that against your qualitative and theoretical understanding of your documents.

This will likely be an iterative process where you refine your model several times. Keep in mind the adage: all models are wrong, some are useful, and a less accurate model may be easier to explain to your stakeholders.

### Check Your Understanding: SVD

Question: What's the most possible topics we could get from this model? Think about what the most singular values are that you could possibly fit in the Σ matrix.

Remember, these singular values exist only on the diagonal, so the most topics we could have will be whichever we have fewer of- unique words or documents in our corpus.

Because there are usually more unique words than there are documents, it will almost always be equal to the number of documents we have, in this case 41.

### Worked Example: LSA

To see this, let's begin to reduce the dimensionality of our TF-IDF matrix using SVD, starting with the greatest number of dimensions.

```python
from sklearn.decomposition import TruncatedSVD
maxDimensions = min(tfidf.shape)-1
svdmodel = TruncatedSVD(n_components=maxDimensions, algorithm="arpack")
lsa = svdmodel.fit_transform(tfidf)
print(lsa)
```

```txt
[[ 3.91364432e-01 -3.38256707e-01 -1.10255485e-01 ... -3.30703329e-04
    2.26445596e-03 -1.29373990e-02]
  [ 2.83139301e-01 -2.03163967e-01  1.72761316e-01 ...  1.98594965e-04
  -4.41931701e-03 -1.84732254e-02]
  [ 3.32869588e-01 -2.67008449e-01 -2.43271177e-01 ...  4.50149502e-03
    1.99200352e-03  2.32871393e-03]
  ...
  [ 1.91400319e-01 -1.25861226e-01  4.36682522e-02 ... -8.51158743e-04
    4.48451964e-03  1.67944132e-03]
  [ 2.33925324e-01 -8.46322843e-03  1.35493523e-01 ...  5.46406784e-03
  -1.11972177e-03  3.86332162e-03]
  [ 4.09480701e-01 -1.78620470e-01 -1.61670733e-01 ... -6.72035999e-02
    9.27745251e-03 -7.60191949e-05]]
```

How should we pick a number of topics to keep? Fortunately, we have the Singular Values to help us understand how much data each topic explains.
Let's take a look and see how much data each topic explains. We will visualize it on a graph.

```python
import matplotlib.pyplot as plt

#this shows us the amount of dropoff in explanation we have in our sigma matrix. 
print(svdmodel.explained_variance_ratio_)

plt.plot(range(maxDimensions), svdmodel.explained_variance_ratio_ * 100)
plt.xlabel("Topic Number")
plt.ylabel("% explained")
plt.title("SVD dropoff")
plt.show()  # show first chart
```

```txt
[0.02053967 0.12553786 0.08088013 0.06750632 0.05095583 0.04413301
  0.03236406 0.02954683 0.02837433 0.02664072 0.02596086 0.02538922
  0.02499496 0.0240097  0.02356043 0.02203859 0.02162737 0.0210681
  0.02004    0.01955728 0.01944726 0.01830292 0.01822243 0.01737443
  0.01664451 0.0160519  0.01494616 0.01461527 0.01455848 0.01374971
  0.01308112 0.01255502 0.01201655 0.0112603  0.01089138 0.0096127
  0.00830014 0.00771224 0.00622448 0.00499762]
```

![png](LSA_files/LSA_18_1.png)

Often a heuristic used by researchers to determine a topic count is to look at the dropoff in percentage of data explained by each topic.

Typically the rate of data explained will be high at first, dropoff quickly, then start to level out. We can pick a point on the "elbow" where it goes from a high level of explanation to where it starts leveling out and not explaining as much per topic. Past this point, we begin to see diminishing returns on the amount of the "stuff" of our documents we can cover quickly. This is also often a good sweet spot between overfitting our model and not having enough topics.

Alternatively, we could set some target sum for how much of our data we want our topics to explain, something like 90% or 95%. However, with a small dataset like this, that would result in a large number of topics, so we'll pick an elbow instead.

Looking at our results so far, a good number in the middle of the "elbow" appears to be around 5-7 topics. So, let's fit a model using only 6 topics and then take a look at what each topic looks like.

(Why is the first topic, "Topic 0," so low? It has to do with how our SVD was setup. Truncated SVD does not mean center the data beforehand, which takes advantage of sparse matrix algorithms by leaving most of the data at zero. Otherwise, our matrix will me mostly filled with the negative of the mean for each column or row, which takes much more memory to store. The math is outside the scope for this lesson, but it's expected in this scenario that topic 0 will be less informative than the ones that come after it, so we'll skip it.)

```python
numDimensions = 7
svdmodel = TruncatedSVD(n_components=numDimensions, algorithm="arpack")
lsa = svdmodel.fit_transform(tfidf)
print(lsa)
```

```txt
[[ 3.91364432e-01 -3.38256707e-01 -1.10255485e-01 -1.57263147e-01
  4.46988327e-01  4.19701195e-02 -1.60554169e-01]
[ 2.83139301e-01 -2.03163967e-01  1.72761316e-01 -2.09939164e-01
-3.26746690e-01  5.57239735e-01 -2.77917582e-01]
[ 3.32869588e-01 -2.67008449e-01 -2.43271177e-01  2.10563091e-01
-1.76563657e-01 -2.99275913e-02  1.16776821e-02]
[ 3.08138678e-01 -2.10715886e-01  1.90232173e-01 -3.35332382e-01
-2.39294420e-01 -2.10772234e-01 -5.00250358e-02]
[ 3.05001339e-01 -2.28993064e-01  2.27384118e-01 -3.12862475e-01
-2.30273991e-01 -3.01470572e-01  2.94344505e-02]
[ 4.61714301e-01 -3.71103910e-01 -6.23885346e-02 -2.07781625e-01
  3.75805961e-01  4.62796547e-02 -2.40105061e-02]
[ 3.99078406e-01 -3.72675621e-01 -4.29488320e-01  3.21312840e-01
-2.06780567e-01 -4.79678166e-02  1.81897768e-02]
[ 2.60635143e-01 -1.90036072e-01 -1.31092747e-02 -1.38136420e-01
  1.37846031e-01  2.59831829e-02  1.28138615e-01]
[ 2.75254100e-01 -1.66002010e-01  1.51344979e-01 -2.03879356e-01
-1.97434785e-01  4.34660579e-01  3.51604210e-01]
[ 2.63962657e-01 -1.51795541e-01  1.03662446e-01 -1.32354362e-01
-8.01919283e-02  1.34144571e-01  4.40821829e-01]
[ 5.39085586e-01  5.51168135e-01 -7.25812593e-02  1.11795245e-02
-2.79031624e-04 -1.68092332e-02  5.49535679e-03]
[ 2.69952815e-01 -1.76699531e-01  5.70356228e-01  4.48630131e-01
  4.28713759e-02 -2.18545514e-02  1.29750415e-02]
[ 6.20096940e-01  6.50488110e-01 -3.76389598e-02  2.84363611e-02
  1.59378698e-02 -1.18479143e-02 -1.67609142e-02]
[ 2.39439789e-01 -1.46548125e-01  5.73647210e-01  4.48872088e-01
  6.91429226e-02 -6.62720018e-02 -5.65690665e-02]
[ 3.46673808e-01 -2.28179603e-01  4.18572442e-01  1.99567055e-01
-9.26169891e-03  1.28870542e-02  6.90447513e-02]
[ 6.16613469e-01  6.59524199e-01 -6.30672750e-02  4.21736740e-03
  1.66141337e-02 -1.39649741e-02 -9.24035248e-04]
[ 4.19959535e-01 -3.55330895e-01 -5.39327447e-02 -2.01473687e-01
  3.73339308e-01  6.42749710e-02  3.85309124e-02]
[ 3.69324851e-01 -3.45008143e-01 -3.46180574e-01  2.57048111e-01
-2.03332217e-01  8.43097532e-03 -3.03449265e-02]
[ 6.27339749e-01  1.62509554e-01  2.45818244e-02 -7.59347178e-02
-6.91425518e-02  5.45427510e-02  2.01009502e-01]
[ 3.10638955e-01 -1.27428647e-01  6.35926253e-01  4.72744826e-01
  8.18397293e-02 -5.48693117e-02 -7.44129304e-02]
[ 5.81561697e-01  6.09748220e-01 -4.20854426e-02  1.91045296e-03
  4.76425507e-03 -2.04751525e-02 -1.90787467e-02]
[ 3.25549596e-01 -2.35619355e-01  1.94586350e-01 -3.99287993e-01
-2.46239345e-01 -3.59189648e-01 -5.52938926e-02]
[ 3.88812327e-01 -3.62768914e-01 -4.48329052e-01  3.68459209e-01
-2.60646554e-01 -7.30511536e-02  3.70734308e-02]
[ 4.01431564e-01 -3.29316324e-01 -1.07594721e-01 -9.11451209e-02
  2.29891158e-01  5.14621207e-03  4.04610197e-02]
[ 1.72871962e-01 -5.46831788e-02  8.30995631e-02 -1.54834480e-01
-1.59427703e-01  3.85080042e-01 -9.72202770e-02]
[ 5.98566537e-01  5.98108991e-01 -6.66814202e-02  3.05305099e-02
  5.34360487e-03 -2.87781213e-02 -2.44070894e-02]
[ 2.59082136e-01 -1.76483028e-01  1.18735256e-01 -1.85860632e-01
-3.24030617e-01  4.76593510e-01 -3.77322924e-01]
[ 2.85857247e-01 -2.16452087e-01  1.56285206e-01 -3.83067065e-01
-2.24662519e-01 -4.59375982e-01 -1.60404615e-02]
[ 3.96454518e-01 -3.51785523e-01 -4.06191581e-01  3.09628775e-01
-1.65348903e-01 -3.42214059e-02 -8.79935957e-02]
[ 5.68307565e-01  5.79236354e-01 -2.49977438e-02 -1.65820193e-03
-1.48330776e-03  4.97525494e-04 -7.56653060e-03]
[ 3.95181458e-01 -3.43909965e-01 -1.12527848e-01 -1.54143147e-01
  4.24627540e-01  3.46146552e-02 -9.53357379e-02]
[ 7.03778529e-02 -4.53018748e-02  4.47075047e-02 -1.29319689e-02
-1.25637206e-04 -3.73101178e-03  2.26633086e-02]
[ 5.87259340e-01  5.91592344e-01 -3.06093001e-02  3.14797614e-02
  9.20390599e-03 -8.28941483e-03 -2.50957867e-02]
[ 2.90241679e-01 -1.59290104e-01  5.44614348e-01  3.72292370e-01
  2.60700775e-02  7.08606085e-03 -4.24466458e-02]
[ 3.73064985e-01 -2.83432129e-01  2.07212226e-01 -1.86820663e-02
  2.03303288e-01  1.46948739e-02  1.10489338e-01]
[ 3.80760325e-01 -3.20618500e-01 -2.67027067e-01  4.74970999e-02
  1.41382144e-01 -1.72863694e-02  8.04289208e-03]
[ 2.76029781e-01 -2.66104786e-01 -3.70078860e-01  3.35161862e-01
-2.59387443e-01 -7.34908946e-02  4.83959546e-02]
[ 2.87419636e-01 -2.05299959e-01  1.46794264e-01 -3.22859868e-01
-2.05122322e-01 -3.24165310e-01 -4.45227118e-02]
[ 1.91400319e-01 -1.25861226e-01  4.36682522e-02 -1.02268922e-01
-2.32049150e-02  1.95768614e-01  5.96553168e-01]
[ 2.33925324e-01 -8.46322843e-03  1.35493523e-01 -1.92794298e-01
-1.74616417e-01  4.49616713e-02 -1.85204985e-01]
[ 4.09480701e-01 -1.78620470e-01 -1.61670733e-01 -8.17899037e-02
  3.68899535e-01  1.60467077e-02 -2.28751397e-01]]
```

And put all our results together in one DataFrame so we can save it to a spreadsheet to save all the work we've done so far. This will also make plotting easier in a moment.

Since we don't know what these topics correspond to yet, for now I'll call the first topic X, the second Y, the third Z, and so on.

```python
data[["X", "Y", "Z", "W", "P", "Q"]] = lsa[:, [1, 2, 3, 4, 5, 6]]
print(data)
```

Let's also mean-center the data, so that the "average" of all our documents lies at the origin when we plot things in a moment. Otherwise, the origin would be (0,0), which is uninformative for our purposes here.

```python
from numpy import mean
data[["X", "Y", "Z", "W", "P", "Q"]] -= data[["X", "Y", "Z", "W", "P", "Q"]].mean()
print(data)
```

```txt
          Author              Title  \
0       dickens        olivertwist   
1      melville               omoo   
2        austen         northanger   
3    chesterton              brown   
4    chesterton        knewtoomuch   
5       dickens    ourmutualfriend   
6        austen               emma   
7       dickens     christmascarol   
8      melville        piazzatales   
9      melville             conman   
10  shakespeare            muchado   
11        dumas      tenyearslater   
12  shakespeare               lear   
13        dumas    threemusketeers   
14        dumas        montecristo   
15  shakespeare              romeo   
16      dickens  greatexpectations   
17       austen         persuasion   
18     melville             pierre   
19        dumas   twentyyearsafter   
20  shakespeare             caesar   
21   chesterton               ball   
22       austen              pride   
23      dickens         bleakhouse   
24     melville          moby_dick   
25  shakespeare       twelfthnight   
26     melville              typee   
27   chesterton           thursday   
28       austen              sense   
29  shakespeare          midsummer   
30      dickens     pickwickpapers   
31        dumas         blacktulip   
32  shakespeare            othello   
33        dumas      maninironmask   
34      dickens    taleoftwocities   
35      dickens   davidcopperfield   
36       austen          ladysusan   
37   chesterton           napoleon   
38     melville           bartleby   
39   chesterton         whitehorse   
40      dickens          hardtimes   

                                                  Item         X         Y  \
0   python-text-analysis/data/dickens-olivertwist.... -0.261657 -0.141328   
1   python-text-analysis/data/melville-omoo.txt.le... -0.126564  0.141689   
2   python-text-analysis/data/austen-northanger.tx... -0.190409 -0.274343   
3   python-text-analysis/data/chesterton-brown.txt... -0.134116  0.159160   
4   python-text-analysis/data/chesterton-knewtoomu... -0.152394  0.196312   
5   python-text-analysis/data/dickens-ourmutualfri... -0.294504 -0.093461   
6    python-text-analysis/data/austen-emma.txt.lemmas -0.296076 -0.460560   
7   python-text-analysis/data/dickens-christmascar... -0.113437 -0.044181   
8   python-text-analysis/data/melville-piazzatales... -0.089402  0.120273   
9   python-text-analysis/data/melville-conman.txt.... -0.075196  0.072590   
10  python-text-analysis/data/shakespeare-muchado....  0.627768 -0.103653   
11  python-text-analysis/data/dumas-tenyearslater.... -0.100100  0.539284   
12  python-text-analysis/data/shakespeare-lear.txt...  0.727088 -0.068711   
13  python-text-analysis/data/dumas-threemusketeer... -0.069949  0.542575   
14  python-text-analysis/data/dumas-montecristo.tx... -0.151580  0.387500   
15  python-text-analysis/data/shakespeare-romeo.tx...  0.736124 -0.094139   
16  python-text-analysis/data/dickens-greatexpecta... -0.278731 -0.085005   
17  python-text-analysis/data/austen-persuasion.tx... -0.268409 -0.377253   
18  python-text-analysis/data/melville-pierre.txt....  0.239109 -0.006490   
19  python-text-analysis/data/dumas-twentyyearsaft... -0.050829  0.604854   
20  python-text-analysis/data/shakespeare-caesar.t...  0.686348 -0.073158   
21  python-text-analysis/data/chesterton-ball.txt.... -0.159020  0.163514   
22  python-text-analysis/data/austen-pride.txt.lemmas -0.286169 -0.479401   
23  python-text-analysis/data/dickens-bleakhouse.t... -0.252717 -0.138667   
24  python-text-analysis/data/melville-moby_dick.t...  0.021916  0.052027   
25  python-text-analysis/data/shakespeare-twelfthn...  0.674709 -0.097754   
26  python-text-analysis/data/melville-typee.txt.l... -0.099883  0.087663   
27  python-text-analysis/data/chesterton-thursday.... -0.139853  0.125213   
28  python-text-analysis/data/austen-sense.txt.lemmas -0.275186 -0.437264   
29  python-text-analysis/data/shakespeare-midsumme...  0.655836 -0.056070   
30  python-text-analysis/data/dickens-pickwickpape... -0.267310 -0.143600   
31  python-text-analysis/data/dumas-blacktulip.txt...  0.031298  0.013635   
32  python-text-analysis/data/shakespeare-othello....  0.668192 -0.061681   
33  python-text-analysis/data/dumas-maninironmask.... -0.082691  0.513542   
34  python-text-analysis/data/dickens-taleoftwocit... -0.206833  0.176140   
35  python-text-analysis/data/dickens-davidcopperf... -0.244019 -0.298099   
36  python-text-analysis/data/austen-ladysusan.txt... -0.189505 -0.401151   
37  python-text-analysis/data/chesterton-napoleon.... -0.128700  0.115722   
38  python-text-analysis/data/melville-bartleby.tx... -0.049262  0.012596   
39  python-text-analysis/data/chesterton-whitehors...  0.068136  0.104421   
40  python-text-analysis/data/dickens-hardtimes.tx... -0.102021 -0.192743   

            Z         W         P         Q  
0  -0.152952  0.466738  0.032626 -0.164769  
1  -0.205628 -0.306997  0.547896 -0.282132  
2   0.214874 -0.156814 -0.039271  0.007463  
3  -0.331021 -0.219545 -0.220116 -0.054240  
4  -0.308552 -0.210525 -0.310814  0.025220  
5  -0.203471  0.395555  0.036936 -0.028225  
6   0.325624 -0.187031 -0.057312  0.013975  
7  -0.133825  0.157595  0.016639  0.123924  
8  -0.199568 -0.177685  0.425317  0.347390  
9  -0.128043 -0.060443  0.124801  0.436607  
10  0.015490  0.019470 -0.026153  0.001281  
11  0.452941  0.062621 -0.031198  0.008760  
12  0.032747  0.035687 -0.021192 -0.020976  
13  0.453183  0.088892 -0.075616 -0.060784  
14  0.203878  0.010488  0.003543  0.064830  
15  0.008528  0.036364 -0.023309 -0.005139  
16 -0.197163  0.393089  0.054931  0.034316  
17  0.261359 -0.183583 -0.000913 -0.034560  
18 -0.071624 -0.049393  0.045199  0.196795  
19  0.477056  0.101589 -0.064213 -0.078628  
20  0.006221  0.024514 -0.029819 -0.023293  
21 -0.394977 -0.226490 -0.368533 -0.059509  
22  0.372770 -0.240897 -0.082395  0.032859  
23 -0.086834  0.249641 -0.004198  0.036246  
24 -0.150524 -0.139678  0.375736 -0.101435  
25  0.034841  0.025093 -0.038122 -0.028622  
26 -0.181550 -0.304281  0.467250 -0.381538  
27 -0.378756 -0.204913 -0.468720 -0.020255  
28  0.313940 -0.145599 -0.043565 -0.092208  
29  0.002653  0.018266 -0.008846 -0.011781  
30 -0.149832  0.444377  0.025271 -0.099550  
31 -0.008621  0.019624 -0.013075  0.018449  
32  0.035791  0.028953 -0.017633 -0.029310  
33  0.376603  0.045819 -0.002258 -0.046661  
34 -0.014371  0.223053  0.005351  0.106275  
35  0.051808  0.161132 -0.026630  0.003828  
36  0.339473 -0.239638 -0.082835  0.044181  
37 -0.318549 -0.185373 -0.333509 -0.048737  
38 -0.097958 -0.003455  0.186425  0.592339  
39 -0.188483 -0.154867  0.035618 -0.189420  
40 -0.077479  0.388649  0.006703 -0.232966  
```

Finally, let's save our progress so far.

```python
data.to_csv("/content/drive/My Drive/Colab Notebooks/text-analysis/data/data.csv", index=False)
data.to_xlsx("/content/drive/My Drive/Colab Notebooks/text-analysis/data/data.xlsx", index=False)
```

## Inspecting LSA Results

### Plotting

Let's plot the results, using a helper we prepared for learners. We'll focus on the X and Y topics for now to illustrate the workflow. We'll return to the other topics in our model as a further exercise.

```python
from helpers import lsa_plot
lsa_plot(data, svdmodel)
```

What do you think these X and Y axes are capturing, conceptually?

To help figure that out, lets color-code by author to see if any patterns are immediately apparent.

```python
colormap = {
    "austen": "red",
    "chesterton": "blue",
    "dickens": "green",
    "dumas": "orange",
    "melville": "cyan",
    "shakespeare": "magenta"
}

lsa_plot(data, svdmodel, groupby="Author", colors=colormap)
```

![png](LSA_files/LSA_25_0.png)

It seems that some of the books by the same author are clumping up together in our plot.

We don't know *why* they are getting arranged this way, since we don't know what more concepts X and Y correspond to. But we can work do some work to figure that out.

### Topics

Let's write a helper to get the strongest words for each topic. This will show the terms with the *highest* and *lowest* association with a topic. In LSA, each topic is a spectra of subject matter, from the kinds of terms on the low end to the kinds of terms on the high end. So, inspecting the *contrast* between these high and low terms (and checking that against our domain knowledge) can help us interpret what our model is identifying.

```python
def showTopics(topic, n):
  terms = vectorizer.get_feature_names_out()
  weights = svdmodel.components_[topic]
  df = pandas.DataFrame({"Term": terms, "Weight": weights})
  tops = df.sort_values(by=["Weight"], ascending=False)[0:n]
  bottoms = df.sort_values(by=["Weight"], ascending=False)[-n:]
  return pandas.concat([tops, bottoms])

topic_words_x = showTopics(1, 5)
topic_words_y = showTopics(2, 5)
```

You can also use a helper we prepared for learners:

```python
from helpers import showTopics
topic_words_x = showTopics(vectorizer, svdmodel, topic_number=1, n=5)
topic_words_y = showTopics(vectorizer, svdmodel, topic_number=2, n=5)
```

Either way, let's look at the terms for the X topic.

What does this topic seem to represent to you? What's the contrast between the top and bottom terms?

```python
print(topic_words_x)
```

```txt
            Term    Weight
8718        thou  0.369606
4026        hath  0.368384
3104        exit  0.219252
8673        thee  0.194711
8783         tis  0.184968
9435          ve -0.083406
555   attachment -0.090431
294           am -0.103122
5312          ma -0.117927
581         aunt -0.139385
```

And the Y topic.

What does this topic seem to represent to you? What's the contrast between the top and bottom terms?

```python
print(topic_words_y)
```

```txt
            Term    Weight
1221    cardinal  0.269191
5318      madame  0.258087
6946       queen  0.229547
4189       honor  0.211801
5746   musketeer  0.203572
294           am -0.112988
5312          ma -0.124932
555   attachment -0.150380
783    behaviour -0.158139
581         aunt -0.216180
```

Now that we have names for our first two topics, let's redo the plot with better axis labels.

```python
lsa_plot(data, svdmodel, groupby="Author", colors=colormap, xlabel="Victorian vs. Elizabethan", ylabel="English vs. French")
```

![png](LSA_files/LSA_33_0.png)

## Check Your Understanding: Intrepreting LSA Results

Finally, let's repeat this process with the other 4 topics, tentative called Z, W, P, and Q.

In the first two topics (X and Y), some authors were clearly separated, but others overlapped. If we hadn't color coded them, we wouldn't be easily able to tell them apart.

But in the next few topics, this flips, with different combinations of authors getting pulled apart and pulled together. This is because these topics (Z, W, P, and Q) highlight different features of the data, *independent* of the features we've already captured above.

Take a few moments to work through the steps above for the remaining axes of our LSA model, and chat with one another about what you think the topics being represented are.

# Word2Vec Pre-trained

## Document/Corpus Embeddings Recap

**Note to instructor:** Run the cells below to load the pretrained Word2Vec model before explaining the below text.

So far, we've seen how word counts, TF-IDF, and LSA can help us embed a document or set of documents into useful vector spaces that allow us to gain insights from text data.

Let's review the embeddings covered thus far:

- **TF-IDF embeddings:** Determines the mathematical significance of words across multiple documents. It's embedding is based on (i) token/word frequency within each document and (ii) how many documents a token appears in
- **LSA embeddings:** Latent Semantic Analysis (LSA) is used to find the hidden topics represented by a group of documents. It involves running single-value decomposition on a document-term matrix (typically the TF-IDF matrix), producing a lower-dimensional vector representation of each document. This vector scores each document's representation in different topic areas which are derived based on word co-occurences. Importantly, LSA is still considered a *bag of words* method since the order of words in a document is not considered.

## Distributional hypothesis: extracting more meaningful representations of text

Compared to TF-IDF, the text representations (a.k.a. embeddings) produced by LSA are arguably more useful since LSA can reveal some of the latent topics referenced throughout a corpus. While LSA gets closer to extracting some of the rich semantic info stored in a corpus, it is limited in the sense that it is a "bag of words" method. That is, it pays no attention to the order in which words appear to create its embedding. A linguist called JR Firth once famously said "You shall know a word by the company it keeps." This means words that repeatedly occur in similar contexts probably have similar meanings. This is often referred to as the distributional hypothesis, and this property of language is exploited in embedding models such as Word2Vec.

## Word embeddings with Word2Vec

Word2vec is a popular word embedding method that relies on the distributional hypothesis to learn meaningful word representations (ie., vectors) which can be used for an assortment of downstream analysis tasks. Unlike with TF-IDF and LSA, which are typically used to produce document and corpus embeddings, Word2Vec focuses on producing a single embedding for every word encountered in a corpus. We'll unpack the full algorithm behind Word2Vec shortly. First, let's see what we can do with word vectors.

### Gensim

We'll be using the Gensim library. The Gensim library comes with many word embedding models, including Word2Vec, GloVe, and fastText. We'll start by exploring one of the pretrained Word2Vec models. We'll discuss the other options later in this episode.

```python
import numpy as np
import gensim.downloader as api
```

Load the Word2Vec embedding model. This can take 3-10 minutes to load.

```python
wv = api.load('word2vec-google-news-300')
```

Gensim calls the pre-trained model object a keyed vectors:

```python
print(type(wv))
```

```txt
<class 'gensim.models.keyedvectors.KeyedVectors'>
```

In this model, each word has a 300-dimensional representation:

```python
 # ("") 
print(wv['whale'].shape)
```

```txt
(300,)
```

How is *whale* represented?

```python
wv['whale']
```

```txt
array([ 0.08154297,  0.41992188, -0.44921875, -0.01794434, -0.24414062,
        -0.21386719, -0.16796875, -0.01831055,  0.32421875, -0.09228516,
        -0.11523438, -0.5390625 , -0.00637817, -0.41601562, -0.02758789,
        0.04394531, -0.15039062, -0.05712891, -0.03344727, -0.10791016,
        0.14453125,  0.17480469,  0.18847656,  0.02282715, -0.05688477,
        -0.13964844,  0.01379395,  0.296875  ,  0.53515625, -0.2421875 ,
        -0.22167969,  0.23046875, -0.20507812, -0.23242188,  0.0123291 ,
        0.14746094, -0.12597656,  0.25195312,  0.17871094, -0.00106812,
        -0.07080078,  0.10205078, -0.08154297,  0.25390625,  0.04833984,
        -0.11230469,  0.11962891,  0.19335938,  0.44140625,  0.31445312,
        -0.06835938, -0.04760742,  0.37890625, -0.18554688, -0.03063965,
        -0.00386047,  0.01062012, -0.15527344,  0.40234375, -0.13378906,
        -0.00946045, -0.06103516, -0.08251953, -0.44335938,  0.29101562,
        -0.22753906, -0.29296875, -0.13671875, -0.08349609, -0.25585938,
        -0.12060547, -0.16113281, -0.27734375,  0.01318359, -0.23730469,
        0.0300293 ,  0.01348877, -0.07226562, -0.02429199, -0.18945312,
        0.05419922, -0.12988281,  0.26953125, -0.11669922,  0.01000977,
        0.05883789, -0.03515625, -0.09375   ,  0.35742188, -0.1875    ,
        -0.06347656,  0.44726562,  0.05761719,  0.3125    ,  0.06347656,
        -0.24121094,  0.3125    ,  0.31054688,  0.11132812, -0.08447266,
        0.06445312, -0.02416992,  0.16113281, -0.1875    ,  0.2109375 ,
        -0.05981445,  0.00524902,  0.13964844,  0.09765625,  0.06835938,
        -0.43945312,  0.01904297,  0.33007812,  0.12011719,  0.08251953,
        -0.08642578,  0.02270508, -0.09472656, -0.21289062,  0.01092529,
        -0.05493164,  0.0625    , -0.0456543 ,  0.06347656, -0.14160156,
        -0.11523438,  0.28125   , -0.09082031, -0.46679688,  0.11035156,
        0.07275391,  0.12988281, -0.32421875,  0.10595703,  0.13085938,
        -0.29101562,  0.02880859,  0.07568359, -0.03637695,  0.16699219,
        0.15917969, -0.08007812,  0.109375  ,  0.4140625 ,  0.30859375,
        0.22558594, -0.22070312,  0.359375  ,  0.08105469,  0.21386719,
        0.59765625,  0.01782227, -0.5859375 ,  0.21777344,  0.18164062,
        -0.08398438,  0.07128906, -0.27148438, -0.11230469, -0.00915527,
        0.10400391,  0.19628906,  0.09912109,  0.09667969,  0.24414062,
        -0.11816406,  0.02758789, -0.26757812, -0.07421875,  0.20410156,
        -0.140625  , -0.03515625,  0.22265625,  0.32226562, -0.18066406,
        -0.30078125, -0.05981445,  0.34765625, -0.2578125 ,  0.0546875 ,
        -0.05541992, -0.46289062, -0.18945312,  0.00668335,  0.15429688,
        0.07275391,  0.07373047, -0.07275391,  0.09765625,  0.03491211,
        -0.33203125, -0.14257812, -0.23046875, -0.13085938, -0.0035553 ,
        0.28515625,  0.25390625, -0.05102539,  0.01318359, -0.16113281,
        0.12353516, -0.39257812, -0.42578125, -0.2578125 , -0.15332031,
        -0.01403809,  0.21972656, -0.04296875,  0.04907227, -0.328125  ,
        -0.46484375,  0.00546265,  0.17089844, -0.10449219, -0.38476562,
        0.13378906,  0.65625   , -0.22363281,  0.15039062,  0.19824219,
        0.3828125 ,  0.10644531,  0.38671875, -0.11816406, -0.00616455,
        -0.19628906,  0.04638672,  0.20507812,  0.36523438,  0.04174805,
        0.45117188, -0.29882812, -0.09228516, -0.31835938,  0.15234375,
        -0.07421875,  0.07128906,  0.25195312,  0.14746094,  0.27148438,
        0.4609375 , -0.4375    ,  0.10302734, -0.49414062, -0.01342773,
        -0.20019531,  0.0456543 ,  0.0402832 , -0.11181641,  0.01489258,
        -0.7421875 , -0.0055542 , -0.21582031, -0.15527344,  0.29296875,
        -0.05981445,  0.02905273, -0.08105469, -0.03955078, -0.17089844,
        0.07080078,  0.00671387, -0.17285156,  0.08544922, -0.11621094,
        0.10253906, -0.24316406, -0.04882812,  0.20410156, -0.27929688,
        -0.21484375,  0.07470703,  0.11767578,  0.6640625 ,  0.29101562,
        0.02404785, -0.65234375,  0.13378906, -0.01867676, -0.07373047,
        -0.18359375, -0.0201416 ,  0.29101562,  0.06640625,  0.04077148,
        -0.10888672,  0.15527344,  0.12792969,  0.375     ,  0.2890625 ,
        0.30078125, -0.15625   , -0.05224609, -0.19042969,  0.10595703,
        0.078125  ,  0.29882812,  0.34179688,  0.04248047,  0.03442383],
      dtype=float32)
```

Once we have our words represented as vectors (of length 300, in this case), we can start using some math to gain additional insights. For instance, we can compute the cosine similarity between two different word vectors using Gensim's similarity function.

How similar are the two words, *whale* and *dolphin*?

```python
wv.similarity('whale','dolphin')
```

```txt
0.77117145
```

How about *whale* and *fish*?

```python
wv.similarity('whale','fish')
```

```txt
0.55177623
```

How about *whale* and...*potato*?

```python
wv.similarity('whale','potato')
```

```txt
0.15530972
```

Our similarity scale seems to be on the right track. Let's take a look at the top 10 words associated with *whale*.

```python
print(wv.most_similar(positive=['whale'],topn=10))
```

```txt
[('whales', 0.8474178910255432), ('humpback_whale', 0.7968777418136597), ('dolphin', 0.7711714506149292), ('humpback', 0.7535837292671204), ('minke_whale', 0.7365031838417053), ('humpback_whales', 0.7337379455566406), ('dolphins', 0.7213870882987976), ('humpbacks', 0.7138717174530029), ('shark', 0.7011443376541138), ('orca', 0.7007412314414978)]
```

Based on our ability to recover similar words, it appears the Word2Vec embedding method produces good (i.e., semantically meaningful) word representations.

### Adding and Subtracting Vectors: King - Man + Woman = Queen

We can also add and subtract word vectors to reveal latent meaning in words. As a canonical example, let's see what happens if we take the word vector representing *King*, subtract the *Man* vector from it, and then add the *Woman* vector to the result. We should get a new vector that closely matches the word vector for *Queen*. We can test this idea out in Gensim with:

```python
print(wv.most_similar(positive=['woman','king'], negative=['man'],topn=3))
```

```txt
[('queen', 0.7118193507194519), ('monarch', 0.6189674139022827), ('princess', 0.5902431011199951)]
```

### Visualizing word vectors with PCA

Similar to how we visualized our texts in the previous lesson to show how they relate to one another, we can visualize how a sample of our words relate.

First, let's produce a 2-dimensional representation:

```python
from sklearn.decomposition import PCA
words = ['man','woman','boy','girl','king','queen','prince','princess']
sample_vectors = np.array([wv[word] for word in words])
pca = PCA(n_components=2)
result = pca.fit_transform(sample_vectors)
```

```python
array([[-1.8593105 , -1.6255254 ],
        [-2.349247  ,  0.27667248],
        [-2.4419675 , -0.47628874],
        [-2.6664796 ,  1.0604485 ],
        [ 2.6521494 , -1.9826698 ],
        [ 1.8861336 ,  0.81828904],
        [ 2.8712058 , -0.69794625],
        [ 1.9075149 ,  2.627021  ]], dtype=float32)
```

And second, let's visualize the results:

```python
import matplotlib.pyplot as plt
plt.scatter(result[:,0], result[:,1])
for i, word in enumerate(words):
  plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.show()
```

![png](WordEmbeddingsIntro_files/WordEmbeddingsIntro_25_0.png)

## Unpacking the Word2Vec Algorithm

How is it that word2vec is able to represent words in such a semantically meaningful way? There are two similar approaches to train a Word2Vec model, and both resulting in meaningful word vectors:

**TODO** Image from Word2Vec research paper, by Mikolov et al

- **Continuous Bag of Words (CBOW)**: The "Continuous Bag of Words" training method takes as an input the words before and after our target word, and tries to guess our target word based on those words.
- **Skip-gram (SG)**: The "skipgram" method flips the task, taking as an input the one target word and trying to predict the surrounding context words.

### Training process

Each time either above task is done, the embeddings (artificial neural network weights, in this case) are slightly adjusted to match the correct answer from the corpus. Word2Vec also selects random words from our corpus that are not related and asks the model to predict that these words are unrelated, in a process called "negative sampling." Negative sampling ensures unrelated words will have embeddings that drift further and further apart, while the standard two tasks bring related embeddings closer together.

Both methods use artificial neural networks as their classification algorithm. Word2vec starts by randomly initializing its embeddings (weights of the neural network models) for all words in the vocabulary. Before the training process, these dimensions are meaningless and the embeddings do not work very well for any task. However, Word2Vec will gradually adjust the neural network weights to get better performance on the prediction task (ie., predicting the missing word or predicting surrounding words in CBOW and SG, respectivitely).

After training on a sufficiently large corpus of text, the neural network will perform well on this prediction task. Once trained, the weights of the model can be used to represent meaningful underlying properties of our words.

Up next, we'll train a model of our own from scratch.

# Word2Vec from Scratch - TODO

## Load in the data

We'll start by fitting a word2vec model to just one of the books in our list, *Moby Dick*.

First, let's continue where we left off in our analysis and get the filename for *Moby Dick*.

```python
from pandas import read_csv
data = read_csv("/content/drive/My Drive/Colab Notebooks/text-analysis/data/data.csv")
single_file = data.loc[data['Title'] == 'moby_dick', 'File'].item()
print(single_file)
```

```txt
'/content/drive/My Drive/Colab Notebooks/text-analysis/data/books/melville-moby_dick.txt'
```

Second, let's load the file contents and inspect a portion of it:

```python
with open(single_file,'r') as f:
  file_contents = f.read()

preview_len = 500
print(file_contents[0:preview_len])
```

```txt
[Moby Dick by Herman Melville 1851]


ETYMOLOGY.

(Supplied by a Late Consumptive Usher to a Grammar School)

The pale Usher--threadbare in coat, heart, body, and brain; I see him
now.  He was ever dusting his old lexicons and grammars, with a queer
handkerchief, mockingly embellished with all the gay flags of all the
known nations of the world.  He loved to dust his old grammars; it
somehow mildly reminded him of his mortality.

"While you take in hand to school others, and to teach them by wha
```

## Restarting the Interpretive Loop

Earlier, we preprocessed our books in a way that removed its underlying structure of words in sentences in paragraphs in chapters.

However, in order to train a model to capture the meaning of words better, we'll need to return re-preprocess our data, this time in a way that retains the sentence-level organization of our data.

Our steps will be:

1. Split the book into sentences, using `punkt`
2. Tokenize, lemmatize, and lowercase the text in each sentence, removing stop words, using the tokenizer we built earlier

### Convert Book to List of Sentences

To split our text into sentences, we will use NLTK's `punkt` sentence tokenizer. This model does a decent job for most applications, even for occasionally complex sentences like we see in the texts in our corpus. However, with very complex paragraphs with lots of punctuation, it can return incorrect results. Still, on data like ours, its errors should be rare enough to not impact our end results.

```python
import nltk
nltk.download('punkt') # dependency of sent_tokenize function
sentences = nltk.sent_tokenize(file_contents)
sentences[300:305]
```

```txt
['How then is this?',
  'Are the green fields gone?',
  'What do they\nhere?',
  'But look!',
  'here come more crowds, pacing straight for the water, and\nseemingly bound for a dive.']
```

### Should We Always Remove Stopwords?

"It clearly makes sense to consider 'not' as a stop word if your task is based on word frequencies (e.g. tfâ€"idf analysis for document classification).

If you're concerned with the context (e.g. sentiment analysis) of the text it might make sense to treat negation words differently. Negation changes the so-called valence of a text. This needs to be treated carefully and is usually not trivial. One example would be the Twitter negation corpus. An explanation of the approach is given in this paper."

"Do we always remove stop words? Are they always useless for us? ğŸ™‹â€�â™€ï¸�
The answer is no! ğŸ™…â€�â™‚ï¸�

We do not always remove the stop words. The removal of stop words is highly dependent on the task we are performing and the goal we want to achieve. For example, if we are training a model that can perform the sentiment analysis task, we might not remove the stop words.

Movie review: â€œThe movie was not good at all.â€�

Text after removal of stop words: â€œmovie goodâ€�"

```python
from nltk.corpus import stopwords
nltk.download('stopwords')
print(stopwords.words("english")) # note negatives and additional context cue words, e.g., before/during/after, above/below, 'not/nor/no
```

```
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
```

### Process Each Sentence

word net is faster, production grade

```python
import spacy 
import en_core_web_sm
spacyt = spacy.load("en_core_web_sm")
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

```python
tokenizer = Our_Tokenizer()
tokens = tokenizer.tokenize(sentence)
print(tokens)
```

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

**Set seed and workers for a fully deterministic run**: Next we'll set some parameters for reproducibility. We'll set the seed so that our vectors get randomly initialized the same way each time this code is run. For a fully deterministically-reproducible run, we'll also limit the model to a single worker thread (workers=1), to eliminate ordering jitter from OS thread scheduling â€" noted in [gensim's documentation](https://radimrehurek.com/gensim/models/word2vec.html)

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

There is at least one sea creature missing from our list â€" a giant squid. The giant squid is only mentioned briefly in Moby Dick, and therefore it could be that our word2vec model was not able to train a good representation of the word "squid".

Think about how rarely occuring or novel entities such as these might be found. In a later lesson, we will explore a task called Named Entity Recognition, which will handle this type of task in a more robust and systematic way.

When using word2vec to reveal items from a category, you risk missing items that are rarely mentioned. For this reason, it's sometimes better to save this task for larger text corpuses, or more widely pretrained models.

#### Exercise? Additional questions you could explore using this method

- **Example**: Train a model on newspaper articles from the 19th century, and collect a list of foods (the topic chosen) referenced throughout the corpus. Do the same for 20th century newspaper articles and compare.

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





# Transformers and BERT

## What are large language models? What is BERT?

For this lesson, we will be learning about large language models (LLMs).

LLMs are the current state of the art when it comes to many tasks, and although LLMs can differ, they are mostly based on a similar architecture to one another.

We will go through the architecture of a highly influential LLM called BERT. BERT stands for Bidirectional Encoder Representations from Transformers. Let's look at each part of this model, starting with the input on the bottom and working toward the output on the top.

![transformers.jpeg](TODO)

This is a complex architecture, but it can be broken down into many of the things we've learned in this course. The model is displayed with the input on the bottom and the output at the top, as is common with neural networks. Let's take a look at one component at a time.

### Tokenizer

First, the input string is broken up by a tokenizer. The tokens are created using a tokenizer which breaks words into smaller lexical units called morphemes rather than words. There are two special types of tokens in this tokenizer. The [CLS] token indicates the start of the document. The [SEP] token indicates the end of a document, or a "segment". Let's look at the tokenizer in action.

```python
!pip install parse bertviz transformers
```

```python
from transformers import BertTokenizer
sentence="My dog Fido is cute. He likes playing."
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
encoding = tokenizer.encode(sentence)
      
#this will give us the ID number of our tokens.
print(encoding)
#this will give us the actual tokens.
print(tokenizer.convert_ids_to_tokens(encoding))
```

```txt
[101, 1422, 3676, 17355, 2572, 1110, 10509, 119, 1124, 7407, 1773, 119, 102]
['[CLS]', 'My', 'dog', 'Fi', '##do', 'is', 'cute', '.', 'He', 'likes', 'playing', '.', '[SEP]']
```

### Embeddings

![embeddings2.jpg](TODO)

Next the model calculates an embedding for each token. Three values are used to calculate our final embedding for each token. The first part is the token embedding, similar to the ones we have discussed with Word2Vec and Glove, only this embedding is trained by the BERT model. For BERT, this algorithm is called WordPiece. The second part is a combination of all the tokens in a given segment, also called a segment embedding. The third part is a positional embedding, which accounts for the locations of words in the document. All three parts are combined as the embedding that is fed into the model. This is how we get our initial input into the model.

### Transformers

Now we are ready to use the main component of BERT: transformers.

Transformers were developed in 2017 by a group of researchers working at Google Brain. This was a revolutionary component that allowed language models to consider all words in a document at the same time in parellel, which sped up model training considerably and opened the door to LLMs.

Transformers make use of something called "self-attention calculation," which mimics how humans focus in on multiple parts of the document and weigh them differently when considering the meaning of a word. Self attention not only factors in the embeddings from other words in the sentence but weighs them depending on their importance.

It is not necessary to understand the exact details of the calculation for this lesson, but if you are interested on the mathematics of self-attention and the details of the calculation, the Illustrated Transformer is an excellent resource: <https://jalammar.github.io/illustrated-transformer/>

![Attention-Heads.jpg](TODO)

You can see in our BERT diagram that each embedding of the word is fed into a transformer called an 'encoder.' Each encoder in a layer runs a self attention calculation and forwards the results to each encoder in the next layer, which runs them on the outputs of the layer before it. Once the attention calculation has been run for each layer, a sophisticated embedding for each input token is output.

One additional detail about this process: it does not happen for just one set of weights. Instead, several independent copies of these encoders are trained and used, all at the same time. Each set of these transformers is called an "attention head."

Each attention head has its own set of weights called parameters that are trained and calculated independently. They are trained using the same type of cloze tasks to fill in masked words that we used to train Word2Vec. All of the outputs of the attention heads are combined together to make a large matrix of values that represents a very robust representation of the input, which we have labelled "T."

Let's take a look at how attention works in an example. Imagine we have two sentences, "The chicken didn't cross the road because it was too tired," and, "The chicken didn't cross the road because it was too wide." These are very similar sentences, but changing one word changes the meaning of both sentences dramatically. For the first sentence, 'it was too tired' refers to the chicken. For the second sentence, 'it was too wide' refers to the road. Ideally our representations for the road or chicken will incorporate these attributes.

```python
import attentionviz as av
sentence_a = "The chicken didn't cross the road because it was too tired"
sentence_b = "The chicken didn't cross the road because it was too wide"
tfviz = av.AttentionViz(sentence_a, sentence_b)
tfviz.hview()
```

This visualization shows how attention works in the BERT model. The different colors represent different attention heads. The left side represents the input embedding and the depth of color shows how much each input weighs in the output of that layer.

Select "Sentence A to Sentence A" on the attention dropdown and mouse over the word "it." In layers 0-7 we can see how different attention heads start to incorporate the embedding of "because" and "too tired" into our embedding for "it." Once we get to layers 8-10, we can see how "chicken" starts to attend to the word "it", indicating that the model has started to incorporate the qualities of being "too tired" that are already part of "it" into the representation for "chicken". Mousing over the word "it" we can also see that it starts to incorporate the embedding built into the word "chicken."

How do you suppose the self attention calculation will change for sentence B? If we look at layers 8-10 we see that "the road" starts to attend to the word "it", rather than the chicken doing so. Self attention can shift to focus on different words depending on the input.

### Output and Classification

![linear-layer.jpg](TODO)

Once the input embeddings have been run through each layer and attention head, all of the outputs are combined together to give us a very robust matrix of values that represent a word and its relationships to other words, which we've called T. Training this component of the model is the vast majority of the work done in creating a pretrained large language model. But now that we have a very complex representation of the word, how do we use it to accomplish a task?

The last step in BERT is the classification layer. During fine-tuning, we add one more layer to our model- a set of connections that calculate the probability that each transformed token T matches each possible output. A much smaller set of test data is then used to train the final layer and refine the other layers of BERT to better suit the task.

## The Power of Transfer Learning

![bert-fine.png](TODO)

Above is a set of images from the creators of BERT showing how it could be easily adapted to different tasks. One of the reasons BERT became so ubiquitous is that it was very effective at transfer learning. Transfer learning means that the underlying model can be repurposed for different tasks.

The underlying large language model for BERT was trained for thousands of compute hours on hundreds of millions of words, but the weights calculated can be reused on a variety of tasks with minimal adaptation. The model does get fine-tuned for each task, but this is much easier than the initial training.

When we adapt BERT for a given NER task, we just need to provide a much smaller set of labelled data to retrain the last step of converting our output into a set of probabilities. These models have had great success at a variety of tasks like parts of speech tagging, translation, document summary, and NER labelling.

State of the art LLMs like GPT-4 operate on this approach. LLMs have grown larger and larger to take advantage of the ability to compute in parallel. Modern LLMs have become so large that they are often run on specialied high performance machines, and only exposed to the public via API. They are scaled up versions, still using transformers as their primary component. LLM's have also become better at so called "zero-shot" tasks, where there is no fine-tuning phase, and instead the model is exposed to novel classes it has never seen outside of its test data. However, fine-tuning is still an important part of maximizing performance.

At the beginning of our carpentries lessons, we used pretrained HuggingFace models to learn about different tasks we could accomplish using NLP. In our next lesson we will fine tune BERT to perform a custom task using a custom dataset.

# Finetuning BERT

## Setup

If you are running this lesson on Google Colab, it is strongly recommended that you enable GPU acceleration. If you are running locally without CUDA, you should be able to run most of the commands, but training will take a long time and you will want to use the pretrained model when using it.

To enable GPU on Colab, click "Edit > Notebook settings" and select GPU. If enabled, this command will return a status window and not an error:

```python
!nvidia-smi
```

```txt
Thu Apr 13 20:14:48 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   63C    P8    14W /  70W |      0MiB / 15360MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                                
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

## Fine Tuning

There are many many prebuilt models for BERT. Why would you want to go through the trouble of training or fine tuning your own?

Perhaps you are looking to do something for which there is no prebuilt model. Or perhaps you simply want better performance based on your own dataset, as training on tasks similar to your dataset will improve performance. For these reasons, you may want to fine tune the original BERT model on your own data. Let's discuss how we might do this using an example.

The standard set of NER labels is designed to be broad: people, organizations and so on. However, it doesn't have to be. We can define our own entities of interest and have our model search for them. For this example, we'll use the task of classifying different elements of restaurant reviews, such as amenities, locations, ratings, cuisine types and so on. How do we start?

The first thing we can do is identify our task. Our task here is Token Classification, or more specifically, Named Entity Recognition. Now that we have an idea of what we're aiming to do, lets look at some of the LLMs provided by HuggingFace.

One special note for this lesson: we will not be writing the code for this from scratch. Doing so is a tough task. Rather, this lesson will focus on creating our own data, adapting existing code and modifying it to achieve the task we want to accomplish.

HuggingFace hosts many instructional Colab notebooks available at: <https://huggingface.co/docs/transformers/notebooks>. We can find an example of Token Classification using PyTorch there which we will modify to suit our needs. Looking at the notebook, we can see it uses a compressed version of BERT, "distilbert." We'll use this model as well.

## Examining our Data

Now, let's take a look at the example data from the dataset used in the example. The dataset used is called the CoNLL2003 dataset.

If possible, it's a good idea to pattern your data output based on what the model is expecting. You will need to make adjustments, but if you have selected a model that is appropriate to the task you can reuse most of the code already in place. We'll start by installing our dependencies.

```python
! pip install datasets transformers seqeval
```

Next, let's look at the CONLL dataset in particular.

```python
from datasets import load_dataset, load_metric
ds = load_dataset("conll2003")
print(ds)
```

```txt
DatasetDict({
    train: Dataset({
        features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
        num_rows: 14041
    })
    validation: Dataset({
        features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
        num_rows: 3250
    })
    test: Dataset({
        features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
        num_rows: 3453
    })
})
```

We can see that the CONLL dataset is split into three sets, training data, validation data, and test data. Training data should make up about 80% of your corpus and is fed into the model to fine tune it. Validation data should be about 10%, and is used to check how the training progress is going as the model is trained. The test data is about 10% withheld until the model is fully trained and ready for testing, so you can see how it handles new documents that the model has never seen before.

Let's take a closer look at a record in the train set so we can get an idea of what our data should look like. The NER tags are the ones we are interested in, so lets print them out and take a look. We'll also select the dataset and then an index for the document to look at an example.

```python
ds["train"][0]
conll_tags = ds["train"].features[f"ner_tags"]
print(conll_tags)
print(ds["train"][0])
```

```txt
Sequence(feature=ClassLabel(names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'], id=None), length=-1, id=None)
{'id': '0', 'tokens': ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'], 'pos_tags': [22, 42, 16, 21, 35, 37, 16, 21, 7], 'chunk_tags': [11, 21, 11, 12, 21, 22, 11, 12, 0], 'ner_tags': [3, 0, 7, 0, 0, 0, 7, 0, 0]}
```

Each document has it's own ID number. We can see that the tokens are a list of words in the document. For each word in the tokens, there are a series of numbers. Those numbers correspond to the labels in the database. Based on this, we can see that the EU is recognized as an ORG and the terms "German" and "British" are labelled as MISC.

These datasets are loaded using specially written loading scripts. We can look this script by searching for the 'conll2003' in huggingface and selecting "Files". The loading script is always named after the dataset. In this case it is "conll2003.py".

<https://huggingface.co/datasets/conll2003/blob/main/conll2003.py>

Opening this file up, we can see that a zip file is downloaded and text files are extracted. We can manually download this ourselves if we would really like to take a closer look. For the sake of convienence, the example we looked just looked at is reproduced below:

```txt
-DOCSTART- -X- -X- O

EU NNP B-NP B-ORG
rejects VBZ B-VP O
German JJ B-NP B-MISC
call NN I-NP O
to TO B-VP O
boycott VB I-VP O
British JJ B-NP B-MISC
lamb NN I-NP O
. . O O
```

This is a simple format, similar to a CSV. Each document is seperated by a blank line. The token we look at is first, then space seperated tags for POS, chunk_tags and NER tags. Many of the token classifications use BIO tagging, which specifies that "B" is the beginning of a tag, "I" is inside a tag, and "O" means that the token outside of our tagging schema.

So, now that we have an idea of what the HuggingFace models expect, let's start thinking about how we can create our own set of data and labels.

## Tagging a dataset

Most of the human time spent training a model will be spent pre-processing and labelling data. If we expect our model to label data with an arbitrary set of labels, we need to give it some idea of what to look for. We want to make sure we have enough data for the model to perform at a good enough degree of accuracy for our purpose. Of course, this number will vary based on what level of performance is "good enough" and the difficulty of the task. While there's no set number, a set of approximately 100,000 tokens is enough to train many NER tasks.

Fortunately, software exists to help streamline the tagging process. One open source example of tagging software is Label Studio. However, it's not the only option, so feel free to select a data labelling software that matches your preferences or needs for a given project. An online demo of Label Studio is available here: <https://labelstud.io/playground>. It's also possible to install locally, although be aware you will need to create new Conda environment to do so.

Select "Named Entity Recognition" as the task to see what the interface would look like if we were doing our own tagging. We can define our own labels by copying in the following code (minus the quotations):

```txt
<View>
  <Labels name="label" toName="text">
    <Label value="Amenity" background="red"/>
    <Label value="Cuisine" background="darkorange"/>
    <Label value="Dish" background="orange"/>
    <Label value="Hours" background="green"/>
    <Label value="Location" background="darkblue"/>
    <Label value="Price" background="blue"/>
    <Label value="Rating" background="purple"/>
    <Label value="Restaurant_Name" background="#842"/>
  </Labels>

  <Text name="text" value="$text"/>
</View>
```

In Label Studio, labels can be applied by hitting a number on your keyboard and highlighting the relevant part of the document. Try doing so on our example text and looking at the output.

Once done, we will have to export our files for use in our model. Label Studio supports a number of different types of labelling tasks, so you may want to use it for tasks other than just NER.

One additional note: There is a github project for direct integration between label studio and HuggingFace available as well. Given that the task selected may vary on the model and you may not opt to use Label Studio for a given project, we will simply point to this project as a possible resource (<https://github.com/heartexlabs/label-studio-transformers>) rather than use it in this lesson.

## Export to desired format

So, let's say you've finished your tagging project. How do we get these labels out of label studio and into our model?

Label Studio supports export into many formats, including one called CoNLL2003. This is the format our test dataset is in. It's a space seperated CSV, with words and their tags.

We'll skip the export step as well, as we already have a prelabeled set of tags in a similar format published by MIT. For more details about supported export formats consult the help page for Label Studio here: <https://labelstud.io/guide/export.html>

At this point, we've got all the labelled data we want. We now need to load our dataset into HuggingFace and then train our model. The following code will be largely based on the example code from HuggingFace, substituting in our data for the CoNLL data.

## Loading the custom dataset

Lets make our own tweaks to the HuggingFace colab notebook. We'll start by importing some key metrics.

```python
import datasets 
from datasets import load_dataset, load_metric, Features
```

The HuggingFace example uses [CONLL 2003 dataset](https://www.aclweb.org/anthology/W03-0419.pdf).

All datasets from huggingface are loaded using scripts. Datasets can be defined from a JSON or CSV file (see the [Datasets documentation](https://huggingface.co/docs/datasets/loading_datasets.html#from-local-files)) but selecting CSV will by default create a new document for every token and NER tag and will not load the documents correctly. So we will use a tweaked version of the Conll loading script instead. Let's take a look at the regular Conll script first:

<https://huggingface.co/datasets/conll2003/tree/main>

The loading script is the python file. Usually the loading script is named after the dataset in question. There are a couple of things we want to change:

1. We want to tweak the metadata with citations to reflect where we got our data. If you created the data, you can add in your own citation here.
2. We want to define our own categories for NER_TAGS, to reflect our new named entities.
3. The order for our tokens and NER tags is flipped in our data files.
4. Delimiters for our data files are tabs instead of spaces.
5. We will replace the method names with ones appropriate for our dataset.

Those modifications have been made in the `mit_restaurants.py` file we prepared for learners. Let's briefly take a look at that file before we proceed with the huggingface script.

Now that we have a modified huggingface script, let's load our data.

```python
ds = load_dataset("/content/drive/My Drive/Colab Notebooks/text-analysis/code/mit_restaurants.py")
```

How does our dataset compare to the CONLL dataset? Let's look at a record and compare.

```python
ds
```

```txt
DatasetDict({
    train: Dataset({
        features: ['id', 'tokens', 'ner_tags'],
        num_rows: 7660
    })
    validation: Dataset({
        features: ['id', 'tokens', 'ner_tags'],
        num_rows: 815
    })
    test: Dataset({
        features: ['id', 'tokens', 'ner_tags'],
        num_rows: 706
    })
})
```

```python
label_list = ds["train"].features["ner_tags"].feature.names
label_list
```

```txt
['O',
'B-Amenity',
'I-Amenity',
'B-Cuisine',
'I-Cuisine',
'B-Dish',
'I-Dish',
'B-Hours',
'I-Hours',
'B-Location',
'I-Location',
'B-Price',
'I-Price',
'B-Rating',
'I-Rating',
'B-Restaurant_Name',
'I-Restaurant_Name']
```

Our data looks pretty similar to the CONLL data now. This is good since we can now reuse many of the methods listed by HuggingFace in their Colab notebook.

## Preprocessing the data

We start by defining some variables that HuggingFace uses later on.

```python
import torch
task = "ner" # Should be one of "ner", "pos" or "chunk"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Next, we create our special BERT tokenizer.

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

And we'll use it on an example:

```python
example = ds["train"][4]
tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
print(tokens)
```

```txt
['[CLS]', 'a', 'great', 'lunch', 'spot', 'but', 'open', 'till', '2', 'a', 'm', 'pass', '##im', '##s', 'kitchen', '[SEP]']
```

Since our words are broken into just words, and the BERT tokenizer sometimes breaks words into subwords, we need to retokenize our words. We also need to make sure that when we do this, the labels we created don't get misaligned. More details on these methods are available through HuggingFace, but we will simply use their code to do this.

```python
word_ids = tokenized_input.word_ids()
aligned_labels = [-100 if i is None else example[f"{task}_tags"][i] for i in word_ids]
label_all_tokens = True

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = ds.map(tokenize_and_align_labels, batched=True)
print(tokenized_datasets)
```

```txt
DatasetDict({
    train: Dataset({
        features: ['id', 'tokens', 'ner_tags', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 7660
    })
    validation: Dataset({
        features: ['id', 'tokens', 'ner_tags', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 815
    })
    test: Dataset({
        features: ['id', 'tokens', 'ner_tags', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 706
    })
})
```

The preprocessed features we've just added will be the ones used to actually train the model.

## Fine-tuning the model

Now that our data is ready, we can download the LLM. Since our task is token classification, we use the `AutoModelForTokenClassification` class, but this will vary based on the task. Before we do though, we want to specify the mapping for ids and labels to our model so it does not simply return CLASS_1, CLASS_2 and so on.

```python
id2label = {
    0: "O",
    1: "B-Amenity",
    2: "I-Amenity",
    3: "B-Cuisine",
    4: "I-Cuisine",
    5: "B-Dish",
    6: "I-Dish",
    7: "B-Hours",
    8: "I-Hours",
    9: "B-Location",
    10: "I-Location",
    11: "B-Price",
    12: "I-Price",
    13: "B-Rating",
    14: "I-Rating",
    15: "B-Restaurant_Name",
    16: "I-Restaurant_Name",
}

label2id = {
    "O": 0,
    "B-Amenity": 1,
    "I-Amenity": 2,
    "B-Cuisine": 3,
    "I-Cuisine": 4,
    "B-Dish": 5,
    "I-Dish": 6,
    "B-Hours": 7,
    "I-Hours": 8,
    "B-Location": 9,
    "I-Location": 10,
    "B-Price": 11,
    "I-Price": 12,
    "B-Rating": 13,
    "I-Rating": 14,
    "B-Restaurant_Name": 15,
    "I-Restaurant_Name": 16,
}
```

```python
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, id2label=id2label, label2id=label2id, num_labels=len(label_list)).to(device)
```

```txt
Some weights of the model checkpoint at distilbert-base-uncased were not used when initializing DistilBertForTokenClassification: ['vocab_projector.bias', 'vocab_projector.weight', 'vocab_layer_norm.bias', 'vocab_transform.weight', 'vocab_transform.bias', 'vocab_layer_norm.weight']
- This IS expected if you are initializing DistilBertForTokenClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing DistilBertForTokenClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of DistilBertForTokenClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

The warning is telling us we are throwing away some weights. The reason we are doing this is because we want to throw away that final classification layer in our fine-tuned BERT model, and replace it with a layer that uses the labels that we have defined.

Next, we configure our trainer. The are lots of settings here but largely the defaults are fine. More detailed documentation on what each of these mean are available through Huggingface:  [`TrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments),

```python
model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    #f"{model_name}-finetuned-{task}",
    f"{model_name}-carpentries-restaurant-ner",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    #push_to_hub=True, #You can have your model automatically pushed to HF if you uncomment this and log in.
)
```

One finicky aspect of the model is that all of the inputs have to be the same size. When the sizes do not match, something called a data collator is used to batch our processed examples together and pad them to the same size.

```python
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer)
```

The last thing we want to define is the metric by which we evaluate how our model did. We will use [`seqeval`](https://github.com/chakki-works/seqeval), but the metric used will vary based on the task.

```python
metric = load_metric("seqeval")
labels = [label_list[i] for i in example[f"{task}_tags"]]
metric.compute(predictions=[labels], references=[labels])
```

```txt
{'Hours': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'number': 1},
'Restaurant_Name': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'number': 1},
'overall_precision': 1.0,
'overall_recall': 1.0,
'overall_f1': 1.0,
'overall_accuracy': 1.0}
```

Per HuggingFace, we need to do a bit of post-processing on our predictions. The following function and description is taken directly from HuggingFace. The function does the following:

- Selected the predicted index (with the maximum logit) for each token
- Converts it to its string label
- Ignore everywhere we set a label of -100

```python
import numpy as np

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
```

Finally, after all of the preparation we've done, we're ready to create a Trainer to train our model.

```python
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

We can now finetune our model by just calling the `train` method. Note that this step will take about 5 minutes if you are running it on a GPU, and 4+ hours if you are not.

```python
print("Training starts NOW")
trainer.train()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>No log</td>
      <td>0.345723</td>
      <td>0.741262</td>
      <td>0.785096</td>
      <td>0.762550</td>
      <td>0.897648</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.619500</td>
      <td>0.304476</td>
      <td>0.775332</td>
      <td>0.812981</td>
      <td>0.793710</td>
      <td>0.907157</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.294300</td>
      <td>0.297899</td>
      <td>0.778443</td>
      <td>0.812500</td>
      <td>0.795107</td>
      <td>0.909409</td>
    </tr>
  </tbody>
</table>

```txt
TrainOutput(global_step=1437, training_loss=0.3906847556266174, metrics={'train_runtime': 85.9468, 'train_samples_per_second': 267.375, 'train_steps_per_second': 16.72, 'total_flos': 117366472959168.0, 'train_loss': 0.3906847556266174, 'epoch': 3.0})
```

We've done it! We've fine-tuned the model for our task. Now that it's trained, we want to save our work so that we can reuse the model whenever we wish. A saved version of this model has also been published through huggingface, so if you are using a CPU, skip the remaining evaluation steps and launch a new terminal so you can participate.

```python
trainer.save_model("/content/drive/MyDrive/text-analysis/code/ft-model")
```

We can run a more detailed evaluation step from HuggingFace if desired, to see how well our model performed. It is likely a good idea to have these metrics so that you can compare your performance to more generic models for the task.

```python
trainer.evaluate()

predictions, labels, _ = trainer.predict(tokenized_datasets["validation"])
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

results = metric.compute(predictions=true_predictions, references=true_labels)
results
```

```txt
{'Amenity': {'precision': 0.6348684210526315,
  'recall': 0.6655172413793103,
  'f1': 0.6498316498316498,
  'number': 290},
  'Cuisine': {'precision': 0.8689138576779026,
  'recall': 0.8140350877192982,
  'f1': 0.8405797101449275,
  'number': 285},
  'Dish': {'precision': 0.797945205479452,
  'recall': 0.9066147859922179,
  'f1': 0.8488160291438981,
  'number': 257},
  'Hours': {'precision': 0.7022900763358778,
  'recall': 0.736,
  'f1': 0.7187499999999999,
  'number': 125},
  'Location': {'precision': 0.800383877159309,
  'recall': 0.8273809523809523,
  'f1': 0.8136585365853658,
  'number': 504},
  'Price': {'precision': 0.7479674796747967,
  'recall': 0.8214285714285714,
  'f1': 0.7829787234042553,
  'number': 112},
  'Rating': {'precision': 0.6805555555555556,
  'recall': 0.7967479674796748,
  'f1': 0.7340823970037453,
  'number': 123},
  'Restaurant_Name': {'precision': 0.8560411311053985,
  'recall': 0.8671875,
  'f1': 0.8615782664941786,
  'number': 384},
  'overall_precision': 0.7784431137724551,
  'overall_recall': 0.8125,
  'overall_f1': 0.7951070336391437,
  'overall_accuracy': 0.9094094094094094}
```

## Using our Model

Now that we've created our model, we don't need any of the preexisting code to actually use it. The code below should run the model. Feel free to compose your own example and see how well the model performs!

```python
from transformers import pipeline
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import TokenClassificationPipeline
import torch

EXAMPLE = "where is a four star restaurant in milwaukee with tapas"

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained("/content/drive/MyDrive/text-analysis/code/ft-model")
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first")
ner_results = nlp(EXAMPLE)
print(ner_results)

```

```txt
[{'entity_group': 'Rating', 'score': 0.9655443, 'word': 'four star', 'start': 11, 'end': 20}, {'entity_group': 'Location', 'score': 0.9490055, 'word': 'milwaukee', 'start': 35, 'end': 44}, {'entity_group': 'Dish', 'score': 0.8788909, 'word': 'tapas', 'start': 50, 'end': 55}]
```

# Corpus Development: Text Data Collection and APIs

## Sources of text data

The best sources of text datasets will ultimately depend on the goals of your project. Some common sources of text data for text analysis include digitized archival materials, newspapers, books, social media, and research articles. For the most part, the datasets and sources you may come across will not have been arranged with a particular project in mind. The burden is therefore on you, as the researcher, to evaluate whether materials are suitable for your corpus. It can be useful to create a list of criteria for how you will decide what to include in your corpus. You may get the best results by piecing together your corpus from materials from various sources that meet your requirements. This will help you to create an intellectually rigorous corpus that meets your project's needs and makes a unique contribution to your area of study.

## Text Data and Restrictions

One of the most important criteria for inclusion in your corpus that you should consider is whether or not you have the right to use the data in the way your project requires. When evaluating data sources for your project, you may need to navigate a variety of legal and ethical issues. We'll briefly mention some of them below, but to learn more about these issues, we recommend the open access book [Building Legal Literacies for Text and Data Mining](https://berkeley.pressbooks.pub/buildinglltdm/).

- **Copyright** - Copyright law in the United States protects original works of authorship and grants the right to reproduce the work, to create derivative works, distribute copies, perform the work publicly, and to share the work publicly. Fair use may create exceptions for some TDM activities, but if you are analyzing copyrighted material, publicly sharing your full text corpus would likely be in violation of copyright.
- **Licensing** - Licenses grant permission to use materials in certain ways while usually restricting others. If you are working with databases or other licensed collections of materials, make sure that you understand the license and how it applies to text and data mining.
- **Terms of Use** - If you are collecting text data from other sources, such as websites or applications, make sure that you understand any retrictions on how the data can be used.
- **Technology Protection Measures** - Some publishers and content hosts protect their copyrighted/licensed materials through encryption. While commercial versions of ebooks, for example, would make for easy content to analyze, circumventing these protections would be illegal in the United States under the Digital Millennium Copyright Act.
- **Privacy** - Before sharing a corpus publicly, consider whether doing so would constitute any legal or ethical violations, especially with regards to privacy. Consulting with digital scholarship librarians at your university or professional organizations in your field would be a good place to learn about privacy issues that might arise with the type of data you are working with.
- **Research Protections** - Depending on the type of corpus you are creating, you might need to consider human subject research protections such as informed consent. Your institution's Institutional Review Board may be able to help you navigate emerging issues surrounding text data that is publicly available but could be sensitive, such as social media data.

## File Format, OCR, and Speech Transcription

Another criteria you may have to consider is the format that you need your files to be in. It may be that your test documents are not in text format, ie. in a file format that can be copied and pasted into a notepad file. Not all data is of this type, for example, there may be documents that are stored as image files or sound files. Or perhaps your documents are in PDF or DOC files.

Fortunately, there exist tools to convert file types like these into text. While these tools are beyond the scope of our lesson, they are still worth mentioning. Optical Character Recognition, or OCR, is a field of study that converts images to text. Tools such as Tesseract, Amazon Textract, or Google's Document AI can perform OCR tasks. Speech transcription will take audio files and convert them to text as well. Google's Speech-to-Text and Amazon Transcribe are two cloud solutions for speech transcription.

Later in this lesson we will be working with OCR text data that has been generated from images of digitized newspapers. As you will see, the quality of text generated by OCR and speech to text software can vary. In order to include a document with imperfect OCR text, you may decide to do some file clean up or remediation. Or you may decide to only include documents with a certain level of OCR accuracy in your corpus.

## Using APIs

When searching through sources, you may come across instructions to access the data through an API. An API, or application programming interface, allows computer programs to talk to one another. In the context of the digital humanities, you can use an API to request and receive specific data from corpora created by libraries, museums, or other cultural organizations or data creators.

There are different types of APIs, but for this lesson, we will be working with a RESTful API, which uses HTTP, or hypertext protocol methods. A RESTful API can be used to post, delete, and get data. You can make a request using a URL, or Uniform Resource Locator, which is sent to a web server using HTTP and returns a response. If you piece together a URL in a certain way, it will give the web server all the info it needs to locate what you are looking for and it will return the correct response. If that sounds familiar, it's because this is how we access websites everyday!

### A few things to keep in mind

- Each API will be different, so you will always want to check their documentation.
- Some APIs will require that you register to receive an API key in order to access their data.
- Just because the data is being made available through an API, doesn't mean that it can be used in your particular project. Remember to check the terms of use.
- Check the data, even if you've used the API before. What format will it be delivered in? Is the quality of the data good enough to work with?

## How to Access an API

For an example of how to access data from an API, we will explore [Chronicling America: Historic American Newspapers](https://chroniclingamerica.loc.gov/about/), a resource produced by the National Digital Newspaper Program (NDNP), a partnership between the National Endowment for the Humanities (NEH) and the Library of Congress (LC). Among other things, this resource offers OCR text data for newspaper pages from 1770 to 1963. The majority of the newspapers included in Chronicling America are in the public domain, but there is a disclaimer from the Library of Congress that newspapers in the resource that were published less than 95 years ago should be evaluated for renewed copyright. The API is public and no API key is required. We'll use Chronicling America's API to explore their data and pull in an OCR text file.

For this lesson, we'll pretend that we're at the start of a project and we are interested in looking at how Wisconsin area newspapers described World War I. We aren't yet sure if we want to focus on any particular newspaper or what methods we want to use. We might want to see what topics were most prominent from year to year. We might want to do a sentiment analysis and see whether positive or negative scores fluctuate over time. The possibilities are endless! But first we want to see what our data looks like.

By adding search/pages/results/? to our source's URL <https://chroniclingamerica.loc.gov/> and adding some of the search parameters that we have already mentioned we can start building our query. We will want to look for newspapers in Wisconsin between 1914 and 1918 that mention our search term "war." And to keep our corpus manageable, we want to see only the first pages using sequence=1. If we specify that we want to be able to see it in a JSON file, that will give us the following query:

<https://chroniclingamerica.loc.gov/search/pages/results/?state=Wisconsin&dateFilterType=yearRange&date1=1914&date2=1918&sort=date&andtext=war&sequence=1&format=json>

Let's take a look at what happens when we type that into our web browser. And let's take a look at what happens when we remove the request to view it in a JSON format.

Now let's explore making requests and getting data using python.

```python
!pip install requests
!pip install pandas
```

```python
import requests
import json
import pandas as pd

from google.colab import drive
drive.mount('/drive')
```

## Making a request

You can make an API call using a get() request. This will give you a response that you can check by accessing .status_code. There are different codes that you might receive with different meanings. If we send a request that can't be found, we get the familiar 404 code. A 200 response means that the request was successful.

```python
#What happens when what you are looking for doesn't exist?
response = requests.get("https://chroniclingamerica.loc.gov/this-api-doesnt-exist")
print(response.status_code)
```

```txt
404
```

```python
#Get json file of your search and check status
#First 20 search results
response20 = requests.get("https://chroniclingamerica.loc.gov/search/pages/results/?state=Wisconsin&dateFilterType=yearRange&date1=1914&date2=1918&sort=date&andtext=war&sequence=1&format=json")
print(response20.status_code)
```

```txt
200
```

Now that we have successfully used an API to call in some of the data we were looking for, let's take a look at our file. We can see that there are 3,941 total items that meet our criteria and that this response has gotten 20 of them for us. We'll save our results to a text file.

```python
# Look at json file
print(response20.json())
```

```txt
{'totalItems': 3941, 'endIndex': 20, 'startIndex': 1, 'itemsPerPage': 20, 'items': [{'sequence': 1, 'county': ['Manitowoc'], 'edition': None, 'frequency': 'Weekly', 'id': '/lccn/sn85033139/1914-01-01/ed-1/seq-1/', 'subject': ['Manitowoc (Wis.)--Newspapers.', 'Wisconsin--Manitowoc.--fast--(OCoLC)fst01225415'], 'city': ['Manitowoc'], 'date': '19140101', 'title': 'The Manitowoc pilot. [volume]', 'end_year': 1932, 'note': ['Archived issues are available in digital format from the Library of Congress Chronicling America online collection.', 'Publisher varies.'], 'state': ['Wisconsin'], 'section_label': '', 'type': 'page', 'place_of_publication': 'Manitowoc, Wis.', 'start_year': 1859, 'edition_label': '', 'publisher': 'Jeremiah Crowley', 'language': ['English'], 'alt_title': [], 'lccn': 'sn85033139', 'country': 'Wisconsin', 'ocr_eng': 'olume IV.\ntCITY COUHCIL NOUS,\npecial meeting of the city council\neld...
```

```python
# Turn file into a python dictionary
data = response20.json()
print(data)
```

```txt
{'totalItems': 3941, 'endIndex': 20, 'startIndex': 1, 'itemsPerPage': 20, 'items': [{'sequence': 1, 'county': ['Manitowoc'], 'edition': None, 'frequency': 'Weekly', 'id': '/lccn/sn85033139/1914-01-01/ed-1/seq-1/', 'subject': ['Manitowoc (Wis.)--Newspapers.', 'Wisconsin--Manitowoc.--fast--(OCoLC)fst01225415'], 'city': ['Manitowoc'], 'date': '19140101', 'title': 'The Manitowoc pilot. [volume]', 'end_year': 1932, 'note': ['Archived issues are available in digital format from the Library of Congress Chronicling America online collection.', 'Publisher varies.'], 'state': ['Wisconsin'], 'section_label': '', 'type': 'page', 'place_of_publication': 'Manitowoc, Wis.', 'start_year': 1859, 'edition_label': '', 'publisher': 'Jeremiah Crowley', 'language': ['English'], 'alt_title': [], 'lccn': 'sn85033139', 'country': 'Wisconsin', 'ocr_eng': 'olume IV.\ntCITY COUHCIL NOUS,\npecial meeting of the city council\neld...
```

```python
#Save a copy of the first 20 results
with open('corpusraw.txt', 'w') as corpusraw_file:
  corpusraw_file.write(json.dumps(data))
```

Next we will look at how we can use the metadata from our results to build a query that use the API to get OCR text from one of the newspaper pages.

The first four key value pairs in the dictionary object tell us about the results of our query, but the fifth one, with the key 'items' is the one that gives us the bulk of the metadata about the newspapers that meet our requirements. To build our query, we need to grab the id so that we can add it to our URL. We could either manually grab it from our text file or we can call it using its index in the list.

```python
#Deal with dictionaries within lists
d = data.get('items')
print(d)
```

```txt
[{'sequence': 1, 'county': ['Manitowoc'], 'edition': None, 'frequency': 'Weekly', 'id': '/lccn/sn85033139/1914-01-01/ed-1/seq-1/', 'subject': ['Manitowoc (Wis.)--Newspapers.', 'Wisconsin--Manitowoc.--fast--(OCoLC)fst01225415'], 'city': ['Manitowoc'], 'date': '19140101', 'title': 'The Manitowoc pilot. [volume]', 'end_year': 1932, 'note': ['Archived issues are available in digital format from the Library of Congress Chronicling America online collection.', 'Publisher varies.'], 'state': ['Wisconsin'], 'section_label': '', 'type': 'page', 'place_of_publication': 'Manitowoc, Wis.', 'start_year': 1859, 'edition_label': '', 'publisher': 'Jeremiah Crowley', 'language': ['English'], 'alt_title': [], 'lccn': 'sn85033139', 'country': 'Wisconsin', 'ocr_eng': 'olume IV.\ntCITY COUHCIL NOUS,\npecial meeting of the city council\neld...
```

```python
d = data.get('items')
newspaper1 = d[0]
idnewspaper1 = newspaper1.get('id')
print(idnewspaper1)
```

```txt
/lccn/sn85033139/1914-01-01/ed-1/seq-1/
```

Now that we have the id we need, we can use it to build our query. Adding it to our source's URL gives us <https://chroniclingamerica.loc.gov/lccn/sn85033139/1914-01-01/ed-1/seq-1/>. To see the OCR for this file, we just need to add ocr.txt to the end of the query to get <https://chroniclingamerica.loc.gov/lccn/sn85033139/1914-01-01/ed-1/seq-1/ocr.txt>.

Now let's take a look at what it looks like when we make the request using the API.

```python
#Grab one OCR file from Chronicling America
responsenewspaper1 = requests.get(f"https://chroniclingamerica.loc.gov{idnewspaper1}ocr.txt")
print(responsenewspaper1.status_code)
```

```txt
200
```

```python
print(responsenewspaper1.text)
```

```txt
olume IV.
tCITY COUHCIL NOUS,
pecial meeting of the city council
eld last Saturday evening to take...
```

# Final Thoughts

## Is text analysis artificial intelligence?

Artificial intelligence is loosely defined as the ability for computer systems to perform tasks that have traditionally required human reasoning and perception.

To the extent that text analysis performs a task that resembles reading, understanding, and analyzing meaning, it can be understood to be part of the definition of artificial intelligence.

The methods in this lesson all demonstrate models that learn from data, specifically, from text corpora that are not structured to explicitly tell the machine anything other than, perhaps, title, author, date, and body of text.

As a method and a tool, it is important to understand the tasks to which it is best suited, and to understand the process well enough to be able to interpret the results, including:

1. whether the results are relevant or meaningful
2. whether the results have been overly influenced by the model or training data
3. how to responsibly use the results

We can describe these as broad-level commitments to ethical research methods, variously construed depending on your stance within your own research.

## Relevance or meaningfulness

As with any research, the relevance or meaningfulness of your results is relative to the research question itself. However, when you have a particular research question (or a particular set of research interests), it can be hard to connect the results of these models back to your bigger picture aims. It can feel like trying to write a book report but all you were given were the table of contents. One reason for this difficulty is that the dimensions of the model are atheoretical. That is, regardless of what research questions you are asking, the models always start from the same starting point: the words of the text, with no understanding of what those words mean to you. Our job is to interpret the meaning of the model's results, or the qualitative work that follows.

The model is making a statistical determination based on the training data it has been fed, and on the training itself, as well as the methods you have used to parse the data set you're analyzing. If you are using a tool like ChatGPT, you may have access only to your own methods, and will need to make an educated guess about the training data and training methods. That doesn't mean you can't use that tool, but it does mean you need to keep what is known and what is obscured about your methods at the forefront as you conduct your research.  

## Training data can influence results

There are numerous examples of how training data (or the language model, ultimately) can negatively influence results. Reproducing bias in the data is probably one of the most discussed negative outcomes. Let's look at one real world example:

In 2016, ProPublica published an investigative report that exposed the clear bias against Black people in computer programs used to determine the likelihood of defendants committing crimes in the future. That bias was built into the tool because the training data that it relied on included historical data about crime statistics, which reflected (and then reproduced) existing racist bias in sentencing.

The creators of the algorithm, Northpointe, claimed however that their models had been properly calibrated to account for race.

What gives?

The heart of the Propublica and Northpointe debate is a difference in how the two parties define fairness and the expected use case of the algorithm in question. This disagreement leads to a difference in which test metric either party believes is the right one to apply to the model. So while it's important to consider how bias and fairness play out in your research, it's also important to think clearly about (a) what fairness means to you and your project and (b) what the right way to measure that conception of fairness is. And whatever decisions you land on, be up front about them.

## Using results in your research

Rarely will results from topic modeling, text analysis, etc. stand *on their own* as evidence of anything. Researchers should be able to explain their method and how they got their results, and be able to talk about the data sets and training models used. As discussed above, though, the nature of the one's models may contain vast numbers of parameters that cannot be reverse engineered or described, at least not easily.

So do you need to understand all your models 100% to use them in your research?

In a sense, sure, why not. But more realistically, you only need to understand the tool pretty well and well enough to fulfill your research commitments that demand you be able to give an account of what you did and how you make sense of your results. Where the line on "enough" gets drawn ultimately depends on your own field and your own project.

## Risk zones

Another area to consider when using any technology, not just NLP, are the [risk zones](https://ethicalexplorer.org/wp-content/uploads/2020/04/Tech-Risk-Zones.pdf) that are introduced. We're talking about unintended consequences, for the most part, but consequences nonetheless.

Let's say you were using BERT to help summarize a large body of texts to understand broad themes and relationships. Could this same method be used to distort the contents of those texts to spread misinformation? How can we mitigate that risk?

In the case of the techniques that underlie many of the text analysis methods you learned in this workshop, is there a chance that the results could reinforce existing biases because of existing biases in the training data?

## Hype cycles and AI

Because this workshop is being introduced shortly after the release of ChatGPT3 by OpenAI, we want to address how AI and tech hype cycles can influence tool selection and use of tech.

The inscrutability of LLMs, the ability of chatbots to output coherent and meaningful text on a seemingly infinite variety of topics, and the rhetoric of the tech industry can make these tools seem magical and unfathomable.

They aren't magical, though the black box nature of the training data and the parameters does lend itself to unfathomability.

Regardless, the output of any of the methods described in this workshop, and by LLMs to come, is the product of mathematical processes and statistical weights. That is why learning some of the methodology behind text analysis is important. Conceivably, we all will use tools based on these methods in the years to come, whether for our research or for more mundane administrative tasks

Understanding how these tools work, even to imperfect degrees, helps hold tech accountable, and enables better use of these tools for apprpriate tasks. Regrdless of the sophistication of the tool, it is humans who attribute meaning to the results, not the machine.