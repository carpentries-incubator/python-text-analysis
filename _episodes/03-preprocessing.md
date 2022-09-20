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

### OCR and Speech Transcription
We open with the assumption that all of your test documents are in text format- that is, in a file format that can be copied and pasted into a notepad file. 
Not all data is of this type, for example, there may be documents that are stored as image files or sound files. Or perhaps your documents are in PDF or DOC files.
Fortunately, there exist tools to convert file types like these into text. While these tools are beyond the scope of our lesson, they are still worth mentioning. 
Optical Character Recognition, or OCR, is a field of study that converts images to text. Tools such as Tesseract, Amazon Textract, or Google's Document AI can perform OCR tasks. 
Speech transcription will take audio files and convert them to text as well. Google's Speech-to-Text and Amazon Transcribe are two cloud solutions for speech transcription.

##Loading text files
To start, we'll say you have a corpus of text files we want to analyze. Let's create a method to list the files we want to analyze. 
To make this method more flexible, we will also use glob to allow us to put in regular expressions so we can filter the files if so desired.
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

corpus_dir = 'C:\\Users\\Desktop\\documents\\'
corpus_file_list = create_file_list(corpus_dir)
print(corpus_file_list)
~~~
{: .language-python}
~~~
['C:\\Users\\Desktop\\documents\\austen-emma.txt', 'C:\\Users\\Desktop\\documents\\austen-persuasion.txt', 'C:\\Users\\Desktop\\documents\\austen-sense.txt', 'C:\\Users\\Desktop\\documents\\bible-kjv.txt', 'C:\\Users\\Desktop\\documents\\blake-poems.txt', 'C:\\Users\\Desktop\\documents\\bryant-stories.txt', 'C:\\Users\\Desktop\\documents\\burgess-busterbrown.txt', 'C:\\Users\\Desktop\\documents\\carroll-alice.txt', 'C:\\Users\\Desktop\\documents\\chesterton-ball.txt', 'C:\\Users\\Desktop\\documents\\chesterton-brown.txt', 'C:\\Users\\Desktop\\documents\\chesterton-thursday.txt', 'C:\\Users\\Desktop\\documents\\edgeworth-parents.txt', 'C:\\Users\\Desktop\\documents\\melville-moby_dick.txt', 'C:\\Users\\Desktop\\documents\\milton-paradise.txt', 'C:\\Users\\Desktop\\documents\\shakespeare-caesar.txt', 'C:\\Users\\Desktop\\documents\\shakespeare-hamlet.txt', 'C:\\Users\\Desktop\\documents\\shakespeare-macbeth.txt', 'C:\\Users\\Desktop\\documents\\whitman-leaves.txt']
~~~
{: .output}

We will use the full corpus, but it might be useful to filter if we have multiple file types in our directory. 
We can filter our list using a regular expression as well. If I want just documents written by Austen, I can filter on part of the file path name.
~~~
austen_list = create_file_list(corpus_dir, '*austen*')
print(austen_list)
~~~
{: .language-python}
~~~
['C:\\Users\\Desktop\\documents\\austen-emma.txt', 'C:\\Users\\Desktop\\documents\\austen-persuasion.txt', 'C:\\Users\\Desktop\\documents\\austen-sense.txt']
~~~
{: .output}

Let's take a closer look at one of the documents in our corpus by pulling out a sentence. We are looking at the first full sentence, which begins with character 50 and ends at character 290.

~~~
preview_len = 290
print(corpus_file_list[0])
sentence = ""
with open(corpus_file_list[0], 'r') as f:
	sentence = f.read(preview_len)[50:preview_len]
print sentence
~~~
{: .language-python}

~~~
C:\\Users\\Desktop\\documents\\austen-emma.txt

Emma Woodhouse, handsome, clever, and rich, with a comfortable home
and happy disposition, seemed to unite some of the best blessings
of existence; and had lived nearly twenty-one years in the world
with very little to distress or vex her.
~~~
{: .output}

## Preprocessing
To prepare our text for use in an NLP model, we want to break the text up into discrete units that we can put into vector space.
Spacy is a python library for Natural Language Processing capable of doing a variety of tasks. 
We will be using spacy's preprocessor for our lessons, but there are other packages in Python such as NLTK, pytorch, and gensim which also implement text analysis tools. 
We'll be customizing the tokenizer later, so we will define a special class for it, and add extra things to our class as we progress through the lesson. We'll also have a copy of our spacy tokenizer.
~~~
import spacy 
import en_core_web_sm

spacyt = en_core_web_sm.load()

class Our_Tokenizer:
    def __init__(self):
        #import spacy tokenizer/language model
        self.nlp = en_core_web_sm.load()
        self.nlp.max_length = 4500000 # increase max number of characters that spacy can process (default = 1,000,000)
	def tokenize(self, document):
		tokens = self.nlp(document)
		return tokens
~~~
{: .language-python}

This will load spacy and the preprocessing model for English. Different languages may have different 
rulesets, and therefore require different preprocessing parsers. 
Running the document we created through the NLP model we loaded performs a variety of tasks for us.
Let's look at these in greater detail.

### Tokenization
Tokenization is the process of breaking down texts (strings of characters) into words, groups of words, and sentences. 
Humans automatically understand words and sentences as discrete units of meaning. 
However, for computers, we have to break up documents containing larger chunks of text into these discrete units of meaning. These are called tokens.
A string of characters needs to be understood by a program as words, or even terms made up of more than one word.
Now let's load our test sentence into our tokenizer.

~~~
tokens = spacyt(sentence)
for t in tokens:
	print(token.text)

~~~
{: .language-python}

~~~
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
~~~
{: .output}

The single sentence has been broken down into a set of tokens. Tokens for spacy are objects with a variety of attributes.
The documentation for these can be found at https://spacy.io/api/token

### Lemmas (Stemming and Lemmatization)
Think about similar words, such as running, ran, and runs.
All of these words have a similar root, but a computer does not know this. Withour preprocessing, these words would be considered dissimilar.
Stemming and Lemmatization are used to group together words that are similar or forms of the same word. 
**Stemming** is removing the conjugation and pluralized endings for words. For example; words like “digitization”, and “digitizing” might chopped down to "digitiz." 
**Lemmatization** is the more sophisticated of the two, and looks for the linguistic base of a word. 
Lemmatization can group words that mean the same thing but may not be grouped through simple “stemming,” such as [bring, brought…]

Similarly, capital letters are considered different from non-capital letters, meaning that capitalized versions of words are considered different from non-capitalized versions. 
Converting all words to lower case ensures that capitalized and non-capitalized versions of words are considered the same.

Spacy also created a lemmatized version of our document. Let's try accessing this by typing the following:
~~~
for token in tokens:
	print(token.lemma)
~~~
{: .language-python}

~~~
4398759943034541363
14931068470291635495
16764210730586636600
13439688181152664240
1939021399366709707
205562458315866255
3806482680584466996
908432558851201422
12288588264913869032
4690420944186131903
908432558851201422
17145585959182355159

...

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
~~~
{: .output}

Spacy stores words by an ID number, and not as a full string, to save space in memory. Many spacy functions will return numbers and not words as you might expect. 
Fortunately, adding an underscore for spacy will return text representations instead. We will also add in the lower case function so that all words are lower case.

~~~
for token in tokens:
	print(str.lower(token.lemma_))
~~~
{: .language-python}

~~~
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

~~~
{: .output}

Notice how words like "best" and "her" have been changed to their root words such as "good" and "she". 
Let's change our tokenizer to save the lower cased, lemmatized versions of words instead of the original words.
~~~
class Our_Tokenizer:
    def __init__(self):
        #import spacy tokenizer/language model
        self.nlp = en_core_web_sm.load()
        self.nlp.max_length = 4500000 # increase max number of characters that spacy can process (default = 1,000,000)
	def tokenize(self, document):
		tokens = self.nlp(document)
		tokens = str.lower(token.lemma_) for token in tokens
		return tokens
~~~
{: .language-python}

### Stop-Words and Punctuation
Stop-words are common words that are often filtered out for more efficient natural language data processing. 
Words such as “the,” “an,” “a,” “of,” “and/or,” “many” don't necessarily tell us a lot about a document's content and are often removed in simpler models. 
Stop lists (groups of stop words) are curated by sorting terms by their collection frequency, or the total number of times that they appear in a document or corpus. 
Punctuation also is something we are not interested in, since we have a "bag of words" assumption which does not care about the position of words or punctuation.
Many open-source software packages for language processing, such as Spacy, include stop lists. Let's look at Spacy's stopword list.

~~~
stopwords = spacyt.Defaults.stop_words
print(stopwords)
~~~
{: .language-python}

~~~
{'side', 'such', 'herself', 'many', 'latterly', 'front', 'down', 'might', 'being', 'sometime', 'hers', 'say', '‘ve', 'why', 'give', 'call', 'put', 'towards', 'how', 'is', 'seem', 'between', 'any', 'next', 'whereafter', 'anyhow', "'d", 'off', 'three', 'somehow', 'one', 'me', 'last', 'over', 'please', 'most', 'thereby', 'their', 'been', 'up', 'quite', 'still', 'would', 'his', 'your', 'show', 'serious', 'thru', 'although', 'namely', 'have', 'could', 'into', 'whereupon', 'always', 'was', 'forty', 'on', 'due', 'wherein', 'see', 'bottom', 'thence', 'beside', 'almost', 'neither', 'there', 'anything', 'whereas', 'eleven', 'the', 'under', '’ll', 'everything', 'others', 'thereupon', 'did', 'while', 'that', 'too', 'becomes', 'and', 'n’t', 'eight', 'unless', 'when', 'yourselves', 'third', 'all', 'a', 'as', "n't", "'re", 'keep', 'around', 'anywhere', 'if', 'our', 'name', 'to', 'per', 'her', 'here', 'whence', 'using', 'were', 'former', 'much', 'toward', 'upon', 'must', 'during', 'hereby', 'something', 'each', 'she', 'own', 'anyway', 'get', 'several', 'afterwards', 'hence', 'hereafter', 'every', 'just', '’re', 'himself', 'ca', 'same', 'thus', 'take', 'twenty', 'these', 'either', 'everyone', '‘m', 'become', 'should', 'whether', 'whatever', 'various', 'someone', 'him', "'s", 'hundred', 'do', 'moreover', 'amongst', 'none', 'can', 'five', 'themselves', 'twelve', 'else', 'never', 'becoming', 'however', 'more', 'nobody', '’m', 'yourself', 'whither', 'they', 'fifty', 'has', 'well', 'seeming', 'first', 'n‘t', 'therefore', 'which', 'go', 'yet', 'enough', 'since', 'where', 'its', "'ve", 'behind', 'wherever', 'fifteen', 'beyond', 'two', 'further', 'part', 'so', 'though', 'be', 'whole', 'anyone', 'or', 'only', 'in', 'mostly', 'against', 'whereby', 'once', 'perhaps', 'by', 'yours', 'out', 'i', 'latter', '‘s', 'we', 'thereafter', 'used', 'who', '‘ll', 'until', 'now', 'rather', 'move', 'already', 'otherwise', 'at', 'made', 'my', 'across', 'some', '‘re', '’d', 'whoever', 'sixty', 'nine', 'it', 'because', 'with', 'very', 'another', 'doing', 'really', 'nevertheless', 'then', 'regarding', 'before', 'this', 'seems', 'whose', 'therein', 'without', 'alone', 'done', 'became', 'an', 'again', 'sometimes', 'above', 'herein', "'m", 'ten', 'cannot', 'ever', 'elsewhere', 'back', 'indeed', 'may', 'meanwhile', 'onto', 'least', 'together', 'us', 'beforehand', 'throughout', 'even', '‘d', 'below', 'had', 'are', 'through', 'will', 'of', 'everywhere', 'along', 'myself', 'six', 'after', 'whom', 'but', 'for', 'make', 'am', 'except', 'often', 'hereupon', 'full', 'about', 'ourselves', 'those', 'what', 'also', 'besides', 'top', 'few', 'mine', 'four', 'less', '’ve', 'them', 'among', 'ours', 'both', 're', 'nor', 'itself', "'ll", 'from', 'nothing', 'formerly', 'he', 'whenever', 'you', 'nowhere', 'empty', 'amount', 'does', 'noone', 'seemed', 'no', 'than', 'via', '’s', 'somewhere', 'within', 'other', 'not'}
~~~
{: .output}

It's possible to add and remove words as well.

~~~
spacyt.Defaults.stop_words.add("zebra")
spacyt.Defaults.stop_words.remove("side")
print(stopwords)
~~~
{: .language-python}

~~~
{'such', 'herself', 'many', 'latterly', 'front', 'down', 'might', 'being', 'sometime', 'hers', 'say', '‘ve', 'why', 'give', 'call', 'put', 'towards', 'how', 'is', 'seem', 'between', 'any', 'next', 'whereafter', 'anyhow', "'d", 'off', 'three', 'somehow', 'one', 'me', 'last', 'over', 'please', 'most', 'thereby', 'their', 'been', 'up', 'quite', 'still', 'would', 'his', 'your', 'show', 'serious', 'thru', 'although', 'namely', 'have', 'could', 'into', 'whereupon', 'always', 'was', 'forty', 'on', 'due', 'wherein', 'see', 'bottom', 'thence', 'beside', 'almost', 'neither', 'there', 'anything', 'whereas', 'eleven', 'the', 'under', '’ll', 'everything', 'others', 'thereupon', 'did', 'while', 'that', 'too', 'becomes', 'and', 'n’t', 'eight', 'unless', 'when', 'yourselves', 'third', 'all', 'a', 'as', "n't", "'re", 'keep', 'around', 'anywhere', 'if', 'our', 'name', 'to', 'per', 'her', 'here', 'whence', 'using', 'were', 'former', 'much', 'toward', 'upon', 'must', 'during', 'hereby', 'something', 'each', 'she', 'own', 'anyway', 'get', 'several', 'afterwards', 'hence', 'hereafter', 'every', 'just', '’re', 'himself', 'ca', 'same', 'thus', 'take', 'twenty', 'these', 'either', 'everyone', '‘m', 'become', 'should', 'whether', 'whatever', 'various', 'someone', 'him', "'s", 'hundred', 'do', 'moreover', 'amongst', 'none', 'can', 'five', 'themselves', 'twelve', 'else', 'never', 'becoming', 'however', 'more', 'nobody', '’m', 'yourself', 'whither', 'they', 'fifty', 'has', 'well', 'seeming', 'first', 'n‘t', 'therefore', 'which', 'go', 'yet', 'enough', 'since', 'where', 'its', "'ve", 'behind', 'wherever', 'fifteen', 'beyond', 'two', 'further', 'part', 'so', 'though', 'be', 'whole', 'anyone', 'or', 'only', 'in', 'mostly', 'against', 'whereby', 'once', 'perhaps', 'by', 'zebra', 'yours', 'out', 'i', 'latter', '‘s', 'we', 'thereafter', 'used', 'who', '‘ll', 'until', 'now', 'rather', 'move', 'already', 'otherwise', 'at', 'made', 'my', 'across', 'some', '‘re', '’d', 'whoever', 'sixty', 'nine', 'it', 'because', 'with', 'very', 'another', 'doing', 'really', 'nevertheless', 'then', 'regarding', 'before', 'this', 'seems', 'whose', 'therein', 'without', 'alone', 'done', 'became', 'an', 'again', 'sometimes', 'above', 'herein', "'m", 'ten', 'cannot', 'ever', 'elsewhere', 'back', 'indeed', 'may', 'meanwhile', 'onto', 'least', 'together', 'us', 'beforehand', 'throughout', 'even', '‘d', 'below', 'had', 'are', 'through', 'will', 'of', 'everywhere', 'along', 'myself', 'six', 'after', 'whom', 'but', 'for', 'make', 'am', 'except', 'often', 'hereupon', 'full', 'about', 'ourselves', 'those', 'what', 'also', 'besides', 'top', 'few', 'mine', 'four', 'less', '’ve', 'them', 'among', 'ours', 'both', 're', 'nor', 'itself', "'ll", 'from', 'nothing', 'formerly', 'he', 'whenever', 'you', 'nowhere', 'empty', 'amount', 'does', 'noone', 'seemed', 'no', 'than', 'via', '’s', 'somewhere', 'within', 'other', 'not'}
~~~
{: .output}

This will only adjust the stopwords for the current session, but it is possible to save them if desired. More information about how to do this can be found in the Spacy documentation.
You might use this stopword list to filter words from documents using spacy, or just by manually iterating through it like a list. 

Let's see what our example looks like without stopwords and punctuation. 
~~~
for token in tokens:
    if token.is_stop == False and token.is_punct == False:
	       print(str.lower(token.lemma_))
~~~
{: .language-python}
~~~
emma
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
~~~
{: .output}

Let's filter out stopwords and punctuation from our custom tokenizer now as well.

~~~
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
~~~

### POS Tagging
Parts of speech (POS) are things such as nouns, verbs and adjectives. 
Technically, POS tagging is also a NLP task, as most documents do not come with POS tags already done.
POS tags often prove useful, so some tokenizers also have built in POS tagging done. Spacy is one such library.
Spacy's POS tags can be used by accessing the pos_ method for each token.
~~~
for token in tokens:
    if token.is_stop == False and token.is_punct == False:
	       print(str.lower(token.lemma_)+" "+token.pos_)
~~~
{: .language-python}

~~~
emma PROPN
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
distress NOUN
vex VERB

 SPACE
~~~
{: .output}

Because our dataset is relatively small, we may find that character names and places weigh very heavily in our early models. 
We also have a number of blank or white space tokens, which we will also want to remove.
We will finish our special tokenizer by removing punctuation and proper nouns from our documents.
~~~
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
~~~
{: .language-python}



Let's test our custom tokenizer on this selection of text to see how it works.

~~~
our_tok = Our_Tokenizer()
tokenset1= our_tok.tokenize(sentence)
print(tokenset1)
~~~
{: .language-python}
~~~
['handsome', 'clever', 'rich', 'comfortable', 'home', 'happy', 'disposition', 'unite', 'good', 'blessing', 'existence', 'live', 'nearly', 'year', 'world', 'little', 'distress', 'vex']
~~~
{: .output}

Let's add two more sentences to our corpus, and then put this representation in vector space. We'll do this using scikit learn. We can specify a tokenizer with sci-kit learn, so we will use the tokenizer we just defined.
Then, we will take a look at all the different terms in our dictionary, which contains a list of all the words that occur in our corpus.
~~~
corp = [sentence, "Happy holidays! Have a happy new year!", "What a handsome, happy, healthy little baby!"] 

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(tokenizer= our_tok.tokenize)
matrix = vectorizer.fit_transform(corp)
vectorizer.get_feature_names_out()
~~~
{: .language-python}
~~~
array(['baby', 'blessing', 'clever', 'comfortable', 'disposition',
       'distress', 'existence', 'good', 'handsome', 'happy', 'healthy',
       'holiday', 'home', 'little', 'live', 'nearly', 'new', 'rich',
       'unite', 'vex', 'world', 'year'], dtype=object)
~~~
{: .output}

Finally, lets take a look a the term-document matrix. Each document is a row, and each column is a dimension that represents a word. The values in each cell are simple word counts.
~~~
print(matrix.toarray())
~~~
{: .language-python}
~~~
[[0 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 0 1 1 1 1 1]
 [0 0 0 0 0 0 0 0 0 2 0 1 0 0 0 0 1 0 0 0 0 1]
 [1 0 0 0 0 0 0 0 1 1 1 0 0 1 0 0 0 0 0 0 0 0]]
~~~
{: .output}

If desired, we could calculate cosine similarity between different documents as well. While we defined cosine similarity ourselves earlier, sci-kit learn also has a method for it that we can use instead.

~~~
from sklearn.metrics.pairwise import cosine_similarity as cs
cs(matrix[0], matrix[1])
cs(matrix[0], matrix[2])
~~~
{: .language-python}
~~~~
array([[0.26726124]])
array([[0.31622777]])
~~~
{: .output}
According to this model, our third sentence is closer to our original sentence than the second one. We could conduct similar analysis over larger groups of text, such as all the documents in our corpus.

This lesson has covered a number of preprocessing steps. We created a list of our files in our corpus, which we can use in future lessons.
We customized a tokenizer from Spacy, to better suit the needs of our corpus, which we can also use moving forward.
Finally, we put our sample sentences in a term-document matrix for the first time and calculated cosine similarity scores between the two. 
Next we will use a more complex model called TF-IDF.