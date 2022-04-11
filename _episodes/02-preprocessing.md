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
To prepare our text for use in an NLP model, we first need to select an appropriate model for our task.
Spacy is a python program capable of doing a variety of tasks. We will be using spacy for our lessons, but there are other packages in Python such as NLTK, pytorch, and gensim which also implement text analysis tools. 
Let's load spacy now.

~~~
import spacy 
eng = spacy.load("en_core_web_sm")
~~~
{: .language-python}

This will load spacy and the preprocessing model for English. Different languages may have different 
rulesets, and therefore require different preprocessing parsers. Now let's load in a test document of 
a single sentence.

~~~
sent = "Apple is looking at buying U.K. startup for $1 billion"
doc = eng(sent)
~~~
{: .language-python}

Running the document we created through the NLP model we loaded performs a variety of tasks for us.
Let's look at these in greater detail.

### Tokenization
Tokenization is the process of breaking down texts (strings of characters) into words, groups of words, and sentences. 
Humans automatically understand words and sentences as discrete units of meaning. 
However, for computers, we have to break up documents containing larger chunks of text into these discrete units of meaning. 
A string of characters needs to be understood by a program as words, or even terms made up of more than one word.

~~~
for token in doc:
	print(token.text)
~~~
{: .language-python}
~~~
Apple
is
looking
at
buying
U.K.
startup
for
$
1
billion
~~~
{: .output}

The single sentence has been broken down into a set of words.

### Lemmas (Stemming and Lemmatization)
Think about similar words, such as running, ran, and runs.
All of these words have a similar root, but a computer does not know this. Withour preprocessing, these words would be considered dissimilar.
Stemming and Lemmatization are used to group together words that are similar or forms of the same word. 
**Stemming** may be familiar if you’ve ever conducted a “wildcard” search in a library catalog 
- using the “*” symbol to indicate that you are looking for any word that begins with “digi”, for example; returning “digital”, “digitization”, and “digitizing.” 
**Lemmatization** is the more sophisticated of the two, and looks for the linguistic base of a word. 
Lemmatization can group words that mean the same thing but may not be grouped through simple “stemming,” such as [bring, brought…]

Spacy also created a lemmatized version of our document. Let's try accessing this by typing the following:
~~~
for token in doc:
	print(token.lemma)
~~~
{: .language-python}

~~~
6418411030699964375
10382539506755952630
16096726548953279178
11667289587015813222
9457496526477982497
14409890634315022856
7622488711881293715
16037325823156266367
11283501755624150392
5533571732986600803
1231493654637052630
~~~
{: .output}

Spacy stores words by an ID number, and not as a full string, to save space in memory. Many functions will return numbers and not words as you might expect. 
Fortunately, adding an underscore will return text representations instead.

~~~
for token in doc:
	print(token.lemma_)
~~~
{: .language-python}

~~~
Apple
be
look
at
buy
U.K.
startup
for
$
1
billion
~~~
{: .output}

Notice how words like "looking" have their ends stripped off. Other irregular words such as "is" have been converted to their lemma of "be" by Spacy as well.

### Stop-Words
Stop-words are common words that are often filtered out for more efficient natural language data processing. 
Words such as “the,” “an,” “a,” “of,” “and/or,” “many” don't necessarily tell us a lot about a document's content and are often removed in simpler models. 
Stop lists (groups of stop words) are curated by sorting terms by their collection frequency, or the total number of times that they appear in a document or corpus. 
Many open-source software packages for language processing, such as Spacy, include stop lists. Let's look at Spacy's stopword list.

~~~
stopwords = eng.Defaults.stop_words
print(stopwords)
~~~
{: .language-python}

~~~
{'side', 'such', 'herself', 'many', 'latterly', 'front', 'down', 'might', 'being', 'sometime', 'hers', 'say', '‘ve', 'why', 'give', 'call', 'put', 'towards', 'how', 'is', 'seem', 'between', 'any', 'next', 'whereafter', 'anyhow', "'d", 'off', 'three', 'somehow', 'one', 'me', 'last', 'over', 'please', 'most', 'thereby', 'their', 'been', 'up', 'quite', 'still', 'would', 'his', 'your', 'show', 'serious', 'thru', 'although', 'namely', 'have', 'could', 'into', 'whereupon', 'always', 'was', 'forty', 'on', 'due', 'wherein', 'see', 'bottom', 'thence', 'beside', 'almost', 'neither', 'there', 'anything', 'whereas', 'eleven', 'the', 'under', '’ll', 'everything', 'others', 'thereupon', 'did', 'while', 'that', 'too', 'becomes', 'and', 'n’t', 'eight', 'unless', 'when', 'yourselves', 'third', 'all', 'a', 'as', "n't", "'re", 'keep', 'around', 'anywhere', 'if', 'our', 'name', 'to', 'per', 'her', 'here', 'whence', 'using', 'were', 'former', 'much', 'toward', 'upon', 'must', 'during', 'hereby', 'something', 'each', 'she', 'own', 'anyway', 'get', 'several', 'afterwards', 'hence', 'hereafter', 'every', 'just', '’re', 'himself', 'ca', 'same', 'thus', 'take', 'twenty', 'these', 'either', 'everyone', '‘m', 'become', 'should', 'whether', 'whatever', 'various', 'someone', 'him', "'s", 'hundred', 'do', 'moreover', 'amongst', 'none', 'can', 'five', 'themselves', 'twelve', 'else', 'never', 'becoming', 'however', 'more', 'nobody', '’m', 'yourself', 'whither', 'they', 'fifty', 'has', 'well', 'seeming', 'first', 'n‘t', 'therefore', 'which', 'go', 'yet', 'enough', 'since', 'where', 'its', "'ve", 'behind', 'wherever', 'fifteen', 'beyond', 'two', 'further', 'part', 'so', 'though', 'be', 'whole', 'anyone', 'or', 'only', 'in', 'mostly', 'against', 'whereby', 'once', 'perhaps', 'by', 'yours', 'out', 'i', 'latter', '‘s', 'we', 'thereafter', 'used', 'who', '‘ll', 'until', 'now', 'rather', 'move', 'already', 'otherwise', 'at', 'made', 'my', 'across', 'some', '‘re', '’d', 'whoever', 'sixty', 'nine', 'it', 'because', 'with', 'very', 'another', 'doing', 'really', 'nevertheless', 'then', 'regarding', 'before', 'this', 'seems', 'whose', 'therein', 'without', 'alone', 'done', 'became', 'an', 'again', 'sometimes', 'above', 'herein', "'m", 'ten', 'cannot', 'ever', 'elsewhere', 'back', 'indeed', 'may', 'meanwhile', 'onto', 'least', 'together', 'us', 'beforehand', 'throughout', 'even', '‘d', 'below', 'had', 'are', 'through', 'will', 'of', 'everywhere', 'along', 'myself', 'six', 'after', 'whom', 'but', 'for', 'make', 'am', 'except', 'often', 'hereupon', 'full', 'about', 'ourselves', 'those', 'what', 'also', 'besides', 'top', 'few', 'mine', 'four', 'less', '’ve', 'them', 'among', 'ours', 'both', 're', 'nor', 'itself', "'ll", 'from', 'nothing', 'formerly', 'he', 'whenever', 'you', 'nowhere', 'empty', 'amount', 'does', 'noone', 'seemed', 'no', 'than', 'via', '’s', 'somewhere', 'within', 'other', 'not'}
~~~
{: .output}

It's possible to add and remove words as well.

~~~
eng.Defaults.stop_words.add("zebra")
eng.Defaults.stop_words.remove("side")
print(stopwords)
~~~
{: .language-python}

~~~
{'such', 'herself', 'many', 'latterly', 'front', 'down', 'might', 'being', 'sometime', 'hers', 'say', '‘ve', 'why', 'give', 'call', 'put', 'towards', 'how', 'is', 'seem', 'between', 'any', 'next', 'whereafter', 'anyhow', "'d", 'off', 'three', 'somehow', 'one', 'me', 'last', 'over', 'please', 'most', 'thereby', 'their', 'been', 'up', 'quite', 'still', 'would', 'his', 'your', 'show', 'serious', 'thru', 'although', 'namely', 'have', 'could', 'into', 'whereupon', 'always', 'was', 'forty', 'on', 'due', 'wherein', 'see', 'bottom', 'thence', 'beside', 'almost', 'neither', 'there', 'anything', 'whereas', 'eleven', 'the', 'under', '’ll', 'everything', 'others', 'thereupon', 'did', 'while', 'that', 'too', 'becomes', 'and', 'n’t', 'eight', 'unless', 'when', 'yourselves', 'third', 'all', 'a', 'as', "n't", "'re", 'keep', 'around', 'anywhere', 'if', 'our', 'name', 'to', 'per', 'her', 'here', 'whence', 'using', 'were', 'former', 'much', 'toward', 'upon', 'must', 'during', 'hereby', 'something', 'each', 'she', 'own', 'anyway', 'get', 'several', 'afterwards', 'hence', 'hereafter', 'every', 'just', '’re', 'himself', 'ca', 'same', 'thus', 'take', 'twenty', 'these', 'either', 'everyone', '‘m', 'become', 'should', 'whether', 'whatever', 'various', 'someone', 'him', "'s", 'hundred', 'do', 'moreover', 'amongst', 'none', 'can', 'five', 'themselves', 'twelve', 'else', 'never', 'becoming', 'however', 'more', 'nobody', '’m', 'yourself', 'whither', 'they', 'fifty', 'has', 'well', 'seeming', 'first', 'n‘t', 'therefore', 'which', 'go', 'yet', 'enough', 'since', 'where', 'its', "'ve", 'behind', 'wherever', 'fifteen', 'beyond', 'two', 'further', 'part', 'so', 'though', 'be', 'whole', 'anyone', 'or', 'only', 'in', 'mostly', 'against', 'whereby', 'once', 'perhaps', 'by', 'zebra', 'yours', 'out', 'i', 'latter', '‘s', 'we', 'thereafter', 'used', 'who', '‘ll', 'until', 'now', 'rather', 'move', 'already', 'otherwise', 'at', 'made', 'my', 'across', 'some', '‘re', '’d', 'whoever', 'sixty', 'nine', 'it', 'because', 'with', 'very', 'another', 'doing', 'really', 'nevertheless', 'then', 'regarding', 'before', 'this', 'seems', 'whose', 'therein', 'without', 'alone', 'done', 'became', 'an', 'again', 'sometimes', 'above', 'herein', "'m", 'ten', 'cannot', 'ever', 'elsewhere', 'back', 'indeed', 'may', 'meanwhile', 'onto', 'least', 'together', 'us', 'beforehand', 'throughout', 'even', '‘d', 'below', 'had', 'are', 'through', 'will', 'of', 'everywhere', 'along', 'myself', 'six', 'after', 'whom', 'but', 'for', 'make', 'am', 'except', 'often', 'hereupon', 'full', 'about', 'ourselves', 'those', 'what', 'also', 'besides', 'top', 'few', 'mine', 'four', 'less', '’ve', 'them', 'among', 'ours', 'both', 're', 'nor', 'itself', "'ll", 'from', 'nothing', 'formerly', 'he', 'whenever', 'you', 'nowhere', 'empty', 'amount', 'does', 'noone', 'seemed', 'no', 'than', 'via', '’s', 'somewhere', 'within', 'other', 'not'}~~~
{: .output}

This will only adjust the stopwords for the current session, but it is possible to save them if desired.
You might use this stopword list to filter words from documents using spacy, or just by manually iterating through it like a list.
This finishes the preprocessing steps we will discuss. The document is now ready for analysis in Spacy. Next we will discuss some of the principles behind natural language models like Spacy.
