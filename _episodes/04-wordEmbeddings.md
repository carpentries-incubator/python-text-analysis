---
title: "Word Embeddings and Complex Models"
teaching: 20
exercises: 20
questions:
- "todo"
objectives:
- "todo"
keypoints:
- "todo"
---
## One hot encoding

The method of using word counts is just one way we might embed a document in vector space. 
Let’s talk about a more complex and valuable way of looking at document embeddings. 
To start, we’ll look at how document embeddings are constructed. 
If we take a document such as “Feed the duck” we can view the document embedding as a sum of the embeddings of each word in the document. 
Let’s imagine we have a vector space with a million different words, and we are just looking at part of the vector space.


|      | dodge | duck | ... | farm | feather | feed | ... | tan | the |
|------|-------|------|-----|------|---------|------|-----|-----|-----|
| feed | 0     | 0    |     | 0    | 0       | 1    |     | 0   | 0   |
| the  | 0     | 0    |     | 0    | 0       | 0    |     | 0   | 1   |
| duck | 0     | 1    |     | 0    | 0       | 0    |     | 0   | 0   |
|------|-------|------|-----|------|---------|------|-----|-----|-----|
| Document | 0     | 1    |     | 0    | 0       | 1    |     | 0   | 1   |



The word embeddings are done in a way so that each word count gives a 1 for the word in the document, and a zero for every other field. 
This kind of encoding is known as “one hot” encoding, where a single value is 1 and all others are 0. 
Once we have all of the word embeddings for each word in the document, we sum them up to get the document embedding.


## Revisiting Assumptions
There are some problems with the model we've constructed.

1)	Some words are more important than others. 
The presence of the word “the” in a document does not tell us a lot about it, 
but the presence of a rarer word such as “farm” tells us quite a bit more about its content.
2)	Words have interrelated meanings. 
This type of word embedding is essentially saying that the word “tan” is as related to the word “duck” 
as the word “feather” is. If I am looking for a “bird with feathers”, a duck would be a good description, 
but neither ‘bird’ nor ‘feather’ are counted as part of the embedding. 

3)	Does not consider the relative positions of words within the document. 
Pronouns and adjectives may modify certain words and not others.
4)	Does not account for words that are ambiguous or have meanings that change based on context? 
How do we handle “duck” as a verb versus “duck” as an animal? How does the meaning and embedding change?

5)	We’re going to get a big empty table very fast if we keep adding words to it. 
Because most documents don’t use all or even most of the English vocabulary, this table will be mostly empty. 
This means processing will be quite slow, on a table which is "sparse", or mostly empty.

## Addressing relative weights of words

We might decide to move away from a one-hot word embedding method to something more nuanced.
Perhaps we might do something more like this instead:

|      | dodge | duck | ... | farm | feather | feed | ... | tan | the |
|------|-------|------|-----|------|---------|------|-----|-----|-----|
| feed | 0     | 0    |     | 0.1  | 0       | 1    |     | 0   | 0   |
| the  | 0     | 0    |     | 0    | 0       | 0    |     | 0   | 1   |
| duck | 0     | 1    |     | 0.2  | 0.5     | 0    |     | 0   | 0   |
|------|-------|------|-----|------|---------|------|-----|-----|-----|
| Document | 0     | 1    |     | 0.3    | 0.5       | 1    |     | 0   | 1   |

By scoring related words as part of our embedding, we convey that the words are related. A search for 
"bird with feathers" might now return our document as a result, since we have some value for "feather" in the embedding.

But how do we determine what these scores should be? 
How do we decide that a mention of duck is worth half a mention of feather, or a fifth of a mention of a farm? 
In real models, weights are not determined by humans, 
but instead trained over a large set of documents using statistical analysis. 
We might measure co-occurence of words within a large group of documents to determine their "relatedness", and use those pretrained weights on smaller corpora for future embeddings.

Determining better ways of computing word embedding scores is a key area of NLP research.
An early way of doing this is TF-IDF.


### tf-idf
We may have the intuition that certain words mean more than others when it comes to document similarity. 
Stop words are the most egregious examples, as words that occur so often as to be borderline meaningless, 
but the inverse also applies. The less likely a word is to occur, the more significance it takes on when it does occur.

Tf-idf stands for term frequency-inverse document frequency, 
and is a weight used as a statistical measure in information retrieval and text mining to evaluate the distinctiveness 
of a word in a collection or a corpus. **TF** stands for term frequency, and measures how frequently a term occurs in a 
document, and is determined by comparing a word’s count with the total number of words in a document. **IDF**, or inverse 
document frequency, measures a term’s importance. When computing TF, all of the terms are considered equally important;
 IDF then weighs down words that appear more frequently, as a way of picking out terms with rare occurrences.  
 
Because tf-idf scores words by their frequency, it’s a useful tool for extracting terms from text. 
We can also embed documents in vector space using TF-IDF scores rather than simple word counts. 
Tf-idf can be used to weaken the impact of stop-words, since due to their common nature, they have very low scores. 

## Addressing issues of context

To figure out an appropriate embedding for a word like duck, we might consider some of the words around it. 
An advanced model might have different embedding values for duck based on context.
One embedding might be used for one sense of the word which co-occurs with other animals, 
and another embedding might be used when duck occurs when used as a verb. 
Various models have different ways of doing this, which can grow complicated. 
Again, a lot of documents are required to tease out these distinctions, which is why pretrained models are required.

## Addressing matrix size

We discussed how most columns will generally be pretty empty when each word is considered a dimension.
Since we are moving away from one-to-one representations of word embeddings, we may also consider moving away from one word per dimension.
It's possible to compress large sparse tables into a fewer number of dimensions while still retaining the characteristics of the original matrix.
This can be thought of as like compressing an image- some information is lost but essential characteristics still remain of the original.
Most contemporary models do not have a clean one word per dimension architecture. 
Instead there's some fixed size of dimensions that may correspond to multiple words or senses of words depending on context.

This gives you some idea of how word and document embeddings grow more complex as models do, but are still based on our original concept of vector space.
Next we will discuss some models and how they may be used to achieve useful tasks.

