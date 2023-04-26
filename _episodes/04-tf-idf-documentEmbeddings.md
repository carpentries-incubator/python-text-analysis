---
title: "Document Embeddings and TF-IDF"
teaching: 20
exercises: 10
questions:
- "What is a document embedding?"
- "What is TF-IDF?"
objectives:
- "Produce TF-IDF matrix on a corpus"
- "Understand how TF-IDF relates to rare/common words"
keypoints:
- "Some words convey more information about a corpus than others"
- "One-hot encodings treat all words equally"
- "TF-IDF encodings weigh overly common words lower"
---

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

Next, let's load a vectorizer from `sklearn` that will help represent our corpus in TF-IDF vector space for us.

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

~~~
(41, 9879)
~~~
{: .output}

Here, `tfidf.shape` shows us the number of rows (books) and columns (words) are in our model.

> ## Check Your Understanding: `max_df` and `min_df`
>
> Try different values for `max_df` and `min_df`. How does increasing/decreasing each value affect the number of columns (words) that get included in the model?
>
> > ## Solution
> >
> > Increasing `max_df` results in more words being included in the more, since a higher `max_df` corresponds to accepting more common words in the model. A higher `max_df` accepts more words likely to be stopwords.
> > 
> > Inversely, increasing `min_df` reduces the number of words in the more, since a higher `min_df` corresponds to removing more rare words from the model. A higher `min_df` removes more words likely to be typos, names of characters, and so on.
> {: .solution}
{: .challenge}


### Inspecting Results

We have a huge number of dimensions in the columns of our matrix (just shy of 10,000), where each one of which represents a word. We also have a number of documents (about forty), each represented as a row.

Let's take a look at some of the words in our documents. Each of these represents a dimension in our model.

```python
vectorizer.get_feature_names_out()[0:5]
```

~~~
array(['15th', '1st', 'aback', 'abandonment', 'abase'], dtype=object)
~~~
{: .output}

What is the weight of those words?

```python
print(vectorizer.idf_[0:5]) # weights for each token
```

~~~
[2.79175947 2.94591015 2.25276297 2.25276297 2.43508453]
~~~
{: .output}

Let's show the weight for all the words:

```python
from pandas import DataFrame
tfidf_data = DataFrame(vectorizer.idf_, index=vectorizer.get_feature_names_out(), columns=["Weight"])
tfidf_data
```

~~~
            Weight
15th        2.791759
1st         2.945910
aback	      2.252763
abandonment	2.252763
abase	      2.435085
...	        ...
zealously	  2.945910
zenith	    2.791759
zest	      2.791759
zigzag	    2.945910
zone	      2.791759
~~~
{: .output}

```python
tfidf_data.sort_values(by="Weight")
```

That was ordered alphabetically. Let's try from lowest to heighest weight:

~~~
              Weight
unaccountable	1.518794
nest	        1.518794
needless	    1.518794
hundred	      1.518794
hunger	      1.518794
...	          ...
incurably	    2.945910
indecent	    2.945910
indeed	      2.945910
incantation	  2.945910
gentlest	    2.945910
~~~
{: .output}

> ## Your Mileage May Vary
> 
> The results above will differ based on how you configured your tokenizer and vectorizer earlier.
{: .callout}

Values are no longer just whole numbers such as 0, 1 or 2. Instead, they are weighted according to how often they occur. More common words have lower weights, and less common words have higher weights.

## TF-IDF Summary

In this lesson, we learned about document embeddings and how they could be done in multiple ways. While one hot encoding is a simple way of doing embeddings, it may not be the best representation.
TF-IDF is another way of performing these embeddings that improves the representation of words in our model by weighting them. TF-IDF is often used as an intermediate step in some of the more advanced models we will construct later.