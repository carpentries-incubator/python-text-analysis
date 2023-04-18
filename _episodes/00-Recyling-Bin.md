## Curse of Dimensionality

Last time, we discussed how every time we added a new word, we would simply add a new dimension to our vector space model. But there is a drawback to simply adding more dimensions to our model. As the number of dimensions increases, the amount of data needed to make good generalizations about that model also goes up. This is sometimes referred to as the **curse of dimensionality**.

This lesson, we will be focusing on how we can load our data into a document-term matrix, while employing various strategies to keep the number of unique words in our model down, which will allow our model to perform better.


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




