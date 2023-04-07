---
title: "Section3: Word Embeddings"
colour: "#fafac8"
start: true
teaching: 20
exercises: 20
questions:
- "todo"
objectives:
- "todo"
keypoints:
- "todo"
---

### Extracting more sophisticated representations of text data
So far, we've seen how word counts, TF-IDF, and LSA can help us extract useful features from text data and embed documents into vector spaces. LSA is one method that moves from simple representations (e.g., word counts) towards representations that reflect semantic meaning. With the help of machine learning, we can extract even more sophisticated representations compared to LSA. Representations built from machine learning models typically have better performance at many tasks such as text/author classificaiton.

The model used by spaCy is something called "FastText". We will discuss how it works later in the lesson. For now, let's focus on what we can do with more sophisticated embeddings.

We'll start by importing spaCy and downloading its medium-sized pre-trained model of English language. In general, larger models are expected to perform "better" and be more accurate overall. We could load spaCy's large model (en_core_web_lg) for optimal performance, but this model has a higher computational cost. A good practice is to first test your code using the small or medium model, and then switch to the large model once everything has been tested.

~~~
import spacy
spacy.cli.download("en_core_web_md") # download the medium-sized model of english language from spacy
~~~
{: .language-python }

Next, we can load the medium sized model and use it to analyze a document containing a single word, "dog".
~~~
nlp = spacy.load("en_core_web_md")
doc = nlp("dog")
~~~
{: .language-python }

We can extract a vector representation of each token in our document. SpaCy's medium model contains 20k unique vectors with 300-dimensions stored for each vector. Since there are over 150,000 words in the English language, some words will use the same vector representation.
~~~
# recall that spaCy will tokenize the text passed to the nlp object. We can print out the first token with
print(doc[0])

# next, we extract the vector attribute from our spaCy token, and print the vector's shape and contents
vector_rep = doc[0].vector
print(vector_rep.shape)
print(vector_rep)
~~~
{: .language-python }

~~~
dog
(300,)
[ 1.2330e+00  4.2963e+00 -7.9738e+00 -1.0121e+01  1.8207e+00  1.4098e+00
 -4.5180e+00 -5.2261e+00 -2.9157e-01  9.5234e-01  6.9880e+00  5.0637e+00
 -5.5726e-03  3.3395e+00  6.4596e+00 -6.3742e+00  3.9045e-02 -3.9855e+00
  1.2085e+00 -1.3186e+00 -4.8886e+00  3.7066e+00 -2.8281e+00 -3.5447e+00
  7.6888e-01  1.5016e+00 -4.3632e+00  8.6480e+00 -5.9286e+00 -1.3055e+00
  8.3870e-01  9.0137e-01 -1.7843e+00 -1.0148e+00  2.7300e+00 -6.9039e+00
  8.0413e-01  7.4880e+00  6.1078e+00 -4.2130e+00 -1.5384e-01 -5.4995e+00
  1.0896e+01  3.9278e+00 -1.3601e-01  7.7732e-02  3.2218e+00 -5.8777e+00
  6.1359e-01 -2.4287e+00  6.2820e+00  1.3461e+01  4.3236e+00  2.4266e+00
 -2.6512e+00  1.1577e+00  5.0848e+00 -1.7058e+00  3.3824e+00  3.2850e+00
  1.0969e+00 -8.3711e+00 -1.5554e+00  2.0296e+00 -2.6796e+00 -6.9195e+00
 -2.3386e+00 -1.9916e+00 -3.0450e+00  2.4890e+00  7.3247e+00  1.3364e+00
  2.3828e-01  8.4388e-02  3.1480e+00 -1.1128e+00 -3.5598e+00 -1.2115e-01
 -2.0357e+00 -3.2731e+00 -7.7205e+00  4.0948e+00 -2.0732e+00  2.0833e+00
 -2.2803e+00 -4.9850e+00  9.7667e+00  6.1779e+00 -1.0352e+01 -2.2268e+00
  2.5765e+00 -5.7440e+00  5.5564e+00 -5.2735e+00  3.0004e+00 -4.2512e+00
 -1.5682e+00  2.2698e+00  1.0491e+00 -9.0486e+00  4.2936e+00  1.8709e+00
  5.1985e+00 -1.3153e+00  6.5224e+00  4.0113e-01 -1.2583e+01  3.6534e+00
 -2.0961e+00  1.0022e+00 -1.7873e+00 -4.2555e+00  7.7471e+00  1.0173e+00
  3.1626e+00  2.3558e+00  3.3589e-01 -4.4178e+00  5.0584e+00 -2.4118e+00
 -2.7445e+00  3.4170e+00 -1.1574e+01 -2.6568e+00 -3.6933e+00 -2.0398e+00
  5.0976e+00  6.5249e+00  3.3573e+00  9.5334e-01 -9.4430e-01 -9.4395e+00
  2.7867e+00 -1.7549e+00  1.7287e+00  3.4942e+00 -1.6883e+00 -3.5771e+00
 -1.9013e+00  2.2239e+00 -5.4335e+00 -6.5724e+00 -6.7228e-01 -1.9748e+00
 -3.1080e+00 -1.8570e+00  9.9496e-01  8.9135e-01 -4.4254e+00  3.3125e-01
  5.8815e+00  1.9384e+00  5.7294e-01 -2.8830e+00  3.8087e+00 -1.3095e+00
  5.9208e+00  3.3620e+00  3.3571e+00 -3.8807e-01  9.0022e-01 -5.5742e+00
 -4.2939e+00  1.4992e+00 -4.7080e+00 -2.9402e+00 -1.2259e+00  3.0980e-01
  1.8858e+00 -1.9867e+00 -2.3554e-01 -5.4535e-01 -2.1387e-01  2.4797e+00
  5.9710e+00 -7.1249e+00  1.6257e+00 -1.5241e+00  7.5974e-01  1.4312e+00
  2.3641e+00 -3.5566e+00  9.2066e-01  4.4934e-01 -1.3233e+00  3.1733e+00
 -4.7059e+00 -1.2090e+01 -3.9241e-01 -6.8457e-01 -3.6789e+00  6.6279e+00
 -2.9937e+00 -3.8361e+00  1.3868e+00 -4.9002e+00 -2.4299e+00  6.4312e+00
  2.5056e+00 -4.5080e+00 -5.1278e+00 -1.5585e+00 -3.0226e+00 -8.6811e-01
 -1.1538e+00 -1.0022e+00 -9.1651e-01 -4.7810e-01 -1.6084e+00 -2.7307e+00
  3.7080e+00  7.7423e-01 -1.1085e+00 -6.8755e-01 -8.2901e+00  3.2405e+00
 -1.6108e-01 -6.2837e-01 -5.5960e+00 -4.4865e+00  4.0115e-01 -3.7063e+00
 -2.1704e+00  4.0789e+00 -1.7973e+00  8.9538e+00  8.9421e-01 -4.8128e+00
  4.5367e+00 -3.2579e-01 -5.2344e+00 -3.9766e+00 -2.1979e+00  3.5699e+00
  1.4982e+00  6.0972e+00 -1.9704e+00  4.6522e+00 -3.7734e-01  3.9101e-02
  2.5361e+00 -1.8096e+00  8.7035e+00 -8.6372e+00 -3.5257e+00  3.1034e+00
  3.2635e+00  4.5437e+00 -5.7290e+00 -2.9141e-01 -2.0011e+00  8.5328e+00
 -4.5064e+00 -4.8276e+00 -1.1786e+01  3.5607e-01 -5.7115e+00  6.3122e+00
 -3.6650e+00  3.3597e-01  2.5017e+00 -3.5025e+00 -3.7891e+00 -3.1343e+00
 -1.4429e+00 -6.9119e+00 -2.6114e+00 -5.9757e-01  3.7847e-01  6.3187e+00
  2.8965e+00 -2.5397e+00  1.8022e+00  3.5486e+00  4.4721e+00 -4.8481e+00
 -3.6252e+00  4.0969e+00 -2.0081e+00 -2.0122e-01  2.5244e+00 -6.8817e-01
  6.7184e-01 -7.0466e+00  1.6641e+00 -2.2308e+00 -3.8960e+00  6.1320e+00
 -8.0335e+00 -1.7130e+00  2.5688e+00 -5.2547e+00  6.9845e+00  2.7835e-01
 -6.4554e+00 -2.1327e+00 -5.6515e+00  1.1174e+01 -8.0568e+00  5.7985e+00]

~~~
{: .output }

All of the 300 dimensions in our model are used in the embedding for this one word!
Like LSA, these are machine driven dimensions that are intended to represent some hidden semantic meaning.
Unlike LSA, they are much harder for a human being to manually interpret.
One interesting property of these more complex embeddings is that they allow us to use consine similarity scores to find similar words.
What happens when we try computing the closest cosine similarity scores?

~~~
# we can determine the hash value of any word stored in spacy using the following code
your_word = "dog"
hash_value = nlp.vocab.strings[your_word]
print(hash_value)

# using this hash value, we can extract this word's vector representation as follows
vector_rep = nlp.vocab.vectors[hash_value]

# using this vector and spaCy's .most_similar function, we can extract some of the most similar tokens (10, in this case)
import numpy as np 
ms = nlp.vocab.vectors.most_similar(
    np.asarray([vector_rep]), n=10)

# the most_similar function returns keys (ms[0]), key indices (ms[1]), and similarity scores (ms[2]) for the top n most similar tokens. We can print the token in string format with the following
words = [nlp.vocab.strings[w] for w in ms[0][0]]
print(words)
~~~
{: .language-python }
~~~
Hash value for dog: 7562983679033046312
['dogsbody', 'wolfdogs', 'Baeg', 'duppy', 'pet(s', 'postcanine', 'Kebira', 'uppies', 'Toropets', 'moggie']
~~~
{: .output }

Notice that not all words are synonyms for dogs.
There are two key reasons for this: 
1. These embeddings are trained by machine learning models, based on the contexts in which they appear. It may be the case that related words such as 'dogsbody' often appear in similar contexts as the word dog over the corpus this model was trained on.
2. Since we are using the medium-sized model rather than the large model, there are fewer vectors available to represent each word (i.e. some words will map onto the same vector). 

> ## Comparing with spaCy's large language model
>
> In a new cell block, download and load spaCy's large language model (Hint: see our first couple of code blocks from this episode). Use this model to determine the 10 most similar words relative to "dog". What do you notice about the results? Do these words seem more comparable to dog than when using the medium-sized model? If you finish this exercise early, try changing your_word to a new reference word and exploring what words spaCy things are similar based on vector similarity (using the same procedure as before, i.e., spaCy's most_similar() function)
>
> > ## Solution
> > ~~~
> > spacy.cli.download("en_core_web_lg") # download spaCy's large model of english language
> > nlp = spacy.load("en_core_web_lg") # load the model
> > 
> > # specify word and extract hash value
> > your_word = "dog"
> > hash_value = nlp.vocab.strings[your_word]
> > print('Hash value for', your_word + ':', hash_value)
> > 
> > # using this hash value, we can extract this word's vector representation as follows
> > vector_rep = nlp.vocab.vectors[hash_value]
> > 
> > # using this vector and spaCy's .most_similar function, we can extract some of the most similar tokens (10, in this case)
> > ms = nlp.vocab.vectors.most_similar(np.asarray([vector_rep]), n=10)
> > 
> > # the most_similar function returns keys (ms[0]), key indices (ms[1]), and similarity scores (ms[2]) for the top n most similar tokens. We can print the token in string format with the following
> > words = [nlp.vocab.strings[w] for w in ms[0][0]]
> > print(words)
> > ~~~
> > {: .language-python }
> > ~~~
> > Hash value for dog: 7562983679033046312
> > ['dog', 'dogs', 'cat', 'puppy', 'pet', 'pup', 'canine', 'wolfdogs', 'dogsled', 'uppy']
> > ~~~
> > {: .output }
> > These words appear to be much more similar to dog than when we used the medium-sized model. This is because the large model has 685k vectors while the medium model only has 20k vectors. Remember to always use the large model once your code is up and running.
> {: .solution}
{: .challenge}



### Distributional hypothesis and Word2Vec
How are these embeddings created? A linguist called JR Firth once famously said “You shall know a word by the company it keeps.” This means words that repeatedly occur in similar contexts probably have similar meanings; often referred to as the distributional hypothesis.
This property is used in embedding models such as FastText and the related model Word2Vec. We'll start with Word2Vec as it's the basis for FastText.
Word2Vec doesn't just use the word itself to determine a representation in vector space, it uses the words surrounding our target word to help determine how it is embedded.

Word2vec starts by randomly initializing its embeddings for all words in the vocabulary.
Before the training process, these dimensions are meaningless and the embeddings do not work very well for any task.
However, Word2Vec will gradually adjust the values of these embeddings by slowly changing them based on training data.

How does Word2Vec adjust these embeddings? Word2Vec looks at a sliding window of words as it does two tasks.
The size of the window is a parameter we set, so for now we will say it is size 2.
Suppose we call the word we are training at position t in the text w(t). The word directly before the training word would be w(t-1) and the word after it, w(t+1).
Word2Vec now does two training tasks, described below:

![Image from Word2Vec research paper, by Mikolov et al](../images/06-word2vecModel.png) 

The “Continuous Bag of Words” training method takes as an input the words before and after our target word, and tries to guess our target word based on those words.
The “skipgram” method flips the task, taking as an input the one target word and trying to predict the surrounding context words.
Each time the task is done, the embeddings are slightly adjusted to match the correct answer from the corpus.
Word2Vec also selects random words from our corpus that are not related and asks the model to predict that these words are unrelated, in a process called “negative sampling.”
Negative sampling ensures unrelated words will have embeddings that drift further and further apart, while the standard two tasks bring related embeddings closer together.

Over time and a large set of data, embeddings will come to reflect the relationships between words.
Understandably it requires a decent amount of data to train Word2Vec. It is not always possible to train smaller datasets specifically for each task.
However, models are often pretrained on larger general sets of data and then refined or used on smaller sets more specifically related to the task we want to do.
We have loaded one such model when we loaded the Spacy language model.

Spacy uses a modified version of Word2Vec called FastText.
FastText is essentially the same as Word2Vec, the only difference is that instead of operating on tokens of entire words, it operates on sets of characters instead.
It also adds special characters for the beginning and end of words to designate their start and end.
In FastText, a word such as 'Hello' would be represented not one token, but six- "<H", "He", "el", "ll", "lo", "o>".
However the same training tasks of CBOW and Skipgrams would still be conducted to train the embeddings.
Surprisingly these tasks still work to capture semantic properties, even on parts of words.

In summary, FastText and Word2Vec are embedding algorithms that convert words into multidimensional embeddings.
While Word2Vec's embeddings are less easily manually interpretable they also capture related meanings in a way that our previous models did not.
These embeddings may tell us more about the contexts in which words appear, and may also be used for visualization and categorization the way our previous models were.
