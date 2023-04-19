---
exercises: 20
keypoints:
- TODO
objectives:
- TODO
questions:
- TODO
teaching: 20
title: WordEmbeddingsIntro
---

## Word Embeddings Outline
#### Notebook1 - Word Embeddings Intro: https://colab.research.google.com/drive/1VJxo0CQ0NzrMcoxs2GXBvc27PRnpR4tq?usp=sharing

1. Recap of embeddings covered so far

2. Distributional hypothesis

    a. intro word2vec

3. show word2vec capability

    a. man/king example

4. Explain how it works using NN

    a. weights associated with each word

    b. explain two mechanisms: CBOW and SG

#### Notebook2 - Training Word2Vec From Scratch: https://colab.research.google.com/drive/16i0ScO0WMtknqSzCARVU1Cb3LEW5zZfF?usp=sharing

5. Train model from scratch to show additional use-cases (start in new notebook)

    a. categorical search using word2vec

    b. exercise: compare CBOW and SG

6. Additional research questions

7. Limitations & alternative word embedding methods

    a. Can't store multiple vectors/meanings per word

    b. It doesn’t have a good technique to deal with ambiguity. Two exact words but in two different contexts will have too close vectors.)

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



## Document/Corpus Embeddings Recap
**Note to instructor:** Run the cells below to load the pretrained Word2Vec model before explaining the below text.

So far, we’ve seen how word counts, TF-IDF, and LSA can help us embed a document or set of documents into useful vector spaces that allow us to gain insights from text data. Let's review the embeddings covered thus far...

##### **TF-IDF embeddings:** Determines the mathematical significance of words across multiple documents. It's embedding is based on (1) token/word frequency within each document and (2) how many documents a token appears in. 

##### **LSA embeddings:** Latent Semantic Analysis (LSA) is used to find the hidden topics represented by a group of documents. It involves running single-value decomposition on a document-term matrix (typically the TF-IDF matrix) — producing a vector representation of each document. This vector scores each document's representation in different topic areas which are derived based on word co-occurences. Importantly, LSA is considered a *bag of words* method since the order of words in a document is not considered. 

## Distributional hypothesis -- extracting more meaningful representations of text 
Compared to TF-IDF, the text representations (a.k.a. embeddings) produced by LSA are arguably more useful since LSA can reveal some of the latent topics referenced throughout a corpus. While LSA gets closer to extracting some of the rich semantic info stored in a corpus, it is limited in the sense that it is a "bag of words" method. That is, it pays no attention to the order in which words appear to create its embedding. A linguist called JR Firth once famously said “You shall know a word by the company it keeps.” This means words that repeatedly occur in similar contexts probably have similar meanings; often referred to as the distributional hypothesis. This property is exploited in embedding models such as Word2Vec.

### Word embeddings with Word2Vec
Word2vec is a popular word embedding method that relies on the distributional hypothesis to learn meaningful word representations (i.e., vectors) which can be used for an assortment of downstream analysis tasks. Unlike with TF-IDF and LSA, which are typically used to produce document and corpus embeddings, Word2Vec focuses on producing a single embedding for every word encountered in a corpus. We'll unpack the full algorithm behind Word2Vec shortly. First, let's see what we can do with meaningful word vectors.


#### Gensim
The Gensim library comes with many word embedding models -- including Word2Vec, GloVe, and fastText. We'll start by exploring one of the pretrained Word2Vec models. We'll discuss the other options later in this episode.


```python
# import numpy because numpy
import numpy as np
# api to load word2vec models
import gensim.downloader as api
```

Load the Word2Vec embedding model. Takes 3- 10 minutes to load.


```python
# wv = api.load('glove-wiki-gigaword-50')
wv = api.load('word2vec-google-news-300') # takes 3-10 minutes to load - started at 9:27, finished 9:30
```


```python
print(type(wv)) # ("gensim calls the pre-trained model object a keyed vectors") 
print(wv['whale'].shape) # 300-dims per word representation
wv['whale'] 
```

    <class 'gensim.models.keyedvectors.KeyedVectors'>
    (300,)





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



One we have our words represented as vectors (of length 300, in this case), we can start using some math to gain additional insights. For instance, we can compute the cosine similarity between two different word vectors using Gensim's similarity function. 

**Note**: In mathy terms, cosine similarity is a dot product of two vectors divided by the product of their lengths. In plain terms, cosine similarity helps evaluate vector similarity in terms of the angle that separates the two vectors, irrespective of vector magnitude. It can take a value ranging from -1 to 1, with...
* 1 indicating that the two vectors share the same angle
* 0 indicating that the two vectors are perpendicular or 90 degrees to one another
* -1 indicating that the two vectors are 180 degrees apart.

How similar are the two words, whale and dolphin?


```python
wv.similarity('whale','dolphin')
```




    0.77117145



How about whale and fish?


```python
wv.similarity('whale','fish')
```




    0.55177623



How about whale and... potato?


```python
wv.similarity('whale','potato')
```




    0.15530972



Our similarity scale seems to be on the right track. Let's take a look at the top 10 words associated with a commonly used term throughout the book, Moby Dick.


```python
print(wv.most_similar(positive=['whale'],topn=10))
```

    [('whales', 0.8474178910255432), ('humpback_whale', 0.7968777418136597), ('dolphin', 0.7711714506149292), ('humpback', 0.7535837292671204), ('minke_whale', 0.7365031838417053), ('humpback_whales', 0.7337379455566406), ('dolphins', 0.7213870882987976), ('humpbacks', 0.7138717174530029), ('shark', 0.7011443376541138), ('orca', 0.7007412314414978)]


Based on our ability to recover similar words, it appears the Word2Vec embedding method produces fairly good (i.e., semantically meaningful) word representations. 

### Adding and Subtracting Vectors: King - Man + Woman = Queen
We can also add and subtract word vectors to reveal latent meaning in words. As a canonical example, let's see what happens if we take the word vector representing "King", subtract the "Man" vector from it, and then add the "Woman" vector to the result. We should get a new vector that closely matches the word vector for Queen. We can test this idea out in Gensim with:




```python
print(wv.most_similar(positive=['woman','king'], negative=['man'],topn=3))
```

    [('queen', 0.7118193507194519), ('monarch', 0.6189674139022827), ('princess', 0.5902431011199951)]


### Visualizing word vectors with PCA


```python
# plot some of the vectors with some help from PCA
words = ['man','woman','boy','girl','king','queen','prince','princess']
sample_vectors = np.array([wv[word] for word in words])
sample_vectors.shape
```




    (8, 50)




```python
# can't visualize 100 dims... let's try 2 instead
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
result = pca.fit_transform(sample_vectors)
result
```




    array([[-1.8593105 , -1.6255254 ],
           [-2.349247  ,  0.27667248],
           [-2.4419675 , -0.47628874],
           [-2.6664796 ,  1.0604485 ],
           [ 2.6521494 , -1.9826698 ],
           [ 1.8861336 ,  0.81828904],
           [ 2.8712058 , -0.69794625],
           [ 1.9075149 ,  2.627021  ]], dtype=float32)




```python
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(result[:,0], result[:,1])
for i, word in enumerate(words):
  plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()
```


    
![png](WordEmbeddingsIntro_files/WordEmbeddingsIntro_25_0.png)
    


## Unpacking the Word2Vec Algorithm




How is it that word2vec is able to represent words in such a semanticallly meaningful way? There are two similar approaches to train a Word2Vec model — both resulting in meaningful word vectors:

**Image from Word2Vec research paper, by Mikolov et al**

#### Continuous Bag of Words (CBOW)
The “Continuous Bag of Words” training method takes as an input the words before and after our target word, and tries to guess our target word based on those words. 

#### Skip-gram (SG)
The “skipgram” method flips the task, taking as an input the one target word and trying to predict the surrounding context words. 

#### Training process
Each time the task is done, the embeddings (artificial neural network weights, in this case) are slightly adjusted to match the correct answer from the corpus. Word2Vec also selects random words from our corpus that are not related and asks the model to predict that these words are unrelated, in a process called “negative sampling.” Negative sampling ensures unrelated words will have embeddings that drift further and further apart, while the standard two tasks bring related embeddings closer together.

Both methods use artificial neural networks as their classification algorithm. Word2vec starts by randomly initializing its embeddings (weights of the neural network models) for all words in the vocabulary. Before the training process, these dimensions are meaningless and the embeddings do not work very well for any task. However, Word2Vec will gradually adjust the neural network weights to get better performance on the prediction task (i.e., predicting the missing word or predicting surrounding words in CBOW and SG, respectivitely). 

#### Unpack how vectors are extracted from model weights
After training on a sufficiently large corpus of text, the neural network will perform well on this prediction task. Once trained, the weights of the model can be used to represent meaningful underlying properties of our words.

## Next up
Training a Word2Vec model from scratch: https://colab.research.google.com/drive/16i0ScO0WMtknqSzCARVU1Cb3LEW5zZfF?usp=sharing
