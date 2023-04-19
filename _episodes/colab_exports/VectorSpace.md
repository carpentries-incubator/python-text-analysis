---
exercises: 20
keypoints:
- TODO
objectives:
- TODO
questions:
- TODO
teaching: 20
title: VectorSpace
---

---
title: "Vector Space and Distance"
teaching: 20
exercises: 20
questions:
- "What are models?"
- "How can we model language?"
- "How does a vector space model work?"
- "How can we quantify similarity?"
- "Why do we use cosine distance?"
objectives:
- "Learn about vector space"
- "Learn about similarity metrics such as cosine similarity and distance."
keypoints:
- "Vectors can be used to represent language."
- "Vectors are objects with direction and magnitude."
- "Documents can be compared to each other using cosine similarity." 
---

Vector Space
==================
Before we start to discuss how NLP models work, we can ask a more general question: What is a model? The particulars may vary,  but in general models are mathematical representations of some concrete, real-world thing. There are many different ways that human languages can be modelled.

We will focus on a concept called Vector Space, because this model has proven successful at many NLP tasks. Vector Space is a complex concept, and so we will be starting with some basic assumptions of the model. Many of these assumptions will later be broken down or ignored. We are starting with a simple model and will scale up in complexity, modifying our assumptions as we go.

1.	We create __embeddings__, or mathematical surrogates, of words and documents in vector space. These embeddings can be represented as sets of coordinates in multidimensional space, or as multi-dimensional matrices.
2.	These embeddings should be based on some sort of __feature extraction__, meaning that meaningful features from our original documents are somehow represented in our embedding. This will make it so that relationships between embeddings in this vector space will correspond to relationships in the actual documents.
3.	To simplify things to start, we will use a __“bag of words”__ assumption as well. We will not consider the placement of words in sentences, their context or their conjugation into different forms (run vs ran) to start. The effect of this assumption is like putting all words from a sentence in a bag and considering them by count without regard to order. Later, we’ll talk about models where this no longer holds true.

Let's suppose we want to model a small, simple set of toy documents. Our entire corpus of documents will only have two words, which we will call word 'to' and word 'be'. We have four documents, A, B, C and D. Below is the label of the document, followed by its text:

- A: be be be be be be be be be be to
- B: to be to be to be to be to be to be to be to be 
- C: to to be be
- D: to be to be

We will start by embedding words using a "one hot" embedding algorithm. Each document is a new row in our table. Every time word 'to' shows up in a document, we add one to our value for the 'to' dimension for that row, and zero to every other dimension. Every time 'be' shows up in our document, we will add one to our value for the 'be' dimension for that row, and zero to every other dimension.

How does this corpus look in vector space? We can display our model using a __document-term matrix__, which looks like the following:

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

    [[ 1 10]
     [ 8  8]
     [ 2  2]
     [ 2  2]]





## Graphing our model
We don't just have to think of our words as columns. We can also think of them as dimensions, and the values as coordinates for each document. 


```python
#matplotlib expects a list of values by column, not by row.
#We can simply turn our table on its edge so rows become columns and vice versa.
corpusT = np.transpose(corpus)
print(corpusT)

```

    [[ 1  8  2  2]
     [10  8  2  2]]



```python
X = corpusT[0]
Y = corpusT[1]
#define some colors for each point. Since points A and B are the same, we'll have them as the same color.
mycolors = ['r','g','b','b']

#display our visualization
plt.scatter(X,Y, c=mycolors)
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.show()
```


    
![png](VectorSpace_files/VectorSpace_5_0.png)
    


What can we do with this simple model? One thing we might want to ask is compare the similarity of two documents. There are two metrics we'll consider: __distance__ and __cosine similarity__. 

Distance seems like an intuitive metric to use, so let's try using it. The Euclidian distance formula makes use of the Pythagorean theorem, where a^2 + b^2 = c^2. We can draw a triangle between two points, and calculate the hypotenuse to find the distance. This distance formula works in two dimensions, but can also be generalized over as many dimensions as we want. Let's use distance to compare A to B, C and D. We'll say the closer two points are, the more similar they are.


```python
from sklearn.metrics.pairwise import euclidean_distances as dist

#What is closest to document D?
D = [corpus[3]]
print(D)
dist(corpus, D)
```

    [array([2, 2])]





    array([[8.06225775],
           [8.48528137],
           [0.        ],
           [0.        ]])



Distance may seem like a decent metric at first. Certainly, it makes sense that document D has zero distance from itself. C and D are also similar, which makes sense given our bag of words assumption. But take a closer look at documents B and D. Document B is just document D copy and pasted 4 times! How can it be less similar to document D than document B?

Distance is highly sensitive to document length. Because document A is shorter than document B, it is closer to document D. While distance may be an intuitive measure of similarity, it is actually highly dependent on document length. 

We need a different metric that will better represent similarity. This is where vectors come in. Vectors are geometric objects with both length and direction. They can be thought of as a ray or an arrow pointing from one point to another.

Vectors can be added, subtracted, or multiplied together, just like regular numbers can. Our model will consider documents as vectors instead of points, going from the origin at (0,0) to each document. Let's visualize this.


```python
#we need the point of origin in order to draw a vector. Numpy has a function to create an array full of zeroes.
origin = np.zeros([1,4])
print(origin)

plt.quiver(origin, origin, X, Y, color=mycolors, angles='xy', scale_units='xy', scale=1)
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.show()
```

    [[0. 0. 0. 0.]]



    
![png](VectorSpace_files/VectorSpace_9_1.png)
    


Document A and document D are headed in exactly the same direction, which matches our intution that both documents are in some way similar to each other. 

__Cosine Similarity__ is a metric which is only concerned with the direction of the vector, not its length. This means the length of a document will no longer factor into our similarity metric. The more similar two vectors are in direction,
the closer the cosine similarity score gets to 1. And the more orthogonal two vectors get, the closer it gets to 0.

You can think of cosine similarity between vectors as signposts aimed out into multidimensional space. Two similar documents going in the same direction have a high cosine similarity, even if one of them is much further away in that direction. Now that we know what cosine similarity is, how does this metric compare our documents?



```python
from sklearn.metrics.pairwise import cosine_similarity as cs

cs(corpus, D)
```




    array([[0.7739573],
           [1.       ],
           [1.       ],
           [1.       ]])



Both A and D are considered similar by this metric. Cosine similarity is used by many models as a measure of similarity between documents and words.


Generalizing over more dimensions
==================

If we want to add another word to our model, we simply add another dimension, which we can represent as another column in our table. Let's add more documents with novel words in it.

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

We can keep adding dimensions for however many words we want to add. It's easy to imagine vector space with two or three dimensions, but visualizing this mentally will rapidly become downright impossible as we add more and more words. Vocabularies for natural languages can easily reach thousands of words.

Keep in mind, it’s not necessary to visualize how a high dimensional vector space looks. These relationships and formulae work over an arbitrary number of dimensions. Our methods for how to measure similarity will carry over, even if drawing a graph is no longer possible.


```python
#add two dimensions to our corpus
corpus = np.hstack((corpus, np.zeros((4,2))))
print(corpus)
```

    [[ 1. 10.  0.  0.]
     [ 8.  8.  0.  0.]
     [ 2.  2.  0.  0.]
     [ 2.  2.  0.  0.]]



```python
E = np.array([[0,2,1,1]])
F = np.array([[2,2,1,1]])

#add document E to our corpus
corpus = np.vstack((corpus, E))
print(corpus)

```

    [[ 1. 10.  0.  0.]
     [ 8.  8.  0.  0.]
     [ 2.  2.  0.  0.]
     [ 2.  2.  0.  0.]
     [ 0.  2.  1.  1.]]


What do you think the most similar document is to document F?


```python
cs(corpus, F)
```




    array([[0.69224845],
           [0.89442719],
           [0.89442719],
           [0.89442719],
           [0.77459667]])



This new document seems most similar to the documents B,C and D.

This principle of using vector space will hold up over an arbitrary number of dimensions, and therefore over a vocabulary of arbitrary size.

This is the essence of vector space modelling- documents are embedded as vectors in very high dimensional space.
How we define these dimensions and the methods for feature extraction may change and become more complex, but the essential idea remains the same.
Next, we will discuss preprocessing, which will break down documents into a format where we can put them into vector space.
