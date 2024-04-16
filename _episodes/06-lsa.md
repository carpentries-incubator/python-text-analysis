---
title: "Latent Semantic Analysis"
teaching: 20
exercises: 10
questions:
- "What is topic modeling?"
- "What is Latent Semantic Analysis (LSA)?"
objectives:
- "Use LSA to explore topics in a corpus"
- "Produce and interpret an LSA plot"
keypoints:
- "Topic modeling helps explore and describe the content of a corpus"
- "LSA defines topics as spectra that the corpus is distributed over"
- "Each dimension (topic) in LSA corresponds to a contrast between positively and negatively weighted words"
---

So far, we've learned the kinds of task NLP can be used for, preprocessed our data, and represented it as a TF-IDF vector space.

Now, we begin to close the loop with Topic Modeling — one of many embedding-related tasks possible with NLP.

![The Interpretive Loop](../images/01-Interpretive_Loop.JPG)

Topic Modeling is a frequent goal of text analysis. Topics are the things that a document is about, by some sense of "about." We could think of topics as:

- discrete categories that your documents belong to, such as fiction vs. non-fiction
- or spectra of subject matter that your documents contain in differing amounts, such as being about politics, cooking, racing, dragons, ...

In the first case, we could use machine learning to predict discrete categories, such as [trying to determine the author of the Federalist Papers](https://towardsdatascience.com/hamilton-a-text-analysis-of-the-federalist-papers-e64cb1764fbf).

In the second case, we could try to determine the least number of topics that provides the most information about how our documents differ from one another, then use those concepts to gain insight about the "stuff" or "story" of our corpus as a whole.

In this lesson we'll focus on this second case, where topics are treated as spectra of subject matter. There are a variety of ways of doing this, and not all of them use the vector space model we have learned. For example:

- Vector-space models:
  - Principle Component Analysis (PCA)
  - Epistemic Network Analysis (ENA)
  - Latent Semantic Analysis (LSA)
- Probability models:
  - Latent Dirichlet Allocation (LDA)

Specifically, we will be discussing Latent Semantic Analysis (LSA). We're narrowing our focus to LSA because it introduces us to concepts and workflows that we will use in the future, in particular that of dimensional reduction.

## What is dimensional reduction?

Think of a map of the Earth. The Earth is a three dimensional sphere, but we often represent it as a two dimensional shape such as a square or circle. We are performing dimensional reduction- taking a three dimensional object and trying to represent it in two dimensions.

![Maps with different projections of the Earth](../images/05-projections.jpg)

Why do we create maps? It can often be helpful to have a two dimensional representation of the Earth. It may be used to get an approximate idea of the sizes and shapes of various countries next to each other, or to determine at a glance what things are roughly in the same direction. 

How do we create maps? There's many ways to do it, depending on what properties are important to us. We cannot perfectly capture area, shape, direction, bearing and distance all in the same model- we must make tradeoffs. Different projections will better preserve different properties we find desirable. But not all the relationships will be preserved- some projections will distort area in certain regions, others will distort directions or proximity. Our technique will likely depend on what our application is and what we determine is valuable.

Dimensional reduction for our data is the same principle. Why do we do dimensional reduction? When we perform dimensional reduction we hope to take our highly dimensional language data and get a useful 'map' of our data with fewer dimensions. We have various tasks we may want our map to help us with. We can determine what words and documents are semantically "close" to each other, or create easy to visualise clusters of points.

How do we do dimensional reduction? There are many ways to do dimensional reduction, in the same way that we have many projections for maps. Like maps, different dimensional reduction techniques have different properties we have to choose between- high performance in tasks, ease of human interpretation, and making the model easily trainable are a few. They are all desirable but not always compatible. When we lose a dimension, we inevitably lose data from our original representation. This problem is multiplied when we are reducing so many dimensions. We try to bear in mind the tradeoffs and find useful models that don't lose properties and relationships we find important. But "importance" depends on your moral theoretical stances. Because of this, it is important to carefully inspect the results of your model, carefully interpret the "topics" it identifies, and check all that against your qualitative and theoretical understanding of your documents.

This will likely be an iterative process where you refine your model several times. Keep in mind the adage: all models are wrong, some are useful, and a less accurate model may be easier to explain to your stakeholders.

## LSA

The assumption behind LSA is that underlying the thousands of words in our vocabulary are a smaller number of hidden ("latent") topics, and that those topics help explain the distribution of the words we see across our documents. In all our models so far, each dimension has corresponded to a single word. But in LSA, each dimension now corresponds to a hidden topic, and each of those in turn corresponds to the words that are most strongly associated with it.

For example, a hidden topic might be [the lasting influence of the Battle of Hastings on the English language](https://museumhack.com/english-language-changed/), with some documents using more words with Anglo-Saxon roots and other documents using more words with Latin roots. This dimension is "hidden" because authors don't usually stamp a label on their books with a summary of the linguistic histories of their words. Still, we can imagine a spectrum between words that are strongly indicative of authors with more Anglo-Saxon diction vs. words strongly indicative of authors with more Latin diction. Once we have that spectrum, we can place our documents along it, then move on to the next hidden topic, then the next, and so on, until we've discussed the fewest, strongest hidden topics that capture the most "story" about our corpus.

LSA requires two steps- first we must create a TF-IDF matrix, which we have already covered in our previous lesson. 

Next, we will perform dimensional reduction using a technique called SVD.

### Worked Example: LSA
In case you are starting from a fresh notebook, you will need to (1), mount Google drive (2) add the helper code to your path, and (3) load the data.csv file.

```python
# Run this cell to mount your Google Drive.
from google.colab import drive
drive.mount('/content/drive')

# Show existing colab notebooks and helpers.py file
from os import listdir
wksp_dir = '/content/drive/My Drive/Colab Notebooks/text-analysis/code'
listdir(wksp_dir)

# Add folder to colab's path so we can import the helper functions
import sys
sys.path.insert(0, wksp_dir)

# Read the data back in.
from pandas import read_csv
data = read_csv("/content/drive/My Drive/Colab Notebooks/text-analysis/data/data.csv")

```

Mathematically, these "latent semantic" dimensions are derived from our TF-IDF matrix, so let's begin there. From the previous lesson:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(input='filename', max_df=.6, min_df=.1) # Here, max_df=.6 removes terms that appear in more than 60% of our documents (overly common words like the, a, an) and min_df=.1 removes terms that appear in less than 10% of our documents (overly rare words like specific character names, typos, or punctuation the tokenizer doesn’t understand)
tfidf = vectorizer.fit_transform(list(data["Lemma_File"]))
print(tfidf.shape)
```

~~~
(41, 9879)
~~~
{: .output}

What do these dimensions mean? We have 41 documents, which we can think of as rows. And we have several thousands of tokens, which is like a dictionary of all the types of words we have in our documents, and which we represent as columns.

#### Dimension Reduction Via Singular Value Decomposition (SVD)
Now we want to reduce the number of dimensions used to represent our documents. We will use a technique called Singular Value Decomposition (SVD) to do so. SVD is a powerful linear algebra tool that works by capturing the underlying patterns and relationships within a given matrix. When applied to a TF-IDF matrix, it identifies the most significant patterns of word co-occurrence across documents and condenses this information into a smaller set of "topics," which are abstract representations of semantic themes present in the corpus. By reducing the number of dimensions, we gradually distill the essence of our corpus into a concise set of topics that capture the key themes and concepts across our documents. This streamlined representation not only simplifies further analysis but also uncovers the latent structure inherent in our text data, enabling us to gain deeper insights into its content and meaning.

To see this, let's begin to reduce the dimensionality of our TF-IDF matrix using SVD, starting with the greatest number of dimensions (min(#rows, #cols)). In this case the maxiumum number of 'topics' corresponds to the number of documents- 41.

```python
from sklearn.decomposition import TruncatedSVD
maxDimensions = min(tfidf.shape)-1
svdmodel = TruncatedSVD(n_components=maxDimensions, algorithm="arpack")
lsa = svdmodel.fit_transform(tfidf)
print(lsa)
```

~~~
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
~~~
{: .output}

Unlike with a globe, we must make a choice of how many dimensions to cut out. We could have anywhere between 41 topics to 2. 

How should we pick a number of topics to keep? Fortunately, the dimension reducing technique we used produces something to help us understand how much data each topic explains.
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

~~~
[0.02053967 0.12553786 0.08088013 0.06750632 0.05095583 0.04413301
  0.03236406 0.02954683 0.02837433 0.02664072 0.02596086 0.02538922
  0.02499496 0.0240097  0.02356043 0.02203859 0.02162737 0.0210681
  0.02004    0.01955728 0.01944726 0.01830292 0.01822243 0.01737443
  0.01664451 0.0160519  0.01494616 0.01461527 0.01455848 0.01374971
  0.01308112 0.01255502 0.01201655 0.0112603  0.01089138 0.0096127
  0.00830014 0.00771224 0.00622448 0.00499762]
~~~
{: .output}

![Image of drop-off of variance explained](../images/05-svd-dropoff.png)

Often a heuristic used by researchers to determine a topic count is to look at the dropoff in percentage of data explained by each topic.

Typically the rate of data explained will be high at first, dropoff quickly, then start to level out. We can pick a point on the "elbow" where it goes from a high level of explanation to where it starts leveling out and not explaining as much per topic. Past this point, we begin to see diminishing returns on the amount of the "stuff" of our documents we can cover quickly. This is also often a good sweet spot between overfitting our model and not having enough topics.

Alternatively, we could set some target sum for how much of our data we want our topics to explain, something like 90% or 95%. However, with a small dataset like this, that would result in a large number of topics, so we'll pick an elbow instead.

Looking at our results so far, a good number in the middle of the "elbow" appears to be around 5-7 topics. So, let's fit a model using only 6 topics and then take a look at what each topic looks like.

> ## Why is the first topic, "Topic 0," so low?
>
> It has to do with how our SVD was setup. Truncated SVD does not mean center the data beforehand, which takes advantage of sparse matrix algorithms by leaving most of the data at zero. Otherwise, our matrix will me mostly filled with the negative of the mean for each column or row, which takes much more memory to store. The math is outside the scope for this lesson, but it's expected in this scenario that topic 0 will be less informative than the ones that come after it, so we'll skip it.
{: .callout}

```python
numDimensions = 7
svdmodel = TruncatedSVD(n_components=numDimensions, algorithm="arpack")
lsa = svdmodel.fit_transform(tfidf)
print(lsa)
```

~~~
[[ 3.91364432e-01 -3.38256707e-01 -1.10255485e-01 -1.57263147e-01
  4.46988327e-01  4.19701195e-02 -1.60554169e-01]
  ...
~~~
{: .output}

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

~~~
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
~~~
{: .output}

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

![Plot results of our LSA model](../images/05-lsa-plot.png)

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

![Plot results of our LSA model, color-coded by author](../images/05-lsa-plot-color.png)

It seems that some of the books by the same author are clumping up together in our plot.

We don't know *why* they are getting arranged this way, since we don't know what more concepts X and Y correspond to. But we can work do some work to figure that out.

### Topics

Let's write a helper to get the strongest words for each topic. This will show the terms with the *highest* and *lowest* association with a topic. In LSA, each topic is a spectra of subject matter, from the kinds of terms on the low end to the kinds of terms on the high end. So, inspecting the *contrast* between these high and low terms (and checking that against our domain knowledge) can help us interpret what our model is identifying.

```python
def show_topics(topic, n):
    terms = vectorizer.get_feature_names_out()
    weights = svdmodel.components_[topic]
    df = pandas.DataFrame({"Term": terms, "Weight": weights})
    tops = df.sort_values(by=["Weight"], ascending=False)[0:n]
    bottoms = df.sort_values(by=["Weight"], ascending=False)[-n:]
    return pandas.concat([tops, bottoms])

topic_words_x = show_topics(1, 5)
topic_words_y = show_topics(2, 5)
```

You can also use a helper we prepared for learners:

```python
from helpers import show_topics
topic_words_x = show_topics(vectorizer, svdmodel, topic_number=1, n=5)
topic_words_y = show_topics(vectorizer, svdmodel, topic_number=2, n=5)
```

Either way, let's look at the terms for the X topic.

What does this topic seem to represent to you? What's the contrast between the top and bottom terms?

```python
print(topic_words_x)
```

~~~
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
~~~
{: .output}

And the Y topic.

What does this topic seem to represent to you? What's the contrast between the top and bottom terms?

```python
print(topic_words_y)
```

~~~
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
~~~
{: .output}

Now that we have names for our first two topics, let's redo the plot with better axis labels.

```python
lsa_plot(data, svdmodel, groupby="Author", colors=colormap, xlabel="Victorian vs. Elizabethan", ylabel="English vs. French")
```

![Plot results of our LSA model, revised with new axis labels](../images/05-lsa-plot-labeled.png)

> ## Check Your Understanding: Intrepreting LSA Results
>
> Let's repeat this process with the other 4 topics, which we tentatively called Z, W, P, and Q.
>
> In the first two topics (X and Y), some authors were clearly separated, but others overlapped. If we hadn't color coded them, we wouldn't be easily able to tell them apart.
>
> But in remaining topics, different combinations of authors get pulled apart or together. This is because these topics (Z, W, P, and Q) highlight different features of the data, *independent* of the features we've already captured above.
>
> Take a few moments to work through the steps above for the remaining dimensions Z, W, P, and Q, and chat with one another about what you think the topics being represented are.
{: .challenge}
