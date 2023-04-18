---
exercises: 20
keypoints:
- TODO
objectives:
- TODO
questions:
- TODO
teaching: 20
title: LSA
---

# Topic Modeling

So far, we've:

1. Warmed up with a hands on card sorting activity and a discussion about the kinds of tasks we have as qualitative or mixed methods researchers
2. Situated ourselves with the kinds of tasks machine learning is well suited for, alongside some of the common ethical risk areas for data-driven research and technology
3. Narrowed our focus to modern Natural Language Processing tasks
4. Introduced vector spaces for thinking about the differences and similarities between several documents at once
5. Prepared our data for processing in Python
6. And introduced word embeddings and TF-IDF matrices as one approach for representing our documents as large vector spaces

We now begin to close the loop with Topic Modeling.

Topic Modeling is a frequent goal of text analysis. Topics are the things that a document is about, by some sense of "about." We could think of topics as:

- discrete categories that your documents belong to, such as fiction vs. non-fiction
- or spectra of subject matter that your documents contain in differing amounts, such as being about politics, cooking, racing, dragons, ...

In the first case, we could use machine learning to predict discrete categories, such as [trying to determine the author of the Federalist Papers](https://towardsdatascience.com/hamilton-a-text-analysis-of-the-federalist-papers-e64cb1764fbf).

In the second case, we could try to determine the least number of topics that provides the most information about how our documents differ from one another, then use those concepts to gain insight about the "stuff" or "story" of our documents.

In this lesson we'll focus on this second case, where topics are treated as spectra of subject matter. There are a variety of ways of doing this, and not all of them use the vector space model we have learned. For example:

- Vector-space models:
    - Principle Component Analysis (PCA)
    - Epistemic Network Analysis (ENA)
    - Linear Discriminant Analysis (LDA)
    - Latent Semantic Analysis (LSA)
- Probability models:
    - Latent Dirichlet Allocation (LDA)

Specifically, we will be discussing Latent Semantic Analysis (LSA). We're narrowing our focus to LSA because it introduces us to concepts and workflows that we will use in the future.

The assumption behind LSA is that underlying the thousands of words in our vocabulary are a smaller number of hidden topics, and that those topics help explain the distribution of the words we see across our documents. In all our models so far, each dimension has corresponded to a single word. But in LSA, each dimension now corresponds to a hidden topic, and each of those in turn corresponds to the words that are most strongly associated with it.

For example, a hidden topic might be [the lasting influence of the Battle of Hastings on the English language](https://museumhack.com/english-language-changed/), with some documents using more words with Anglo-Saxon roots and other documents using more words with Latin roots. This dimension is "hidden" because authors don't usually stamp a label on their books with a summary of the linguistic histories of their words. Still, we can imagine a spectrum between words that are strongly indicative of authors with more Anglo-Saxon diction vs. words strongly indicative of authors with more Latin diction. Once we have that spectrum, we can place our documents along it, then move on to the next hidden topic, then the next, and so on, until we've discussed the fewest, strongest hidden topics that capture the most "story" about our documents.

## Setting Up TF-IDF

Mathematically, these "latent semantic" dimensions are derived from our TF-IDF matrix, so let's begin there.

First we need to download the data we'll be using and do a little python setup.


```python
!pip install parse
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: parse in /usr/local/lib/python3.9/dist-packages (1.19.0)
    /bin/bash: svn: command not found



```python
# Run this cell to mount your Google Drive.
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).



```python
# add folder to colab's path so we can import the helper functions
import sys
wksp_dir = '/content/drive/My Drive/Colab Notebooks/text-analysis'
sys.path.insert(0, wksp_dir)
listdir(wksp_dir)

# and import our functions from helpers.py
from helpers import create_file_list, parse_into_dataframe, lemmatize_files
```


```python
# get list of files to analyze
data_dir = wksp_dir + '/data/'
corpus_file_list = create_file_list(data_dir, "*.txt")

# parse filelist into a dataframe
data = parse_into_dataframe(data_dir + "{Author}-{Title}.txt", corpus_file_list)
data
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




Next we'll load our tokenizer for processing our documents.

Then we'll preprocess each file, creating a "lemmatized" copy of each.

*Note: This can take several minutes to run. It took me around 7 minutes when I was writing these instructions.*


```python
import spacy, logging
tokenizer = spacy.load('en_core_web_sm', disable=["parser", "ner", "textcat", "senter"])
tokenizer.max_length = 4500000
def lemmatize_files(tokenizer, corpus_file_list, pos_set={"ADJ", "ADV", "INTJ", "NOUN", "VERB"}, stop_set=set()):
    """
    Example:
        data["Lemma_File"] = lemmatize_files(tokenizer, corpus_file_list)
    """
    logging.warning("This function is computationally intensive. It may take several minutes to finish running.")
    N = len(corpus_file_list)
    lemma_filename_list = []
    for i, filename in enumerate(corpus_file_list):
        logging.info(f"{i+1} out of {N}: Lemmatizing {filename}")
        lemma_filename = filename + ".lemmas"
        lemma_filename_list.append(lemma_filename)
        open(lemma_filename, "w", encoding="utf-8").writelines(
            token.lemma_.lower() + "\n"
            for token in tokenizer(open(filename, "r", encoding="utf-8").read())
            if token.pos_ in pos_set
            and token.lemma_.lower() not in stop_set
            # and token.text_.lower() not in stop_set
        )

    return lemma_filename_list

data["LemmaFile"] = lemmatize_files(tokenizer, corpus_file_list)
```

    WARNING:root:This function is computationally intensive. It may take several minutes to finish running.



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-14-96b4175b727c> in <cell line: 26>()
         24     return lemma_filename_list
         25 
    ---> 26 data["LemmaFile"] = lemmatize_files(tokenizer, corpus_file_list)
    

    <ipython-input-14-96b4175b727c> in lemmatize_files(tokenizer, corpus_file_list, pos_set, stop_set)
         16         open(lemma_filename, "w", encoding="utf-8").writelines(
         17             token.lemma_.lower() + "\n"
    ---> 18             for token in tokenizer(open(filename, "r", encoding="utf-8").read())
         19             if token.pos_ in pos_set
         20             and token.lemma_.lower() not in stop_set


    /usr/local/lib/python3.9/dist-packages/spacy/language.py in __call__(self, text, disable, component_cfg)
       1009                 error_handler = proc.get_error_handler()
       1010             try:
    -> 1011                 doc = proc(doc, **component_cfg.get(name, {}))  # type: ignore[call-arg]
       1012             except KeyError as e:
       1013                 # This typically happens if a component is not initialized


    /usr/local/lib/python3.9/dist-packages/spacy/pipeline/trainable_pipe.pyx in spacy.pipeline.trainable_pipe.TrainablePipe.__call__()


    /usr/local/lib/python3.9/dist-packages/spacy/pipeline/tok2vec.py in predict(self, docs)
        123             width = self.model.get_dim("nO")
        124             return [self.model.ops.alloc((0, width)) for doc in docs]
    --> 125         tokvecs = self.model.predict(docs)
        126         return tokvecs
        127 


    /usr/local/lib/python3.9/dist-packages/thinc/model.py in predict(self, X)
        313         only the output, instead of the `(output, callback)` tuple.
        314         """
    --> 315         return self._func(self, X, is_train=False)[0]
        316 
        317     def finish_update(self, optimizer: Optimizer) -> None:


    /usr/local/lib/python3.9/dist-packages/thinc/layers/chain.py in forward(model, X, is_train)
         53     callbacks = []
         54     for layer in model.layers:
    ---> 55         Y, inc_layer_grad = layer(X, is_train=is_train)
         56         callbacks.append(inc_layer_grad)
         57         X = Y


    /usr/local/lib/python3.9/dist-packages/thinc/model.py in __call__(self, X, is_train)
        289         """Call the model's `forward` function, returning the output and a
        290         callback to compute the gradients via backpropagation."""
    --> 291         return self._func(self, X, is_train=is_train)
        292 
        293     def initialize(self, X: Optional[InT] = None, Y: Optional[OutT] = None) -> "Model":


    /usr/local/lib/python3.9/dist-packages/thinc/layers/chain.py in forward(model, X, is_train)
         53     callbacks = []
         54     for layer in model.layers:
    ---> 55         Y, inc_layer_grad = layer(X, is_train=is_train)
         56         callbacks.append(inc_layer_grad)
         57         X = Y


    /usr/local/lib/python3.9/dist-packages/thinc/model.py in __call__(self, X, is_train)
        289         """Call the model's `forward` function, returning the output and a
        290         callback to compute the gradients via backpropagation."""
    --> 291         return self._func(self, X, is_train=is_train)
        292 
        293     def initialize(self, X: Optional[InT] = None, Y: Optional[OutT] = None) -> "Model":


    /usr/local/lib/python3.9/dist-packages/thinc/layers/with_array.py in forward(model, Xseq, is_train)
         30 ) -> Tuple[SeqT, Callable]:
         31     if isinstance(Xseq, Ragged):
    ---> 32         return cast(Tuple[SeqT, Callable], _ragged_forward(model, Xseq, is_train))
         33     elif isinstance(Xseq, Padded):
         34         return cast(Tuple[SeqT, Callable], _padded_forward(model, Xseq, is_train))


    /usr/local/lib/python3.9/dist-packages/thinc/layers/with_array.py in _ragged_forward(model, Xr, is_train)
         85 ) -> Tuple[Ragged, Callable]:
         86     layer: Model[ArrayXd, ArrayXd] = model.layers[0]
    ---> 87     Y, get_dX = layer(Xr.dataXd, is_train)
         88 
         89     def backprop(dYr: Ragged) -> Ragged:


    /usr/local/lib/python3.9/dist-packages/thinc/model.py in __call__(self, X, is_train)
        289         """Call the model's `forward` function, returning the output and a
        290         callback to compute the gradients via backpropagation."""
    --> 291         return self._func(self, X, is_train=is_train)
        292 
        293     def initialize(self, X: Optional[InT] = None, Y: Optional[OutT] = None) -> "Model":


    /usr/local/lib/python3.9/dist-packages/thinc/layers/chain.py in forward(model, X, is_train)
         53     callbacks = []
         54     for layer in model.layers:
    ---> 55         Y, inc_layer_grad = layer(X, is_train=is_train)
         56         callbacks.append(inc_layer_grad)
         57         X = Y


    /usr/local/lib/python3.9/dist-packages/thinc/model.py in __call__(self, X, is_train)
        289         """Call the model's `forward` function, returning the output and a
        290         callback to compute the gradients via backpropagation."""
    --> 291         return self._func(self, X, is_train=is_train)
        292 
        293     def initialize(self, X: Optional[InT] = None, Y: Optional[OutT] = None) -> "Model":


    /usr/local/lib/python3.9/dist-packages/thinc/layers/chain.py in forward(model, X, is_train)
         53     callbacks = []
         54     for layer in model.layers:
    ---> 55         Y, inc_layer_grad = layer(X, is_train=is_train)
         56         callbacks.append(inc_layer_grad)
         57         X = Y


    /usr/local/lib/python3.9/dist-packages/thinc/model.py in __call__(self, X, is_train)
        289         """Call the model's `forward` function, returning the output and a
        290         callback to compute the gradients via backpropagation."""
    --> 291         return self._func(self, X, is_train=is_train)
        292 
        293     def initialize(self, X: Optional[InT] = None, Y: Optional[OutT] = None) -> "Model":


    /usr/local/lib/python3.9/dist-packages/thinc/layers/maxout.py in forward(model, X, is_train)
         51     W = model.get_param("W")
         52     W = model.ops.reshape2f(W, nO * nP, nI)
    ---> 53     Y = model.ops.gemm(X, W, trans2=True)
         54     Y += model.ops.reshape1f(b, nO * nP)
         55     Z = model.ops.reshape3f(Y, Y.shape[0], nO, nP)


    KeyboardInterrupt: 



```python
lemmas_file_list = []
pos_set = {"ADJ", "ADV", "INTJ", "NOUN", "VERB"}
for i, filename in enumerate(corpus_file_list):
  print(i+1, "out of", len(corpus_file_list), "Lemmatizing", filename)
  lemma_filename = filename + ".lemmas"
  lemmas_file_list.append(lemma_filename)
  open(lemma_filename, "w", encoding="utf-8").writelines(
    token.lemma_.lower() + "\n"
    for token in tokenizer(open(filename, "r", encoding="utf-8").read())
    if token.pos_ in pos_set
  )

print(lemmas_file_list)
```

    1 out of 41 Lemmatizing python-text-analysis/data/dickens-olivertwist.txt
    2 out of 41 Lemmatizing python-text-analysis/data/dumas-montecristo.txt
    3 out of 41 Lemmatizing python-text-analysis/data/melville-typee.txt
    4 out of 41 Lemmatizing python-text-analysis/data/dumas-twentyyearsafter.txt
    5 out of 41 Lemmatizing python-text-analysis/data/dumas-blacktulip.txt
    6 out of 41 Lemmatizing python-text-analysis/data/dumas-tenyearslater.txt
    7 out of 41 Lemmatizing python-text-analysis/data/dickens-davidcopperfield.txt
    8 out of 41 Lemmatizing python-text-analysis/data/chesterton-thursday.txt
    9 out of 41 Lemmatizing python-text-analysis/data/shakespeare-lear.txt
    10 out of 41 Lemmatizing python-text-analysis/data/shakespeare-midsummer.txt
    11 out of 41 Lemmatizing python-text-analysis/data/chesterton-whitehorse.txt
    12 out of 41 Lemmatizing python-text-analysis/data/chesterton-ball.txt
    13 out of 41 Lemmatizing python-text-analysis/data/dickens-taleoftwocities.txt
    14 out of 41 Lemmatizing python-text-analysis/data/chesterton-napoleon.txt
    15 out of 41 Lemmatizing python-text-analysis/data/shakespeare-romeo.txt
    16 out of 41 Lemmatizing python-text-analysis/data/chesterton-brown.txt
    17 out of 41 Lemmatizing python-text-analysis/data/dickens-pickwickpapers.txt
    18 out of 41 Lemmatizing python-text-analysis/data/austen-pride.txt
    19 out of 41 Lemmatizing python-text-analysis/data/dickens-hardtimes.txt
    20 out of 41 Lemmatizing python-text-analysis/data/melville-moby_dick.txt
    21 out of 41 Lemmatizing python-text-analysis/data/austen-emma.txt
    22 out of 41 Lemmatizing python-text-analysis/data/shakespeare-othello.txt
    23 out of 41 Lemmatizing python-text-analysis/data/melville-conman.txt
    24 out of 41 Lemmatizing python-text-analysis/data/dickens-ourmutualfriend.txt
    25 out of 41 Lemmatizing python-text-analysis/data/dickens-greatexpectations.txt
    26 out of 41 Lemmatizing python-text-analysis/data/shakespeare-muchado.txt
    27 out of 41 Lemmatizing python-text-analysis/data/chesterton-knewtoomuch.txt
    28 out of 41 Lemmatizing python-text-analysis/data/austen-northanger.txt
    29 out of 41 Lemmatizing python-text-analysis/data/dickens-christmascarol.txt
    30 out of 41 Lemmatizing python-text-analysis/data/austen-persuasion.txt
    31 out of 41 Lemmatizing python-text-analysis/data/melville-bartleby.txt
    32 out of 41 Lemmatizing python-text-analysis/data/austen-sense.txt
    33 out of 41 Lemmatizing python-text-analysis/data/dumas-threemusketeers.txt
    34 out of 41 Lemmatizing python-text-analysis/data/melville-piazzatales.txt
    35 out of 41 Lemmatizing python-text-analysis/data/shakespeare-caesar.txt
    36 out of 41 Lemmatizing python-text-analysis/data/melville-pierre.txt
    37 out of 41 Lemmatizing python-text-analysis/data/dumas-maninironmask.txt
    38 out of 41 Lemmatizing python-text-analysis/data/austen-ladysusan.txt
    39 out of 41 Lemmatizing python-text-analysis/data/melville-omoo.txt
    40 out of 41 Lemmatizing python-text-analysis/data/dickens-bleakhouse.txt
    41 out of 41 Lemmatizing python-text-analysis/data/shakespeare-twelfthnight.txt
    ['python-text-analysis/data/dickens-olivertwist.txt.lemmas', 'python-text-analysis/data/dumas-montecristo.txt.lemmas', 'python-text-analysis/data/melville-typee.txt.lemmas', 'python-text-analysis/data/dumas-twentyyearsafter.txt.lemmas', 'python-text-analysis/data/dumas-blacktulip.txt.lemmas', 'python-text-analysis/data/dumas-tenyearslater.txt.lemmas', 'python-text-analysis/data/dickens-davidcopperfield.txt.lemmas', 'python-text-analysis/data/chesterton-thursday.txt.lemmas', 'python-text-analysis/data/shakespeare-lear.txt.lemmas', 'python-text-analysis/data/shakespeare-midsummer.txt.lemmas', 'python-text-analysis/data/chesterton-whitehorse.txt.lemmas', 'python-text-analysis/data/chesterton-ball.txt.lemmas', 'python-text-analysis/data/dickens-taleoftwocities.txt.lemmas', 'python-text-analysis/data/chesterton-napoleon.txt.lemmas', 'python-text-analysis/data/shakespeare-romeo.txt.lemmas', 'python-text-analysis/data/chesterton-brown.txt.lemmas', 'python-text-analysis/data/dickens-pickwickpapers.txt.lemmas', 'python-text-analysis/data/austen-pride.txt.lemmas', 'python-text-analysis/data/dickens-hardtimes.txt.lemmas', 'python-text-analysis/data/melville-moby_dick.txt.lemmas', 'python-text-analysis/data/austen-emma.txt.lemmas', 'python-text-analysis/data/shakespeare-othello.txt.lemmas', 'python-text-analysis/data/melville-conman.txt.lemmas', 'python-text-analysis/data/dickens-ourmutualfriend.txt.lemmas', 'python-text-analysis/data/dickens-greatexpectations.txt.lemmas', 'python-text-analysis/data/shakespeare-muchado.txt.lemmas', 'python-text-analysis/data/chesterton-knewtoomuch.txt.lemmas', 'python-text-analysis/data/austen-northanger.txt.lemmas', 'python-text-analysis/data/dickens-christmascarol.txt.lemmas', 'python-text-analysis/data/austen-persuasion.txt.lemmas', 'python-text-analysis/data/melville-bartleby.txt.lemmas', 'python-text-analysis/data/austen-sense.txt.lemmas', 'python-text-analysis/data/dumas-threemusketeers.txt.lemmas', 'python-text-analysis/data/melville-piazzatales.txt.lemmas', 'python-text-analysis/data/shakespeare-caesar.txt.lemmas', 'python-text-analysis/data/melville-pierre.txt.lemmas', 'python-text-analysis/data/dumas-maninironmask.txt.lemmas', 'python-text-analysis/data/austen-ladysusan.txt.lemmas', 'python-text-analysis/data/melville-omoo.txt.lemmas', 'python-text-analysis/data/dickens-bleakhouse.txt.lemmas', 'python-text-analysis/data/shakespeare-twelfthnight.txt.lemmas']



```python
# %%shell

# cd python-text-analysis/data
# zip lemmas *.lemmas
```


```python
# %%shell

# cd python-text-analysis/data
# unzip lemmas.zip
```

And after that, we'll convert our lemmatized files to TF-IDF matrix format, just as we did in the previous lesson.

Recall, `max_df=.6` removes terms that appear in more than 60% of our documents (overly common words like the, a, an) and `min_df=.1` removes terms that appear in less than 10% of our documents (overly rare words like specific character names, typos, or punctuation the tokenizer doesn't understand). We're looking for that sweet spot where terms are frequent enough for us to build theoretical understanding of what they mean for our corpus, but not so frequent that they can't help us tell our documents apart.


```python
from sklearn.feature_extraction.text import TfidfVectorizer
lemmas_file_list = create_file_list("python-text-analysis/data", "*.txt.lemmas")
vectorizer = TfidfVectorizer(input='filename', max_df=.6, min_df=.1)
tfidf = vectorizer.fit_transform(lemmas_file_list)
print(tfidf.shape)
```

    (41, 9879)


## How LSA Works: SVD of TF-IDF Matrix

What do these dimensions mean? We have 41 documents, which we can think of as rows. And we have tens of thousands of tokens, which is like a dictionary of all the types of words we have in our documents, and which we represent as columns.

Now we want to reduce the number of dimensions used to represent our documents. Sure, we *could* talk through each of the tens of thousands words in our dictionary. And we *could* talk through each of our individual 41 documents. Qualitative researchers are capable of great things. But those approaches won't take advantage of our model, they require a HUGE page burden to walk a reader through, and they put all the pressure on you to notice cross-cutting themes on your own. Instead, we can use a technique from linear algebra--Singlar Value Decomposition (SVD)--to reduce those tens of thousands of words or 41 documents to a smaller set of more cross-cutting dimensions.

We won’t deeply dive into all the mathematics of SVD, but will discuss what happens in abstract. The mathematical technique we are using is called “SVD” because we are "decomposing" our original matrix and creating a special matrix with "singular values." Any matrix M of arbitrary size can always be split or decomposed into three matrices that multiply together to make M. There are often many non-unique ways to do this decomposition. 

The three resulting matrices are called U, Σ, and Vt.

![Image of SVD. Visualisation of the priciple of SVD by CMG Lee.](images/05-svd.png)

The U matrix is a matrix where there are documents as rows, and different topics as columns. The scores in each cell show how much each document is "about" each topic. 

The Vt matrix is a matrix where there are a set of terms as columns, and different topics as rows. Again, the values in each cell correspond to how much a given word indicates a given topic.

The Σ matrix is special, and the one from which SVD gets its name. Nearly every cell in the matrix is zero. Only the diagonal cells are filled in: there are singular values in each row and column. Each singular value represent the amount of variation in our data explained by each topic--how much of the "stuff" of the story that topic covers.

A good deal of variation can often be explained by a relatively small number of topics, and often the variation each topic describes shrinks with each new topic.  Because of this, we can truncate or remove individual rows with the lowest singular values, since they provide the least amount of information.

Once this truncation happens, we can multiply together our three matrices and end up with a smaller matrix with topics instead of words as dimensions.

![Image of singular value decomposition with columns and rows truncated. ](images/05-truncatedsvd.png)

This allows us to focus our account of our documents on a narrower set of cross-cutting topics. This does come at a price though. When we reduce dimensionality, our model loses information about our dataset. Our hope is that the information lost was unimportant. But "importance" depends on your moral theoretical stances. Because of this, it is important to carefully inspect the results of your model, carefully interpret the "topics" it identifies, and check all that against your qualitative and theoretical understanding of your documents.

This will likely be an iterative process where you refine your model several times. Keep in mind the adage: all models are wrong, some are useful, and a less accurate model may be preferred if it is easier to explain to your stakeholders.

Question: What's the most possible topics we could get from this model? Think about what the most singular values are that you could possibly fit in the Σ matrix.

Remember, these singular values exist only on the diagonal, so the most topics we could have will be whichever we have fewer of- unique words or documents in our corpus.

Because there are usually more unique words than there are documents, it will almost always be equal to the number of documents we have, in this case 41.

To see this, let's begin to reduce the dimensionality of our TF-IDF matrix using SVD, starting with the greatest number of dimensions.


```python
from sklearn.decomposition import TruncatedSVD
maxDimensions = min(tfidf.shape)-1
svdmodel = TruncatedSVD(n_components=maxDimensions, algorithm="arpack")
lsa = svdmodel.fit_transform(tfidf)
print(lsa)
```

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

    [0.02053967 0.12553786 0.08088013 0.06750632 0.05095583 0.04413301
     0.03236406 0.02954683 0.02837433 0.02664072 0.02596086 0.02538922
     0.02499496 0.0240097  0.02356043 0.02203859 0.02162737 0.0210681
     0.02004    0.01955728 0.01944726 0.01830292 0.01822243 0.01737443
     0.01664451 0.0160519  0.01494616 0.01461527 0.01455848 0.01374971
     0.01308112 0.01255502 0.01201655 0.0112603  0.01089138 0.0096127
     0.00830014 0.00771224 0.00622448 0.00499762]



    
![png](LSA_files/LSA_18_1.png)
    


Often a heuristic used by researchers to determine a topic count is to look at the dropoff in percentage of data explained by each topic. 

Typically the rate of data explained will be high at first, dropoff quickly, then start to level out. We can pick a point on the "elbow" where it goes from a high level of explanation to where it starts levelling out and not explaining as much per topic. Past this point, we begin to see diminishing returns on the amount of the "stuff" of our documents we can cover quickly. This is also often a good sweet spot between overfitting our model and not having enough topics.

Alternatively, we could set some target sum for how much of our data we want our topics to explain, something like 90% or 95%. However, with a small dataset like this, that would result in a large number of topics, so we'll pick an elbow instead.

Looking at our results so far, a good number in the middle of the "elbow" appears to be around 5-7 topics. So, let's fit a model using only 6 topics and then take a look at what each topic looks like.

(Why is the first topic, "Topic 0," so low? It has to do with how our SVD was setup. Truncated SVD does not mean center the data beforehand, which takes advantage of sparse matrix algorithms by leaving most of the data at zero. Otherwise, our matrix will me mostly filled with the negative of the mean for each column or row, which takes much more memory to store. The math is outside the scope for this lesson, but it's expected in this scenario that topic 0 will be less informative than the ones that come after it, so we'll skip it.)


```python
numDimensions = 7
svdmodel = TruncatedSVD(n_components=numDimensions, algorithm="arpack")
lsa = svdmodel.fit_transform(tfidf)
print(lsa)
```

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


And put all our results together in one DataFrame so we can save it to a spreadsheet to save all the work we've done so far. This will also make plotting easier in a moment.

Since we don't know what these topics correspond to yet, for now I'll call the first topic X, the second Y, the third Z, and so on.


```python
import parse
import pandas
# "python-text-analysis/data/{Author}-{Title}.txt"
def parse_into_dataframe(pattern, items, col_name="Item"):
    results = []
    p = parse.compile(pattern)
    for item in items:
        result = p.search(item)
        if result is not None:
            result.named[col_name] = item
            results.append(result.named)
    
    return pandas.DataFrame.from_dict(results)
```


```python
import pandas
from numpy import mean
data = parse_into_dataframe("python-text-analysis/data/{Author}-{Title}.txt", lemmas_file_list)
# skipping w[0] because it is less informative in a truncated svd
# xs = [w[1] for w in lsa]
# ys = [w[2] for w in lsa]
# zs = [w[3] for w in lsa]
# ws = [w[4] for w in lsa]
# ps = [w[5] for w in lsa]
# qs = [w[6] for w in lsa]
# data = pandas.DataFrame({
#     "Author": authors,
#     "Title": titles,
#     # "X": xs - mean(xs),
#     # "Y": ys - mean(ys),
#     # "Z": zs - mean(zs),
#     # "W": ws - mean(ws),
#     # "P": ps - mean(ps),
#     # "Q": qs - mean(qs)
# })

data[["X", "Y", "Z", "W", "P", "Q"]] = lsa[:, [1, 2, 3, 4, 5, 6]]
data[["X", "Y", "Z", "W", "P", "Q"]] -= data[["X", "Y", "Z", "W", "P", "Q"]].mean()

data.to_csv("results.csv")
print(data)
```

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


## Inspecting Results

Let's plot the results, color-coding by author to see if any patterns are immediately apparent. We'll focus on the X and Y topics for now to illustrate the workflow. We'll return to the other topics in our model as a further exercise.


```python
colormap = {
    "austen": "red",
    "chesterton": "blue",
    "dickens": "green",
    "dumas": "orange",
    "melville": "cyan",
    "shakespeare": "magenta"
}

data["Color"] = data["Author"].map(colormap)

xR2 = round(svdmodel.explained_variance_ratio_[1] * 100, 2)
yR2 = round(svdmodel.explained_variance_ratio_[2] * 100, 2)
for author, books in data.groupby(by="Author"):
  books.plot(
      "X", "Y",
      kind="scatter",
      ax=plt.gca(),
      figsize=[5, 5],
      label=author,
      c="Color",
      xlim=[-1, 1],
      ylim=[-1, 1],
      title="My LSA Plot",
      xlabel=f"Topic X ({xR2}%)",
      ylabel=f"Topic Y ({yR2}%)"
  )
```


    
![png](LSA_files/LSA_25_0.png)
    


It seems that some of the books by the same author are clumping up and getting arrange in our plot, much like how we arranged cards in the card sorting activity much earlier.

We don't know *why* they are getting arranged this way, since we don't know what more meaningful concepts X and Y correspond to. But we can work do some work to figure that out.

Let's write a helper to get the strongest words for each topic. This will show the terms with the *highest* and *lowest* association with a topic. In LSA, each topic is a spectra of subject matter, from the kinds of terms on the low end to the kinds of terms on the high end. So, inspecting the *contrast* between these high and low terms (and checking that against our domain knowledge) can help us interpret what our model is identifying.


```python
def showTopics(topic, n):
  terms = vectorizer.get_feature_names_out()
  weights = svdmodel.components_[topic]
  df = pandas.DataFrame({"Term": terms, "Weight": weights})
  tops = df.sort_values(by=["Weight"], ascending=False)[0:n]
  bottoms = df.sort_values(by=["Weight"], ascending=False)[-n:]
  return pandas.concat([tops, bottoms])
```

Let's use it to get the terms for the X topic.

What does this topic seem to represent to you? What's the contrast between the top and bottom terms?


```python
print(showTopics(1, 5))
```

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


And the Y topic.

What does this topic seem to represent to you? What's the contrast between the top and bottom terms?


```python
print(showTopics(2, 5))
```

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


Now that we have names for our first two topics, let's redo the plot with better axis labels.


```python
for author, books in data.groupby(by="Author"):
  books.plot(
      "X", "Y",
      kind="scatter",
      ax=plt.gca(),
      figsize=[5, 5],
      label=author,
      c="Color",
      xlim=[-1, 1],
      ylim=[-1, 1],
      title="My LSA Plot",
      xlabel=f"Victorian vs. Elizabethan ({xR2}%)",
      ylabel=f"English vs. French ({yR2}%)"
    )
```


    
![png](LSA_files/LSA_33_0.png)
    


Finally, let's repeat this process with the other 4 topics, tentative called Z, W, P, and Q.

In the first two topics (X and Y), some authors were clearly separated, but others overlapped. If we hadn't color coded them, we wouldn't be easily able to tell them apart.

But in the next few topics, this flips, with different combinations of authors getting pulled apart and pulled together. This is because these topics (Z, W, P, and Q) highlight different features of the data, *independent* of the features we've already captured above.


```python
zR2 = round(svdmodel.explained_variance_ratio_[3] * 100, 2)
wR2 = round(svdmodel.explained_variance_ratio_[4] * 100, 2)
for author, books in data.groupby(by="Author"):
  books.plot(
      "Z", "W",
      kind="scatter",
      ax=plt.gca(),
      figsize=[5, 5],
      label=author,
      c="Color",
      xlim=[-1, 1],
      ylim=[-1, 1],
      title="My LSA Plot",
      xlabel=f"Topic Z ({zR2}%)",
      ylabel=f"Topic W ({wR2}%)"
    )
```

I wonder what these topics correspond to?


```python
print(showTopics(3, 5))
```


```python
print(showTopics(4, 5))
```


```python
for author, books in data.groupby(by="Author"):
  books.plot(
      "Z", "W",
      kind="scatter",
      ax=plt.gca(),
      figsize=[5, 5],
      label=author,
      c="Color",
      xlim=[-1, 1],
      ylim=[-1, 1],
      title="My LSA Plot",
      xlabel=f"Common vs. Royal ({zR2}%)",
      ylabel=f"Sea vs. Contractions ({wR2}%)"
    )
```

So far, Austen and Dickens have been lumped together, as have Chesterton and Mellville. But in these last two topics, P and Q, we can see their differences.


```python
pR2 = round(svdmodel.explained_variance_ratio_[5] * 100, 2)
qR2 = round(svdmodel.explained_variance_ratio_[6] * 100, 2)
for author, books in data.groupby(by="Author"):
  books.plot(
      "P", "Q",
      kind="scatter",
      ax=plt.gca(),
      figsize=[5, 5],
      label=author,
      c="Color",
      xlim=[-1, 1],
      ylim=[-1, 1],
      title="My LSA Plot",
      xlabel=f"Topic P ({pR2}%)",
      ylabel=f"Topic Q ({qR2}%)"
    )
```

I wonder what these topics correspond to?


```python
print(showTopics(5, 5))
```


```python
print(showTopics(6, 5))
```


```python
for author, books in data.groupby(by="Author"):
  books.plot(
      "P", "Q",
      kind="scatter",
      ax=plt.gca(),
      figsize=[5, 5],
      label=author,
      c="Color",
      xlim=[-1, 1],
      ylim=[-1, 1],
      title="My LSA Plot",
      xlabel=f"Land vs. Sea ({pR2}%)",
      ylabel=f"Exploration vs. Labor ({qR2}%)"
    )
```
