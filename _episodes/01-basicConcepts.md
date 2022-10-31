---
title: "Getting Started"
teaching: 10
exercises: 0
questions:
- "What does NLP do?"
objectives:
- "Learn basic concept of topic modelling."
- "Learn basic concept of named entity recognition."
- "Learn basic concept of search."
- "Learn basic concept of document summarization."
keypoints:
- "Text is the main input of NLP."
- "Models are used to transform text into output, which will vary based on task."
- "Topic Modelling has topics as the output."
- "Named Entity recognition has word labels as the output."
- "Search has relevant documents as the output."
- "Document summarization has a summary as the output."
---

##Limitations of Digital Humanities
There are a wide variety of fields of study within the humanities. When planning this workshop, a informal survey was sent to discover what would be of interest to those identifying as humanists.
We received responses from art and literature scholars, historians, political scientists, psychologists, sociologists, librarians and many others.
Each of those fields brings a variety of research interests and methods to focus on a wide variety of questions.
Because the humanities is a broad field, we will touch on a variety of techniques and tasks which natural language processing can help accomplish.
We will also discuss some details about how these models work, so that researchers can understand the concepts underlying common tasks and how text analysis approaches these problems.
However, these methods are not infallible or without bias. They are simply another tool you can use to analyze texts and should be critically considered in the same way any other tool would be.
The goal of this workshop is not to replace or discredit existing humanist methods, but to help humanists learn about some of the things text analysis can help do and how these models work.

##What is Natural Language Processing?

Natural Language Processing, or NLP, attempts to process human languages using computer models.

##What does NLP do?
There are many possible uses for NLP. Machine Learning and Artificial Intelligence can be thought of as a set of computer
algorithms used to take a piece of text as an input and produce a desired output. What distinguishes NLP
from other types of machine learning is that text and human language is the main input for NLP tasks.

A model is a mathematical construct designed to turn our text input into a desired output,
which will vary based on the task. We can think of the various tasks NLP can do as different types
of desired outputs, which may require different models.

##What can I do with NLP
Some of the many functions of NLP include topic modelling and categorization,
named entity recognition, search, summarization and more.

##Topic Modelling
Topic modeling is a type of analysis that attempts to categorize texts.
Documents might be made to match categories defined by the user, in a process called supervised learning.
For example, we might set a number of authors as “categories” and try to identify which author wrote a text.
Alternatively, the computer might be asked to come up with a set number of topics, and create categories without precoded documents,
in a process called unsupervised learning. Supervised learning requires human labelling and intervention, where
unsupervised learning does not.

![Topic Modelling Graph](images/01-topicmodelling.png)


##Named Entity Recognition
The task of Named Entity Recognition is trying to label words belonging to a certain group.
The entities we are looking to recognize may be proper nouns, quantities, or even just words belonging to a certain category, such as animals.
A possible application of this would be to track co-occurrence of characters in different chapters in a book.

![Named Entity Recognition](images/01-ner.png)

#Search
Search attempts to retrieve documents that are similar to your query.
In order to do this, there must be some way to compute the similarity between documents.
A search query can be thought of as a small input document, and the outputs could be relevant documents stored in the corpus.

![Search and Document Summarization](images/01-search.png)

#Document Summarization
Document summarization takes documents which are longer, and attempts to output a document with the same meaning by finding
relevant snippets or by generating a smaller document that conveys the meaning of the first document.

#Text Prediction
Text prediction attempts to predict future text inputs from a user based on previous text inputs. Predictive text is used in search engines and also on smartphones to help correct inputs and speed up the process of text input.

#Preprocessing
Despite the variety of tasks, many NLP models have related underlying models and techniques to process data.
But before data can be processed, it has to be prepared by the machine for analysis in a step called “preprocessing.”
Our next lesson will discuss some of the steps of preprocessing in greater detail.

{% include links.md %}
