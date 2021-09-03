---
title: "Text analysis: Key concepts"
teaching: 0
exercises: 0
questions:
- ""
objectives:
- "Learn basic concepts to understand text analysis."
- "Learn which concepts are necessary for pre-processing a text."
- "Learn the limitations of text analysis as evidence."
keypoints:
- "FIXME"
---
##Key Concepts for Text Analysis

Practioners of digital humanities often employ text analysis to look for patterns across a single text, or across a corpus of text that may be too large for a single human to read. Python includes several libraries that address text analysis, including nltk and sci-kit. Before we get to those, we'll discuss some fundamental concepts of text analysis, especially as they relate to pre-processing of the text (to make it amenable to analysis) and the methods and limitations of some common methods. 

###Tokenization

Tokenization is the process of breaking down texts (strings of characters) into words, groups of words, and sentences. A string of characters needs to be understood by a program as words, or even terms made up of more than one word.
[https://librarycarpentry.org/lc-tdm/04-tokenising-text/index.html] (https://librarycarpentry.org/lc-tdm/04-tokenising-text/index.html) and [https://librarycarpentry.org/lc-tdm/08-counting-tokens/index.html] (https://librarycarpentry.org/lc-tdm/08-counting-tokens/index.html). 

###Stop-Words

Stop-words are common words that are filtered out for more efficient natural language data processing. These words are filtered out because they don’t carry much significance about a corpus, such as “the,” “an,” “a,” “of,” “and/or,” “many.” Stop lists (groups of stop words) are curated by sorting terms by their collection frequency, or the total number of times that they appear in a document or corpus. Many open-source software packages for language processing, such as Python, include stop lists. 

###Lemmas (Stemming and Lemmatization)

Stemming and Lemmatization are used to group together words that are similar or forms of the same word. **Stemming** may be familiar if you’ve ever conducted a “wildcard” search in a library catalog - using the “*” symbol to indicate that you are looking for any word that begins with “digi”, for example; returning “digital”, “digitization”, and “digitizing.” **Lemmatization** is the more sophisticated of the two, and looks for the linguistic base of a word. Lemmatization can group words that mean the same thing but may not be grouped through simple “stemming,” such as [bring, brought…]

###tf-idf

Tf-idf stands for term frequency-inverse document frequency, and is a weight used as a statistical measure in information retrieval and text mining to evaluate the distinctiveness of a word in a collection or a corpus. **TF** stands for term frequency, and measures how frequently a term occurs in a document, and is determined by comparing a word’s count with the total number of words in a document. **IDF**, or inverse document frequency, measures a term’s importance. When computing TF, all of the terms are considered equally important; IDF then weighs down words that appear more frequently, as a way of picking out terms with rare occurrences.  
Because tf-idf sorts words by their frequency, it’s a useful tool for extracting terms from text. Tf-idf can be used to filter out stop-words. It can also be used to identify similar texts in a corpus. 

A more thorough discussion of Tf-idf and its applications for document analysis can be found at the *Programming Historian*: [https://programminghistorian.org/en/lessons/analyzing-documents-with-tfidf#tf-idf-definition-and-background] (https://programminghistorian.org/en/lessons/analyzing-documents-with-tfidf#tf-idf-definition-and-background)

###Topic Modeling

Topic modeling is a type of analysis that looks categories, or topics, within the text to determine what a set of documents are about. The topic model is based on a statistical model of occurrences of words. LDA, or Latent Dirichlet Allocation, is a commonly used model for natual language processing. An excellent introduction to topic modeling for the humanities is Ted Underwood's [Topic modeling made just simple enough] (https://tedunderwood.com/2012/04/07/topic-modeling-made-just-simple-enough/). 
