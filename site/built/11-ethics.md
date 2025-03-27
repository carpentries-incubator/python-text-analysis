---
title: "Ethics and Text Analysis"
teaching: 20
exercises: 20
questions:
- "Is text analysis artificial intelligence?"
- "How can training data influence results?"
- "What are the risk zones to consider when using text analysis for research?"
objectives:
- "Understand how text analysis fits into the larger picture of artificial intelligence"
- "Be able to consider the tool against your research objectives"
- "Consider the drawbacks and inherent biases that may be present in large language models"
keypoints:
- "Text analysis is a tool and can't assign meaning to results"
- "As researchers we are responsible for understanding and explaining our methods and results"
---


## Is text analysis artificial intelligence? 

Artificial intelligence is loosely defined as the ability for computer systems to perform tasks that have traditionally required human reasoning and perception. 
* To the extent that text analysis performs a task that resembles reading, understanding, and analyzing meaning, it can be understood to be part of the definition of artificial intelligence. 
* The methods in this lesson all demonstrate models that learn from data - specifically, from text corpora that are not structured to explicitly tell the machine anything other than, perhaps, title, author, date, and body of text.
* As a method and a tool, it is important to understand the tasks to which it is best suited, and to understand the process well enough to be able to interpret the results, including:

  1. whether the results are relevant or meaningful
  2. whether the results have been overly influenced by the model or training data 
  3. how to responsibly use the results 

We can describe these as commitments to ethical research methods. 

## Relevance or meaningfulness
As with any research, the relevance or meaningfulness of your results is relative to the research question itself. However, when you have a particular research question (or a particular set of research interests), it can be hard to connect the results of these models back to your bigger picture aims. It can feel like trying to write a book report but all you were given were the table of contents. One reason for this difficulty is that the dimensions of the model are atheoretical. That is, regardless of what research questions you are asking, the models always start from the same starting point: the words of the text, with no understanding of what those words mean to you. Our job is to interpret the meaning of the model’s results, or the qualitative work that follows. 

The model is making a statistical determination based on the training data it has been fed, and on the training itself, as well as the methods you have used to parse the data set you're analyzing. If you are using a tool like ChatGPT, you may have access only to your own methods, and will need to make an educated guess about the training data and training methods. That doesn't mean you can't use that tool, but it does mean you need to keep what is known and what is obscured about your methods at the forefront as you conduct your research.  

> *Exercise*: You use LSA as a method to identify important topics that are common across a set of popular 19th century English novels, and conclude that X is most common. How might you explain this result and why you used LSA? 


## Training data can influence results

There are numerous examples of how training data - or the language model, ultimately - can negatively influence results. Reproducing bias in the data is probably one of the most discussed negative outcomes. Let's look at one real world example:

In 2016, ProPublica published an investigative report that exposed the clear bias against Black people in computer programs used to determine the likelihood of defendants committing crimes in the future. That bias was built into the tool because the training data that it relied on included historical data about crime statistics, which reflected - and then reproduced - existing racist bias in sentencing. 

> *Exercise*: How might a researcher avoid introducing bias into their methodology when using pre-trained data to conduct text analysis? 

## Using your research

Rarely will results from topic modeling, text analysis, etc. stand on their own as evidence of anything. Researchers should be able to explain their method and how they got their results, and be able to talk about the data sets and training models used. As discussed above, though, the nature of the large language models that may underlie the methods used to do LSA topic modeling, identify relationships between words using Word2Vec, or summarize themes using BERT, is that they contain vast numbers of parameters that cannot be reverse engineered or described. The tool can still be part of the explanation, and any results that may change due to the randommness of the LLM can be called out, for example. 

## Risk zones

Another area to consider when using any technology are the risk zones that are introduced. We're talking about unintended consequences, for the most part, but consequences nonethless. 

Let's say you were using BERT to help summarize a large body of texts to understand broad themes and relationships. Could this same method be used to distort the contents of those texts to spread misinformation? How can we mitigate that risk? 

In the case of the LLMs that underlie many of the text analysis methods you learned in this workshop, is there a chance that the results could reinforce existing biases because of existing biases in the training data? Consider this example:

> *Exercise*: You are identifying topics across a large number of archival texts from hundreds of 20th century collections documenting LGBTQ organizations. You are using a LLM where the training data is petabytes of data collected over a decade of web crawling, starting in 2013. What risks are introduced by this method and how might they be anticipated and mitigated? 

## Hype cycles and AI

Because this workshop is being introduced shortly after the release of ChatGPT3 by OpenAI, we want to address how AI and tech hype cycles can influence tool selection and use of tech. The inscrutability of LLMs, the ability of chatbots to output coherent and meaningful text on a seemingly infinite variety of topics, and the rhetoric of the tech industry can make these tools seem magical and unfathomable. They aren't magical, though the black box nature of the training data and the parameters does lend itself to unfathomability. Regardless, the output of any of the methods described in this workshop, and by LLMs to come, is the product of mathematical processes and statistical weights. That is why learning some of the methodology behind text analysis is important, even if it takes much longer to become fluent in LSA or Word2Vec. We all will use tools based on these methods in the years to come, whether for our research or for more mundane administrative tasks. Understanding something about how these tools work helps hold tech accountable, and enables better use of these tools for apprpriate tasks. Regrdless of the sophistication of the tool, it is humans who attribute meaning to the results and not the machine. 

