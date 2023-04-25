---
exercises: 20
keypoints:
- TODO
objectives:
- TODO
questions:
- TODO
teaching: 20
title: IntroToTasks
---

##Context for Digital Humanists

There are a wide variety of fields of study within the humanities. Each of those fields brings a variety of research interests and methods to focus on a wide variety of questions. For this reason, we will touch on a variety of techniques and tasks which natural language processing can help accomplish. We will discuss different metrics which can be used to evaluate how well we accomplished those tasks. We will also discuss models commonly used and how they work, so that researchers can understand the underlying approach used by these tools.

These methods are not infallible or without bias. They are simply another tool you can use to analyze texts and should be critically considered in the same way any other tool would be. The goal of this workshop is not to replace or discredit existing humanist methods, but to help humanists learn new tools to help them accomplish their research.

##What is Natural Language Processing?

Natural Language Processing, or NLP, attempts to process human languages using mathematic computer models.

##What does NLP do? 

There are many possible uses for NLP. Machine Learning and Artificial Intelligence can be thought of as a set of computer algorithms used to take data as an input and produce a desired output. What distinguishes NLP from other types of machine learning is that text and human language is the main input for NLP tasks.

##What can I do with NLP? 

There are many tasks you can accomplish using NLP. We're going to explore some of these tasks in this lesson. We will start by using looking at some of the tasks achievable using the popular "HuggingFace" library.

Navigating to https://huggingface.co/tasks, we can see that there are many tasks achievable using NLP. 

What do these different tasks mean? Let's take a look at an example. Conversational tasks are also known as chatbots. A user engages in conversation with a bot. Let's click on this task now.

https://huggingface.co/tasks/conversational

Huggingface usefully provides an online demo as well as a description of the task. On the right, we can see there is a demo of a particular model that does this task. Give conversing with the chatbot a try.

If we scroll down, there is also a link to sample models and datasets HuggingFace has made available that can do variations of this task.  Documentation on how to use the model is available by scrolling down the page. Model specific information is available by clicking on the model. Let's go back to the https://huggingface.co/tasks




##Chatbot in Python

We've got an overview of what different tasks we can accomplish. Now let's try getting started with doing these tasks in Python. We won't worry too much about how this model works for the time being, but will instead just focusing trying it out. We'll start by running a chatbot, just like the one we used online.


```python
!pip install transformers
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting transformers
      Downloading transformers-4.28.1-py3-none-any.whl (7.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m7.0/7.0 MB[0m [31m31.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting tokenizers!=0.11.3,<0.14,>=0.11.1
      Downloading tokenizers-0.13.3-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (7.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m7.8/7.8 MB[0m [31m25.9 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.9/dist-packages (from transformers) (4.65.0)
    Requirement already satisfied: requests in /usr/local/lib/python3.9/dist-packages (from transformers) (2.27.1)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.9/dist-packages (from transformers) (1.22.4)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.9/dist-packages (from transformers) (2022.10.31)
    Requirement already satisfied: filelock in /usr/local/lib/python3.9/dist-packages (from transformers) (3.11.0)
    Collecting huggingface-hub<1.0,>=0.11.0
      Downloading huggingface_hub-0.13.4-py3-none-any.whl (200 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m200.1/200.1 kB[0m [31m15.7 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.9/dist-packages (from transformers) (23.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.9/dist-packages (from transformers) (6.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.9/dist-packages (from huggingface-hub<1.0,>=0.11.0->transformers) (4.5.0)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.9/dist-packages (from requests->transformers) (1.26.15)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.9/dist-packages (from requests->transformers) (3.4)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.9/dist-packages (from requests->transformers) (2.0.12)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.9/dist-packages (from requests->transformers) (2022.12.7)
    Installing collected packages: tokenizers, huggingface-hub, transformers
    Successfully installed huggingface-hub-0.13.4 tokenizers-0.13.3 transformers-4.28.1


NLP tasks often need to be broken down into simpler subtasks to be executed in a particular order. These are called "pipelines" since the output from one subtask is used as the input to the next subtask. We will define a "pipeline" in Python. Feel free to prompt the chatbot as you wish.


```python
from transformers import pipeline, Conversation
converse = pipeline("conversational", model="microsoft/DialoGPT-medium")

conversation_1 = Conversation("Going to the movies tonight - any suggestions?")
conversation_2 = Conversation("What's the last book you have read?")
converse([conversation_1, conversation_2])
```

    The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
    A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
    The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
    A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.





    [Conversation id: 91dc8c91-cec7-4826-8a26-2d6c06298696 
     user >> Going to the movies tonight - any suggestions? 
     bot >> The Big Lebowski ,
     Conversation id: f7b2a7b4-a941-4f0f-88a3-3153626278e8 
     user >> What's the last book you have read? 
     bot >> The Last Question ]



##Group Activity and Discussion
> Break out into groups and look at a couple of tasks for HuggingFace. The groups will be based on general categories for each task. Discuss possible applications of this type of model to your field of research. Try to brainstorm possible applications for now, don't worry about technical implementation.

1. Tasks that seek to convert non-text into text

> https://huggingface.co/tasks/image-to-text
> https://huggingface.co/tasks/text-to-image
> https://huggingface.co/tasks/automatic-speech-recognition
> https://huggingface.co/tasks/image-to-text

2. Searching and classifying documents as a whole

> https://huggingface.co/tasks/text-classification

> https://huggingface.co/tasks/sentence-similarity

3. Classifying individual words- Sequence based tasks

> https://huggingface.co/tasks/token-classification

> https://huggingface.co/tasks/translation

4. Interactive and generative tasks such as conversation and question answering

> https://huggingface.co/tasks/conversational

> https://huggingface.co/tasks/question-answering

Briefly present a summary of some of the tasks you explored. What types of applications could you see this type of task used in? How might this be relevant to a research question you have? Summarize these tasks and present your findings to the group.
> {: .discussion}


##Summary and Outro

We've looked at a variety of different tasks you can accomplish with NLP, and used Python to generate text based on one of the models available through HuggingFace. Next lesson we will be working on better understanding what is happening in these models.
