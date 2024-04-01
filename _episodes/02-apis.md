---
exercises: 20
keypoints:
- You will need to evaluate the suitability of data for inclusion in your corpus and will need to take into consideration issues such as legal/ethical restrictions and data quality among others. 
- Making an API request is a common way to access data. 
- You can build a query using a source's URL and combine it with get() in Python to make a request for data. 
- 
objectives:
- Become familiar with legal and ethical considerations for data collection.
- Practice making an API request to a cultural heritage institution and interpreting responses. 
questions:
- How do I know what data I can use for my corpus? 
- How can I use an API to acquire data?
teaching: 20
title: APIs

---

# Corpus Development- Text Data Collection and APIs

## Sources of text data

The best sources of text datasets will ultimately depend on the goals of your project. Some common sources of text data for text analysis include digitized archival materials, newspapers, books, social media, and research articles. For the most part, the datasets and sources you may come across will not have been arranged with a particular project in mind. The burden is therefore on you, as the researcher, to evaluate whether materials are suitable for your corpus. It can be useful to create a list of criteria for how you will decide what to include in your corpus. You may get the best results by piecing together your corpus from materials from various sources that meet your requirements. This will help you to create an intellectually rigorous corpus that meets your project's needs and makes a unique contribution to your area of study. 

## Text Data and Restrictions

One of the most important criteria for inclusion in your corpus that you should consider is whether or not you have the right to use the data in the way your project requires. When evaluating data sources for your project, you may need to navigate a variety of legal and ethical issues. We’ll briefly mention some of them below, but to learn more about these issues, we recommend the open access book[ Building Legal Literacies for Text and Data Mining](https://berkeley.pressbooks.pub/buildinglltdm/). 


*   **Copyright** - Copyright law in the United States protects original works of authorship and grants the right to reproduce the work, to create derivative works, distribute copies, perform the work publicly, and to share the work publicly. Fair use may create exceptions for some TDM activities, but if you are analyzing copyrighted material, publicly sharing your full text corpus would likely be in violation of copyright. 
*   **Licensing** - Licenses grant permission to use materials in certain ways while usually restricting others. If you are working with databases or other licensed collections of materials, make sure that you understand the license and how it applies to text and data mining. 
*   **Terms of Use** - If you are collecting text data from other sources, such as websites or applications, make sure that you understand any retrictions on how the data can be used. 
*   **Technology Protection Measures** - Some publishers and content hosts protect their copyrighted/licensed materials through encryption. While commercial versions of ebooks, for example, would make for easy content to analyze, circumventing these protections would be illegal in the United States under the Digital Millennium Copyright Act. 
*   **Privacy** - Before sharing a corpus publicly, consider whether doing so would constitute any legal or ethical violations, especially with regards to privacy. Consulting with digital scholarship librarians at your university or professional organizations in your field would be a good place to learn about privacy issues that might arise with the type of data you are working with. 
*   **Research Protections** - Depending on the type of corpus you are creating, you might need to consider human subject research protections such as informed consent. Your institution’s Institutional Review Board may be able to help you navigate emerging issues surrounding text data that is publicly available but could be sensitive, such as social media data. 




## OCR and Speech Transcription

Another criteria you may have to consider is the format that you need your files to be in. It may be that your test documents are not in text format- that is, in a file format that can be copied and pasted into a notepad file. Not all data is of this type, for example, there may be documents that are stored as image files or sound files. Or perhaps your documents are in PDF or DOC files. 

Fortunately, there exist tools to convert file types like these into text. While these tools are beyond the scope of our lesson, they are still worth mentioning. Optical Character Recognition, or OCR, is a field of study that converts images to text. Tools such as Tesseract, Amazon Textract, or Google’s Document AI can perform OCR tasks. Speech transcription will take audio files and convert them to text as well. Google’s Speech-to-Text and Amazon Transcribe are two cloud solutions for speech transcription.

Later in this lesson we will be working with OCR text data that has been generated from images of digitized newspapers. As you will see, the quality of text generated by OCR and speech to text software can vary. In order to include a document with imperfect OCR text, you may decide to do some file clean up or remediation. Or you may decide to only include documents with a certain level of OCR accuracy in your corpus. 


## Using APIs

When searching through sources, you may come across instructions to access the data through an API. An API, or application programming interface, allows computer programs to talk to one another. In the context of the digital humanities, you can use an API to request and receive specific data from corpora created by libraries, museums, or other cultural organizations or data creators. 

There are different types of APIs, but for this lesson, we will be working with a RESTful API, which uses HTTP, or hypertext protocol methods. A RESTful API can be used to post, delete, and get data. You can make a request using a URL, or Uniform Resource Locator, which is sent to a web server using HTTP and returns a response. If you piece together a URL in a certain way, it will give the web server all the info it needs to locate what you are looking for and it will return the correct response. If that sounds familiar, it’s because this is how we access websites everyday! 

### A few things to keep in mind: 

*   Each API will be different, so you will always want to check their documentation. 
*   Some APIs will require that you register to receive an API key in order to access their data. 
*   Just because the data is being made available through an API, doesn’t mean that it can be used in your particular project. Remember to check the terms of use. 
*   Check the data - even if you’ve used the API before. What format will it be delivered in? Is the quality of the data good enough to work with?




## How to Access an API

For an example of how to access data from an API, we will explore [Chronicling America: Historic American Newspapers](https://chroniclingamerica.loc.gov/about/), a resource produced by the National Digital Newspaper Program (NDNP), a partnership between the National Endowment for the Humanities (NEH) and the Library of Congress (LC). Among other things, this resource offers OCR text data for newspaper pages from 1770 to 1963. The majority of the newspapers included in Chronicling America are in the public domain, but there is a disclaimer from the Library of Congress that newspapers in the resource that were published less than 95 years ago should be evaluated for renewed copyright. The API is public and no API key is required. We'll use Chronicling America's API to explore their data and pull in an OCR text file. 

For this lesson, we’ll pretend that we’re at the start of a project and we are interested in looking at how Wisconsin area newspapers described World War I. We aren’t yet sure if we want to focus on any particular newspaper or what methods we want to use. We might want to see what topics were most prominent from year to year. We might want to do a sentiment analysis and see whether positive or negative scores fluctuate over time. The possibilities are endless! But first we want to see what our data looks like. 

By adding search/pages/results/? to our source’s URL https://chroniclingamerica.loc.gov/ and adding some of the search parameters that we have already mentioned we can start building our query. We will want to look for newspapers in Wisconsin between 1914 and 1918 that mention our search term “war.” And to keep our corpus manageable, we want to see only the first pages using sequence=1. If we specify that we want to be able to see it in a JSON file, that will give us the following query: 

https://chroniclingamerica.loc.gov/search/pages/results/?state=Wisconsin&dateFilterType=yearRange&date1=1914&date2=1918&sort=date&andtext=war&sequence=1&format=json

Let’s take a look at what happens when we type that into our web browser. And let’s take a look at what happens when we remove the request to view it in a JSON format. 

Now let's explore making requests and getting data using python. 


```python
!pip install requests
!pip install pandas
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: requests in /usr/local/lib/python3.9/dist-packages (2.27.1)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.9/dist-packages (from requests) (1.26.15)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.9/dist-packages (from requests) (2022.12.7)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.9/dist-packages (from requests) (2.0.12)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.9/dist-packages (from requests) (3.4)
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: pandas in /usr/local/lib/python3.9/dist-packages (1.5.3)
    Requirement already satisfied: numpy>=1.20.3 in /usr/local/lib/python3.9/dist-packages (from pandas) (1.22.4)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.9/dist-packages (from pandas) (2022.7.1)
    Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.9/dist-packages (from pandas) (2.8.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.9/dist-packages (from python-dateutil>=2.8.1->pandas) (1.16.0)



```python
import requests
import json
import pandas as pd

from google.colab import drive
drive.mount('/drive')
```

    Mounted at /drive


## Making a request

You can make an API call using a get() request. This will give you a response that you can check by accessing .status_code. There are different codes that you might receive with different meanings. If we send a request that can't be found, we get the familiar 404 code. A 200 response means that the request was successful. 


```python
#What happens when what you are looking for doesn't exist?
response = requests.get("https://chroniclingamerica.loc.gov/this-api-doesnt-exist")
print(response.status_code)
```

    404



```python
#Get json file of your search and check status
#First 20 search results
response20 = requests.get("https://chroniclingamerica.loc.gov/search/pages/results/?state=Wisconsin&dateFilterType=yearRange&date1=1914&date2=1918&sort=date&andtext=war&sequence=1&format=json")
print(response20.status_code)
```

    200


Now that we have successfully used an API to call in some of the data we were looking for, let's take a look at our file. We can see that there are 3,941 total items that meet our criteria and that this response has gotten 20 of them for us. We'll save our results to a text file. 


```python
# Look at json file
print(response20.json())
```

    {'totalItems': 3941, 'endIndex': 20, 'startIndex': 1, 'itemsPerPage': 20, 'items': [{'sequence': 1, 'county': ['Manitowoc'], ...}



```python
# Turn file into a python dictionary
data = response20.json()
print(data)
```

    {'totalItems': 3941, 'endIndex': 20, 'startIndex': 1, 'itemsPerPage': 20, ... }



```python
#Save a copy of the first 20 results
with open('corpusraw.txt', 'w') as corpusraw_file:
     corpusraw_file.write(json.dumps(data))
```

Next we will look at how we can use the metadata from our results to build a query that use the API to get OCR text from one of the newspaper pages. 

The first four key value pairs in the dictionary object tell us about the results of our query, but the fifth one, with the key 'items' is the one that gives us the bulk of the metadata about the newspapers that meet our requirements. To build our query, we need to grab the id so that we can add it to our URL. We could either manually grab it from our text file or we can call it using its index in the list. 


```python
#Deal with dictionaries within lists
d = data.get('items')
print(d)
```

    [{'sequence': 1, 'county': ['Manitowoc'], 'edition': None, 'frequency': 'Weekly', 'id': '/lccn/sn85033139/1914-01-01/ed-1/seq-1/',... }]



```python
d = data.get('items')
newspaper1 = d[0]
idnewspaper1 = newspaper1.get('id')
print(idnewspaper1)



```

    /lccn/sn85033139/1914-01-01/ed-1/seq-1/


Now that we have the id we need, we can use it to build our query. Adding it to our source's URL gives us https://chroniclingamerica.loc.gov/lccn/sn85033139/1914-01-01/ed-1/seq-1/. To see the OCR for this file, we just need to add ocr.txt to the end of the query to get https://chroniclingamerica.loc.gov/lccn/sn85033139/1914-01-01/ed-1/seq-1/ocr.txt. 

Now let's take a look at what it looks like when we make the request using the API.


```python
#Grab one OCR file from Chronicling America
responsenewspaper1 = requests.get("https://chroniclingamerica.loc.gov/lccn/sn85033139/1914-01-01/ed-1/seq-1/ocr.txt")
print(responsenewspaper1.status_code)
```

    200



```python
print(responsenewspaper1.text)
```

    olume IV.
    tCITY COUHCIL NOUS,
    pecial meeting of the city council
    eld last Saturday evening to take
    i winding up the electric light
    purchase matter and to consider
    otor lire truck purchase matter,
    ler, Gcorgenson and Schroeder
    absent...
