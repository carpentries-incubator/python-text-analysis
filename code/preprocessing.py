#Need to understand libraries and how they work to do import
import nltk
import spacy
import gensim
from spacy.lang.en import English
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
import pickle

#make sure nltk is up to date, download wordnet to get synonyms and lemmas
nltk.download('wordnet')
nltk.download('stopwords')


#STEP 1: Load in parser/stopwords/corpus

#load in corpus of reuters newspaper texts. This is built into NLTK. Need to understand how to call functions from a library.
guten = nltk.corpus.gutenberg.words()
#k = PlaintextCorpusReader('C:\\Users\\Karl\\Desktop\\nltk-demo\\data', '.*')
#guten = k.words()

#loading in spacy English parser
spacy.load("en_core_web_sm")
parser = English()

en_stop = set(nltk.corpus.stopwords.words('english'))

text_data = []

#Step 2: Preprocess your data.
print("Start preprocess")
#we want to process every word in our corpus, using a for loop.
for word in guten:
    #step 2.1, tokenize words
    tokens = []
    unproc_tokens = parser(word)
    for token in unproc_tokens:
        if token.orth_.isspace():
            continue
        else:
            tokens.append(token.lower_)

    #step 2.2 filters out words of less than 3 characters
    tokens = [token for token in tokens if len(token) > 3]
    #step 2.3 filters out stopwords
    tokens = [token for token in tokens if token not in en_stop]
    #step 2.4 lemmaize words, in other words, removed endings so stop, stopped, stopping all same root/lemma
    for token in tokens:
        lemma = wn.morphy(word)
        if lemma is not None:
            token = lemma
    #step 2.5 save lemmas
    if tokens == []:
        continue
    else:
        text_data.append(tokens)
        #print(tokens)

#Generate dictionary that keeps track of word counts and convert to corpus object. Needed for running the lda model.
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]
print("End preprocessing")


###STEP 3: Set hyperparameters.
#The "parameters" of the model are tweaked automatically by our library.
#We are changing the parameters used to set our model, also known as "hyperparameters"

#NUM_TOPICS-  the number of topics we want to see. Too many topics means each topic overspecialized, too few means they're too general. You may have to tweak this variable.
#note the passes variable- more passes is a tighter model fit, but also more runtime.
NUM_TOPICS = 10

#PASS_CT - running the model more than once results in slightly different topics. doing multiple passes smooths out the topics.
#it's possible to "overfit" the model so that it doesn't generalize well to new topics.
#10-15 passes common, but this can take a long time on a desktop! We'll set this to 1.
PASS_CT = 10

##STEP 4: Build Model
#now we run the model. 
print("Model building starts")
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=PASS_CT)
print("Model completed!")

topics = ldamodel.print_topics(num_words=10)
for topic in topics:
    print(topic)


#STEP 5- save our work. Creating all of this takes a long time.
#By saving we can do some analysis on the objects we just created without rerunning them over and over. 

pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')
ldamodel.save('modelten.gensim')

