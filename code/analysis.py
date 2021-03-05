#same libraries as before. probably not all of them are used here.
#import nltk
#import spacy
import gensim
import pickle
import pyLDAvis
import pyLDAvis.gensim
import nltk

#load our preprocessed dictionary, corpus and model
dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('modelten.gensim')


lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
pyLDAvis.save_html(lda_display, "guten_out.html")
#pyLDAvis.enable_notebook()
#pyLDAvis.show(lda_display)



#pyLDAvis.enable_notebook()
#lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
#pyLDAvis.save_html(lda_display, 'LDA_Visualization.html')
