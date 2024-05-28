# -*- codeing = utf-8 -*-
# @File :LDA_TFIDF.py
# @Software :PyCharm
import gensim
import gensim.corpora as corpora
from gensim import models
from gensim.models.coherencemodel import CoherenceModel

#0 data
PATH = r'F:\data.csv'
file_object = open(PATH, encoding='utf-8', errors='ignore').read().split('\n')
data_set = []
for i in range(len(file_object)):
    result = []
    seg_list = file_object[i].split()
    for w in seg_list:
        result.append(w)
    data_set.append(result)

#1 Dictionary
id2word = corpora.Dictionary(data_set)# create
id2word.filter_extremes(no_below=3, no_above=0.5, keep_n=3000)
id2word.save_as_text("dictionary") # save

#2 corpus for text
corpus = [id2word.doc2bow(text) for text in data_set]

#3 TF-IDF
tfidf_model = models.TfidfModel(corpus=corpus, dictionary=id2word)
tfidf_model.save('tfidf.model') # save
tfidf_model = models.TfidfModel.load('tfidf.model') # load
corpus_tfidf = [tfidf_model[doc] for doc in corpus]

#4 LDA_model
Lda = gensim.models.ldamodel.LdaModel

#5 Coherence
def Coherence(num_topics):
    ldamodel = Lda(corpus=corpus_tfidf,id2word=id2word,num_topics=num_topics, random_state=100,update_every=1,chunksize=100,passes=10,alpha='auto',per_word_topics=True)
    cm = CoherenceModel(model=ldamodel, corpus=corpus_tfidf, texts=data_set ,coherence='c_v')
    coherence = cm.get_coherence()
    return coherence