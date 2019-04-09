# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 22:48:08 2019

@author: Vinay
"""

import re
import nltk
import string
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_colwidth",200)
warnings.filterwarnings("ignore",category = DeprecationWarning)

train = pd.read_csv('C:\\Users\\ADMIN\\.spyder-py3\\train_tweets.csv')
test = pd.read_csv('C:\\Users\\ADMIN\\.spyder-py3\\test_tweets.csv')

train[train['label']==0].head(10)
train[train['label']==1].head(10)

train.shape, test.shape
train["label"].value_counts()

length_train = train['tweet'].str.len()
length_test = test['tweet'].str.len()

plt.hist(length_train,bins=20,label="train_tweets")
plt.hist(length_test,bins=20,label="test_tweets")
plt.legend()
plt.show()

combi=train.append(test,ignore_index=True)
combi.shape

def remove_pattern(input_txt,pattern):
    r = re.findall(pattern,input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)
    return input_txt

combi['tidy_tweet']=np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*")
combi.tidy_tweet

combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
combi.head(10)

# =============================================================================
# def drop_short(input_text):
#     for m in input_text.split():
#         spli = m.split()
#         for i in spli:
#             print(i)
#             if len(str(i))>3:
#                 
#     return input_text
# 
# combi['tidy_tweet2'] = combi['tidy_tweet'].apply(drop_short)
# combi['tidy_tweet'].apply(drop_short) #(combi['tidy_tweet'])     
# split1=combi['tidy_tweet'][1].split()
# split1
# combi['tidy_tweet'][1]
# =============================================================================

########  IMP  ###############
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
combi.head()

#Tokenizing the tweets
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())
tokenized_tweet.head()

from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])   #stemming

tokenized_tweet.head()

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i]=' '.join(tokenized_tweet[i])

tokenized_tweet.head()
combi['tidy_tweet']=tokenized_tweet

all_words = ' '.join([text for text in combi['tidy_tweet']])
all_words

from wordcloud import WordCloud

wordcloud = WordCloud(width=800,height=500,random_state=25,max_font_size=110).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()

normal_words =' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

negative_words = ' '.join([text for text in combi["tidy_tweet"][combi['label'] == 1]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()

def hashtag_extract(x):
    hashtags=[]
    for i in x:
        ht = re.findall(r"#(\w+)",i)
        hashtags.append(ht)
    
    return hashtags

HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0])

HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1])

HT_regular = sum(HT_regular,[])

HT_negative = sum(HT_negative,[])


import nltk
from nltk import FreqDist

sentence='This is my sentence'
tokens = nltk.word_tokenize(sentence)
fdist=FreqDist(tokens)
fdist
data = pd.DataFrame({'word':list(fdist.keys()),'count':list(fdist.values())})
data

a1 = nltk.FreqDist(HT_regular)
d1 = pd.DataFrame({'hashtag':list(a1.keys()),'count':list(a1.values())})

a2 = nltk.FreqDist(HT_negative)
d2 = pd.DataFrame({'hashtag':list(a2.keys()),'count':list(a2.values())})

d1 = d1.nlargest(columns='count',n=20)
plt.figure(figsize=(16,5))
sns.barplot(data=d1,x="hashtag",y="count")
d2 = d2.nlargest(columns='count',n=20)
plt.figure(figsize=(16,5))
sns.barplot(data=d2,x="hashtag",y="count")

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim

bow_vectorizer = CountVectorizer(max_df=0.90,min_df=2,max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])
bow.shape

tfidf_vectorizer=TfidfVectorizer(max_df=0.90,min_df=2,max_features=1000, stop_words='english')
tfidf=tfidf_vectorizer.fit_transform(combi['tidy_tweet'])
tfidf.shape


tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) #tokenizing
tokenized_tweet

model_w2v = gensim.models.Word2Vec(
        tokenized_tweet,
        size=200, # desired no. of features/independent variables 
        window=5, #context window size
        min_count=2,
        sg=1, #1 for skipgram model
        hs=0,
        negative=10, #for negative sampling
        workers=2, #no. of cores
        seed=34)

model_w2v.train(tokenized_tweet, total_examples=len(combi['tidy_tweet']), epochs=20)

model_w2v.wv.most_similar(positive='dinner')

model_w2v.wv.most_similar(positive='black')
model_w2v.wv.most_similar(positive='racist')

model_w2v['food']

len(model_w2v['food'])
def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        except KeyError: # handling the case where the token is not in vocabulary
                         
            continue
    if count != 0:
        vec /= count
    return vec

wordvec_arrays = np.zeros((len(tokenized_tweet), 200))

for i in range(len(tokenized_tweet)):
    wordvec_arrays[i,:] = word_vector(tokenized_tweet[i], 200)
    
wordvec_df = pd.DataFrame(wordvec_arrays)
wordvec_df.shape

from tqdm import tqdm
tqdm.pandas(dessc="progress-bar")
from gensim.models.doc2vec import LabeledSentence

def add_label(twt):
    output = []
    for i, s in zip(twt.index, twt):
        output.append(LabeledSentence(s,["tweet_" + str(i)]))
    return output

labeled_tweets = add_label(tokenized_tweet) #label all the tweets with unique ids
labeled_tweets[:6]


model_d2v = gensim.models.Doc2Vec(dm=1, # dm = 1 for ‘distributed memory’ model 
                                  dm_mean=1, # dm = 1 for using mean of the context word vectors
                                  vector_size=200, # no. of desired features
                                  window=5, # width of the context window
                                  negative=7, # if > 0 then negative sampling will be used
                                  min_count=5, # Ignores all words with total frequency lower than 2.
                                  workers=3, # no. of cores
                                  alpha=0.1, # learning rate
                                  seed = 23)

model_d2v.build_vocab([i for i in tqdm(labeled_tweets)])
model_d2v.train(labeled_tweets, total_examples=len(combi['tidy_tweet']), epochs=15)

docvec_arrays = np.zeros((len(tokenized_tweet),200))

for i in range(len(combi)):
    docvec_arrays[i,:] = model_d2v.docvecs[i].reshape((1,200))

docvec_df = pd.DataFrame(docvec_arrays)    
docvec_df.shape

#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# BAG-OF-WORDS FEATURES
# fit LogReg to BOW features

train_bow = bow[:31962,:]
test_bow = bow[31962:,:]

xtrain_bow, xvalid_bow, ytrain_bow, yvalid_bow = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow,ytrain_bow)

prediction = lreg.predict_proba(xvalid_bow)
prediction
prediction_int = prediction[:,1] >= 0.3
prediction_int
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid_bow, prediction_int)

test_pred=lreg.predict_proba(test_bow)
test_pred_int=test_pred[:,1]>=0.3
test_pred_int = test_pred_int.astype(np.int)
test['label']=test_pred_int
submission=test[['id','label']]
submission.to_csv('sub_lreg_bow.csv', index=False)


# TFIDF Features

train_tfidf=tfidf[:31962,:]
test_tfidf = tfidf[31962:,:]

xtrain_tfidf=train_tfidf[ytrain_bow.index]
xvalid_tfidf = train_tfidf[yvalid_bow.index]

lreg.fit(xtrain_tfidf,ytrain_bow)
prediction=lreg.predict_proba(xvalid_tfidf)
prediction_int=prediction[:,1]>=0.3
prediction_int=prediction_int.astype(np.int)
f1_score(yvalid_bow,prediction_int)


# Word2Vec features

train_w2v = wordvec_df.iloc[:31962,:]
test_w2v = wordvec_df.iloc[31962:,:]

xtrain_w2v = train_w2v.iloc[ytrain_bow.index,:]
xvalid_w2v = train_w2v.iloc[yvalid_bow.index,:]

lreg.fit(xtrain_w2v,ytrain_bow)
prediction = lreg.predict_proba(xvalid_w2v)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)
f1_score(yvalid_bow, prediction_int)

# Doc2Vec Features

