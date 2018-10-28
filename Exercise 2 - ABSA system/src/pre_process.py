import pandas as pd
import numpy as np
import time as t

#import tensorflow
#import keras
import nltk
from   nltk.tokenize import TweetTokenizer
from   nltk.stem.snowball import SnowballStemmer
from   nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier

def pre_process(data):
    
    # Transform 'Polarity' feature into numeric levels, 0 for negative and 1 for positive
    idx_n=data.loc[data['Polarity']=='negative'].index
    idx_p=data.loc[data['Polarity']=='positive'].index
    idx_neutr=data.loc[data['Polarity']=='neutral'].index
    
    data.loc[idx_n,'Polarity']=-1
    data.loc[idx_p,'Polarity']=1
    data.loc[idx_neutr,'Polarity']=0
    
    # Sentence ID (necessary for merging with spaCy dependencies parsing)
    data['sent_id']=range(0,len(data))
    
    # Transform the feature 'Aspect category' into dummies, otherwise pb in the modeling part
    df_asp=pd.get_dummies(data[['Aspect category']])
    data=pd.concat([data, df_asp], axis = 1)
    data.drop(['Aspect category'], axis=1, inplace=True)
    
    # Lowercase
    data['Term']=list(map(lambda i: data['Term'][i].lower(),range(len(data))))
    
    # Add the 'Term' word to the 'Text' feature
    data['Text']= list(map(lambda i: data['Term'][i]+' : '+data['Text'][i],range(len(data))))
    txt=data['Text']
    
    # Remove punctuation
    txt=txt.str.replace("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([0-9])","")
    
    # Tokenization
    txt=txt.apply(nltk.word_tokenize)
    
    # Lowercase
    txt=list(map(lambda i: [item.lower() for item in txt[i]],range(len(txt))))
    
    # Remove 1-letter word
    for i in range(0,len(txt)):
        c=[]
        for k in range(0,len(txt[i])):
            if len(txt[i][k])==1 :
                c.append(k)
        if c!=[] :
            txt[i]=np.delete(txt[i], c).tolist()

    # Remove stopwords
    stopwords_ = nltk.corpus.stopwords.words('english')
    txt=list(map(lambda i: [y for y in txt[i] if y not in stopwords_],range(len(txt))))

    # Stemming each word
    stemmer = SnowballStemmer('english')
    txt=list(map(lambda i: [stemmer.stem(y) for y in txt[i]],range(len(txt))))
    
    # Detokenize cleaned dataframe for vectorizing
    txt = list(map(lambda i: " ".join(txt[i]), range(len(txt))))
        
    
    stemmer = SnowballStemmer('english')
    data['Term']=list(map(lambda i: stemmer.stem(data['Term'][i]), range(len(data))))
    temp=pd.DataFrame({'polarity': data['Polarity'],'text': txt})
    
    featureList = []
    for i in range(0,len(txt)):
        featureList.extend(txt[i].split())
    
    ex_featureList=list(featureList)
    featureList=list(set(featureList))
    
    return data, temp, featureList
