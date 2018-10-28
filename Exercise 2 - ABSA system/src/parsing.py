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
import spacy

def parsing(txt):
    nlp = spacy.load('en')
    length = len(txt)
    
    asdf = pd.DataFrame(columns=['sent_id',
                                 'Term',
                                 'POS',
                                 'Det_POS',
                                 'Dep',
                                 'Head',
                                 'Head_POS',
                                 'Vec_Norm'])
    j = 0
    for i in range(length):
        doc_en = nlp(txt[i])
        token_attributes = [(i,
                             token.orth_,
                             token.pos_,
                             token.tag_,
                             token.dep_,
                             token.head,
                             token.head.pos_,
                             token.vector_norm) for token in doc_en]
        for r in token_attributes:
            asdf.loc[j] = r
            j += 1

    df_pars=pd.get_dummies(asdf[['POS', 'Det_POS', 'Dep', 'Head_POS']])
    asdf=pd.concat([asdf, df_pars], axis = 1)
    asdf.drop(['POS','Det_POS', 'Dep', 'Head','Head_POS' ], axis=1, inplace=True)

    return asdf

