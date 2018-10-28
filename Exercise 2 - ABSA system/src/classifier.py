
# coding: utf-8

# In[ ]:

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
from pre_process import pre_process
from parsing import parsing
from sklearn.linear_model import LogisticRegression

# coding: utf-8

# In[ ]:

# CLASSIFIER.PY
class Classifier:
    """The Classifier"""

    #############################################
    
    
    def score_cv(model, X, y, scoring='accuracy'):
        # Function that computes the score using cross validation
        s = cross_val_score(model, X, y, cv = 5, scoring=scoring)
        return np.mean(s)

    
    #############################################

    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        self.trainfile=trainfile
        data=pd.read_csv(self.trainfile,sep='\t', header=None)
        data.columns=['Polarity','Aspect category','Term','Character','Text']
        self.featureList=[]
        
        # Pre-processing
        data, temp, self.featureList = pre_process(data)
        
        # Parsing
        #pars=parsing(temp['text'])
        
        # BOW Model
        label_column = ['label']
        columns = label_column + list(map(lambda w: w + "_bow",self.featureList))
        labels = []
        rows = []

        # 10 min 30 sec approx
        for idx in range(0,len(temp)):
            #print(idx)
            current_row = []
            current_label = temp.loc[idx, "polarity"]
            labels.append(current_label)
            current_row.append(current_label)

            temp.text[idx]=temp.text[idx].split()

            # add bag-of-words
            tokens = set(temp.loc[idx, "text"])
            for _, word in enumerate(self.featureList):
                current_row.append(1 if word in tokens else 0)

            rows.append(current_row)

            df = pd.DataFrame(rows, columns=columns)

        self.final=df.copy()
        for c in list(data.columns)[1:]:
            self.final[c]=data[c]
            
        #self.df_final=pd.merge(self.final, pars, on=['sent_id', 'Term'])
      
        self.df_final =self.final.copy()
        self.df_final=self.df_final.drop(['sent_id','Term'],axis=1)
        
        # Model
        col_names = [col for col in self.df_final.columns if col not in ['label','Text','Character','Head']]
        X_train=self.df_final[col_names].copy()
        y_train=self.df_final['label']

        ###########################################
        ###########################################
        # BEST MODEL
        ###########################################
        ###########################################
        regLog = LogisticRegression(C=0.5,random_state=5)
        self.model = regLog.fit(X_train,y_train)
        #rf=RandomForestClassifier()
        #self.model = rf.fit(X_train,y_train)


        ###########################################
        ###########################################
        ###########################################

    #############################################

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        self.datafile=datafile
        test=pd.read_csv(self.datafile,sep='\t', header=None)
        test.columns=['Polarity','Aspect category','Term','Character','Text']
        self.featureList2=[]
        
        # Pre-processing
        test, temp, self.featureList2 = pre_process(test)
        
        # Parsing
        #pars_test=parsing(temp['text'])
        
        # BOW Model
        label_column = ['label']
        columns = label_column + list(map(lambda w: w + "_bow",self.featureList))
        labels = []
        rows = []

        for idx in range(0,len(temp)):
            #print(idx)
            current_row = []
            current_label = temp.loc[idx, "polarity"]
            labels.append(current_label)
            current_row.append(current_label)

            temp.text[idx]=temp.text[idx].split()

            # add bag-of-words
            tokens = set(temp.loc[idx, "text"])
            for _, word in enumerate(self.featureList):
                current_row.append(1 if word in tokens else 0)

            rows.append(current_row)

            df2 = pd.DataFrame(rows, columns=columns)

        final_test=df2.copy()
        for c in list(test.columns)[1:]:
            final_test[c]=test[c]
            
        #df_final_test=pd.merge(final_test, pars_test, on=['sent_id', 'Term'])

        df_final_test=final_test.copy()

        df_final_test=df_final_test.drop(['sent_id','Term'],axis=1)

       # To be sure to have the same features in X_train and X_test
        # if more features in train (final) : we add them to the test dataset
        if self.df_final.shape[1] > df_final_test.shape[1]:
            missing_feat = list(set(self.df_final.columns) - set(df_final_test.columns))
            for f in missing_feat:
                df_final_test[f] = np.zeros(len(df_final_test))
        # if more features in test (final_test) : we remove them from the test dataset because the features are determined
        # in the training phase and cannot be modified during the test phase
        if self.df_final.shape[1] < df_final_test.shape[1]:
            excess_feat = list(set(df_final_test.columns) - set(self.df_final.columns))
            for f in excess_feat:
                df_final_test = df_final_test.drop(f, 1)


        # Predict
        col_names_test = [col for col in df_final_test.columns if col not in ['label','Text','Character','Head']]
        X_test=df_final_test[col_names_test].copy()
        y_test=df_final_test['label']

        pred = self.model.predict(X_test)
        idx_pos=np.where(pred==1)[0]
        idx_neg=np.where(pred==-1)[0]
        idx_ntr=np.where(pred==0)[0]

        pred=pred.astype(str)

        pred[idx_pos]='positive'
        pred[idx_neg]='negative'
        pred[idx_ntr]='neutral'

        return pred

