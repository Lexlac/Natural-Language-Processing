# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, model_from_json
from keras.layers import Input, LSTM, Dense
import json
import pickle
import argparse

#---------------------------------------------------------------------------------------------------------
#Load the data
#---------------------------------------------------------------------------------------------------------

def load_data(path_to_data):
    df_train = pd.read_csv(path_to_data, encoding='utf8')
    return df_train


#---------------------------------------------------------------------------------------------------------
#Preprocessing the input
#---------------------------------------------------------------------------------------------------------

slots = ['name', 'eatType', 'food', 'priceRange', 'customer rating', 'area', 'familyFriendly', 'near']
vochar = {'<name>', '<near>', '<food>', '<eattype>', '<bos>', '<eos>'}
tokens = {'<name>', '<near>', '<food>', '<eattype>', '<bos>', '<eos>'}

#Function: Pre-processing the MR variable
def preprocess(line):
    mr = []

    words = line.split(',')
    for w in words:
        for s in slots:
            w = w.strip()
            if w.startswith(s):
                val = w[len(s)+1:].replace(']', '')
                mr.append((s, val))
    return mr

#Function: Creating a dictionary of unique MR slots, also adding the non-specified keys
def encode_vector(mrs, s_to_id):
    vec = np.zeros(len(s_to_id))

    visible = set()
    for k,v in mrs:
        visible.add(k)
        if k in ['name', 'near']:
            vec[s_to_id[k]] = 1
        else:
            vec[s_to_id[(k,v)]] = 1

    for not_visible in set(slots) - visible:
        vec[s_to_id[(k, 'not visible')]] = 1

    return vec

#Function: Pre-processing of input for training
def pre_processing_input(df_train):
    dic = {}

    #Pre-processing MR type tokens
    slots = ['name', 'eatType', 'food', 'priceRange', 'customer rating', 'area', 'familyFriendly', 'near']
    for line in df_train.mr:
        words = line.split(',')
        for w in words:
            for s in slots:
                w = w.strip()
                if w.startswith(s):
                    if s not in dic:
                        dic[s] = set()

                    val = w[len(s)+1:].replace(']', '')
                    dic[s].add(val)

    #MR type is converted to a unique ID to be taken as input for the feature vector
    s_to_id = {'name':0, 'near':1}
    i = 2
    for k, v in dic.items():
        if k not in ['name', 'near']:
            for a in v:
                s_to_id[(k,a)] = i
                i += 1


    #Not-visible attributes
    not_visible = ['eatType', 'food', 'priceRange', 'customer rating', 'area', 'familyFriendly', 'near']
    for a in not_visible:
        s_to_id[(a, 'not visible')] = len(s_to_id)


    new_mr = [preprocess(line) for line in df_train.mr]
    X_feature_vectors = np.array([encode_vector(x,s_to_id) for x in new_mr])

    return (X_feature_vectors ,  new_mr, s_to_id)


#---------------------------------------------------------------------------------------------------------
#Preprocessing the characters
#---------------------------------------------------------------------------------------------------------

def pre_processing_char (df_train , new_mr, vochar):
    # Change the name, near, food, eatType values from the MR to a specific static token
    sentences = df_train.ref.values

    proc_sentences = []
    for i in range(len(sentences)):
        s = sentences[i]
        mr = new_mr[i]
        
        #replace corresponding value in all MRs
        for k,v in mr:
            if k == 'name':
                s = s.replace(v, ' <name> ')
            elif k == 'near':
                s = s.replace(v, ' <near> ')
            elif k == 'food':
                s = s.replace(v, ' <food> ')
            elif k == 'eatType':
                s = s.replace(v, ' <eatType> ')
                
        #get rid of uppercase characters       
        proc_sentences.append(s.lower())

        
    
    for s in proc_sentences:

        # add key-value pairs of every character of each sentence to vochar dic
        for char in s:
            vochar.update(char)
            
    #transform vochar to list
    vochar = list(vochar)
    #re-map ID, character to a dictionnary
    c_to_id = {vochar[i]:i for i in range(len(vochar))}
    
    lists = []
    
    #create a list of vochar IDs to use for one-hot encoding later
    for s in proc_sentences:
        #put every bos token ID in a list
        sent_id = [c_to_id['<bos>']]

        words = s.split(' ')
        for i in range(len(words)):
            word = words[i]

            if word == '<name>':
                sent_id.append(c_to_id['<name>'])
            elif word == '<near>':
                sent_id.append(c_to_id['<near>'])
            elif word == '<food>':
                sent_id.append(c_to_id['<food>'])
            elif word == 'eatType':
                sent_id.append(c_to_id['<eattype>'])
            else:
                # For character in word, append character
                for char in word:
                    sent_id.append(c_to_id[char])

                # Suppress whitespace after the last word
                if i < len(words) - 1:
                    sent_id.append(c_to_id[' '])

        sent_id.append(c_to_id['<eos>'])
        lists.append(sent_id)
        
    max_seq_len = 150
    X_data = [] 
    
    #one-hot encoding of characters
    for i in range(len(lists)):
        b = lists[i]
        #create a matrix for each sentence of size (max_seq_length, vochar_length)
        S = np.zeros((max_seq_len, len(vochar)))
        for j in range(len(b)):
            if j >= len(vochar):
                break
            
            vec = np.zeros(len(vochar))
            #encode 1 when character corresponds
            vec[b[j]] = 1
            S[j] = vec
        X_data.append(S)

    X_data = np.array(X_data)
    
    return (X_data, proc_sentences, vochar)

#---------------------------------------------------------------------------------------------------------
#Train the model
#---------------------------------------------------------------------------------------------------------

def train (X_feature_vectors,X_data,proc_sentences,s_to_id, vochar):
    
    #Define our input data for decoder and encoder:
    encoder_input_data = X_feature_vectors.reshape((len(proc_sentences), 1, len(s_to_id)))
    decoder_input_data = X_data

    #Padding of the target data: 
    npad = ((0, 0), (0, 1), (0, 0))
    decoder_target_data = np.pad(X_data[:,1:,:], pad_width=npad, mode='constant', constant_values=0)
    
    #Definition of our model parameters:
    batch_size = 64  
    epochs = 10
    latent_dim = 256
    num_samples = len(proc_sentences)  
    
    num_encoder_tokens = len(s_to_id)
    num_decoder_tokens = len(vochar)

    #----------------- MODEL
    
    #Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    _, state_h, state_c = encoder(encoder_inputs)
    #We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]
    
    #Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences, and to return internal states as well. We don't use the 
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()
    
    # We train our model while monitoring the loss on a held-out set of 20% of the samples (validation split).
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit([encoder_input_data, decoder_input_data], 
              decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2)
  
    # We are ready for inference. To decode a test sentence, we will repeatedly: Encode the input sentence and retrieve the
    # initial decoder state + run one step of the decoder with this initial state and a "start of sequence" token as target. The
    # output will be the next target character + append the target character predicted and repeat.
    # Below is the inference setup : (the decode function will be part of the test_model.py file)
    encoder_model = Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    
    #(source of the model above : https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)
    #-----------------
    
    return (model , encoder_model, decoder_model)

def save_model(path, encoder_model,decoder_model, vochar, s_to_id ):
    
    #1. Save encoder
    encoder_model_json = encoder_model.to_json()
    with open(path + "encoder.json", "w") as json_file:
        json_file.write(encoder_model_json)
    encoder_model.save_weights(path + "encoder.h5")
    
    #2. Save decoder
    decoder_model_json = decoder_model.to_json()
    with open(path + "decoder.json", "w") as json_file:
        json_file.write(decoder_model_json)
    decoder_model.save_weights(path + "decoder.h5")
    
    #3. Save vochar 
    with open(path + "vochar", 'wb') as fp:
        pickle.dump(vochar, fp)
    
    #4. Save s_to_id
    np.save(path +'s_to_id.npy', s_to_id) 

    

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_dataset', dest='train_set', nargs=1, help='path containing training data')
    parser.add_argument('-output_model_file', dest='output', nargs=1,  help='path to save the model')
	
    opts, args = parser.parse_known_args()
    path_to_train=args[1]
    path_to_model = args[3]
    
    print("Loading data....")
    df_train = load_data(path_to_train)
    
    print("Pre processing input....")
    X_feature_vectors , new_mr , s_to_id= pre_processing_input(df_train)
    
    print("Pre processing char....")
    X_data,proc_sentences, vochar = pre_processing_char(df_train,  new_mr, vochar)
    
    print("Training....")
    model, encoder_model, decoder_model = train(X_feature_vectors, X_data, proc_sentences, s_to_id, vochar)
    
    print("Saving the models....")
    save_model(path_to_model, encoder_model,decoder_model, vochar, s_to_id)

    print("End.")

