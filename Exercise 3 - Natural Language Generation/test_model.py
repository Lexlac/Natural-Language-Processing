
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from keras.models import Model, model_from_json
from keras.layers import Input, LSTM, Dense
import json
import pickle


slots = ['name', 'eatType', 'food', 'priceRange', 'customer rating', 'area', 'familyFriendly', 'near']
vochar = {'<name>', '<near>', '<food>', '<eattype>', '<bos>', '<eos>'}
tokens = {'<name>', '<near>', '<food>', '<eattype>', '<bos>', '<eos>'}




#---------------------------------------------------------------------------------------------------------
# Load Data
#---------------------------------------------------------------------------------------------------------
def load_data(path_to_data):
    df_test = pd.read_csv(path_to_data, encoding='utf8')
    df_test.columns=["mr"]
    return df_test


#---------------------------------------------------------------------------------------------------------
# Load Model
#---------------------------------------------------------------------------------------------------------
def load_model(path): 
    
    # A. Encoder
    # load json and create encoder model
    json_file = open(path + "encoder.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model_encoder = model_from_json(loaded_model_json)
    # load weights into new encoder model
    loaded_model_encoder.load_weights(path + "encoder.h5")
    
    
    # B. Decoder
    # load json and create decoder model
    json_file = open(path + "decoder.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model_decoder = model_from_json(loaded_model_json)
    # load weights into new encoder model
    loaded_model_decoder.load_weights(path + "decoder.h5")
  

    # C. Vocab 
    with open (path + "vochar", 'rb') as fp:
        vocab = pickle.load(fp)
        
    # D. type2id
    type2id = np.load(path +"s_to_id.npy").item()
    
    print("Loaded model from disk")
    return (loaded_model_encoder , loaded_model_decoder , vocab )


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
# Train and Post Processing
#---------------------------------------------------------------------------------------------------------
max_decoder_seq_length = 150
def decode_sequence(input_seq, encoder_model, decoder_model):
    
    num_encoder_tokens = len(type2id)
    num_decoder_tokens = len(vocab)
    
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, char2id['<bos>']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = id2char[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '<eos>' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]
        
        # At most 2 sentences in the utterance
        if decoded_sentence.count('.') >= 2:
            decoded_sentence = ".".join(decoded_sentence.split(".", 2)[:2])+'.'
        
        if decoded_sentence.count('.') > 1:
            l = list(decoded_sentence)
            l[decoded_sentence.find('.')+2] = decoded_sentence[decoded_sentence.find('.')+2].upper()
            decoded_sentence = "".join(l)
            
    return decoded_sentence

def post_processing(X, type2id, processed_mrs, encoder_model, decoder_model):   
    # Decode every sentence (which is for now a binary vector)
    results = []
    results = map(lambda i: decode_sequence(X[i].reshape((1,1,len(type2id))), encoder_model, decoder_model), range(len(X))) 

    # Replace slot placeholders by their values
    # Name
    results = map(lambda i: results[i].replace('<name>', dict(processed_mrs[i])['name']) if 'name' in 
                  dict(processed_mrs[i]) else results[i], range(len(X)))
    
    # Near
    results = map(lambda i: results[i].replace('<near>', dict(processed_mrs[i])['near']) if 'near' in 
                  dict(processed_mrs[i]) else results[i], range(len(X)))
    
    # Food
    results = map(lambda i: results[i].replace('<food>', dict(processed_mrs[i])['food']) if 'food' in 
                  dict(processed_mrs[i]) else results[i].replace('<food>',''), range(len(X)))
    
    # Eat Type
    results = map(lambda i: results[i].replace('<eattype>', dict(processed_mrs[i])['eatType']) if 'eatType' in
                  dict(processed_mrs[i]) else results[i], range(len(X)))
    
    # End of sentence
    results = map(lambda i: results[i].replace('<eos>','') if '<eos>' in results[i] else results[i], range(len(X)))
    
    return results


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_dataset', dest='train_set', nargs=1, help='path containing training data')
    parser.add_argument('-input_model_file', dest='model', nargs=1,  help='path to save the model')
    parser.add_argument('-ouput_test_file', dest='output_file', nargs=1,  help='path to save the model')


    opts, args = parser.parse_known_args()
    path_to_test=args[1]
    path_to_model = args[3]
    path_to_output= args[5]
    
    print("Loading the test set ...")
    test_set= load_data(path_to_test)
    
    print("Loading the model ...")
    loaded_model_encoder , loaded_model_decoder , vocab  = load_model(path_to_model)
    
    print("Preprocessing the test set ...")
    X_feature_vectors ,  processed_mrs_test, type2id= pre_processing_input(test_set)    
    
    print("Predicting ...")
    char2id = {vocab[i]:i for i in range(len(vocab))}
    id2char = {v:k for k,v in char2id.items()}
    results = post_processing(X_feature_vectors, type2id, processed_mrs_test, loaded_model_encoder, loaded_model_decoder)
    
    np.savetxt(path_to_output, results, fmt="%s")
    
    print("Predictions done")

