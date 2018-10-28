NLP Course - CentraleSupélec
Exercise #3 - NLG

Rafaëlle Aygalenq, Sarah Lina Hammoutene, Dora Linda Kocsis, Noémie Quéré

The purpose of this exercise is to design and develop a deep learning model and a training algorithm over a training set for Natural Language Generation, and to evaluate its performance on a test set. 
The dataset is taken from: http://www.macs.hw.ac.uk/InteractionLab/E2E/


Approach 1 : LSTM (word level model)
The first approach is based on Long Short-term Memory neural networks. We perform preprocessing consisting of delexicalizing sentences, splitting data and postprocessing consisting of the opposite operations (joining and relexicalization). 


Delexicalization is about the slot value pairs contained in the Meaning Representation data: the slot represents the type of information and the value represents the information itself. This has the advantage of creating more generic sentences which can increase the performance of the model. For example the following sentence :


‘The Rice Boat is an Indian restaurant in the city centre near the Express by Holiday Inn, it is kid friendly highly rated and costs 20-25 euros.’
becomes :
                                                   
‘slot_val_name is an Indian restaurant in the city centre near slot_val_near, it is kid friendly slot_val_customerrating rated and costs slot_val_priceRange.’
                                        
Then, the idea is that by by breaking longer MRs into multiple smaller MRs that have at most 3 slots, we will be able to generate multiple utterances which when recombined will be more coherent and informative than attempting to directly generate a single long MR.


As said before, for the modeling part we used a LSTM neural network because of its good properties on sentence modeling with the aim of generating stylized responses that contain the natural variations of the human language. We used two different LSTM: one for encoding and one for decoding. 


At the end, we post-processed the output of the model in order to have a “correct” sentence. The first post-processing task is relexicalisation. It consists of replacing the slot_val_. into their original values. The second post-processing task is about concatenating the splits we made previously into sentences. For this part, in order to have a more ‘human generated’ sentence, if the utterance is made of several sentences we added a condition stating that the second sentence should not begin with the ‘name’ of the restaurant but with ‘It’. 


The results with this approach were not satisfying and it was really time consuming. We were limited by the capacity of our computers to train the model (it did not work with more than 5 epochs during the training phase). Thus we decided to choose another approach which is based on a character level training.
________________


Approach 2 : Char2Char model


In this approach, we chose to implement a character-level sequence-to-sequence model. The model takes two input vectors, a feature vector and a one-hot vectorization of the sentences character by character and a target output that is similar to the one-hot vector input, except that it is offset by one timestep.


Vector encoding
First, we encoded the MR column as a feature vector: for every possibility every token (near, food, priceRange etc.) we created a dictionary containing the token, its name and its index. We used this to encode each line of the MR column into a list of 0 and 1 that we stored into an array of shape (number of lines, number of token possibilities).
Second, we hot-encoded every sentence for every character after delexicalized the following tokens <name>,  <near>, <food> and <eatType>.


Encoder
The encoder is a RNN layer that takes as input the two encoded vectors and returns its own internal state. We don’t use the outputs of the encoder RNN, and only keep the state. It is used as the ‘conditioning’ of the decoder in the next step.


Decoder
The decoder is a second RNN layer. It predicts the next characters of the target sequence. It takes the state learned from the previous encoder, and uses it to obtain information about what it is supposed to generate, that is to say the same sequences as the target sequences but offset by one timestep in the future.


Decoding
Finally, the last step is to decode the target output we get out of the decoder and relexicalize the obtained sentences (replace the slot placeholders <name>,  <near>, <food> and <eatType> by their original values).


The results with this approach are more accurate, we obtained correct sentences even if all the slots are not always appearing in the predicted sentences, the main ones are present. 


source of the model: https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
