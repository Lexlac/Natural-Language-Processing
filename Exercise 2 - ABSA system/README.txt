NLP Exercise 2: ABSA system

1. Students who contributed to the deliverable:

	- Rafaëlle Aygalenq
	- Sarah Lina Hammoutene
	- Dora Linda Kocsis
	- Noémie Quéré

2. Description of our final system

	a) Feature representation

We first implemented a pre-processing function which will be applied to both training data and testing data. This function is composed of several tasks:
	- transforming categorical features : the levels of the ‘Polarity’ feature, which are {‘positive’,’neutral’,’negative’}, are turned into numerical levels {1,0,-1}. The ‘Aspect category’ feature is transformed into dummies.
	- concerning the feature ‘Text’ : we removed the punctuation, put all words in lowercase, tokenized each sentence, remove stopwords, stemmed each word to keep only the root and reduce the future number of features and then we detokenize all sentences.

Then we implemented a Bag-Of-Words model based on the feature ‘Text’ which contains the pre-processed sentences. Vocabulary made of every word present among all sentences in the training file is used as features and we put the value 1 when a word in present in the initial sentence, 0 otherwise. At the end we join the dataset resulting of the BOW model and the initial one to obtain a dataset with all features (words and initial ones). All of these tasks are applied on the training set and on the test set in order at the end to have the exact same features for both. A small verification of the features’ exact matching is made in order not to have a problem during the modelling part (especially for the creation of the dummies of the dependency parsing, all the different tags may not be present in both datasets so we just make sure that at the end we have the same features in both train and test datasets). 

We also tried a system with a different set of features.We also chose to apply dependency parsing using spaCy on the feature ‘Text’ containing the sentences. We selected several tokens including part-of-speech tagging , detailed POS, Dep, Head, Head POS and vec_norm. Then we transformed the POS, Det_POS, Des and Head_POS features into dummies and we created a feature ‘Sent_id’ containing the ID of each sentence for the initial dataframe and the dependency parsing data frame. 
This model was then merged with the BOW data frame . We merged them on multiple keys - sent_id and term - and then dropped those so our final dataset contains words features from the BOW model, the initial features and the features from the dependency parsing. We chose not to keep this model including dependency parsing because of some reasons explained in the following section so our final feature set is a Bag of Words model with some initial features. 



	b) Type of classification model

Given that we face a multi class classification problem where the goal is to predict if a term is positive, negative or neutral,  we should explore the capabilities of classification algorithms. We chose to implement the following models:
1- Logistic Regression: Logistic regression belongs to the same family as linear regression, or the generalized linear models. In both cases its main aim is to link an event to a linear combination of features. However, for logistic regression we assume that the target variable is following a Binomial distribution. 
2- Random Forests (RF)  : Random Forest is an ensemble machine learning method, that works by developing a multitude of decision trees. The aim is to make decision trees more independent by adding randomness in features choice. For each tree, it firstly draws a bootstrap sample, obtained by repeatedly sampling observations from the original sample with replacement. Then, the tree is estimated on that sample by applying feature randomization. This means, that searching for the  optimal node is preceded by a random sampling of a subset of predictors. Finally, the result may either be an average or weighted average of all the terminal nodes that are reached, or, in the case of categorical variables, a voting majority.
3- Neural Networks (NNET) :  A neural network is an association into a graph, complex, of artificial neurons. Neural networks are distinguished by their architecture (layers, complete), their level of complexity (number of neurons) and their activation function. An artificial neuron is a mathematical function conceived as a model of biological neurons. It is characterized by an  internal state, input signals and an activation function which is operating a transformation of an affine combination of input signals. This affine combination is defined by a weight vector associated to the neuron and for which the values are estimating during the training part. For this project, we used a multilayer perceptron. This is a network composed of successive layers which are ensembles of neurons with no connection between them.
For each of the previous classification models, we perform parameter tuning to make a better fit. At the end, the selected model is the one that shows the best train and test accuracies.
After running the models, we have noticed that adding the parsing doesn’t improve the accuracy on the test dataset, so we have decided to remove it (the code is still in the files the call to the function has been commented) in order to save some computational time. 
The model that gave the best result after parameters tuning is the logistic regression with C = 0.5.


3. Accuracy that we get on the dev dataset :
The obtained accuracy is : ACCURACY: 77.39
