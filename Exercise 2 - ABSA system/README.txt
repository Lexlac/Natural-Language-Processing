{\rtf1\ansi\ansicpg1252\cocoartf1504\cocoasubrtf760
{\fonttbl\f0\froman\fcharset0 Times-Roman;\f1\fnil\fcharset0 HelveticaNeue;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red237\green236\blue236;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;\cssrgb\c94510\c94118\c94118;}
\paperw11900\paperh16840\margl1440\margr1440\vieww25400\viewh13620\viewkind0
\deftab720
\pard\pardeftab720\sl280\partightenfactor0

\f0\fs24 \cf2 \expnd0\expndtw0\kerning0
NLP Exercise 2: ABSA system\
\
1. Students who contributed to the deliverable:\
\
	- Rafa\'eblle Aygalenq\
	- Sarah Lina Hammoutene\
	- Dora Linda Kocsis\
	- No\'e9mie Qu\'e9r\'e9\
\
2. Description of our final system\
\
	a) Feature representation\
\
We first implemented a pre-processing function which will be applied to both training data and testing data. This function is composed of several tasks:\
	- transforming categorical features : the levels of the \'91Polarity\'92 feature, which are \{\'91positive\'92,\'92neutral\'92,\'92negative\'92\}, are turned into numerical levels \{1,0,-1\}. The \'91Aspect category\'92 feature is transformed into dummies.\
	- concerning the feature \'91Text\'92 : we removed the punctuation, put all words in lowercase, tokenized each sentence, remove stopwords, stemmed each word to keep only the root and reduce the future number of features and then we detokenize all sentences.\
\
Then we implemented a Bag-Of-Words model based on the feature \'91Text\'92 which contains the pre-processed sentences. Vocabulary made of every word present among all sentences in the training file is used as features and we put the value 1 when a word in present in the initial sentence, 0 otherwise. At the end we join the dataset resulting of the BOW model and the initial one to obtain a dataset with all features (words and initial ones). All of these tasks are applied on the training set and on the test set in order at the end to have the exact same features for both. A small verification of the features\'92 exact matching is made in order not to have a problem during the modelling part (especially for the creation of the dummies of the dependency parsing, all the different tags may not be present in both datasets so we just make sure that at the end we have the same features in both train and test datasets). \
\
We also tried a system with a different set of features.We also chose to apply dependency parsing using spaCy on the feature \'91Text\'92 containing the sentences. We selected several tokens including part-of-speech tagging , detailed POS, Dep, Head, Head POS and vec_norm. Then we transformed the POS, Det_POS, Des and Head_POS features into dummies and we created a feature \'91Sent_id\'92 containing the ID of each sentence for the initial dataframe and the dependency parsing data frame. \
This model was then merged with the BOW data frame . We merged them on multiple keys - sent_id and term - and then dropped those so our final dataset contains words features from the BOW model, the initial features and the features from the dependency parsing. We chose not to keep this model including dependency parsing because of some reasons explained in the following section so our final feature set is a Bag of Words model with some initial features. \
\
\
\
	b) Type of classification model\
\
Given that we face a multi class classification problem where the goal is to predict if a term is positive, negative or neutral,  we should explore the capabilities of classification algorithms. We chose to implement the following models:\
1- Logistic Regression: Logistic regression belongs to the same family as linear regression, or the generalized linear models. In both cases its main aim is to link an event to a linear combination of features. However, for logistic regression we assume that the target variable is following a Binomial distribution. \
2- Random Forests (RF)  : Random Forest is an ensemble machine learning method, that works by developing a multitude of decision trees. The aim is to make decision trees more independent by adding randomness in features choice. For each tree, it firstly draws a bootstrap sample, obtained by repeatedly sampling observations from the original sample with replacement. Then, the tree is estimated on that sample by applying feature randomization. This means, that searching for the  optimal node is preceded by a random sampling of a subset of predictors. Finally, the result may either be an average or weighted average of all the terminal nodes that are reached, or, in the case of categorical variables, a voting majority.\
3- Neural Networks (NNET) :  A neural network is an association into a graph, complex, of artificial neurons. Neural networks are distinguished by their architecture (layers, complete), their level of complexity (number of neurons) and their activation function. An artificial neuron is a mathematical function conceived as a model of biological neurons. It is characterized by an  internal state, input signals and an activation function which is operating a transformation of an affine combination of input signals. This affine combination is defined by a weight vector associated to the neuron and for which the values are estimating during the training part. For this project, we used a multilayer perceptron. This is a network composed of successive layers which are ensembles of neurons with no connection between them.\
For each of the previous classification models, we perform parameter tuning to make a better fit. At the end, the selected model is the one that shows the best train and test accuracies.\
After running the models, we have noticed that adding the parsing doesn\'92t improve the accuracy on the test dataset, so we have decided to remove it (the code is still in the files the call to the function has been commented) in order to save some computational time. \
The model that gave the best result after parameters tuning is the logistic regression with C = 0.5.\
\
\
3. Accuracy that we get on the dev dataset :\
The obtained accuracy is : 
\f1\fs28 \cb3 ACCURACY: 77.39}