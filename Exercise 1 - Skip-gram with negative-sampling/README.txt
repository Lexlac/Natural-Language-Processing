SkipGram with Negative Sampling
____________________
Included Files
____________________

main.2.py
README.txt

____________________
Train/test the data
____________________

Train: python main.2.py --text testfile.txt --model Model
Test: python main.2.py --text data.csv --model Model --test

____________________
Specifications
____________________

The vocabulary dictionary is built with the stemmed version of the words, for both the training and the evaluation.
Thus, the similarity is also computed with the stemmed words.
If you wish to compute the similarity with the complete words, you can comment out the stemming part in both the train and test.

If your test file is only composed of stemmed words, the line of code MUST BE commented out. Otherwise the stemmed words will be stemmed again, resulting in unknown words in the vocabulary.
