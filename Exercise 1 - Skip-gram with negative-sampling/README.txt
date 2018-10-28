{\rtf1\ansi\ansicpg1252\cocoartf1561\cocoasubrtf200
{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 HelveticaNeue;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red0\green0\blue0;\red58\green62\blue68;
}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;\csgray\c0\c0;\cssrgb\c29412\c30980\c33725;
}
\paperw11900\paperh16840\margl1440\margr1440\vieww10600\viewh12420\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf2 \cb3 SkipGram with Negative Sampling\
____________________\
Included Files\
____________________\
\
main.2.py\
README.txt\
\
____________________\
Train/test the data\
____________________\
\
Train: 
\f1 \expnd0\expndtw0\kerning0
python main.2.py --text testfile.txt --model Model\
Test: python main.2.py --text data.csv --model Model --test\
\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0 \cf2 \kerning1\expnd0\expndtw0 ____________________\
Specifications\
____________________\
\
The vocabulary dictionary is built with the stemmed version of the words, for both the training and the evaluation.\
Thus, the similarity is also computed with the stemmed words.\
If you wish to compute the similarity with the complete words, you can comment out the stemming part in both the train and test.\
\
If your test file is only composed of stemmed words, the line of code MUST BE commented out. Otherwise the stemmed words will be stemmed again, resulting in unknown words in the vocabulary.\
 }