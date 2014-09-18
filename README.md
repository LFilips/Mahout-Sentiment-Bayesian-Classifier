SentimentBayesianClassifier
===========================


Repository that implement Sentiment Analysis using a Naive Bayes Classifier, using Apache Mahout 0.9.

The Bayesian Classifier need a Dictionary file, in which each word has a sentiment associated, this is an example:


The example is in italian language:

positivo,bello     
positivo,intelligente     
negativo,brutto     
neutrale,sedia     

This is the same in english:

positive,good     
positive,nice     
positive,smart     
negative,ugly     
negative,bad     
neutral,chair     


These word will we used for training a model, and then used to classify new text.

The project uses term frequency vector for represent text, this is better in case of Sentiment Analysis,
for a simple label classification, like spam filter, tf-idf could be better.
