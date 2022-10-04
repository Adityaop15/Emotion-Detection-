# Adity sharma

# [EMOTION PREDICTION ON TWITTER DATA USING NLP] 
* Emotion Predications an increasingly popular method of natural language processing. In this article, you will learn how to perform Twitter Sentiment Analysis using Python programming ,ML,DNN and BERT.
* The dataset used is a Twitter dataset that was downloaded from various sources. The tweets include the user’s name, the text tweet and emoticons. Emoji’s are not included in the tweet. The dataset has 113300 tweets containing three columns. The first column contains the Tweet comments, the second contains the sentiment score and the third column contains the actual tweet. The emotion score takes 10 values. A score of 0:Anger,1:Criticism, 2:Fear, 3:Hate, 4:Joy, 5:Love, 6:Offensive, 7:Optimism, 8:Sadness, 9:Surprise, I gave emotions. 
* Built a client facing API using steamlit

![](/images/1_7kxtIjXsYeFHy4GcglUo-w.png)


# [Approach & Methodology/Techniques:]
The approach In this section, we will present the dataset used, and our methodologies to recognize emotions of English tweets using :
1.	ML: (Support Vector Machines, Decision Tree, Logistic Regression, Multinomial Naïve Bayes)
2.	RNN: Neural Network where the output from the previous step is fed as input to the current step
3.	LSTM: LSTM stands for Long-Short Term Memory. LSTM is a type of recurrent neural network but is better than traditional recurrent neural networks in terms of memory.
4.	BERT: BERT stands for Bidirectional Encoder Representations from Transformers and it is a state-of-the-art machine learning model used for NLP tasks

![](/images/matrix_results.png)

# [Data Preprocessing:]
1.Data Preprocessing Pre-processing is the set of transformations applied to the data before doing the actual analytics. 
2.Data collected from various sources exist in raw format which needs to be processed before it is analyzed
3.Removal of punctuation and HTML tags,Tokenization,Stop-word removal,tdfi and counter vector.


![](/images/matrix_results.png)

# [Model Description:]
Proposed Models for Tweets Sentiment Classification:

•	Support Vector Machines (SVM)
•	Decision Tree (DT)
•	Logistic Regression (LR)
•	Stochastic Gradient Descent (SGD)
•	Multinomial Naïve Bayes(MNB)
•	recurrent Neural Network(RNN)
•	BI-LSTM
•	A-LSTM 
•	BERT



![](/images/matrix_results.png)



!# [Accuracy:]

As far as the accuracy of the model is concerned BERT better than the LSTM model which in turn performs better than RNN and other ML models. BERT has shown the accuracy up to 76.00% which when compared to Bi-LSTM which has an accuracy of 71.00% is quiet higher in the field of Machine Learning.

Classification Report

BERT shows the best results for the emotion criticism with the precision score of 0.76, Bi-LSTM-0.71 and 0.68 for the ML model respectively


![](/images/matrix_results.png)






