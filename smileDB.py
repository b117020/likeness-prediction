# -*- coding: utf-8 -*-
"""

@author: Devdarshan
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes
from sklearn.metrics import accuracy_score
import pickle
#read data
train = pd.read_csv('hm_train.csv', error_bad_lines=False)
Test_data = pd.read_csv('hm_test.csv')
train = train.fillna("no info")
train.head()
Test_X = Test_data['cleaned_hm']
Test_id = Test_data['hmid']
#X_train, y_train = train['Issue'].values, train['Product'].values

#train test split
Train_X = train['cleaned_hm']

#Train_X = train.iloc[:, 2:3].values
Train_Y = train.iloc[:, 4].values

train_X, test_X, train_Y, test_Y = model_selection.train_test_split(train['cleaned_hm'],train['predicted_category'],test_size=0.3)

#encoding
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
train_Y = Encoder.fit_transform(train_Y)
test_Y = Encoder.fit_transform(test_Y)

#vectorisation
Tfidf_vect = TfidfVectorizer(max_features=2600)
Tfidf_vect.fit(train['cleaned_hm'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
train_X_Tfidf = Tfidf_vect.transform(train_X)
test_X_Tfidf = Tfidf_vect.transform(test_X)

# fit the training dataset on the classifier  
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)


'''with open('model.pkl', 'wb') as handle:
    pickle.dump(Naive, handle, pickle.HIGHEST_PROTOCOL)
with open('model.pkl', 'rb') as handle:
    model1 = pickle.load(handle)  

'''
predictions_NB = Naive.predict(Test_X_Tfidf)
predictions = Naive.predict(test_X_Tfidf)


#predictions_NB = model1.predict(Test_X_Tfidf)
#inverse the encoded words
reversed = Encoder.inverse_transform(predictions_NB)

new_series = pd.Series(reversed)
new_series = new_series.to_frame('predicted_category')
final_result = pd.concat([Test_id, new_series], axis=1)
final_result.to_csv('test.csv') 
#print(reversed)
#accuracy_score function to get the accuracy
rev_test = Encoder.inverse_transform(predictions)
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions, test_Y)*100)



