# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:39:52 2019

@author: Devdarshan
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
np.random.seed(32)


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Conv1D, MaxPooling1D, Dropout
from keras.utils.np_utils import to_categorical

df = pd.read_csv("hm_train.csv")

train_text, test_text, train_y, test_y = train_test_split(df['cleaned_hm'],df['predicted_category'],test_size = 0.3)

MAX_NB_WORDS = 20000

# get the raw text data
texts_train = train_text.astype(str)
texts_test = test_text.astype(str)

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, char_level=False)
tokenizer.fit_on_texts(train_text)
sequences = tokenizer.texts_to_sequences(texts_train)
sequences_test = tokenizer.texts_to_sequences(texts_test)
#Y_sequences = tokenizer.texts_to_sequences(train_y)
#Y_sequences_test = tokenizer.texts_to_sequences(test_y)

Encoder = LabelEncoder()
train_Y = Encoder.fit_transform(train_y)
test_Y = Encoder.fit_transform(test_y)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

type(tokenizer.word_index), len(tokenizer.word_index)

index_to_word = dict((i, w) for w, i in tokenizer.word_index.items())

" ".join([index_to_word[i] for i in sequences[0]])

seq_lens = [len(s) for s in sequences]
print("average length: %0.1f" % np.mean(seq_lens))
print("max length: %d" % max(seq_lens))

MAX_SEQUENCE_LENGTH = 150

# pad sequences with 0s
x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', x_train.shape)
print('Shape of data test tensor:', x_test.shape)

y_train = train_Y
y_test = test_Y

y_train = to_categorical(np.asarray(y_train))
print('Shape of label tensor:', y_train.shape)



from keras.layers import Dense, Input, Flatten
from keras.layers import GlobalAveragePooling1D, Embedding
from keras.models import Model

EMBEDDING_DIM = 50
N_CLASSES = 7

# input: a sequence of MAX_SEQUENCE_LENGTH integers
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
embedded_sequences = embedding_layer(sequence_input)

average = GlobalAveragePooling1D()(embedded_sequences)
predictions = Dense(N_CLASSES, activation='sigmoid')(average)

model = Model(sequence_input, predictions)
model.compile(loss='binary_crossentropy',
              optimizer='adam')

model.fit(x_train, y_train, validation_split=0.1,
          nb_epoch=10, batch_size=128)

output_test = model.predict(x_test)
result = []
result = [[max(i)] for i in output_test]
for i in enumerate(output_test):
    for e in range(7):
        if output_test[i][e] == result[i] :
            reversed[i] = Encoder.inverse_transform(output_test[i][e])
        else:
            output_test[e][f] = 0
    
reversed = Encoder.inverse_transform(output_test)

new_series = pd.Series(reversed)
new_series = new_series.to_frame('predicted_category')
final_result = pd.concat([Test_id, new_series], axis=1)
final_result.to_csv('test.csv') 
print("test auc:", roc_auc_score(y_test,output_test[:,1]))
print("Accuracy Score -> ",accuracy_score(output_test[:,1], y_test)*100)